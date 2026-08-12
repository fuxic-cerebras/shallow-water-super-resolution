"""Processed-layer field transforms: destaggering and vector-aware augmentation (D-04).

Everything here operates on **storage order** `[..., channel, y, x]` with channels
`[eta, u, v]`, which is what the HDF5 layer holds. The solver's own `[x, y]` order stops at
`swe_sr.solver.runner`; `tests/data/test_processing.py` cross-checks these against the
solver-order implementations in `swe_sr.solver.diagnostics` so the two cannot drift.

Destaggering (D011) moves `u` and `v` from cell faces to cell centers, so all three
channels share the `eta` coordinate grid and endpoint-aligned interpolation becomes exact
for every channel.

Augmentation is a more delicate story than `docs/DATASET.md` suggests -- see D018 and
`augmentation_symmetry_error` below. The transforms are implemented correctly and are
geometrically exact on cell-centered fields, but they are **not** symmetries of this
solver's dynamics, so they are disabled by default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

ETA, U, V = 0, 1, 2


# -- Destaggering (D011) --------------------------------------------------------------


def destagger(fields: np.ndarray) -> np.ndarray:
    """Move `u` and `v` from cell faces to cell centers, in storage order.

    Accepts any leading batch/time dimensions: the last three axes must be
    `(channel, y, x)`. `u` is averaged along x and `v` along y, because `u[.., y, i]` is
    `u_{i+1/2, y}` on an east face and `v[.., j, x]` is `v_{x, j+1/2}` on a north face.
    The wall-adjacent column and row take half the single stored face value, the west and
    south wall faces being zero and never stored.

    Returns a new array; `eta` passes through untouched since it is already cell-centered.
    """
    if fields.shape[-3] != 3:
        raise ValueError(f"expected 3 channels [eta, u, v] on axis -3, got shape {fields.shape}")
    centered = np.array(fields, copy=True)

    u = fields[..., U, :, :]
    centered[..., U, :, 1:] = 0.5 * (u[..., :, :-1] + u[..., :, 1:])
    centered[..., U, :, 0] = 0.5 * u[..., :, 0]

    v = fields[..., V, :, :]
    centered[..., V, 1:, :] = 0.5 * (v[..., :-1, :] + v[..., 1:, :])
    centered[..., V, 0, :] = 0.5 * v[..., 0, :]

    return centered


# -- Vector-aware augmentation --------------------------------------------------------
#
# These act on cell-centered fields, where they are geometrically exact: the grid is
# uniform and square, so flipping an axis or swapping axes maps grid points onto grid
# points. What they are NOT is dynamically valid for this solver (D018).


def reflect_x(fields: np.ndarray) -> np.ndarray:
    """Mirror across the y axis. `u` is a normal component under this reflection, so it
    changes sign; `eta` and `v` do not."""
    flipped = np.array(fields[..., :, :, ::-1], copy=True)
    flipped[..., U, :, :] *= -1.0
    return flipped


def reflect_y(fields: np.ndarray) -> np.ndarray:
    """Mirror across the x axis; `v` changes sign."""
    flipped = np.array(fields[..., :, ::-1, :], copy=True)
    flipped[..., V, :, :] *= -1.0
    return flipped


def transpose_xy(fields: np.ndarray) -> np.ndarray:
    """Swap the x and y axes, which also swaps `u` with `v`.

    Forgetting the channel swap is the classic error here: the array would look plausible
    while every velocity vector pointed the wrong way. `docs/DATASET.md` requires a unit
    test before this is used, and `tests/data/test_processing.py` provides it.
    """
    swapped = np.swapaxes(fields, -1, -2)
    return np.stack(
        [swapped[..., ETA, :, :], swapped[..., V, :, :], swapped[..., U, :, :]],
        axis=-3,
    )


def identity(fields: np.ndarray) -> np.ndarray:
    return np.array(fields, copy=True)


AUGMENTATIONS = {
    "identity": identity,
    "reflect_x": reflect_x,
    "reflect_y": reflect_y,
    "transpose": transpose_xy,
}

# The eight-element dihedral group would be the natural full set, but only these are named
# by docs/DATASET.md, and D018 explains why none is enabled by default.
DEFAULT_AUGMENTATIONS: tuple[str, ...] = ()


@dataclass(frozen=True)
class AugmentationPolicy:
    """Which augmentations to apply, and how.

    A single transform is drawn per *sample pair* and applied to both the coarse and the
    fine member, so a pair stays consistent. Applying different transforms to the two
    members would silently destroy the correspondence the model is trained on.
    """

    names: tuple[str, ...] = DEFAULT_AUGMENTATIONS

    def __post_init__(self) -> None:
        unknown = set(self.names) - set(AUGMENTATIONS)
        if unknown:
            raise ValueError(
                f"unknown augmentation(s) {sorted(unknown)}; available: {sorted(AUGMENTATIONS)}"
            )

    @property
    def enabled(self) -> bool:
        return bool(self.names)

    def draw(self, rng: np.random.Generator) -> str:
        """Pick one transform, including the identity, so some samples stay untransformed."""
        if not self.names:
            return "identity"
        choices = ("identity", *self.names)
        return str(choices[int(rng.integers(len(choices)))])

    def apply(self, name: str, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
        transform = AUGMENTATIONS[name]
        return tuple(transform(array) for array in arrays)


def augmentation_symmetry_error(
    name: str,
    nodes: int = 64,
    steps: int = 400,
    *,
    use_coriolis: bool = True,
    use_beta: bool = True,
) -> float:
    """Measure how far an augmentation is from being a symmetry of the solver.

    Evolves a state, then compares transforming-then-evolving against
    evolving-then-transforming. Zero means the transform commutes with the dynamics and the
    augmentation generates states the solver could genuinely produce.

    This is the check `docs/DATASET.md` asks for, promoted to library code because its
    result is a scientific finding rather than a test detail. Measured values are recorded
    in D018: with the project's rotating beta-plane every transform lands near 0.9, and
    with rotation disabled the reflections still land near 0.05 because the C-grid
    staggering is itself not reflection-symmetric.

    Returns the maximum absolute discrepancy relative to the evolved field's peak
    amplitude.
    """
    # Imported here rather than at module scope: the solver depends on nothing in the data
    # layer, and this keeps that direction of dependency one-way for everything else.
    from swe_sr.solver.config import SolverConfig
    from swe_sr.solver.initial_conditions import GaussianBump, GaussianBumpsIC
    from swe_sr.solver.runner import solve

    transform = AUGMENTATIONS[name]
    config = SolverConfig(n_x=nodes, n_y=nodes, use_coriolis=use_coriolis, use_beta=use_beta)
    specification = GaussianBumpsIC(
        bumps=(
            GaussianBump(1.3, 1.9e5, 2.2e5, 8.0e4),
            GaussianBump(-1.0, -1.5e5, -1.1e5, 9.0e4),
        )
    )

    baseline = solve(config, specification, sample_steps=[0, steps])
    initial, evolved = baseline.fields[0], baseline.fields[-1]
    scale = float(np.abs(evolved).max())
    if scale == 0.0:
        raise RuntimeError("evolved field is identically zero; cannot form a relative error")

    class _StoredIC:
        """Feeds a prepared array back through the solver's IC interface."""

        def __init__(self, storage_order_eta: np.ndarray) -> None:
            self._eta = storage_order_eta

        def evaluate(self, _config: SolverConfig) -> np.ndarray:
            return np.asarray(self._eta.T)  # storage [y, x] -> solver [x, y]

    transformed_initial = transform(initial)[ETA]
    evolved_from_transformed = solve(
        config,
        _StoredIC(transformed_initial),  # type: ignore[arg-type]
        sample_steps=[steps],
    ).fields[-1]

    return float(np.abs(evolved_from_transformed - transform(evolved)).max() / scale)


def channel_names() -> Sequence[str]:
    return ("eta", "u", "v")
