"""Headless solver entry point.

`docs/ARCHITECTURE.md` requires the refactored solver to "expose pure data, not plotting
side effects", and requires that generation "must not accumulate every internal time
step". This module is the only place that converts the solver's `[x, y]` working order
into the stored `[time, channel, y, x]` order (D016).

    result = solve(config, initial_condition, sample_steps)
    result.fields.shape == (len(sample_steps), 3, n_y, n_x)   # channels are [eta, u, v]

Nothing here imports matplotlib; `tests/solver/test_headless.py` enforces that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from swe_sr.solver.config import SolverConfig
from swe_sr.solver.diagnostics import DiagnosticsAccumulator, RunDiagnostics
from swe_sr.solver.initial_conditions import InitialCondition
from swe_sr.solver.numerics import ShallowWaterKernel, SolverState

CHANNEL_NAMES = ("eta", "u", "v")


@dataclass(frozen=True)
class SolveResult:
    """Sampled trajectory in storage order, plus provenance and diagnostics.

    `fields` is `[time, channel, y, x]` with channels `[eta, u, v]`. These are the raw
    staggered velocities; destaggering to cell centers happens in the processed layer
    (D011), not here.
    """

    fields: np.ndarray
    times: np.ndarray
    sample_steps: np.ndarray
    config: SolverConfig
    diagnostics: RunDiagnostics

    def __post_init__(self) -> None:
        expected = (len(self.sample_steps), 3, self.config.n_y, self.config.n_x)
        if self.fields.shape != expected:
            raise ValueError(f"fields shape {self.fields.shape} does not match {expected}")

    @property
    def eta(self) -> np.ndarray:
        return self.fields[:, 0]

    @property
    def x(self) -> np.ndarray:
        return self.config.x

    @property
    def y(self) -> np.ndarray:
        return self.config.y


def _to_storage_order(state: SolverState) -> np.ndarray:
    """Stack `[eta, u, v]` and transpose `[x, y] -> [y, x]`.

    The solver indexes `A[i, j]` as `(x_i, y_j)` (MODEL_NOTES.md section 3) while storage
    is `[..., y, x]` (D016). This single transpose is the entire bridge between the two
    conventions; `tests/solver/test_orientation.py` pins it.
    """
    return np.stack([state.eta.T, state.u.T, state.v.T], axis=0)


def solve(
    config: SolverConfig,
    initial_condition: InitialCondition,
    sample_steps: Sequence[int] | np.ndarray,
    *,
    diagnostic_stride: int = 1,
) -> SolveResult:
    """Integrate the shallow-water equations and return the requested snapshots.

    Args:
        config: fully resolved solver configuration.
        initial_condition: analytic specification, evaluated on `config`'s own grid so no
            state is ever derived by resizing another (D002).
        sample_steps: step indices to record, where 0 is the initial state. Must be
            sorted, unique, and non-negative. Only these snapshots are held in memory.
        diagnostic_stride: evaluate diagnostics every this many steps. 1 checks every
            step; larger values trade diagnostic resolution for speed on long runs.

    Returns:
        A `SolveResult` whose `fields` are `[time, 3, n_y, n_x]` in `[eta, u, v]` order.
    """
    steps = np.asarray(sample_steps, dtype=np.int64)
    if steps.ndim != 1 or steps.size == 0:
        raise ValueError("sample_steps must be a non-empty 1-D sequence")
    if np.any(steps < 0):
        raise ValueError("sample_steps must be non-negative")
    if np.any(np.diff(steps) <= 0):
        raise ValueError("sample_steps must be strictly increasing and unique")
    if diagnostic_stride < 1:
        raise ValueError(f"diagnostic_stride must be >= 1, got {diagnostic_stride}")

    eta_initial = initial_condition.evaluate(config)
    state = SolverState.from_initial_eta(eta_initial)
    kernel = ShallowWaterKernel(config)
    accumulator = DiagnosticsAccumulator(config, state)

    total_steps = int(steps[-1])
    # Preallocate exactly the requested snapshots; never the whole trajectory.
    fields = np.empty((steps.size, 3, config.n_y, config.n_x), dtype=np.float64)

    wanted = set(int(s) for s in steps)
    index_of = {int(s): i for i, s in enumerate(steps)}

    if 0 in wanted:
        fields[index_of[0]] = _to_storage_order(state)

    for step in range(1, total_steps + 1):
        kernel.step(state)
        if step % diagnostic_stride == 0 or step == total_steps:
            accumulator.update(state, step_index=step)
        if step in wanted:
            fields[index_of[step]] = _to_storage_order(state)

    diagnostics = accumulator.finalize(state)
    times = steps.astype(np.float64) * config.dt

    return SolveResult(
        fields=fields,
        times=times,
        sample_steps=steps,
        config=config,
        diagnostics=diagnostics,
    )


def sample_schedule(discard_steps: int, stride: int, count: int) -> np.ndarray:
    """Build the saved-step schedule for a trajectory.

    docs/DATASET.md specifies a spin-up discard followed by evenly spaced saves: the
    primary pair nominally discards 288 steps and saves every 24, the backup discards 576
    and saves every 48, for 128 snapshots each. Returning explicit integer step indices
    keeps physical times exact multiples of `dt` rather than accumulated rounded
    intervals.
    """
    if discard_steps < 0:
        raise ValueError(f"discard_steps must be non-negative, got {discard_steps}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    return discard_steps + stride * np.arange(count, dtype=np.int64)
