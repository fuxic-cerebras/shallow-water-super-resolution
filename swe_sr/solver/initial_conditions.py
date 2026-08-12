"""Analytic initial conditions, evaluated directly on whichever grid is asked for.

This is the mechanism behind D002: the same analytic specification is evaluated on the
coarse and the fine coordinates independently, so neither state is ever obtained by
resizing the other. Specifications are plain data so they serialize into the immutable
IC registry and reproduce exactly.

Arrays are returned in the solver's `[x, y]` order.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from swe_sr.solver.config import SolverConfig

# Sampling ranges from docs/DATASET.md.
BUMP_COUNT_CHOICES = (1, 2, 3)
AMPLITUDE_RANGE_M = (0.5, 1.5)
WIDTH_RANGE_M = (65.0e3, 120.0e3)
WALL_MARGIN_SIGMAS = 2.0


@dataclass(frozen=True)
class GaussianBump:
    """One Gaussian surface perturbation."""

    amplitude: float
    center_x: float
    center_y: float
    width: float

    def evaluate(self, mesh_x: np.ndarray, mesh_y: np.ndarray) -> np.ndarray:
        """Evaluate the bump on the given mesh.

        The two axis terms are each divided by `2 sigma^2` before being summed, rather
        than summed and then divided. That is algebraically identical but not
        bit-identical, and it is the form the reference script uses at swe.py:160, so
        matching it keeps single-bump parity exact to the last bit.
        """
        denominator = 2 * self.width**2
        exponent_x = (mesh_x - self.center_x) ** 2 / denominator
        exponent_y = (mesh_y - self.center_y) ** 2 / denominator
        return np.asarray(self.amplitude * np.exp(-(exponent_x + exponent_y)))


@dataclass(frozen=True)
class GaussianBumpsIC:
    """A sum of one to three Gaussian bumps -- the in-distribution training family."""

    bumps: tuple[GaussianBump, ...]
    family: str = "gaussian_bumps"

    def evaluate(self, config: SolverConfig) -> np.ndarray:
        mesh_x, mesh_y = config.mesh
        eta = np.zeros_like(mesh_x)
        for bump in self.bumps:
            eta += bump.evaluate(mesh_x, mesh_y)
        return eta

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "bumps": [asdict(b) for b in self.bumps]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GaussianBumpsIC:
        return cls(bumps=tuple(GaussianBump(**b) for b in payload["bumps"]))

    def resolvability(self, config: SolverConfig) -> dict[str, float]:
        """`sigma/dx` diagnostics, which docs/DATASET.md requires recording per grid.

        Below roughly 2 the bump is barely represented on the grid, which is the regime
        the coarse member of the primary pair deliberately sits in.
        """
        spacing = min(config.dx, config.dy)
        ratios = [bump.width / spacing for bump in self.bumps]
        return {
            "min_sigma_over_dx": float(min(ratios)),
            "max_sigma_over_dx": float(max(ratios)),
            "grid_spacing_m": float(spacing),
        }


@dataclass(frozen=True)
class RingIC:
    """An annular perturbation with zero initial velocity.

    Deliberately outside the Gaussian training family; used only for the `ring_ood`
    evaluation workload, which is generated after model selection.
    """

    amplitude: float
    radius: float
    width: float
    center_x: float = 0.0
    center_y: float = 0.0
    family: str = "ring"

    def evaluate(self, config: SolverConfig) -> np.ndarray:
        mesh_x, mesh_y = config.mesh
        radial = np.sqrt((mesh_x - self.center_x) ** 2 + (mesh_y - self.center_y) ** 2)
        return np.asarray(
            self.amplitude * np.exp(-((radial - self.radius) ** 2) / (2 * self.width**2))
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RingIC:
        fields = {k: v for k, v in payload.items() if k != "family"}
        return cls(**fields)


InitialCondition = GaussianBumpsIC | RingIC


def initial_condition_from_dict(payload: dict[str, Any]) -> InitialCondition:
    """Rebuild an initial condition from its registry entry."""
    family = payload.get("family")
    if family == "gaussian_bumps":
        return GaussianBumpsIC.from_dict(payload)
    if family == "ring":
        return RingIC.from_dict(payload)
    raise ValueError(f"unknown initial-condition family {family!r}")


class InitialConditionRejected(ValueError):
    """A drawn specification failed a validity check and must be redrawn."""


def _check_admissible(candidate: GaussianBumpsIC, config: SolverConfig) -> None:
    """Apply the rejection rules from docs/DATASET.md.

    Rejects a candidate whose bumps come within `2 sigma` of a wall, whose total depth is
    non-positive anywhere, or whose evaluated field is not finite.
    """
    half_x = config.length_x / 2
    half_y = config.length_y / 2
    for bump in candidate.bumps:
        margin = WALL_MARGIN_SIGMAS * bump.width
        if abs(bump.center_x) + margin > half_x or abs(bump.center_y) + margin > half_y:
            raise InitialConditionRejected(
                f"bump at ({bump.center_x:.3e}, {bump.center_y:.3e}) with sigma "
                f"{bump.width:.3e} violates the {WALL_MARGIN_SIGMAS} sigma wall margin"
            )

    eta = candidate.evaluate(config)
    if not np.all(np.isfinite(eta)):
        raise InitialConditionRejected("evaluated initial elevation is not finite")
    if float((config.depth + eta).min()) <= 0.0:
        raise InitialConditionRejected(
            f"total depth H + eta reaches {float((config.depth + eta).min()):.3f} m, "
            "which is not positive"
        )


def sample_gaussian_bumps_ic(
    seed: int,
    config: SolverConfig,
    *,
    max_attempts: int = 100,
) -> GaussianBumpsIC:
    """Draw one admissible initial condition for `seed`, redrawing on rejection.

    The draw is a pure function of `seed`, so the registry reproduces exactly. `config`
    only supplies the domain and resting depth for the admissibility checks; because the
    accepted specification is analytic, it can then be evaluated on any grid.
    """
    rng = np.random.default_rng(seed)
    half_x = config.length_x / 2
    half_y = config.length_y / 2

    for _ in range(max_attempts):
        count = int(rng.choice(BUMP_COUNT_CHOICES))
        bumps = []
        for _ in range(count):
            width = float(rng.uniform(*WIDTH_RANGE_M))
            magnitude = float(rng.uniform(*AMPLITUDE_RANGE_M))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            margin = WALL_MARGIN_SIGMAS * width
            # Draw centers inside the admissible box so rejection is rare rather than
            # the primary sampling mechanism.
            limit_x = half_x - margin
            limit_y = half_y - margin
            if limit_x <= 0 or limit_y <= 0:
                raise ValueError(
                    f"width {width:.3e} m cannot satisfy a {WALL_MARGIN_SIGMAS} sigma "
                    f"margin in a {config.length_x:.3e} m domain"
                )
            bumps.append(
                GaussianBump(
                    amplitude=sign * magnitude,
                    center_x=float(rng.uniform(-limit_x, limit_x)),
                    center_y=float(rng.uniform(-limit_y, limit_y)),
                    width=width,
                )
            )
        candidate = GaussianBumpsIC(bumps=tuple(bumps))
        try:
            _check_admissible(candidate, config)
        except InitialConditionRejected:
            continue
        return candidate

    raise RuntimeError(
        f"could not draw an admissible initial condition for seed {seed} in {max_attempts} attempts"
    )


def shipped_demo_ic(config: SolverConfig) -> GaussianBumpsIC:
    """The reference script's own initial condition (swe.py:160).

    Provided so parity tests can exercise the exact shipped case. Note its sigma of 50 km
    is narrower than this project's 65-120 km sampling range.
    """
    return GaussianBumpsIC(
        bumps=(
            GaussianBump(
                amplitude=1.0,
                center_x=config.length_x / 2.7,
                center_y=config.length_y / 4,
                width=0.05e6,
            ),
        )
    )
