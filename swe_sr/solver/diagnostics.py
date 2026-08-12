"""Physical diagnostics for solver runs (task P-03).

These check that a run is physically admissible, independently of any machine-learning
concern. The conventions are fixed by D014:

    M = sum_ij eta * dx * dy
    E = sum_ij [ 0.5 * (H + eta) * (u_c^2 + v_c^2) + 0.5 * g * eta^2 ] * dx * dy

Mass is conserved to roundoff by construction -- the nonlinear upwind continuity equation
makes the closed-basin flux budget exact -- so a mass drift above roundoff is a real bug.
Energy is *not* conserved by this scheme, so its drift is reported, never asserted.

Per D012 the exact wall checks belong on raw staggered arrays, which is what this module
receives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from swe_sr.solver.config import SolverConfig
from swe_sr.solver.numerics import SolverState

# Mass is exact to roundoff; this bound accommodates accumulation over ~10^4 steps.
MASS_DRIFT_TOLERANCE = 1e-12


def total_mass(eta: np.ndarray, config: SolverConfig) -> float:
    """Domain-integrated elevation anomaly, the varying part of `sum(H + eta)` (D014)."""
    return float(eta.sum() * config.dx * config.dy)


def destagger_u(u_staggered: np.ndarray) -> np.ndarray:
    """Move `u` from east faces to cell centers along x (D011).

    `u[i]` lives at `x_{i+1/2}`, so the cell-centered value averages the two bracketing
    faces. The west face of column 0 is the wall, where `u = 0` and nothing is stored, so
    that column is half the single interior face value.
    """
    centered = np.empty_like(u_staggered)
    centered[0, :] = 0.5 * u_staggered[0, :]
    centered[1:, :] = 0.5 * (u_staggered[:-1, :] + u_staggered[1:, :])
    return centered


def destagger_v(v_staggered: np.ndarray) -> np.ndarray:
    """Move `v` from north faces to cell centers along y (D011)."""
    centered = np.empty_like(v_staggered)
    centered[:, 0] = 0.5 * v_staggered[:, 0]
    centered[:, 1:] = 0.5 * (v_staggered[:, :-1] + v_staggered[:, 1:])
    return centered


def total_energy(state: SolverState, config: SolverConfig) -> float:
    """Domain-integrated energy under the D014 convention, at cell centers."""
    u_c = destagger_u(state.u)
    v_c = destagger_v(state.v)
    kinetic = 0.5 * (config.depth + state.eta) * (u_c**2 + v_c**2)
    potential = 0.5 * config.gravity * state.eta**2
    return float((kinetic + potential).sum() * config.dx * config.dy)


@dataclass
class RunDiagnostics:
    """Accumulated physical diagnostics for one solver run."""

    initial_mass: float
    initial_energy: float
    final_mass: float = 0.0
    final_energy: float = 0.0
    max_abs_mass_drift: float = 0.0
    min_total_depth: float = float("inf")
    max_abs_velocity: float = 0.0
    max_gravity_cfl: float = 0.0
    max_advective_cfl: float = 0.0
    non_finite_steps: list[int] = field(default_factory=list)
    max_wall_normal_velocity: float = 0.0

    @property
    def relative_mass_drift(self) -> float:
        """Peak mass drift relative to the initial mass.

        Falls back to an absolute measure when the initial mass is ~0, which happens for
        an antisymmetric initial condition whose bumps cancel.
        """
        scale = abs(self.initial_mass)
        if scale < 1e-30:
            return self.max_abs_mass_drift
        return self.max_abs_mass_drift / scale

    @property
    def relative_energy_drift(self) -> float:
        scale = abs(self.initial_energy)
        if scale < 1e-30:
            return abs(self.final_energy - self.initial_energy)
        return abs(self.final_energy - self.initial_energy) / scale

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_mass": self.initial_mass,
            "final_mass": self.final_mass,
            "relative_mass_drift": self.relative_mass_drift,
            "initial_energy": self.initial_energy,
            "final_energy": self.final_energy,
            "relative_energy_drift": self.relative_energy_drift,
            "min_total_depth": self.min_total_depth,
            "max_abs_velocity": self.max_abs_velocity,
            "max_gravity_cfl": self.max_gravity_cfl,
            "max_advective_cfl": self.max_advective_cfl,
            "max_wall_normal_velocity": self.max_wall_normal_velocity,
            "non_finite_steps": list(self.non_finite_steps),
        }


class DiagnosticsAccumulator:
    """Updates `RunDiagnostics` as a run proceeds, without storing every step."""

    def __init__(self, config: SolverConfig, initial_state: SolverState) -> None:
        self.config = config
        self._initial_mass = total_mass(initial_state.eta, config)
        self.diagnostics = RunDiagnostics(
            initial_mass=self._initial_mass,
            initial_energy=total_energy(initial_state, config),
        )
        self.update(initial_state, step_index=0)

    def update(self, state: SolverState, step_index: int) -> None:
        config = self.config
        diag = self.diagnostics

        if not (
            np.all(np.isfinite(state.eta))
            and np.all(np.isfinite(state.u))
            and np.all(np.isfinite(state.v))
        ):
            diag.non_finite_steps.append(step_index)
            return

        mass = total_mass(state.eta, config)
        diag.max_abs_mass_drift = max(diag.max_abs_mass_drift, abs(mass - self._initial_mass))
        diag.min_total_depth = min(diag.min_total_depth, float((config.depth + state.eta).min()))

        peak_velocity = float(max(np.abs(state.u).max(), np.abs(state.v).max()))
        diag.max_abs_velocity = max(diag.max_abs_velocity, peak_velocity)

        spacing = min(config.dx, config.dy)
        # The gravity-wave Courant number is the binding constraint for this scheme; the
        # advective one is reported because the continuity equation is solved nonlinearly.
        diag.max_gravity_cfl = max(
            diag.max_gravity_cfl, config.gravity_wave_speed * config.dt / spacing
        )
        diag.max_advective_cfl = max(diag.max_advective_cfl, peak_velocity * config.dt / spacing)

        # Only the east and north walls carry an explicitly stored zero (D012).
        diag.max_wall_normal_velocity = max(
            diag.max_wall_normal_velocity,
            float(np.abs(state.u[-1, :]).max()),
            float(np.abs(state.v[:, -1]).max()),
        )

    def finalize(self, state: SolverState) -> RunDiagnostics:
        self.diagnostics.final_mass = total_mass(state.eta, self.config)
        self.diagnostics.final_energy = total_energy(state, self.config)
        return self.diagnostics


class DiagnosticsFailure(RuntimeError):
    """A run violated a physical admissibility check."""


def assert_admissible(
    diagnostics: RunDiagnostics,
    *,
    mass_drift_tolerance: float = MASS_DRIFT_TOLERANCE,
    require_positive_depth: bool = True,
) -> None:
    """Raise if a completed run is not physically admissible.

    Deliberately silent about energy drift, which this scheme does not conserve.
    """
    problems: list[str] = []
    if diagnostics.non_finite_steps:
        first = diagnostics.non_finite_steps[0]
        problems.append(
            f"non-finite values at {len(diagnostics.non_finite_steps)} sampled step(s), "
            f"first at step {first}"
        )
    if diagnostics.relative_mass_drift > mass_drift_tolerance:
        problems.append(
            f"relative mass drift {diagnostics.relative_mass_drift:.3e} exceeds "
            f"tolerance {mass_drift_tolerance:.3e}"
        )
    if require_positive_depth and diagnostics.min_total_depth <= 0.0:
        problems.append(f"total depth reaches {diagnostics.min_total_depth:.6f} m, not positive")
    if diagnostics.max_gravity_cfl >= 1.0:
        problems.append(
            f"gravity-wave CFL {diagnostics.max_gravity_cfl:.3f} is at or above 1; "
            "the scheme is unstable"
        )
    if diagnostics.max_wall_normal_velocity > 0.0:
        problems.append(
            f"wall-normal velocity reaches {diagnostics.max_wall_normal_velocity:.3e} m/s "
            "at the east or north wall, where it must be exactly zero"
        )
    if problems:
        raise DiagnosticsFailure("; ".join(problems))
