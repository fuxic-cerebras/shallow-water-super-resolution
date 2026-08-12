"""Physical diagnostics and destaggering (task P-03).

Covers the G1 acceptance requirement that "all physical diagnostics pass", and pins the
three decisions this project made where the specification was silent or contradictory:
D011 (destaggering), D012 (where wall checks belong), and D014 (mass/energy convention).
"""

from __future__ import annotations

import numpy as np
import pytest

from swe_sr.solver.config import SolverConfig
from swe_sr.solver.diagnostics import (
    DiagnosticsAccumulator,
    DiagnosticsFailure,
    RunDiagnostics,
    assert_admissible,
    destagger_u,
    destagger_v,
    total_energy,
    total_mass,
)
from swe_sr.solver.initial_conditions import GaussianBump, GaussianBumpsIC, shipped_demo_ic
from swe_sr.solver.numerics import SolverState
from swe_sr.solver.runner import sample_schedule, solve


def _ic() -> GaussianBumpsIC:
    return GaussianBumpsIC(
        bumps=(GaussianBump(amplitude=1.0, center_x=2.0e5, center_y=-1.2e5, width=8.0e4),)
    )


# -- Destaggering (D011) --------------------------------------------------------------


def test_destagger_u_averages_bracketing_faces() -> None:
    """`u[i]` is at `x_{i+1/2}`, so the centered value averages faces `i-1` and `i`."""
    u_staggered = np.array([[1.0], [3.0], [5.0], [7.0]])
    centered = destagger_u(u_staggered)
    # Column 0's west face is the wall, where u = 0 and nothing is stored.
    assert centered[0, 0] == pytest.approx(0.5 * 1.0)
    assert centered[1, 0] == pytest.approx(0.5 * (1.0 + 3.0))
    assert centered[2, 0] == pytest.approx(0.5 * (3.0 + 5.0))
    assert centered[3, 0] == pytest.approx(0.5 * (5.0 + 7.0))


def test_destagger_v_operates_along_y() -> None:
    """The v operator must act on the y axis, not x.

    Uses a field that varies only in y so an axis mix-up cannot pass.
    """
    v_staggered = np.array([[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]])
    centered = destagger_v(v_staggered)
    assert centered[0, 0] == pytest.approx(0.5)
    assert centered[0, 1] == pytest.approx(2.0)
    assert centered[0, 2] == pytest.approx(4.0)
    # Constant along x, as the input is.
    np.testing.assert_allclose(centered[0], centered[1])


def test_destagger_preserves_a_uniform_interior_flow() -> None:
    """A uniform interior face velocity destaggers to the same uniform interior value.

    Only the wall-adjacent column is halved, which is the correct cell average given the
    zero wall face and is exactly why D012 moves the hard zero check to the raw arrays.
    """
    u_staggered = np.full((6, 4), 2.0)
    centered = destagger_u(u_staggered)
    np.testing.assert_allclose(centered[1:], 2.0)
    np.testing.assert_allclose(centered[0], 1.0)


def test_destaggering_shifts_a_ramp_by_half_a_cell() -> None:
    """On a linear ramp, destaggering recovers the value half a cell to the west.

    This is the quantitative form of the D011 problem: interpreting a face-centered
    velocity as cell-centered misplaces it by dx/2, and dx differs between the coarse and
    fine grids, so the error does not cancel between input and target.
    """
    n = 8
    faces = np.arange(n, dtype=float)[:, None] + 0.5  # u_{i+1/2} = i + 0.5
    centered = destagger_u(np.repeat(faces, 3, axis=1))
    expected_interior = np.arange(1, n, dtype=float)[:, None]  # cell centers i = 1..n-1
    np.testing.assert_allclose(centered[1:], np.repeat(expected_interior, 3, axis=1))


def test_destaggered_shapes_are_unchanged() -> None:
    state = np.zeros((5, 7))
    assert destagger_u(state).shape == (5, 7)
    assert destagger_v(state).shape == (5, 7)


# -- Mass and energy (D014) -----------------------------------------------------------


def test_total_mass_is_the_area_weighted_elevation_integral() -> None:
    config = SolverConfig(n_x=10, n_y=20)
    eta = np.full((config.n_x, config.n_y), 0.5)
    expected = 0.5 * config.n_x * config.n_y * config.dx * config.dy
    assert total_mass(eta, config) == pytest.approx(expected)


def test_total_energy_matches_a_hand_computed_case() -> None:
    """Energy on a state at rest is pure available potential energy."""
    config = SolverConfig(n_x=6, n_y=6)
    eta = np.full((6, 6), 0.25)
    state = SolverState(eta=eta, u=np.zeros((6, 6)), v=np.zeros((6, 6)))
    expected = 0.5 * config.gravity * 0.25**2 * 36 * config.dx * config.dy
    assert total_energy(state, config) == pytest.approx(expected)


def test_total_energy_includes_kinetic_contribution() -> None:
    """A uniform interior flow adds kinetic energy above the resting value."""
    config = SolverConfig(n_x=8, n_y=8)
    eta = np.zeros((8, 8))
    resting = SolverState(eta=eta, u=np.zeros((8, 8)), v=np.zeros((8, 8)))
    moving = SolverState(eta=eta, u=np.full((8, 8), 0.5), v=np.zeros((8, 8)))
    assert total_energy(resting, config) == pytest.approx(0.0)
    assert total_energy(moving, config) > 0.0


# -- Run-level diagnostics ------------------------------------------------------------


def test_mass_is_conserved_to_roundoff_over_a_full_length_trajectory() -> None:
    """The nonlinear upwind continuity equation conserves closed-basin mass exactly.

    A drift above roundoff here means the flux budget is broken, which is why the
    tolerance is tight rather than generous.
    """
    config = SolverConfig(n_x=64, n_y=64)
    steps = sample_schedule(288, 24, 128)
    result = solve(config, _ic(), sample_steps=steps, diagnostic_stride=8)

    assert result.diagnostics.relative_mass_drift < 1e-13
    assert_admissible(result.diagnostics)


@pytest.mark.parametrize("nodes", [32, 128])
def test_paired_resolutions_are_both_physically_admissible(nodes: int) -> None:
    """Both members of the primary pair pass every diagnostic at the shared time step."""
    fine = SolverConfig(n_x=128, n_y=128)
    config = SolverConfig(n_x=nodes, n_y=nodes, dt_override=fine.dt)
    result = solve(config, _ic(), sample_steps=sample_schedule(288, 24, 16), diagnostic_stride=4)

    diagnostics = result.diagnostics
    assert_admissible(diagnostics)
    assert diagnostics.min_total_depth > 0.0
    assert not diagnostics.non_finite_steps
    assert diagnostics.max_wall_normal_velocity == 0.0
    # The shared step is set by the fine grid, so the coarse run is well inside its bound.
    assert diagnostics.max_gravity_cfl <= 0.1 + 1e-12


def test_wall_normal_velocity_is_exactly_zero_on_raw_arrays() -> None:
    """D012: the east and north walls carry an exact stored zero.

    Asserted as exact equality, not a tolerance, because the kernel writes the zero.
    """
    config = SolverConfig(n_x=48, n_y=48)
    result = solve(config, _ic(), sample_steps=sample_schedule(0, 40, 6))
    for frame in range(result.fields.shape[0]):
        stored_u = result.fields[frame, 1]  # [y, x]
        stored_v = result.fields[frame, 2]
        assert np.all(stored_u[:, -1] == 0.0), "east wall u must be exactly zero"
        assert np.all(stored_v[-1, :] == 0.0), "north wall v must be exactly zero"


def test_energy_drift_is_reported_but_not_asserted() -> None:
    """The scheme does not conserve energy, so a nonzero drift must not fail a run."""
    config = SolverConfig(n_x=48, n_y=48)
    result = solve(config, _ic(), sample_steps=sample_schedule(288, 24, 32), diagnostic_stride=8)
    assert result.diagnostics.relative_energy_drift >= 0.0
    assert_admissible(result.diagnostics)  # passes despite any energy drift
    assert "relative_energy_drift" in result.diagnostics.to_dict()


def test_cfl_is_measured_from_the_resolved_state_not_hard_coded() -> None:
    """docs/DATASET.md requires computing the actual CFL bound rather than assuming it."""
    config = SolverConfig(n_x=128, n_y=128)
    result = solve(config, _ic(), sample_steps=[0, 100], diagnostic_stride=1)
    # cfl_factor defaults to 0.1, and dt derives from this grid, so the gravity Courant
    # number should land on 0.1 rather than on some assumed constant.
    assert result.diagnostics.max_gravity_cfl == pytest.approx(0.1, rel=1e-12)
    assert 0.0 < result.diagnostics.max_advective_cfl < result.diagnostics.max_gravity_cfl


def test_resolvability_diagnostic_reports_sigma_over_dx() -> None:
    """docs/DATASET.md requires recording `sigma/dx` for every grid.

    The coarse member of the primary pair sits near 2, which is the deliberately
    under-resolved regime that makes the super-resolution task non-trivial.
    """
    narrow = GaussianBumpsIC(
        bumps=(GaussianBump(amplitude=1.0, center_x=0.0, center_y=0.0, width=65.0e3),)
    )
    coarse = narrow.resolvability(SolverConfig(n_x=32, n_y=32))
    fine = narrow.resolvability(SolverConfig(n_x=128, n_y=128))
    assert coarse["min_sigma_over_dx"] == pytest.approx(2.015, abs=0.01)
    assert fine["min_sigma_over_dx"] == pytest.approx(8.255, abs=0.01)


# -- Failure detection ----------------------------------------------------------------


def test_assert_admissible_rejects_non_finite_values() -> None:
    diagnostics = RunDiagnostics(initial_mass=1.0, initial_energy=1.0, min_total_depth=100.0)
    diagnostics.non_finite_steps.append(7)
    with pytest.raises(DiagnosticsFailure, match="non-finite"):
        assert_admissible(diagnostics)


def test_assert_admissible_rejects_mass_drift() -> None:
    diagnostics = RunDiagnostics(
        initial_mass=1.0,
        initial_energy=1.0,
        max_abs_mass_drift=1e-3,
        min_total_depth=100.0,
    )
    with pytest.raises(DiagnosticsFailure, match="mass drift"):
        assert_admissible(diagnostics)


def test_assert_admissible_rejects_negative_depth() -> None:
    diagnostics = RunDiagnostics(initial_mass=1.0, initial_energy=1.0, min_total_depth=-0.5)
    with pytest.raises(DiagnosticsFailure, match="not positive"):
        assert_admissible(diagnostics)


def test_assert_admissible_rejects_unstable_cfl() -> None:
    diagnostics = RunDiagnostics(
        initial_mass=1.0, initial_energy=1.0, min_total_depth=100.0, max_gravity_cfl=1.4
    )
    with pytest.raises(DiagnosticsFailure, match="CFL"):
        assert_admissible(diagnostics)


def test_assert_admissible_rejects_leaking_walls() -> None:
    diagnostics = RunDiagnostics(
        initial_mass=1.0,
        initial_energy=1.0,
        min_total_depth=100.0,
        max_wall_normal_velocity=1e-9,
    )
    with pytest.raises(DiagnosticsFailure, match="wall-normal"):
        assert_admissible(diagnostics)


def test_an_unstable_configuration_is_actually_caught() -> None:
    """A negative-control run: a CFL factor far above the stable range must be rejected.

    docs/VALIDATION.md requires negative tests, not only passing ones. Without this, the
    diagnostics could be vacuously green.
    """
    config = SolverConfig(n_x=32, n_y=32, cfl_factor=8.0)
    # Overflow to inf and then NaN is the expected behavior of a divergent run, so the
    # numpy warnings are suppressed here rather than globally, where they would hide a
    # genuine problem in a run that is supposed to be stable.
    with np.errstate(all="ignore"):
        result = solve(config, _ic(), sample_steps=[0, 400], diagnostic_stride=10)
    with pytest.raises(DiagnosticsFailure):
        assert_admissible(result.diagnostics)


def test_diagnostics_accumulator_records_the_initial_state() -> None:
    """Step 0 is included, so a bad initial condition cannot slip past unmeasured."""
    config = SolverConfig(n_x=16, n_y=16)
    state = SolverState.from_initial_eta(shipped_demo_ic(config).evaluate(config))
    accumulator = DiagnosticsAccumulator(config, state)
    assert accumulator.diagnostics.min_total_depth < float("inf")
    assert accumulator.diagnostics.initial_mass == pytest.approx(total_mass(state.eta, config))
