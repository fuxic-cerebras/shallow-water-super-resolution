"""Runner contract: orientation, headlessness, sampling, and determinism."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from swe_sr.solver.config import SolverConfig
from swe_sr.solver.initial_conditions import GaussianBump, GaussianBumpsIC, shipped_demo_ic
from swe_sr.solver.runner import CHANNEL_NAMES, sample_schedule, solve


def _asymmetric_ic() -> GaussianBumpsIC:
    """A bump that is off-center in both axes, so an x/y swap cannot go unnoticed."""
    return GaussianBumpsIC(
        bumps=(GaussianBump(amplitude=1.0, center_x=3.0e5, center_y=-1.0e5, width=8.0e4),)
    )


def test_no_plotting_import_in_generation_path() -> None:
    """Importing the solver must not pull in matplotlib.

    docs/ARCHITECTURE.md requires generation to run headlessly, and the G1 gate requires
    "no plotting import in generation". Checked in a subprocess so a matplotlib already
    imported by another test cannot mask a real dependency.
    """
    program = (
        "import sys;"
        "import swe_sr.solver.runner, swe_sr.solver.diagnostics,"
        " swe_sr.solver.initial_conditions, swe_sr.solver.numerics, swe_sr.solver.config;"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] "
        "in {'matplotlib', 'viz_tools', 'pyplot'});"
        "print(leaked);"
        "sys.exit(1 if leaked else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, f"plotting modules leaked into the solver: {completed.stdout}"


def test_fields_are_in_storage_order_with_y_before_x() -> None:
    """`solve` returns `[time, channel, y, x]` even on a non-square grid.

    The solver works in `[x, y]`; storage is `[y, x]` (D016). A square grid would let a
    missing transpose pass silently, so this uses distinct node counts.
    """
    config = SolverConfig(n_x=24, n_y=36)
    result = solve(config, _asymmetric_ic(), sample_steps=[0, 5])

    assert result.fields.shape == (2, 3, 36, 24)
    assert CHANNEL_NAMES == ("eta", "u", "v")


def test_transpose_places_values_at_the_right_physical_coordinates() -> None:
    """The stored array indexes as `[y, x]`, verified against the analytic peak location.

    This is the check that actually catches a transposed grid: it locates the bump's peak
    in the stored array and confirms the recovered coordinates match the specification.
    """
    config = SolverConfig(n_x=41, n_y=61)
    center_x, center_y = 2.0e5, -1.5e5
    initial_condition = GaussianBumpsIC(
        bumps=(GaussianBump(amplitude=1.0, center_x=center_x, center_y=center_y, width=6.0e4),)
    )
    stored_eta = solve(config, initial_condition, sample_steps=[0]).fields[0, 0]

    row, column = np.unravel_index(int(np.argmax(stored_eta)), stored_eta.shape)
    # row indexes y, column indexes x.
    assert config.y[row] == pytest.approx(center_y, abs=config.dy)
    assert config.x[column] == pytest.approx(center_x, abs=config.dx)


def test_initial_velocities_satisfy_wall_conditions() -> None:
    config = SolverConfig(n_x=20, n_y=20)
    result = solve(config, shipped_demo_ic(config), sample_steps=[0])
    assert np.all(result.fields[0, 1] == 0.0)
    assert np.all(result.fields[0, 2] == 0.0)


def test_identical_config_and_seed_reproduce_identical_arrays() -> None:
    """docs/VALIDATION.md: identical configs and seeds reproduce identical arrays."""
    config = SolverConfig(n_x=32, n_y=32)
    initial_condition = _asymmetric_ic()
    first = solve(config, initial_condition, sample_steps=[0, 50, 100])
    second = solve(config, initial_condition, sample_steps=[0, 50, 100])

    np.testing.assert_array_equal(first.fields, second.fields)
    np.testing.assert_array_equal(first.times, second.times)
    assert config.config_hash == SolverConfig(n_x=32, n_y=32).config_hash


def test_sampled_snapshots_are_independent_of_the_sampling_request() -> None:
    """Asking for extra snapshots must not change the ones already requested.

    Guards the streaming sampler: if it accidentally mutated or aliased state, requesting
    a denser schedule would perturb the shared steps.
    """
    config = SolverConfig(n_x=24, n_y=24)
    initial_condition = _asymmetric_ic()
    sparse = solve(config, initial_condition, sample_steps=[0, 60])
    dense = solve(config, initial_condition, sample_steps=[0, 10, 20, 30, 40, 50, 60])

    np.testing.assert_array_equal(sparse.fields[0], dense.fields[0])
    np.testing.assert_array_equal(sparse.fields[-1], dense.fields[-1])


def test_snapshots_are_not_aliased_to_each_other() -> None:
    """Each stored snapshot is a distinct state, not repeated references to one buffer."""
    config = SolverConfig(n_x=20, n_y=20)
    result = solve(config, _asymmetric_ic(), sample_steps=[0, 25, 50])
    assert not np.array_equal(result.fields[0], result.fields[1])
    assert not np.array_equal(result.fields[1], result.fields[2])


def test_times_are_exact_multiples_of_dt() -> None:
    """docs/DATASET.md requires deriving times from resolved configs, not rounded intervals."""
    config = SolverConfig(n_x=20, n_y=20)
    steps = [0, 7, 21]
    result = solve(config, shipped_demo_ic(config), sample_steps=steps)
    for index, step in enumerate(steps):
        assert result.times[index] == step * config.dt
    assert result.times.dtype == np.float64


def test_sample_schedule_matches_documented_primary_and_backup_cadence() -> None:
    """The schedules in docs/DATASET.md, as explicit integer step indices (D017)."""
    primary = sample_schedule(discard_steps=288, stride=24, count=197)
    assert primary[0] == 288
    assert primary[-1] == 288 + 24 * 196 == 4992
    assert primary.size == 197

    backup = sample_schedule(discard_steps=576, stride=48, count=197)
    assert backup[0] == 576
    assert backup[-1] == 576 + 48 * 196 == 9984
    assert backup.size == 197


def test_paired_releases_cover_matching_physical_durations() -> None:
    """D017: doubling the backup's stride *and* step cap is what keeps the pairs aligned.

    The backup's `dt` is half the primary's because its fine grid has half the spacing, so
    an equal step count would leave it covering half the primary's evolution. This pins the
    property that justifies the stride choice, and the 0.392 percent residual mismatch that
    the endpoint convention makes unavoidable.
    """
    primary_dt = SolverConfig(n_x=128, n_y=128).dt
    backup_dt = SolverConfig(n_x=256, n_y=256).dt
    assert backup_dt == pytest.approx(primary_dt / 2, rel=0.005)

    primary_duration = 4992 * primary_dt
    backup_duration = 9984 * backup_dt
    assert primary_duration / 3600 == pytest.approx(34.86, abs=0.01)
    assert backup_duration / 3600 == pytest.approx(34.72, abs=0.01)

    mismatch = abs(primary_duration - backup_duration) / primary_duration
    assert mismatch == pytest.approx(0.00392, abs=1e-5)

    # Saved intervals also stay aligned, which is what makes the frames comparable.
    assert 24 * primary_dt == pytest.approx(603.36, abs=0.01)
    assert 48 * backup_dt == pytest.approx(600.99, abs=0.01)


def test_paired_resolutions_share_bit_identical_saved_times() -> None:
    """Within a pair, LR and HR saved times must be exactly equal (docs/DATASET.md).

    This is what D003's shared fine-grid time step buys, and docs/VALIDATION.md gates on
    it. Equality must be exact, not approximate.
    """
    fine = SolverConfig(n_x=128, n_y=128)
    coarse = SolverConfig(n_x=32, n_y=32, dt_override=fine.dt)
    steps = sample_schedule(288, 24, 8)

    initial_condition = _asymmetric_ic()
    fine_result = solve(fine, initial_condition, sample_steps=steps)
    coarse_result = solve(coarse, initial_condition, sample_steps=steps)

    np.testing.assert_array_equal(fine_result.times, coarse_result.times)
    assert fine_result.fields.shape[2:] == (128, 128)
    assert coarse_result.fields.shape[2:] == (32, 32)


def test_coarse_and_fine_initial_states_are_independent_analytic_evaluations() -> None:
    """D002: neither grid's state is obtained by resizing the other.

    Evidence that both are analytic evaluations: the coarse initial field equals the
    specification evaluated on coarse coordinates exactly, which a resampling of the fine
    field would not.
    """
    initial_condition = _asymmetric_ic()
    fine = SolverConfig(n_x=128, n_y=128)
    coarse = SolverConfig(n_x=32, n_y=32, dt_override=fine.dt)

    coarse_stored = solve(coarse, initial_condition, sample_steps=[0]).fields[0, 0]
    np.testing.assert_array_equal(coarse_stored, initial_condition.evaluate(coarse).T)


@pytest.mark.parametrize(
    "bad_steps",
    [[], [-1, 0], [5, 5], [10, 3]],
    ids=["empty", "negative", "duplicate", "unsorted"],
)
def test_invalid_sample_steps_are_rejected(bad_steps: list[int]) -> None:
    config = SolverConfig(n_x=12, n_y=12)
    with pytest.raises(ValueError):
        solve(config, shipped_demo_ic(config), sample_steps=bad_steps)
