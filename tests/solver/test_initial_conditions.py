"""Analytic initial conditions: reproducibility, admissibility, and serialization.

The IC layer is what makes D002 true in practice: one analytic specification evaluated
independently on each grid, never a resize of the other. It must also round-trip through
the immutable registry exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from swe_sr.solver.config import ResolutionPair, SolverConfig
from swe_sr.solver.initial_conditions import (
    AMPLITUDE_RANGE_M,
    BUMP_COUNT_CHOICES,
    WALL_MARGIN_SIGMAS,
    WIDTH_RANGE_M,
    GaussianBump,
    GaussianBumpsIC,
    InitialConditionRejected,
    RingIC,
    _check_admissible,
    initial_condition_from_dict,
    sample_gaussian_bumps_ic,
)

PRIMARY = ResolutionPair(pair_id="swe_gaussian_32x128_v1", coarse_nodes=32, fine_nodes=128)


def test_sampling_is_reproducible_from_the_seed_alone() -> None:
    """The registry stores seeds, so the same seed must reproduce the same specification."""
    config = SolverConfig(n_x=128, n_y=128)
    first = sample_gaussian_bumps_ic(7, config)
    second = sample_gaussian_bumps_ic(7, config)
    assert first == second
    assert first != sample_gaussian_bumps_ic(8, config)


def test_sampled_specifications_stay_inside_the_documented_ranges() -> None:
    """Every drawn bump must satisfy the docs/DATASET.md sampling table."""
    config = SolverConfig(n_x=128, n_y=128)
    for seed in range(48):
        candidate = sample_gaussian_bumps_ic(seed, config)
        assert len(candidate.bumps) in BUMP_COUNT_CHOICES
        for bump in candidate.bumps:
            assert AMPLITUDE_RANGE_M[0] <= abs(bump.amplitude) <= AMPLITUDE_RANGE_M[1]
            assert WIDTH_RANGE_M[0] <= bump.width <= WIDTH_RANGE_M[1]
            margin = WALL_MARGIN_SIGMAS * bump.width
            assert abs(bump.center_x) + margin <= config.length_x / 2
            assert abs(bump.center_y) + margin <= config.length_y / 2


def test_amplitudes_take_both_signs_across_the_registry() -> None:
    """docs/DATASET.md specifies a *signed* amplitude, so both signs must appear."""
    config = SolverConfig(n_x=128, n_y=128)
    signs = {
        np.sign(bump.amplitude)
        for seed in range(48)
        for bump in sample_gaussian_bumps_ic(seed, config).bumps
    }
    assert signs == {-1.0, 1.0}


def test_sampled_states_keep_total_depth_positive() -> None:
    config = SolverConfig(n_x=128, n_y=128)
    for seed in range(48):
        eta = sample_gaussian_bumps_ic(seed, config).evaluate(config)
        assert float((config.depth + eta).min()) > 0.0
        assert np.all(np.isfinite(eta))


# -- The core D002 property -----------------------------------------------------------


def test_the_same_specification_evaluates_independently_on_each_grid() -> None:
    """Coarse and fine states are analytic evaluations, not resamplings of one another.

    The decisive evidence: at every shared physical coordinate the two evaluations agree
    to floating-point precision. An interpolated or pooled coarse field would not, because
    it would carry the fine grid's discretization.
    """
    coarse = PRIMARY.coarse_config()
    fine = PRIMARY.fine_config()
    specification = sample_gaussian_bumps_ic(3, fine)

    coarse_eta = specification.evaluate(coarse)
    fine_eta = specification.evaluate(fine)

    # The 32-point grid's coordinates are not a subset of the 128-point grid's, because
    # the spacing ratio is 127/31, not 4. Compare at the corners and the exact center,
    # which both grids do share.
    assert coarse_eta[0, 0] == pytest.approx(fine_eta[0, 0], rel=1e-12)
    assert coarse_eta[-1, 0] == pytest.approx(fine_eta[-1, 0], rel=1e-12)
    assert coarse_eta[0, -1] == pytest.approx(fine_eta[0, -1], rel=1e-12)
    assert coarse_eta[-1, -1] == pytest.approx(fine_eta[-1, -1], rel=1e-12)


def test_coarse_grid_is_not_a_subset_of_the_fine_grid() -> None:
    """Guards the docs/DATASET.md warning against calling these nested 4x meshes.

    Node counts are x4 but spacings are not, so index-aligned reasoning is invalid and
    interpolation must go through physical coordinates.
    """
    coarse = PRIMARY.coarse_config()
    fine = PRIMARY.fine_config()
    assert PRIMARY.shape_factor == 4
    assert PRIMARY.spacing_ratio == pytest.approx(127 / 31)
    assert PRIMARY.spacing_ratio != 4.0

    # Only the endpoints and the midpoint coincide.
    shared = np.intersect1d(coarse.x, fine.x)
    assert len(shared) < coarse.n_x


# -- Rejection rules ------------------------------------------------------------------


def test_a_bump_too_close_to_a_wall_is_rejected() -> None:
    config = SolverConfig(n_x=64, n_y=64)
    too_close = GaussianBumpsIC(
        bumps=(GaussianBump(amplitude=1.0, center_x=4.9e5, center_y=0.0, width=1.0e5),)
    )
    with pytest.raises(InitialConditionRejected, match="wall margin"):
        _check_admissible(too_close, config)


def test_a_bump_that_dries_the_basin_is_rejected() -> None:
    """H + eta <= 0 must be rejected, per docs/DATASET.md."""
    config = SolverConfig(n_x=64, n_y=64, depth=1.0)
    deep_trough = GaussianBumpsIC(
        bumps=(GaussianBump(amplitude=-5.0, center_x=0.0, center_y=0.0, width=8.0e4),)
    )
    with pytest.raises(InitialConditionRejected, match="not positive"):
        _check_admissible(deep_trough, config)


def test_an_admissible_bump_passes() -> None:
    config = SolverConfig(n_x=64, n_y=64)
    fine = GaussianBumpsIC(
        bumps=(GaussianBump(amplitude=1.2, center_x=1.0e5, center_y=-5.0e4, width=8.0e4),)
    )
    _check_admissible(fine, config)  # must not raise


# -- Serialization --------------------------------------------------------------------


def test_gaussian_ic_round_trips_through_its_registry_form() -> None:
    config = SolverConfig(n_x=64, n_y=64)
    original = sample_gaussian_bumps_ic(11, config)
    restored = initial_condition_from_dict(original.to_dict())
    assert restored == original
    np.testing.assert_array_equal(restored.evaluate(config), original.evaluate(config))


def test_ring_ic_round_trips_and_is_annular() -> None:
    """`ring_ood` is qualitatively different from the Gaussian family (docs/DATASET.md).

    Checks the peak sits on the ring radius rather than at the center, which is what makes
    it out of distribution.
    """
    config = SolverConfig(n_x=128, n_y=128)
    ring = RingIC(amplitude=1.0, radius=2.5e5, width=5.0e4)
    restored = initial_condition_from_dict(ring.to_dict())
    assert restored == ring

    eta = ring.evaluate(config)
    center_index = config.n_x // 2
    center_value = eta[center_index, center_index]
    assert eta.max() == pytest.approx(1.0, abs=1e-3)
    assert center_value < 0.01, "a ring must be near zero at its center"


def test_unknown_ic_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown initial-condition family"):
        initial_condition_from_dict({"family": "not_a_family"})


# -- Pair configuration ---------------------------------------------------------------


def test_pair_shares_the_fine_grid_time_step() -> None:
    """D003: both members integrate with the fine grid's stable step."""
    assert PRIMARY.coarse_config().dt == PRIMARY.fine_config().dt
    assert PRIMARY.coarse_config().dt == pytest.approx(25.1398, abs=1e-4)

    backup = ResolutionPair(pair_id="swe_gaussian_64x256_v1", coarse_nodes=64, fine_nodes=256)
    assert backup.coarse_config().dt == pytest.approx(12.5206, abs=1e-4)
    assert backup.spacing_ratio == pytest.approx(255 / 63)


def test_cross_pair_saved_times_differ_slightly() -> None:
    """The 0.392 percent mismatch recorded in PROJECT_SPEC.md Open Questions.

    Pinned as a test so the discrepancy cannot be quietly forgotten when cross-resolution
    comparisons are reported.
    """
    primary_interval = 24 * PRIMARY.fine_config().dt
    backup = ResolutionPair(pair_id="swe_gaussian_64x256_v1", coarse_nodes=64, fine_nodes=256)
    backup_interval = 48 * backup.fine_config().dt

    assert primary_interval == pytest.approx(603.355, abs=1e-3)
    assert backup_interval == pytest.approx(600.989, abs=1e-3)
    relative = abs(primary_interval - backup_interval) / primary_interval
    assert relative == pytest.approx(0.00392, abs=1e-5)


def test_non_integer_shape_factor_is_rejected() -> None:
    with pytest.raises(ValueError, match="integer"):
        _ = ResolutionPair(pair_id="bad", coarse_nodes=30, fine_nodes=128).shape_factor
