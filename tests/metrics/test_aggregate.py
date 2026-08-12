"""The aggregation protocol (docs/VALIDATION.md, E-series).

The property that matters most is that trajectories carry equal weight regardless of frame
count, since that is what stops a longer trajectory from dominating a reported score.
"""

from __future__ import annotations

import numpy as np
import pytest

from swe_sr.metrics.aggregate import (
    SnapshotMetric,
    TrajectoryAggregate,
    aggregate_by_trajectory,
    bootstrap_confidence_interval,
    paired_bootstrap_difference,
)


def _samples(spec: dict[str, list[float]]) -> list[SnapshotMetric]:
    return [
        SnapshotMetric(trajectory_id=key, frame=index, value=value)
        for key, values in spec.items()
        for index, value in enumerate(values)
    ]


def test_trajectories_carry_equal_weight_regardless_of_frame_count() -> None:
    """The central rule: "this prevents a trajectory with more usable frames from dominating".

    Trajectory A contributes 100 frames at 1.0, B contributes 1 frame at 3.0. Pooling snapshots
    would give ~1.02; equal-weight trajectory averaging gives 2.0.
    """
    samples = _samples({"a": [1.0] * 100, "b": [3.0]})
    result = aggregate_by_trajectory(samples, metric="mse")

    assert result.mean == pytest.approx(2.0)
    pooled = float(np.mean([s.value for s in samples]))
    assert pooled == pytest.approx(1.0198, abs=1e-3)
    assert result.mean != pytest.approx(pooled)


def test_within_trajectory_mean_comes_first() -> None:
    """Step 2 of the protocol: each trajectory reduces over its own frames before combining."""
    result = aggregate_by_trajectory(_samples({"a": [1.0, 3.0], "b": [10.0, 20.0]}), metric="mse")
    means = {t.trajectory_id: t.mean for t in result.trajectory_means}
    assert means == {"a": pytest.approx(2.0), "b": pytest.approx(15.0)}
    assert result.mean == pytest.approx(8.5)


def test_reports_mean_median_std_and_interval() -> None:
    """docs/VALIDATION.md requires all four where meaningful."""
    result = aggregate_by_trajectory(
        _samples({f"t{i}": [float(i)] for i in range(8)}), metric="rel_l2"
    )
    assert result.trajectories == 8
    assert result.snapshots == 8
    assert result.mean == pytest.approx(3.5)
    assert result.median == pytest.approx(3.5)
    # Sample standard deviation, ddof=1: these trajectories are a sample of possible ICs.
    assert result.std == pytest.approx(np.std(np.arange(8.0), ddof=1))
    assert result.ci_low < result.mean < result.ci_high


def test_frame_counts_are_recorded_per_trajectory() -> None:
    result = aggregate_by_trajectory(_samples({"a": [1.0, 2.0, 3.0], "b": [4.0]}), metric="mse")
    counts = {t.trajectory_id: t.frames for t in result.trajectory_means}
    assert counts == {"a": 3, "b": 1}


def test_a_single_trajectory_yields_no_interval() -> None:
    """One sample supports no interval; a zero-width one would read as infinite precision."""
    result = aggregate_by_trajectory(_samples({"only": [1.0, 2.0]}), metric="mse")
    assert result.trajectories == 1
    assert np.isnan(result.ci_low) and np.isnan(result.ci_high)
    assert result.std == 0.0


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="no samples"):
        aggregate_by_trajectory([], metric="mse")


def test_aggregation_rule_travels_with_the_number() -> None:
    """CLAUDE.md forbids reporting a metric without its aggregation rule."""
    payload = aggregate_by_trajectory(_samples({"a": [1.0], "b": [2.0]}), metric="mse").to_dict()
    assert payload["trajectory_equal_weight"] is True
    assert "equal-weight across trajectories" in payload["aggregation"]
    assert payload["confidence"] == 0.95


# -- Bootstrap ------------------------------------------------------------------------


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    """An interval that moves between reruns of the same data is not reportable."""
    values = np.array([1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0])
    first = bootstrap_confidence_interval(values, seed=7)
    second = bootstrap_confidence_interval(values, seed=7)
    assert first == second
    assert bootstrap_confidence_interval(values, seed=8) != first


def test_bootstrap_interval_brackets_the_mean_and_narrows_with_more_data() -> None:
    rng = np.random.default_rng(3)
    small = rng.normal(loc=1.0, scale=0.2, size=8)
    large = rng.normal(loc=1.0, scale=0.2, size=200)

    small_low, small_high = bootstrap_confidence_interval(small, seed=1)
    large_low, large_high = bootstrap_confidence_interval(large, seed=1)
    assert small_low < small.mean() < small_high
    assert (large_high - large_low) < (small_high - small_low)


def test_bootstrap_resamples_trajectories_not_snapshots() -> None:
    """Resampling correlated snapshots would understate the interval badly.

    Demonstrated by the interval width depending on the number of *trajectories*: 8 trajectories
    each averaging 100 near-identical frames must give the same interval as 8 trajectories of one
    frame at the same means.
    """
    means = [1.0, 1.2, 0.9, 1.4, 1.1, 0.8, 1.3, 1.0]
    dense = aggregate_by_trajectory(
        _samples({f"t{i}": [m] * 100 for i, m in enumerate(means)}), metric="mse", seed=5
    )
    sparse = aggregate_by_trajectory(
        _samples({f"t{i}": [m] for i, m in enumerate(means)}), metric="mse", seed=5
    )
    assert dense.ci_low == pytest.approx(sparse.ci_low)
    assert dense.ci_high == pytest.approx(sparse.ci_high)
    assert dense.snapshots == 800 and sparse.snapshots == 8


# -- Paired comparison ----------------------------------------------------------------


def _aggregates(values: dict[str, float]) -> list[TrajectoryAggregate]:
    return [TrajectoryAggregate(trajectory_id=k, mean=v, frames=1) for k, v in values.items()]


def test_paired_bootstrap_detects_a_consistent_improvement() -> None:
    """A small but consistent per-trajectory gain should give an interval excluding zero.

    This is the case an unpaired comparison would miss, because between-trajectory spread is
    much larger than the difference.
    """
    baseline = {f"t{i}": 1.0 + i * 0.5 for i in range(8)}  # wide spread
    improved = {key: value - 0.05 for key, value in baseline.items()}  # uniformly better

    result = paired_bootstrap_difference(_aggregates(improved), _aggregates(baseline), seed=2)
    assert result["mean_difference"] == pytest.approx(-0.05)
    assert result["ci_high"] < 0.0, "a uniform improvement should exclude zero"


def test_paired_bootstrap_reports_no_difference_when_there_is_none() -> None:
    values = {f"t{i}": 1.0 + i * 0.3 for i in range(8)}
    result = paired_bootstrap_difference(_aggregates(values), _aggregates(values), seed=2)
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)


def test_paired_bootstrap_refuses_mismatched_trajectory_sets() -> None:
    """Comparing different data silently is worse than failing to compare."""
    left = _aggregates({"a": 1.0, "b": 2.0, "c": 3.0})
    right = _aggregates({"a": 1.0, "b": 2.0, "d": 3.0})
    with pytest.raises(ValueError, match="different trajectory sets"):
        paired_bootstrap_difference(left, right)


def test_paired_bootstrap_needs_at_least_two_trajectories() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        paired_bootstrap_difference(_aggregates({"a": 1.0}), _aggregates({"a": 2.0}))
