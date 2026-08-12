"""The aggregation protocol from `docs/VALIDATION.md` (E-series support).

The protocol is explicit and the ordering matters:

1. compute metrics per snapshot and per channel;
2. aggregate over time within each trajectory;
3. aggregate trajectories with **equal weight**;
4. report mean, median, standard deviation, and a 95% bootstrap confidence interval over
   trajectories where meaningful.

Step 3 is the point. Pooling every snapshot would let a trajectory with more usable frames
dominate the result, and since the frames within one trajectory are ~97% correlated at the
saved cadence, a pooled standard error would also be badly overconfident. Equal-weight
trajectory averaging is what makes the reported spread mean something.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Bootstrap resamples for the trajectory-level confidence interval. 10,000 is cheap here
# because it resamples ~8 trajectory means, not the underlying frames.
BOOTSTRAP_RESAMPLES = 10_000
CONFIDENCE = 0.95


@dataclass
class SnapshotMetric:
    """One metric value for one snapshot of one trajectory."""

    trajectory_id: str
    frame: int
    value: float


@dataclass
class TrajectoryAggregate:
    """Step 2: a single trajectory's mean over its own frames."""

    trajectory_id: str
    mean: float
    frames: int


@dataclass
class AggregateResult:
    """Step 3 and 4: the trajectory-level distribution and its summary.

    Every field is named so a reported number carries its aggregation rule with it, which
    `CLAUDE.md` requires.
    """

    metric: str
    trajectory_means: list[TrajectoryAggregate] = field(default_factory=list)
    mean: float = float("nan")
    median: float = float("nan")
    std: float = float("nan")
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    trajectories: int = 0
    snapshots: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "aggregation": (
                "per-snapshot, then within-trajectory mean, then equal-weight across trajectories"
            ),
            "trajectory_equal_weight": True,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": CONFIDENCE,
            "trajectories": self.trajectories,
            "snapshots": self.snapshots,
        }


def aggregate_by_trajectory(
    samples: Sequence[SnapshotMetric],
    *,
    metric: str,
    seed: int = 0,
) -> AggregateResult:
    """Apply the four-step protocol to per-snapshot values.

    `seed` fixes the bootstrap resampling so a reported confidence interval is reproducible;
    an interval that moves between reruns of the same data is not a reportable number.
    """
    if not samples:
        raise ValueError(f"no samples supplied for metric {metric!r}")

    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        grouped[sample.trajectory_id].append(sample.value)

    # Step 2: within-trajectory mean over that trajectory's own frames.
    trajectory_means = [
        TrajectoryAggregate(trajectory_id=key, mean=float(np.mean(values)), frames=len(values))
        for key, values in sorted(grouped.items())
    ]
    values = np.array([aggregate.mean for aggregate in trajectory_means], dtype=np.float64)

    # Step 3: equal weight across trajectories, regardless of frame count.
    low, high = bootstrap_confidence_interval(values, seed=seed)
    return AggregateResult(
        metric=metric,
        trajectory_means=trajectory_means,
        mean=float(values.mean()),
        median=float(np.median(values)),
        # Sample standard deviation over trajectories. ddof=1 because these are a sample of
        # possible initial conditions, not the population.
        std=float(values.std(ddof=1)) if values.size > 1 else 0.0,
        ci_low=low,
        ci_high=high,
        trajectories=values.size,
        snapshots=len(samples),
    )


def bootstrap_confidence_interval(
    values: np.ndarray,
    *,
    seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, resampling trajectories.

    Resamples *trajectories*, not snapshots: snapshots within a trajectory are strongly
    correlated, so resampling them would understate the interval badly. Returns `(nan, nan)`
    for a single trajectory rather than a zero-width interval, since one sample supports no
    interval at all and a zero-width one would read as infinite precision.
    """
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def paired_bootstrap_difference(
    left: Sequence[TrajectoryAggregate],
    right: Sequence[TrajectoryAggregate],
    *,
    seed: int = 0,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
) -> dict[str, float]:
    """Paired bootstrap on the per-trajectory difference between two methods.

    Paired because both methods are evaluated on the *same* trajectories, so the pairing
    removes trajectory-to-trajectory variation that would otherwise swamp the comparison. This
    is what `docs/DATASET.md` means by paired trajectory bootstrapping.

    Returns the mean difference and its interval. An interval excluding zero is evidence one
    method is better on this test set; it is not a claim about physical generalization.
    """
    left_by_id = {aggregate.trajectory_id: aggregate.mean for aggregate in left}
    right_by_id = {aggregate.trajectory_id: aggregate.mean for aggregate in right}
    shared = sorted(set(left_by_id) & set(right_by_id))
    if len(shared) < 2:
        raise ValueError(
            f"paired bootstrap needs at least 2 shared trajectories, found {len(shared)}"
        )
    if set(left_by_id) != set(right_by_id):
        raise ValueError(
            "the two methods were evaluated on different trajectory sets; a paired comparison "
            "would silently compare different data"
        )

    differences = np.array([left_by_id[key] - right_by_id[key] for key in shared], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(resamples, differences.size), replace=True)
    means = draws.mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return {
        "mean_difference": float(differences.mean()),
        "ci_low": float(np.quantile(means, tail)),
        "ci_high": float(np.quantile(means, 1.0 - tail)),
        "trajectories": float(len(shared)),
        "confidence": confidence,
    }
