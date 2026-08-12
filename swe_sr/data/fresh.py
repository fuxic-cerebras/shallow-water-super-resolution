"""Fresh post-training evaluation workloads (task E-02).

`docs/DATASET.md` defines two, generated only after model selection and absent from any
training manifest:

- `fresh_id`: a new seed from the same Gaussian family, a clean repeatability check;
- `ring_ood`: an annular perturbation with zero initial velocity, qualitatively unlike the
  Gaussian training family.

They are reported separately and never mixed into the held-out test score.

The seeds live in a reserved band far above the training registry's 0-47 so a fresh workload
provably cannot reuse a training initial condition. `assert_disjoint_from_registry` checks that
rather than trusting the convention, because a fresh workload that silently overlapped training
would turn the headline generalization claim into a restatement of the training score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from swe_sr.data.registry import TOTAL_TRAJECTORIES, trajectory_id_for_seed
from swe_sr.solver.config import SolverConfig
from swe_sr.solver.initial_conditions import (
    InitialCondition,
    RingIC,
    sample_gaussian_bumps_ic,
)

# Reserved band, well clear of the registry's 0-47. The gap is deliberate: an off-by-a-few
# mistake cannot land inside the training range.
FRESH_SEED_BASE = 10_000
SCENARIOS = ("fresh_id", "ring_ood")

# Ring geometry. The radius sits well inside the basin and the width is comparable to the
# Gaussian family's, so the difference from training is the *shape* rather than the scale --
# otherwise a degradation could be attributed to an unseen amplitude instead of an unseen form.
RING_AMPLITUDE_M = 1.0
RING_RADIUS_M = 2.5e5
RING_WIDTH_M = 6.0e4


@dataclass(frozen=True)
class FreshWorkload:
    """One evaluation-only workload: a scenario name, its seeds, and its initial conditions."""

    scenario: str
    trajectories: tuple[tuple[int, str, InitialCondition], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "evaluation_only": True,
            "seed_base": FRESH_SEED_BASE,
            "trajectories": [
                {
                    "seed": seed,
                    "trajectory_id": trajectory_id,
                    "initial_condition": ic.to_dict(),
                }
                for seed, trajectory_id, ic in self.trajectories
            ],
        }

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(seed for seed, _, _ in self.trajectories)

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(tid for _, tid, _ in self.trajectories)


def build_fresh_workload(
    scenario: str,
    *,
    count: int = 4,
    reference_config: SolverConfig | None = None,
) -> FreshWorkload:
    """Construct a fresh workload deterministically from its scenario name.

    `count` defaults to 4 trajectories: enough for the trajectory-level bootstrap the
    aggregation protocol uses to produce a meaningful interval, without the cost of a full
    48-trajectory release for something that is only ever evaluated.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; available: {sorted(SCENARIOS)}")
    if count < 2:
        raise ValueError(
            f"count must be at least 2 so a trajectory bootstrap is defined, got {count}"
        )
    config = reference_config or SolverConfig(n_x=128, n_y=128)

    trajectories = []
    for index in range(count):
        # Offset by scenario so the two workloads never share a seed with each other either.
        seed = FRESH_SEED_BASE + SCENARIOS.index(scenario) * 1_000 + index
        if scenario == "fresh_id":
            initial_condition: InitialCondition = sample_gaussian_bumps_ic(seed, config)
        else:
            # Vary the ring's centre slightly per trajectory so the workload is not four
            # copies of one state, while keeping the annular form fixed.
            offset = (index - (count - 1) / 2) * 4.0e4
            initial_condition = RingIC(
                amplitude=RING_AMPLITUDE_M,
                radius=RING_RADIUS_M,
                width=RING_WIDTH_M,
                center_x=offset,
                center_y=-offset,
            )
        trajectories.append((seed, trajectory_id_for_seed(seed), initial_condition))
    return FreshWorkload(scenario=scenario, trajectories=tuple(trajectories))


class FreshWorkloadLeakage(ValueError):
    """A fresh workload overlaps the training registry, which would invalidate its purpose."""


def assert_disjoint_from_registry(workload: FreshWorkload) -> None:
    """Fail if a fresh workload could share an initial condition with the training registry.

    Checked on both seed and trajectory ID. `docs/VALIDATION.md` requires fresh workloads to be
    absent from the training manifest, and an overlap here would quietly turn a generalization
    result into a restatement of the training score.
    """
    registry_seeds = set(range(TOTAL_TRAJECTORIES))
    registry_ids = {trajectory_id_for_seed(seed) for seed in registry_seeds}

    overlapping_seeds = registry_seeds & set(workload.seeds)
    if overlapping_seeds:
        raise FreshWorkloadLeakage(
            f"{workload.scenario} reuses training seed(s) {sorted(overlapping_seeds)}"
        )
    overlapping_ids = registry_ids & set(workload.trajectory_ids)
    if overlapping_ids:
        raise FreshWorkloadLeakage(
            f"{workload.scenario} reuses training trajectory ID(s) {sorted(overlapping_ids)}"
        )
    if min(workload.seeds) < FRESH_SEED_BASE:
        raise FreshWorkloadLeakage(
            f"{workload.scenario} has a seed below the reserved base {FRESH_SEED_BASE}"
        )
