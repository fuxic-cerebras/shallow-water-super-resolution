"""Fresh post-training workloads (E-02, E-03).

`docs/DATASET.md` requires these to be absent from any training manifest, and
`docs/VALIDATION.md` requires normalization and weights frozen, no fine-tuning, and results
reported separately. The tests enforce those as properties rather than conventions.

Two of them exist because I got the corresponding thing wrong first: the lead-time coverage
test, because a 12-snapshot default silently restricted the workload to 2.0-3.9 h and made the
model look 14x worse than bicubic when over the matched range it is 3.2x better; and the ring
diversity caveat, because near-duplicate trajectories give a tight bootstrap interval that reads
as precision.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from swe_sr.data.fresh import (
    FRESH_SEED_BASE,
    SCENARIOS,
    FreshWorkloadLeakage,
    assert_disjoint_from_registry,
    build_fresh_workload,
)
from swe_sr.data.generate import GenerationConfig, generate_dataset
from swe_sr.data.registry import TOTAL_TRAJECTORIES, build_registry, trajectory_id_for_seed
from swe_sr.evaluate_fresh import evaluate_fresh
from swe_sr.train import train
from swe_sr.training.config import TrainingConfig

DATA = GenerationConfig(
    dataset_id="test_fresh",
    pair_id="swe_gaussian_32x128_v1",
    resolution_family="swe_gaussian_x4",
    coarse_nodes=32,
    fine_nodes=128,
    discard_steps=12,
    sample_stride=6,
    snapshot_count=4,
    diagnostic_stride=6,
    trajectory_limit=2,
)


# -- Workload construction -------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_workloads_are_reproducible_from_the_scenario_name(scenario: str) -> None:
    assert build_fresh_workload(scenario) == build_fresh_workload(scenario)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_workloads_cannot_reuse_a_training_initial_condition(scenario: str) -> None:
    """docs/VALIDATION.md: fresh workloads are absent from the training manifest.

    An overlap would turn the generalization claim into a restatement of the training score, so
    this is checked on both seed and trajectory ID rather than trusting the seed convention.
    """
    workload = build_fresh_workload(scenario, count=6)
    assert_disjoint_from_registry(workload)
    assert min(workload.seeds) >= FRESH_SEED_BASE
    registry_ids = {trajectory_id_for_seed(s) for s in range(TOTAL_TRAJECTORIES)}
    assert not set(workload.trajectory_ids) & registry_ids


def test_the_two_scenarios_do_not_share_seeds() -> None:
    """Otherwise `fresh_id` and `ring_ood` would not be independent workloads."""
    first = set(build_fresh_workload("fresh_id", count=8).seeds)
    second = set(build_fresh_workload("ring_ood", count=8).seeds)
    assert not first & second


def test_leakage_is_detected_when_it_exists() -> None:
    """Negative control: the guard must fire, not merely pass on good input."""
    workload = build_fresh_workload("fresh_id", count=2)
    leaking = replace(
        workload,
        trajectories=(
            (0, trajectory_id_for_seed(0), workload.trajectories[0][2]),
            *workload.trajectories[1:],
        ),
    )
    with pytest.raises(FreshWorkloadLeakage, match="training seed"):
        assert_disjoint_from_registry(leaking)


def test_ring_scenario_is_annular_not_gaussian() -> None:
    """`ring_ood` must be qualitatively unlike the training family (docs/DATASET.md)."""
    from swe_sr.solver.config import SolverConfig
    from swe_sr.solver.initial_conditions import RingIC

    workload = build_fresh_workload("ring_ood", count=2)
    config = SolverConfig(n_x=64, n_y=64)
    for _, _, initial_condition in workload.trajectories:
        assert isinstance(initial_condition, RingIC)
        eta = initial_condition.evaluate(config)
        centre = eta[config.n_x // 2, config.n_y // 2]
        # A ring is near zero at its own centre; a Gaussian bump peaks there.
        assert abs(float(centre)) < 0.2 * float(eta.max())


def test_fresh_id_scenario_uses_the_training_family() -> None:
    """`fresh_id` is a repeatability check, so it must stay in-distribution."""
    from swe_sr.solver.initial_conditions import GaussianBumpsIC

    for _, _, initial_condition in build_fresh_workload("fresh_id", count=3).trajectories:
        assert isinstance(initial_condition, GaussianBumpsIC)


def test_too_few_trajectories_is_rejected() -> None:
    """A bootstrap over one trajectory is undefined, so the workload must refuse it."""
    with pytest.raises(ValueError, match="at least 2"):
        build_fresh_workload("fresh_id", count=1)


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        build_fresh_workload("hurricane")


# -- Evaluation ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fresh")
    manifest = generate_dataset(DATA, registry=build_registry(), output_root=root, verbose=False)
    config = replace(
        TrainingConfig(),
        model_config="configs/model/edsr_x4.yaml",
        manifest=str(root / "processed" / manifest.dataset_id / "manifest.json"),
        run_root=str(root / "runs"),
        stage="smoke",
        batch_size=2,
        max_epochs=1,
        max_steps=4,
        warmup_steps=1,
    )
    return train(config, verbose=False).run_dir


def test_fresh_evaluation_spans_the_training_lead_time_range_by_default(trained: Path) -> None:
    """The defect this test exists for: a short default silently biases the comparison.

    With 12 of 197 snapshots the workload covered only 2.0-3.9 h -- the regime where this model
    is weakest -- making it look 14x worse than bicubic, when over the matched range it is 3.2x
    better. Defaulting to the manifest's own count keeps fresh and test scores comparable.
    """
    report = evaluate_fresh(trained, "fresh_id", count=2)
    lead_time = report["lead_time_hours"]
    assert lead_time["matches_training_range"] is True
    assert report["snapshots_per_trajectory"] == DATA.snapshot_count


def test_a_restricted_range_is_flagged_rather_than_hidden(trained: Path) -> None:
    """Restricting coverage is allowed, but it must be visible in the report."""
    report = evaluate_fresh(trained, "fresh_id", count=2, snapshot_count=2)
    assert report["lead_time_hours"]["matches_training_range"] is False


def test_fresh_report_records_that_nothing_was_refitted_or_tuned(trained: Path) -> None:
    """docs/VALIDATION.md: freeze normalization and weights, no fine-tuning."""
    report = evaluate_fresh(trained, "fresh_id", count=2)
    assert report["evaluation_only"] is True
    assert report["fine_tuned"] is False
    assert "not refitted" in report["normalization_source"]
    assert (
        "never mixed into the held-out test score"
        in (report["reference_notes"]["reported_separately"])
    )


def test_ring_workload_carries_its_diversity_caveat(trained: Path) -> None:
    """A tight interval from near-duplicate trajectories must not read as precision."""
    report = evaluate_fresh(trained, "ring_ood", count=2)
    assert report["reference_notes"]["ring_diversity_caveat"]
    assert "near" in report["reference_notes"]["ring_diversity_caveat"]
    # And the caveat is absent where it does not apply.
    other = evaluate_fresh(trained, "fresh_id", count=2)
    assert other["reference_notes"]["ring_diversity_caveat"] is None


def test_baselines_and_model_are_evaluated_on_the_same_fresh_states(trained: Path) -> None:
    report = evaluate_fresh(trained, "ring_ood", count=2)
    counts = {
        name: method["aggregate_macro_mse_normalized"]["snapshots"]
        for name, method in report["methods"].items()
    }
    assert len(set(counts.values())) == 1
    assert set(report["methods"]) >= {"nearest", "bicubic"}


def test_fresh_report_is_written_separately_per_scenario(trained: Path) -> None:
    """Scenarios are reported separately and must not overwrite each other."""
    evaluate_fresh(trained, "fresh_id", count=2)
    evaluate_fresh(trained, "ring_ood", count=2)
    for scenario in SCENARIOS:
        path = trained / f"evaluation_fresh_{scenario}.json"
        assert path.is_file()
        assert json.loads(path.read_text())["scenario"] == scenario
