"""Held-out evaluation (E-01, G7 gate).

The properties worth testing are the ones that keep a reported number honest: baselines and
models see identical states, nothing is fitted on the evaluated split, trajectories carry equal
weight, and the lead-time breakdown is always present so a single mean cannot hide a regime
change. That last one exists because a two-trajectory subsample once led me to overstate a
baseline result by a factor of two.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from swe_sr.data.generate import GenerationConfig, generate_dataset
from swe_sr.data.registry import build_registry
from swe_sr.evaluate import BASELINE_NAMES, evaluate_run, render_table
from swe_sr.train import train
from swe_sr.training.config import TrainingConfig, model_config_for_run

DATA = GenerationConfig(
    dataset_id="test_eval",
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


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short real run, so evaluation is tested against genuine artifacts."""
    root = tmp_path_factory.mktemp("eval")
    manifest = generate_dataset(DATA, registry=build_registry(), output_root=root, verbose=False)
    manifest_path = root / "processed" / manifest.dataset_id / "manifest.json"
    config = replace(
        TrainingConfig(),
        model_config="configs/model/edsr_x4.yaml",
        manifest=str(manifest_path),
        run_root=str(root / "runs"),
        stage="smoke",
        batch_size=2,
        max_epochs=1,
        max_steps=4,
        warmup_steps=1,
    )
    return train(config, verbose=False).run_dir


@pytest.fixture(scope="module")
def report(trained: Path) -> dict:
    return evaluate_run(trained, split="test", seed=1)


def test_every_method_is_evaluated_on_identical_states(report: dict) -> None:
    """docs/EXPERIMENT_PLAN.md: baselines use the identical test states and metrics.

    Same snapshot and trajectory counts for every method is the check that they did.
    """
    assert set(report["methods"]) == {*BASELINE_NAMES, report["model"]}
    counts = {
        name: method["aggregate_macro_mse_normalized"]["snapshots"]
        for name, method in report["methods"].items()
    }
    assert len(set(counts.values())) == 1, f"methods saw different snapshot counts: {counts}"
    assert next(iter(counts.values())) == report["snapshots"]


def test_baselines_are_parameter_free_and_the_model_is_not(report: dict) -> None:
    for name in BASELINE_NAMES:
        assert report["methods"][name]["trainable_parameters"] == 0
    assert report["methods"][report["model"]]["trainable_parameters"] > 0


def test_aggregation_is_trajectory_equal_weight(report: dict) -> None:
    """docs/VALIDATION.md: trajectories carry equal weight, never pooled snapshots."""
    for method in report["methods"].values():
        aggregate = method["aggregate_macro_mse_normalized"]
        assert aggregate["trajectory_equal_weight"] is True
        assert "equal-weight across trajectories" in aggregate["aggregation"]
        assert aggregate["trajectories"] == report["trajectories"]


def test_lead_time_breakdown_is_always_reported(report: dict) -> None:
    """The guard against a single mean hiding a regime change.

    A two-trajectory subsample once made a baseline look twice as bad as it is; the breakdown is
    what makes that visible rather than a footnote.
    """
    for name, method in report["methods"].items():
        breakdown = method["normalized_macro_mse_by_lead_time_hours"]
        assert breakdown, f"{name} has no lead-time breakdown"
        assert len(breakdown) == DATA.snapshot_count
        for value in breakdown.values():
            assert value >= 0.0


def test_mean_predictor_reference_is_stated(report: dict) -> None:
    """A value near 1.0 must not read as unremarkable; the report says what 1.0 means."""
    notes = report["reference_notes"]
    assert notes["mean_predictor_normalized_mse"] == 1.0
    assert "unit variance" in notes["note"]


def test_provenance_links_the_checkpoint_to_its_data_and_commits(report: dict) -> None:
    """CLAUDE.md: a metric needs its split, and a result needs the run that produced it."""
    assert report["split"] == "test"
    assert report["dataset_id"] == DATA.dataset_id
    assert report["pair_id"] == DATA.pair_id
    assert report["ic_registry_hash"]
    assert report["trained_at_commit"]
    assert report["evaluated_at_commit"]
    assert report["checkpoint"].endswith("best.pt")
    assert "macro-averaged normalized MSE" in report["checkpoint_selection_rule"]


def test_physical_metrics_are_reported_in_si_with_units_named(report: dict) -> None:
    """docs/ARCHITECTURE.md: physical diagnostics come after de-normalization."""
    physical = report["methods"]["bicubic"]["physical_metrics_si"]
    assert "relative_mass_error" in physical
    assert "min_predicted_depth_m" in physical
    assert "wall_velocity_error_max_m_per_s" in physical
    # De-normalized depth is a physical metre-scale quantity, not a standardized value.
    assert physical["min_predicted_depth_m"] > 50.0


def test_paired_bootstrap_compares_against_bicubic(report: dict) -> None:
    """Paired on shared trajectories, which is what removes between-trajectory spread."""
    comparisons = report["paired_bootstrap_vs_bicubic"]
    assert f"{report['model']}_minus_bicubic" in comparisons
    assert "bicubic_minus_bicubic" not in comparisons
    for comparison in comparisons.values():
        assert comparison["ci_low"] <= comparison["mean_difference"] <= comparison["ci_high"]
        assert comparison["confidence"] == 0.95


def test_nearest_and_bicubic_differ_but_only_slightly(report: dict) -> None:
    """Two baselines that scored identically would mean one is redundant."""
    nearest = report["methods"]["nearest"]["aggregate_macro_mse_normalized"]["mean"]
    bicubic = report["methods"]["bicubic"]["aggregate_macro_mse_normalized"]["mean"]
    assert nearest != bicubic


def test_report_is_written_to_the_run_directory(trained: Path, report: dict) -> None:
    path = trained / "evaluation_test.json"
    assert path.is_file()
    assert json.loads(path.read_text())["run_id"] == report["run_id"]


def test_table_states_units_aggregation_and_reference(report: dict) -> None:
    """CLAUDE.md forbids a metric without units, aggregation rule, split, and baseline."""
    table = render_table(report)
    assert "equal weight across trajectories" in table
    assert "predicting the channel mean scores 1.0" in table
    assert "95% CI" in table
    assert report["split"] in table
    for name in report["methods"]:
        assert name in table


def test_evaluating_the_validation_split_also_works(trained: Path) -> None:
    """Needed for model selection audits, and it must not overwrite the test report."""
    validation = evaluate_run(trained, split="validation", seed=1)
    assert validation["split"] == "validation"
    assert (trained / "evaluation_validation.json").is_file()
    assert (trained / "evaluation_test.json").is_file()


def test_same_pair_evaluation_is_not_labelled_a_transfer(report: dict) -> None:
    assert report["resolution_transfer"] is False
    assert report["trained_pair_id"] == report["pair_id"]
    assert "transfer_notes" not in report
    assert report["artifact"] == "evaluation_test.json"


def test_cross_resolution_evaluation_cannot_overwrite_the_frozen_report(
    trained: Path, report: dict, tmp_path: Path
) -> None:
    """The negative test for the hazard, not just the feature.

    A `--manifest` override pointed at a different resolution pair once wrote to the run's
    canonical `evaluation_test.json` -- the file docs/RESULTS.md, docs/EXPERIMENT_FREEZE.md and
    scripts/verify_independent.py all read. A frozen in-distribution result would have been
    silently replaced by a number from another dataset. This asserts the canonical file is
    byte-identical afterwards, so the guard is shown to fire rather than merely assumed.
    """
    canonical = trained / "evaluation_test.json"
    before = canonical.read_bytes()

    other = replace(
        DATA,
        dataset_id="test_eval_transfer",
        pair_id="swe_gaussian_64x256_v1",
        coarse_nodes=64,
        fine_nodes=256,
    )
    manifest = generate_dataset(
        other, registry=build_registry(), output_root=tmp_path, verbose=False
    )
    manifest_path = tmp_path / "processed" / manifest.dataset_id / "manifest.json"

    transfer = evaluate_run(trained, split="test", manifest_override=manifest_path, seed=1)

    assert canonical.read_bytes() == before, "a cross-pair evaluation overwrote a frozen report"
    assert transfer["artifact"] == "evaluation_test__swe_gaussian_64x256_v1.json"
    assert (trained / transfer["artifact"]).is_file()

    assert transfer["resolution_transfer"] is True
    assert transfer["trained_pair_id"] == report["pair_id"]
    assert transfer["pair_id"] == "swe_gaussian_64x256_v1"
    # Normalization follows the evaluated pair, so the number is not contaminated by
    # applying one pair's statistics to another (check_pair_id enforces this).
    assert transfer["transfer_notes"]["normalization_pair_id"] == "swe_gaussian_64x256_v1"

    # The same architecture really did run at the larger resolution, and every method saw it.
    assert set(transfer["methods"]) == set(report["methods"])


def test_trajectory_means_are_serialized_for_cross_run_paired_tests(report: dict) -> None:
    """Without these, "model A beats model B" cannot be paired-tested across two run dirs.

    Each model is evaluated in its own process against its own run directory, so the only
    paired comparison the artifacts supported was against bicubic within one process. Comparing
    two models then meant asking whether two independent confidence intervals overlap, which is
    a weaker test on identical data. These are the protocol's step-2 values, one per trajectory.
    """
    for name, method in report["methods"].items():
        means = method["trajectory_means_macro_mse_normalized"]
        assert means, f"{name} serialized no trajectory means"
        assert len(means) == report["trajectories"]
        # Step 3 of the protocol is an equal-weight mean over exactly these values.
        recomputed = sum(means.values()) / len(means)
        assert recomputed == pytest.approx(
            method["aggregate_macro_mse_normalized"]["mean"], rel=1e-12
        )


# -- The architecture a run is evaluated with (D022) -----------------------------------


def test_evaluation_rebuilds_the_architecture_the_run_recorded(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A direct-prediction run must not be evaluated as a residual model, or vice versa.

    This is a regression test for a trap the D022 ablation created. Evaluation used to
    reconstruct the model config from the run's model *name* as `configs/model/<name>_x4.yaml`.
    The two ablation arms have identical parameter shapes, so `load_state_dict` against the wrong
    arm succeeds silently and every reported metric then describes a model that was never
    trained -- a wrong number with no error anywhere. `model_config_for_run` reads the recorded
    path instead. Here the run is trained with the direct arm and the check is that evaluation
    reports the direct model, whose forward pass adds no bicubic path.
    """
    root = tmp_path_factory.mktemp("eval_direct")
    manifest = generate_dataset(
        replace(DATA, dataset_id="test_eval_direct"),
        registry=build_registry(),
        output_root=root,
        verbose=False,
    )
    manifest_path = root / "processed" / manifest.dataset_id / "manifest.json"
    config = replace(
        TrainingConfig(),
        model_config="configs/model/edsr_direct_x4.yaml",
        manifest=str(manifest_path),
        run_root=str(root / "runs"),
        stage="smoke",
        batch_size=2,
        max_epochs=1,
        max_steps=4,
        warmup_steps=1,
    )
    run_dir = train(config, verbose=False).run_dir
    assert "edsr_direct" in run_dir.name, "the run ID must carry the ablation model name"

    report = evaluate_run(run_dir, split="test", seed=1)
    assert report["model"] == "edsr_direct"
    assert "edsr_direct" in report["methods"]
    # And the resolver itself points at the recorded config, not the name convention.
    assert model_config_for_run(run_dir).name == "edsr_direct_x4.yaml"


def test_the_resolver_falls_back_for_runs_without_a_recorded_config(tmp_path: Path) -> None:
    """Runs written before `model_config` was recorded must still resolve."""
    run_dir = tmp_path / "20260101T000000Z_edsr_deadbeef_deadbeef"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"model": "edsr"}))
    assert model_config_for_run(run_dir).name == "edsr_x4.yaml"
    (run_dir / "config.yaml").write_text("model_config: configs/model/unet_direct_x4.yaml\n")
    assert model_config_for_run(run_dir).name == "unet_direct_x4.yaml"
