"""Final comparison report (E-03).

The report is where a misleading number would do the most damage, so the tests target the
reporting rules from `CLAUDE.md` rather than formatting: units, aggregation, split and baseline
always stated; planned values never presented as measured; and a stale artifact flagged rather
than silently trusted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from swe_sr.report import build_report


def _evaluation(model: str, stage: str = "diagnostic") -> dict:
    def method(mean: float, params: int) -> dict:
        return {
            "method": "m",
            "trainable_parameters": params,
            "normalized_metrics": {"rel_l2_eta": 0.3, "rel_l2_u": 0.4, "rel_l2_v": 0.5},
            "physical_metrics_si": {"relative_mass_error": 0.02},
            "aggregate_macro_mse_normalized": {
                "mean": mean,
                "ci_low": mean - 0.05,
                "ci_high": mean + 0.05,
                "median": mean,
                "std": 0.1,
                "trajectories": 8,
                "snapshots": 1576,
                "trajectory_equal_weight": True,
                "aggregation": "per-snapshot, then within-trajectory mean, then equal-weight "
                "across trajectories",
                "confidence": 0.95,
            },
            "normalized_macro_mse_by_lead_time_hours": {"2.010": 0.01, "34.860": 1.10},
            "seconds_per_frame": 0.03,
        }

    return {
        "run_id": f"20260101T000000Z_{model}_aaaaaaaa_bbbbbbbb",
        "stage": stage,
        "model": model,
        "split": "test",
        "checkpoint": "checkpoints/best.pt",
        "checkpoint_selection_rule": "lowest macro-averaged normalized MSE on the full "
        "validation split",
        "dataset_id": "swe_gaussian_32x128_v1",
        "pair_id": "swe_gaussian_32x128_v1",
        "ic_registry_hash": "deadbeef",
        "trained_at_commit": "abc1234",
        "evaluated_at_commit": "abc1234",
        "seed": 0,
        "trajectories": 8,
        "snapshots": 1576,
        "methods": {
            "nearest": method(0.47, 0),
            "bicubic": method(0.468, 0),
            model: method(0.143, 1_517_571),
        },
        "paired_bootstrap_vs_bicubic": {
            f"{model}_minus_bicubic": {
                "mean_difference": -0.325,
                "ci_low": -0.511,
                "ci_high": -0.152,
                "trajectories": 8.0,
                "confidence": 0.95,
            }
        },
        "reference_notes": {"mean_predictor_normalized_mse": 1.0, "note": "unit variance"},
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "run_edsr"
    directory.mkdir()
    (directory / "evaluation_test.json").write_text(json.dumps(_evaluation("edsr")))
    return directory


def test_report_states_units_aggregation_split_and_baseline(run_dir: Path) -> None:
    """CLAUDE.md: never report a metric without all four."""
    text = build_report([run_dir])
    assert "equal weight" in text
    assert "dimensionless" in text
    assert "split: **test**" in text
    assert "predicting the channel mean" in text
    assert "resampling **trajectories**" in text or "resampling *trajectories*" in text


def test_report_records_provenance(run_dir: Path) -> None:
    text = build_report([run_dir])
    assert "swe_gaussian_32x128_v1" in text
    assert "deadbeef" in text
    assert "abc1234" in text
    assert "checkpoints/best.pt" in text


def test_a_non_full_stage_is_called_out(run_dir: Path) -> None:
    """A diagnostic run must not be readable as the frozen experiment."""
    text = build_report([run_dir])
    assert "not the frozen T-03 experiment" in text
    assert "I-02" in text


def test_a_full_stage_run_carries_no_such_warning(tmp_path: Path) -> None:
    directory = tmp_path / "run_full"
    directory.mkdir()
    (directory / "evaluation_test.json").write_text(json.dumps(_evaluation("edsr", stage="full")))
    assert "not the frozen T-03 experiment" not in build_report([directory])


def test_lead_time_table_is_always_included(run_dir: Path) -> None:
    """The aggregate hides a regime change, so the stratified table is not optional."""
    text = build_report([run_dir])
    assert "by lead time" in text
    assert "worse than bicubic at short lead times" in text


def test_limitations_state_the_decorrelation_and_augmentation_findings(run_dir: Path) -> None:
    """A reader must not have to rediscover D002's consequence or D018 from the numbers."""
    text = build_report([run_dir])
    assert "independent" in text and "decorrelate" in text
    assert "D018" in text
    assert "No claim of physical generalization" in text


def test_baselines_are_listed_once_across_multiple_runs(tmp_path: Path) -> None:
    """Repeating identical baselines per model would imply they were measured separately."""
    first = tmp_path / "a"
    second = tmp_path / "b"
    for directory, model in ((first, "edsr"), (second, "unet")):
        directory.mkdir()
        (directory / "evaluation_test.json").write_text(json.dumps(_evaluation(model)))
    text = build_report([first, second])
    assert text.count("\n| bicubic |") == 1
    assert "| edsr |" in text and "| unet |" in text


def test_missing_evaluation_artifact_is_an_explicit_error(tmp_path: Path) -> None:
    """A report built from nothing must fail loudly, not emit an empty table."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=re.escape("run `python -m swe_sr.evaluate` first")):
        build_report([empty])


def test_a_run_without_an_artifact_is_skipped_not_faked(tmp_path: Path) -> None:
    """docs/PROJECT_SPEC.md: empty cells are populated only by generated results."""
    good = tmp_path / "good"
    good.mkdir()
    (good / "evaluation_test.json").write_text(json.dumps(_evaluation("edsr")))
    missing = tmp_path / "missing"
    missing.mkdir()

    text = build_report([good, missing])
    assert "| edsr |" in text
    assert "| unet |" not in text


def test_a_stale_fresh_artifact_is_flagged_rather_than_trusted(run_dir: Path) -> None:
    """An artifact predating lead-time spans cannot be compared safely, so say so.

    Silently omitting the span is what made an earlier fresh comparison misleading; an
    unverifiable span must read as unverifiable.
    """
    stale = {
        "run_id": "x",
        "stage": "pilot",
        "model": "edsr",
        "scenario": "fresh_id",
        "methods": {
            "bicubic": {
                "aggregate_macro_mse_normalized": {"mean": 0.4, "ci_low": 0.3, "ci_high": 0.5}
            }
        },
    }
    (run_dir / "evaluation_fresh_fresh_id.json").write_text(json.dumps(stale))
    text = build_report([run_dir])
    assert "UNKNOWN" in text
    assert "regenerate this artifact" in text


def test_fresh_section_states_it_is_reported_separately(run_dir: Path) -> None:
    """docs/DATASET.md forbids folding fresh results into the test score."""
    fresh = {
        "run_id": "x",
        "stage": "diagnostic",
        "model": "edsr",
        "scenario": "ring_ood",
        "lead_time_hours": {"first": 2.01, "last": 34.86, "matches_training_range": True},
        "methods": {
            "bicubic": {
                "aggregate_macro_mse_normalized": {"mean": 1.33, "ci_low": 1.32, "ci_high": 1.34}
            }
        },
    }
    (run_dir / "evaluation_fresh_ring_ood.json").write_text(json.dumps(fresh))
    text = build_report([run_dir])
    assert "never mixed into the held-out test score" in text
    assert "within-workload repeatability" in text
