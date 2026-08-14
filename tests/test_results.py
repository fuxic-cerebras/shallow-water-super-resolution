"""Tests for the committed results index (D024).

The index is the only path by which a measured number reaches the documentation, so its
guards are load bearing: a registry that names the same run twice, arms evaluated on different
manifests, or a baseline that disagrees between runs would all produce a table that looks
authoritative and compares different things.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from swe_sr import results
from swe_sr.results import RegistryError

RUNS_PRESENT = all(
    (results.RUNS_ROOT / arm["run_id"] / "evaluation_test.json").is_file()
    for arm in results.load_registry()["arms"]
)
needs_runs = pytest.mark.skipif(
    not RUNS_PRESENT,
    reason="run artifacts are gitignored; present only where the experiments were executed",
)

BANDS = [
    {"label": "<= 12 h", "max": 12.0},
    {"label": "16-24 h", "min": 16.0, "max": 24.0},
    {"label": "> 24 h", "min": 24.0},
]


def _registry(tmp_path: Path, arms: list[dict[str, Any]]) -> Path:
    path = tmp_path / "runs.yaml"
    path.write_text(yaml.safe_dump({"schema": 1, "arms": arms, "lead_time_bands": BANDS}))
    return path


def _arm(name: str, **overrides: Any) -> dict[str, Any]:
    arm = {
        "id": name,
        "label": name,
        "run_id": f"20260101T000000Z_{name}_deadbeef_cafebabe",
        "role": "primary",
        "experiment": "T-03",
    }
    arm.update(overrides)
    return arm


# ------------------------------------------------------------------------------------------
# registry validation
# ------------------------------------------------------------------------------------------


def test_the_committed_registry_is_valid() -> None:
    registry = results.load_registry()
    assert registry["arms"]
    assert results.frozen_run_ids() <= results.registered_run_ids()


def test_duplicate_arm_ids_are_rejected(tmp_path: Path) -> None:
    arms = [_arm("edsr"), _arm("edsr", run_id="20260101T000001Z_edsr_deadbeef_cafebabe")]
    with pytest.raises(RegistryError, match="duplicate arm ids"):
        results.load_registry(_registry(tmp_path, arms))


def test_one_run_under_two_ids_is_rejected(tmp_path: Path) -> None:
    """Two ids for one run would double-count it in any table that selected both."""
    arms = [_arm("edsr"), _arm("edsr_again", run_id=_arm("edsr")["run_id"])]
    with pytest.raises(RegistryError, match="same run_id"):
        results.load_registry(_registry(tmp_path, arms))


def test_a_reference_to_an_unknown_arm_is_rejected(tmp_path: Path) -> None:
    arms = [_arm("convmixer_droppath", role="variant", reference="convmixer")]
    with pytest.raises(RegistryError, match="unknown arm convmixer"):
        results.load_registry(_registry(tmp_path, arms))


# ------------------------------------------------------------------------------------------
# lead-time bands
# ------------------------------------------------------------------------------------------


def test_band_means_average_within_half_open_intervals() -> None:
    curve = {"1.0": 0.1, "11.0": 0.3, "16.5": 1.0, "23.0": 2.0, "30.0": 5.0}
    means = results._band_means(curve, BANDS, context="test")
    assert means == {"<= 12 h": pytest.approx(0.2), "16-24 h": pytest.approx(1.5), "> 24 h": 5.0}


def test_a_lead_time_exactly_on_a_boundary_is_an_error() -> None:
    """Which band it belongs to would depend on a convention the band labels do not state."""
    curve = {"1.0": 0.1, "12.0": 0.3, "20.0": 1.0, "30.0": 5.0}
    with pytest.raises(RegistryError, match="exactly on band boundary"):
        results._band_means(curve, BANDS, context="test")


def test_an_empty_band_is_an_error() -> None:
    with pytest.raises(RegistryError, match="no snapshots fall in band"):
        results._band_means({"1.0": 0.1}, BANDS, context="test")


# ------------------------------------------------------------------------------------------
# building the index from artifacts
# ------------------------------------------------------------------------------------------


def test_a_registered_run_without_an_evaluation_is_an_error(tmp_path: Path) -> None:
    registry = yaml.safe_load(_registry(tmp_path, [_arm("edsr")]).read_text())
    with pytest.raises(RegistryError, match="does not exist"):
        results.build_index(registry, runs_root=tmp_path / "runs")


@needs_runs
def test_the_committed_index_is_reproducible_from_the_artifacts() -> None:
    """The index must be built, never edited. This is the check that enforces it."""
    rebuilt = results.render_index(results.build_index())
    assert rebuilt == results.INDEX_PATH.read_text()


@needs_runs
def test_every_arm_shares_one_manifest_split_and_snapshot_count() -> None:
    index = results.build_index()
    assert index["provenance"]["split"] == "test"
    assert index["provenance"]["snapshots"] == 1576
    assert index["provenance"]["trajectories"] == 8


@needs_runs
def test_paired_comparisons_state_a_reason_when_unavailable() -> None:
    """The frozen artifacts predate D021 and carry no per-trajectory means. Say so, do not omit."""
    paired = results.build_index()["paired_vs_reference"]
    assert "unavailable" in paired["unet_direct"]
    assert "predate D021" in paired["unet_direct"]["unavailable"]

    droppath = paired["convmixer_droppath"]
    assert droppath["reference"] == "convmixer"
    # Negative favours the arm, and this is the one A-05 result that resolves.
    assert droppath["mean_difference"] == pytest.approx(-0.00561, abs=5e-6)
    assert droppath["excludes_zero"] is True
    assert droppath["arm_better_on"] == "6 of 8"


@needs_runs
def test_gap_is_measured_at_the_selected_checkpoint_not_the_last_epoch() -> None:
    """ConvMixer early-stops, so its final training loss comes from discarded weights."""
    arm = results.build_index()["arms"]["convmixer"]
    training = arm["training"]
    assert training["best_epoch"] == 21 and training["epochs"] == 36
    assert training["gap"] == pytest.approx(
        training["best_validation_mse"] / training["train_mse_at_best_epoch"]
    )
    assert training["gap"] == pytest.approx(3.61, abs=0.005)
    # The end-of-run loss is lower, which is exactly why it is the wrong denominator.
    assert training["final_train_mse"] < training["train_mse_at_best_epoch"]
