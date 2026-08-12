"""Dataset validation gates, and the negative tests that prove they bite (V-01).

`docs/VALIDATION.md` names six negative tests: split overlap, mismatched saved times,
coordinate-hash mismatch, wrong vector reflection, wrong normalization pair ID, and invalid
non-x4 output. The first three and the fifth are validator concerns and live here; wrong
vector reflection is in `test_processing.py` and non-x4 output belongs to the model gate.

Every corruption here is applied to a *copy* of a released dataset. Released artifacts are
immutable, and a test that mutated one would poison every later test in the session.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from swe_sr.data.generate import GenerationConfig, generate_dataset
from swe_sr.data.registry import build_registry
from swe_sr.data.validate import validate_dataset

CONFIG = GenerationConfig(
    dataset_id="test_validate",
    pair_id="swe_gaussian_32x128_v1",
    resolution_family="swe_gaussian_x4",
    coarse_nodes=32,
    fine_nodes=128,
    discard_steps=12,
    sample_stride=6,
    snapshot_count=3,
    diagnostic_stride=6,
    trajectory_limit=1,
)


@pytest.fixture(scope="module")
def released(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A clean released dataset; returns the release root containing raw/ and processed/."""
    root = tmp_path_factory.mktemp("validate")
    generate_dataset(CONFIG, registry=build_registry(), output_root=root, verbose=False)
    return root


def _processed(root: Path) -> Path:
    return root / "processed" / CONFIG.dataset_id / "manifest.json"


def _corrupt_copy(released: Path, tmp_path: Path) -> Path:
    """Copy a release so a test can corrupt it without touching the immutable original."""
    target = tmp_path / "copy"
    shutil.copytree(released, target)
    return target


def _failing_check_names(root: Path) -> set[str]:
    report = validate_dataset(_processed(root))
    return {result.name for result in report.failures}


def _fails_matching(root: Path, fragment: str) -> bool:
    """True if some failing check's name contains `fragment`.

    Substring matching keeps these negative tests robust against cosmetic relabelling of a
    check, which is presentation rather than behaviour.
    """
    return any(fragment in name for name in _failing_check_names(root))


# -- The happy path --------------------------------------------------------------------


def test_a_clean_release_passes_every_gate(released: Path) -> None:
    report = validate_dataset(_processed(released))
    assert report.passed, report.render()
    # Guard against a validator that silently stops running checks.
    assert len(report.results) >= 12, f"expected at least 12 checks, ran {len(report.results)}"


def test_report_renders_all_check_names(released: Path) -> None:
    report = validate_dataset(_processed(released))
    rendered = report.render()
    assert "RESULT: PASS" in rendered
    for result in report.results:
        assert result.name in rendered


def test_raw_manifest_can_be_validated_without_normalization(released: Path) -> None:
    """The raw manifest carries no normalization block (D019), so that gate is skippable."""
    raw_manifest = released / "raw" / CONFIG.dataset_id / "manifest.json"
    report = validate_dataset(raw_manifest, check_normalization=False)
    assert report.passed, report.render()


def test_validating_a_raw_manifest_with_normalization_required_fails(released: Path) -> None:
    """Asking for the normalization gate against the raw manifest must fail loudly.

    Otherwise a caller could believe normalization was verified when it was never present.
    """
    raw_manifest = released / "raw" / CONFIG.dataset_id / "manifest.json"
    report = validate_dataset(raw_manifest, check_normalization=True)
    assert not report.passed
    assert any("normalization" in failure.name for failure in report.failures)


# -- Negative tests: each gate must actually fire --------------------------------------


def test_split_overlap_is_detected(released: Path, tmp_path: Path) -> None:
    """docs/VALIDATION.md negative test: split overlap.

    Relabels a test trajectory as train, the exact leakage that would invalidate every
    number a run produces.
    """
    root = _corrupt_copy(released, tmp_path)
    path = _processed(root)
    payload = json.loads(path.read_text())
    for record in payload["trajectories"]:
        if record["split"] == "test":
            record["split"] = "train"
            break
    path.write_text(json.dumps(payload))

    failures = _failing_check_names(root)
    assert "split membership agrees with the IC registry" in failures


def test_duplicated_trajectory_across_splits_is_detected(released: Path, tmp_path: Path) -> None:
    """A trajectory present in two splits at once, rather than merely mislabelled."""
    root = _corrupt_copy(released, tmp_path)
    path = _processed(root)
    payload = json.loads(path.read_text())
    duplicate = dict(payload["trajectories"][0])
    duplicate["split"] = "validation" if duplicate["split"] != "validation" else "test"
    payload["trajectories"].append(duplicate)
    path.write_text(json.dumps(payload))

    failures = _failing_check_names(root)
    assert "trajectory IDs and seeds disjoint across splits" in failures


def test_mismatched_saved_times_are_detected(released: Path, tmp_path: Path) -> None:
    """docs/VALIDATION.md negative test: mismatched saved times.

    Perturbs one stored time array so it no longer matches the manifest. Within a pair the
    times must be exactly equal (D003), so even a tiny offset must fail.
    """
    root = _corrupt_copy(released, tmp_path)
    manifest = json.loads(_processed(root).read_text())
    trajectory = root / "raw" / CONFIG.dataset_id / manifest["trajectories"][0]["relative_path"]
    with h5py.File(trajectory, "r+") as handle:
        handle["time"][1] = float(handle["time"][1]) + 1e-6

    failures = _failing_check_names(root)
    assert "within-pair saved times are exactly equal" in failures


def test_coordinate_hash_mismatch_is_detected(released: Path, tmp_path: Path) -> None:
    """docs/VALIDATION.md negative test: coordinate-hash mismatch.

    A shifted grid with unchanged fields is precisely the failure a coordinate hash exists to
    catch, because every array shape still looks right.
    """
    root = _corrupt_copy(released, tmp_path)
    manifest = json.loads(_processed(root).read_text())
    trajectory = root / "raw" / CONFIG.dataset_id / manifest["trajectories"][0]["relative_path"]
    with h5py.File(trajectory, "r+") as handle:
        handle["fine_x"][...] = np.asarray(handle["fine_x"][:]) + 1.0

    assert _fails_matching(root, "coordinate hashes")
    # The physical-domain gate should also notice the endpoints no longer coincide.
    assert _fails_matching(root, "physical domain")


def test_tampered_array_is_detected_by_checksum(released: Path, tmp_path: Path) -> None:
    """The integrity gate: a single altered value must fail the checksum."""
    root = _corrupt_copy(released, tmp_path)
    manifest = json.loads(_processed(root).read_text())
    trajectory = root / "raw" / CONFIG.dataset_id / manifest["trajectories"][0]["relative_path"]
    with h5py.File(trajectory, "r+") as handle:
        handle["hr"][0, 0, 0, 0] = np.float32(float(handle["hr"][0, 0, 0, 0]) + 0.5)

    failures = _failing_check_names(root)
    assert "array checksums recomputed from arrays" in failures


def test_broken_wall_condition_is_detected(released: Path, tmp_path: Path) -> None:
    """D012: the east and north walls must hold exact zeros on the raw arrays."""
    root = _corrupt_copy(released, tmp_path)
    manifest = json.loads(_processed(root).read_text())
    trajectory = root / "raw" / CONFIG.dataset_id / manifest["trajectories"][0]["relative_path"]
    with h5py.File(trajectory, "r+") as handle:
        handle["hr"][0, 1, 0, -1] = np.float32(1e-9)  # east-wall u

    assert _fails_matching(root, "wall velocities")


def test_negative_depth_is_detected(released: Path, tmp_path: Path) -> None:
    """A state that dries the basin is physically inadmissible and must not pass."""
    root = _corrupt_copy(released, tmp_path)
    manifest = json.loads(_processed(root).read_text())
    trajectory = root / "raw" / CONFIG.dataset_id / manifest["trajectories"][0]["relative_path"]
    with h5py.File(trajectory, "r+") as handle:
        handle["hr"][0, 0, 5, 5] = np.float32(-500.0)  # far below the 100 m resting depth

    failures = _failing_check_names(root)
    assert "all values finite and total depth positive" in failures


def test_non_finite_value_is_detected(released: Path, tmp_path: Path) -> None:
    root = _corrupt_copy(released, tmp_path)
    manifest = json.loads(_processed(root).read_text())
    trajectory = root / "raw" / CONFIG.dataset_id / manifest["trajectories"][0]["relative_path"]
    with h5py.File(trajectory, "r+") as handle:
        handle["hr"][0, 0, 3, 3] = np.float32("nan")

    failures = _failing_check_names(root)
    assert "all values finite and total depth positive" in failures


def test_wrong_normalization_pair_id_is_detected(released: Path, tmp_path: Path) -> None:
    """docs/VALIDATION.md negative test: wrong normalization pair ID.

    The dangerous case, because nothing else looks wrong: training would proceed on
    mis-scaled inputs and produce plausible numbers.
    """
    root = _corrupt_copy(released, tmp_path)
    path = _processed(root)
    payload = json.loads(path.read_text())
    payload["normalization"]["pair_id"] = "swe_gaussian_64x256_v1"
    path.write_text(json.dumps(payload))

    failures = _failing_check_names(root)
    assert "recomputed training normalization matches the manifest" in failures


def test_normalization_fitted_on_the_wrong_split_is_detected(
    released: Path, tmp_path: Path
) -> None:
    """Fitting on anything but train is leakage, and the manifest records which split was used."""
    root = _corrupt_copy(released, tmp_path)
    path = _processed(root)
    payload = json.loads(path.read_text())
    payload["normalization"]["fitted_on_split"] = "test"
    path.write_text(json.dumps(payload))

    failures = _failing_check_names(root)
    assert "recomputed training normalization matches the manifest" in failures


def test_altered_normalization_statistics_are_detected(released: Path, tmp_path: Path) -> None:
    """Recomputation from arrays must beat trusting the recorded numbers."""
    root = _corrupt_copy(released, tmp_path)
    path = _processed(root)
    payload = json.loads(path.read_text())
    payload["normalization"]["channels"]["eta"]["mean"] *= 1.5
    payload["normalization"]["channels"]["eta"]["mean"] += 0.1
    path.write_text(json.dumps(payload))

    failures = _failing_check_names(root)
    assert "recomputed training normalization matches the manifest" in failures


def test_inconsistent_shape_factor_is_detected(released: Path, tmp_path: Path) -> None:
    """docs/VALIDATION.md negative test: invalid non-x4 declaration."""
    root = _corrupt_copy(released, tmp_path)
    path = _processed(root)
    payload = json.loads(path.read_text())
    payload["grid"]["shape_factor"] = 3
    path.write_text(json.dumps(payload))

    failures = _failing_check_names(root)
    assert "manifest schema and grid contract" in failures


def test_wrong_saved_times_in_the_manifest_are_detected(released: Path, tmp_path: Path) -> None:
    """Times must be exact multiples of the shared step, not rounded intervals."""
    root = _corrupt_copy(released, tmp_path)
    path = _processed(root)
    payload = json.loads(path.read_text())
    payload["time"]["saved_times"] = [round(t, 1) for t in payload["time"]["saved_times"]]
    path.write_text(json.dumps(payload))

    failures = _failing_check_names(root)
    assert "within-pair saved times are exactly equal" in failures


def test_a_broken_check_reports_as_a_failure_not_a_crash(released: Path, tmp_path: Path) -> None:
    """A validator that dies partway through hides the checks it never reached.

    Removing a trajectory file should surface as failures, with the remaining gates still
    reported.
    """
    root = _corrupt_copy(released, tmp_path)
    manifest = json.loads(_processed(root).read_text())
    trajectory = root / "raw" / CONFIG.dataset_id / manifest["trajectories"][0]["relative_path"]
    trajectory.unlink()

    report = validate_dataset(_processed(root))
    assert not report.passed
    assert len(report.results) >= 12, "validation stopped early instead of reporting every gate"
