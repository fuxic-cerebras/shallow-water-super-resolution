"""Independent dataset validation (task D-03/V-01 gate).

    python -m swe_sr.data.validate --manifest data/processed/swe_gaussian_32x128_v1/manifest.json

Implements every pre-training gate in `docs/VALIDATION.md`. The governing principle is that
this recomputes from the stored arrays rather than trusting what the generator recorded:
`docs/VALIDATION.md` requires the verifier to recompute "checksums, normalization statistics,
split disjointness, and selected metrics from arrays; author logs are not proof."

Every check reports independently, so a run surfaces all failures at once instead of
stopping at the first. Exit code is non-zero if any check fails, which is what makes this
usable as a gate in CI or before a training launch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from swe_sr.data.manifest import DatasetManifest, coordinate_hash, load_manifest
from swe_sr.data.normalization import Normalization, fit_from_manifest
from swe_sr.data.registry import split_for_seed, trajectory_id_for_seed
from swe_sr.data.storage import (
    COARSE_ARRAY,
    FINE_ARRAY,
    read_coordinates,
    read_fields,
    read_metadata,
    read_times,
    verify_trajectory_checksums,
)

# Mass is exact to roundoff by construction; this bound allows accumulation over ~10^4 steps.
MASS_DRIFT_TOLERANCE = 1e-12
# Recomputed statistics should match the manifest to near machine precision, since both are
# float64 reductions over the same bytes. A loose bound here would defeat the point.
NORMALIZATION_TOLERANCE = 1e-9


@dataclass
class CheckResult:
    """Outcome of one named check."""

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}" + (f"\n         {self.detail}" if self.detail else "")


@dataclass
class ValidationReport:
    """All check results for one dataset."""

    dataset_id: str
    manifest_path: Path
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append(CheckResult(name=name, passed=passed, detail=detail))

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> list[CheckResult]:
        return [result for result in self.results if not result.passed]

    def render(self) -> str:
        lines = [f"dataset : {self.dataset_id}", f"manifest: {self.manifest_path}", ""]
        lines += [str(result) for result in self.results]
        lines += [
            "",
            f"{sum(r.passed for r in self.results)}/{len(self.results)} checks passed",
            "RESULT: PASS" if self.passed else f"RESULT: FAIL ({len(self.failures)} failing)",
        ]
        return "\n".join(lines)


def _guard(report: ValidationReport, name: str, check: Callable[[], str]) -> None:
    """Run one check, converting an exception into a failure rather than a crash.

    A validator that dies partway through hides the checks it never reached, which is the
    opposite of what a gate should do.
    """
    try:
        detail = check()
        report.add(name, True, detail)
    except AssertionError as error:
        report.add(name, False, str(error))
    except Exception as error:  # a broken check is a failed check, not a crash
        report.add(name, False, f"{type(error).__name__}: {error}")


def validate_dataset(manifest_path: Path, *, check_normalization: bool = True) -> ValidationReport:
    """Run every documented pre-training gate against a released dataset."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)

    # Arrays live under raw/, whether the manifest handed to us is the raw or processed one.
    raw_dir = manifest_path.parent
    if not (raw_dir / "trajectories").is_dir():
        candidate = raw_dir.parent.parent / "raw" / manifest.dataset_id
        if (candidate / "trajectories").is_dir():
            raw_dir = candidate

    report = ValidationReport(dataset_id=manifest.dataset_id, manifest_path=manifest_path)

    checks: list[tuple[str, Callable[[], str]]] = [
        ("manifest schema and grid contract", lambda: _check_manifest(manifest)),
        ("array checksums recomputed from arrays", lambda: _check_checksums(manifest, raw_dir)),
        ("shapes match node counts and x4 contract", lambda: _check_shapes(manifest, raw_dir)),
        ("coarse and fine share the physical domain", lambda: _check_domain(manifest, raw_dir)),
        ("coordinate hashes match the manifest", lambda: _check_hashes(manifest, raw_dir)),
        ("within-pair saved times are exactly equal", lambda: _check_times(manifest, raw_dir)),
        ("all values finite and total depth positive", lambda: _check_finite(manifest, raw_dir)),
        ("wall velocities exactly zero on raw arrays", lambda: _check_walls(manifest, raw_dir)),
        ("trajectory IDs and seeds disjoint across splits", lambda: _check_splits(manifest)),
        ("split membership agrees with the IC registry", lambda: _check_registry(manifest)),
        ("solver mass drift within tolerance", lambda: _check_mass_drift(manifest)),
    ]
    if check_normalization:
        checks.append(
            (
                "recomputed training normalization matches the manifest",
                lambda: _check_normalization(manifest, raw_dir),
            )
        )
    for name, check in checks:
        _guard(report, name, check)
    return report


# -- Individual checks ----------------------------------------------------------------
#
# Each returns a human-readable detail string on success and raises AssertionError with an
# explanation on failure.


def _check_manifest(manifest: DatasetManifest) -> str:
    assert manifest.schema_version >= 1, f"unexpected schema version {manifest.schema_version}"
    assert manifest.trajectories, "manifest declares no trajectories"
    assert manifest.shape_factor >= 2, f"shape factor {manifest.shape_factor} is not an upscale"
    assert manifest.fine_nodes == manifest.coarse_nodes * manifest.shape_factor, (
        f"{manifest.coarse_nodes} x {manifest.shape_factor} != {manifest.fine_nodes}"
    )
    expected_ratio = (manifest.fine_nodes - 1) / (manifest.coarse_nodes - 1)
    assert abs(manifest.spacing_ratio - expected_ratio) < 1e-12, (
        f"spacing ratio {manifest.spacing_ratio} does not match the endpoint convention "
        f"value {expected_ratio}"
    )
    assert manifest.endpoint_convention == "both_endpoints_included", (
        f"unexpected endpoint convention {manifest.endpoint_convention!r}"
    )
    assert manifest.ic_registry_hash, "manifest does not record an IC registry hash"
    assert len(manifest.saved_times) == len(manifest.sample_steps), (
        "saved_times and sample_steps have different lengths"
    )
    return (
        f"{len(manifest.trajectories)} trajectories, {len(manifest.saved_times)} frames, "
        f"x{manifest.shape_factor} node factor, spacing ratio {manifest.spacing_ratio:.6f}, "
        f"commit {manifest.git_commit}"
    )


def _check_checksums(manifest: DatasetManifest, raw_dir: Path) -> str:
    total = 0
    for record in manifest.trajectories:
        mismatched = verify_trajectory_checksums(raw_dir / record.relative_path, record.arrays)
        assert not mismatched, (
            f"trajectory {record.trajectory_id} has mismatched checksums for {mismatched}"
        )
        total += len(record.arrays)
    return f"{total} arrays recomputed, all matching"


def _check_shapes(manifest: DatasetManifest, raw_dir: Path) -> str:
    frames = len(manifest.saved_times)
    coarse_expected = (frames, 3, manifest.coarse_nodes, manifest.coarse_nodes)
    fine_expected = (frames, 3, manifest.fine_nodes, manifest.fine_nodes)
    for record in manifest.trajectories:
        path = raw_dir / record.relative_path
        coarse_shape = read_fields(path, COARSE_ARRAY).shape
        fine_shape = read_fields(path, FINE_ARRAY).shape
        assert coarse_shape == coarse_expected, (
            f"{record.trajectory_id}: coarse shape {coarse_shape} != {coarse_expected}"
        )
        assert fine_shape == fine_expected, (
            f"{record.trajectory_id}: fine shape {fine_shape} != {fine_expected}"
        )
    return f"all trajectories are {coarse_expected} -> {fine_expected}"


def _check_domain(manifest: DatasetManifest, raw_dir: Path) -> str:
    for record in manifest.trajectories:
        coords = read_coordinates(raw_dir / record.relative_path)
        for axis in ("x", "y"):
            coarse = coords[f"coarse_{axis}"]
            fine = coords[f"fine_{axis}"]
            # Exact equality: endpoint-aligned interpolation depends on it (D011).
            assert coarse[0] == fine[0], (
                f"{record.trajectory_id}: {axis} lower bound differs ({coarse[0]!r} vs {fine[0]!r})"
            )
            assert coarse[-1] == fine[-1], (
                f"{record.trajectory_id}: {axis} upper bound differs "
                f"({coarse[-1]!r} vs {fine[-1]!r})"
            )
    return "coarse and fine endpoints coincide exactly on both axes"


def _check_hashes(manifest: DatasetManifest, raw_dir: Path) -> str:
    for record in manifest.trajectories:
        coords = read_coordinates(raw_dir / record.relative_path)
        coarse_hash = coordinate_hash(coords["coarse_x"], coords["coarse_y"])
        fine_hash = coordinate_hash(coords["fine_x"], coords["fine_y"])
        assert coarse_hash == manifest.coarse_coordinate_hash, (
            f"{record.trajectory_id}: coarse coordinate hash mismatch"
        )
        assert fine_hash == manifest.fine_coordinate_hash, (
            f"{record.trajectory_id}: fine coordinate hash mismatch"
        )
    coarse_short = manifest.coarse_coordinate_hash[:12]
    return f"coarse {coarse_short}, fine {manifest.fine_coordinate_hash[:12]}"


def _check_times(manifest: DatasetManifest, raw_dir: Path) -> str:
    expected = np.asarray(manifest.saved_times, dtype=np.float64)
    steps = np.asarray(manifest.sample_steps, dtype=np.int64)
    derived = steps * manifest.shared_time_step
    assert np.array_equal(expected, derived), (
        "manifest saved_times are not exact multiples of the shared time step; "
        "they must be derived from the resolved config, not rounded intervals"
    )
    for record in manifest.trajectories:
        stored = read_times(raw_dir / record.relative_path)
        assert np.array_equal(stored, expected), (
            f"{record.trajectory_id}: stored times differ from the manifest"
        )
    return (
        f"{len(expected)} frames, dt={manifest.shared_time_step:.6f} s, "
        f"interval={expected[1] - expected[0]:.3f} s, duration={expected[-1] / 3600:.2f} h"
    )


def _check_finite(manifest: DatasetManifest, raw_dir: Path) -> str:
    minimum_depth = float("inf")
    for record in manifest.trajectories:
        path = raw_dir / record.relative_path
        depth = float(read_metadata(path)["depth"])
        for name in (COARSE_ARRAY, FINE_ARRAY):
            fields = read_fields(path, name)
            assert np.isfinite(fields).all(), f"{record.trajectory_id}/{name} has non-finite values"
            total_depth = float((depth + fields[:, 0]).min())
            assert total_depth > 0.0, (
                f"{record.trajectory_id}/{name}: total depth reaches {total_depth:.6f} m"
            )
            minimum_depth = min(minimum_depth, total_depth)
    return f"all values finite; minimum total depth {minimum_depth:.4f} m"


def _check_walls(manifest: DatasetManifest, raw_dir: Path) -> str:
    """D012: the exact zero belongs on the raw staggered arrays, east and north walls."""
    for record in manifest.trajectories:
        path = raw_dir / record.relative_path
        for name in (COARSE_ARRAY, FINE_ARRAY):
            fields = read_fields(path, name)
            assert np.all(fields[:, 1, :, -1] == 0.0), (
                f"{record.trajectory_id}/{name}: east-wall u is not exactly zero"
            )
            assert np.all(fields[:, 2, -1, :] == 0.0), (
                f"{record.trajectory_id}/{name}: north-wall v is not exactly zero"
            )
    return "east-wall u and north-wall v are exactly zero everywhere"


def _check_splits(manifest: DatasetManifest) -> str:
    seen_ids: dict[str, str] = {}
    seen_seeds: dict[int, str] = {}
    counts: dict[str, int] = {}
    for record in manifest.trajectories:
        previous = seen_ids.get(record.trajectory_id)
        assert previous is None or previous == record.split, (
            f"trajectory {record.trajectory_id} appears in both {previous!r} and {record.split!r}"
        )
        assert record.seed not in seen_seeds or seen_seeds[record.seed] == record.split, (
            f"seed {record.seed} appears in both {seen_seeds[record.seed]!r} and {record.split!r}"
        )
        seen_ids[record.trajectory_id] = record.split
        seen_seeds[record.seed] = record.split
        counts[record.split] = counts.get(record.split, 0) + 1
    assert len(seen_ids) == len(manifest.trajectories), "duplicate trajectory IDs in the manifest"
    return ", ".join(f"{split}={count}" for split, count in sorted(counts.items()))


def _check_registry(manifest: DatasetManifest) -> str:
    """The manifest's split labels and IDs must agree with what the registry derives.

    Guards against a manifest that is internally consistent but disagrees with the immutable
    registry every other artifact references.
    """
    for record in manifest.trajectories:
        expected_split = split_for_seed(record.seed)
        assert record.split == expected_split, (
            f"seed {record.seed} is labelled {record.split!r} but the registry assigns "
            f"{expected_split!r}"
        )
        expected_id = trajectory_id_for_seed(record.seed)
        assert record.trajectory_id == expected_id, (
            f"seed {record.seed} has ID {record.trajectory_id} but the registry "
            f"derives {expected_id}"
        )
    return f"all {len(manifest.trajectories)} entries agree with the registry derivation"


def _check_mass_drift(manifest: DatasetManifest) -> str:
    worst = 0.0
    worst_id = ""
    for record in manifest.trajectories:
        for resolution in ("coarse", "fine"):
            drift = float(record.diagnostics[resolution]["relative_mass_drift"])
            if drift > worst:
                worst, worst_id = drift, f"{record.trajectory_id}/{resolution}"
    assert worst <= MASS_DRIFT_TOLERANCE, (
        f"worst relative mass drift {worst:.3e} at {worst_id} exceeds {MASS_DRIFT_TOLERANCE:.0e}"
    )
    return f"worst relative mass drift {worst:.3e} (tolerance {MASS_DRIFT_TOLERANCE:.0e})"


def _check_normalization(manifest: DatasetManifest, raw_dir: Path) -> str:
    """Recompute the training statistics from arrays and compare against the manifest."""
    assert manifest.normalization, (
        "manifest carries no normalization block; validate the processed manifest, not the raw one"
    )
    recorded = Normalization.from_dict(manifest.normalization)
    assert recorded.split == "train", (
        f"normalization was fitted on {recorded.split!r}; it must be fitted on the train split only"
    )
    assert recorded.pair_id == manifest.pair_id, (
        f"normalization pair ID {recorded.pair_id!r} does not match manifest {manifest.pair_id!r}"
    )

    # The persisted mean/std are derived values; `from_dict` rebuilds them from the
    # count/sum/sum_squared accumulators and ignores them. That makes them decorative, and a
    # hand-edited manifest could display statistics that disagree with its own accumulators:
    # anyone reading the JSON would see one number while the loader used another. Check the
    # recorded derived values against the recorded accumulators explicitly.
    channels = manifest.normalization["channels"]
    for index, name in enumerate(("eta", "u", "v")):
        block = channels[name]
        stats = recorded.channels[index]
        for attribute, recorded_value in (("mean", block["mean"]), ("std", block["std"])):
            derived = float(getattr(stats, attribute))
            scale = max(abs(derived), 1e-12)
            assert abs(float(recorded_value) - derived) / scale < NORMALIZATION_TOLERANCE, (
                f"{name} {attribute}: manifest records {recorded_value!r} but its own "
                f"accumulators derive {derived!r}"
            )

    recomputed = fit_from_manifest(manifest, raw_dir, split="train")
    for index, name in enumerate(("eta", "u", "v")):
        for attribute in ("mean", "scale"):
            expected = float(getattr(recorded, attribute)[index])
            actual = float(getattr(recomputed, attribute)[index])
            scale = max(abs(expected), 1e-12)
            assert abs(expected - actual) / scale < NORMALIZATION_TOLERANCE, (
                f"{name} {attribute}: manifest {expected!r} but recomputed {actual!r}"
            )
    return "; ".join(
        f"{name} mean={recorded.mean[i]:+.6e} scale={recorded.scale[i]:.6e}"
        for i, name in enumerate(("eta", "u", "v"))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="path to a dataset manifest")
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="skip the normalization recomputation (useful against a raw manifest)",
    )
    args = parser.parse_args(argv)

    report = validate_dataset(args.manifest, check_normalization=not args.skip_normalization)
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
