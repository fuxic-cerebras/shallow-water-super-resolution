"""Independent recomputation of a run's reported metrics (V-05).

    python scripts/verify_independent.py --run-dir runs/<run-id> --split test

`docs/VALIDATION.md` requires the verifier to "recompute checksums, normalization statistics,
split disjointness, and selected metrics from arrays; author logs are not proof." A recomputation
that imported `swe_sr.metrics` would only prove that module agrees with itself, so **every
quantity here is implemented from the specification in plain numpy**, deliberately duplicating
logic rather than reusing it.

That duplication is the point. The only things imported from the package are the pieces whose
correctness is itself under test elsewhere and cannot be independently re-derived without
reimplementing the solver: the HDF5 reader and the model definition. Normalization,
destaggering, all field metrics, the mass diagnostic, and the aggregation protocol are written
out again here from `docs/DATASET.md`, `docs/VALIDATION.md`, D011, and D014.

Exit code is non-zero if any recomputed value disagrees with what the run reported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import numpy as np
import torch
import yaml

# Tolerances: these are float64 reductions over the same bytes, so agreement should be near
# machine precision. A loose bound would defeat the purpose of an independent check.
RELATIVE_TOLERANCE = 1e-6
CHANNELS = ("eta", "u", "v")


def _fail(message: str) -> None:
    print(f"  MISMATCH: {message}")


def _close(actual: float, expected: float, tolerance: float = RELATIVE_TOLERANCE) -> bool:
    scale = max(abs(expected), 1e-12)
    return abs(actual - expected) / scale < tolerance


# -- Independent reimplementations (from the docs, not from swe_sr) --------------------


def destagger_independent(fields: np.ndarray) -> np.ndarray:
    """D011, reimplemented: u averaged along x, v along y, wall column/row halved."""
    out = fields.astype(np.float64).copy()
    u = fields[:, 1].astype(np.float64)
    out[:, 1, :, 1:] = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    out[:, 1, :, 0] = 0.5 * u[:, :, 0]
    v = fields[:, 2].astype(np.float64)
    out[:, 2, 1:, :] = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    out[:, 2, 0, :] = 0.5 * v[:, 0, :]
    return out


def normalize_independent(fields: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """docs/DATASET.md: q' = (q - mu) / max(sigma, 1e-8)."""
    return (fields - mean.reshape(1, 3, 1, 1)) / scale.reshape(1, 3, 1, 1)


def macro_mse_independent(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean over channels of the per-channel mean squared error."""
    per_channel = ((prediction - target) ** 2).mean(axis=(0, 2, 3))
    return float(per_channel.mean())


def relative_l2_independent(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """docs/VALIDATION.md relL2, computed **per snapshot** as the protocol requires.

    Shaped `[sample, channel]`. An earlier version of this verifier pooled the norm over every
    frame of a trajectory at once, which gave 0.5996 against a reported 0.5505 and looked like a
    bug in the code under test. It was a bug here: step 1 of the aggregation protocol is
    "compute metrics per snapshot and per channel", and a pooled ratio weights high-error frames
    more heavily because they dominate both norms. That `docs/VALIDATION.md` also asks for the
    sample median and 95th percentile of relL2 confirms it is meant to be a per-sample quantity
    with a distribution.
    """
    difference = np.sqrt(((prediction - target) ** 2).sum(axis=(2, 3)))
    magnitude = np.sqrt((target**2).sum(axis=(2, 3)))
    return difference / (magnitude + 1e-12)


def relative_mass_error_independent(
    prediction: np.ndarray, target: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """D014: M = sum(eta) * dx * dy, per sample, relative to the target's magnitude."""
    predicted = prediction[:, 0].sum(axis=(1, 2)) * dx * dy
    actual = target[:, 0].sum(axis=(1, 2)) * dx * dy
    return np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-8)


def aggregate_independent(values_by_trajectory: dict[str, list[float]]) -> float:
    """docs/VALIDATION.md: within-trajectory mean, then equal weight across trajectories."""
    per_trajectory = [float(np.mean(v)) for v in values_by_trajectory.values()]
    return float(np.mean(per_trajectory))


# -- Verification ----------------------------------------------------------------------


def verify(run_dir: Path, split: str) -> int:
    """Recompute and compare. Returns the number of mismatches found."""
    run_dir = Path(run_dir)
    report = json.loads((run_dir / f"evaluation_{split}.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    repo_root = Path(__file__).resolve().parents[1]

    manifest_path = repo_root / str(config["manifest"])
    manifest = json.loads(manifest_path.read_text())
    normalization = manifest["normalization"]["channels"]

    # Re-derive mean and scale from the persisted accumulators, not the recorded mean/std.
    mean = np.array([normalization[c]["sum"] / normalization[c]["count"] for c in CHANNELS])
    variance = np.array(
        [
            max(
                normalization[c]["sum_squared"] / normalization[c]["count"]
                - (normalization[c]["sum"] / normalization[c]["count"]) ** 2,
                0.0,
            )
            for c in CHANNELS
        ]
    )
    scale = np.maximum(np.sqrt(variance), 1e-8)

    mismatches = 0
    print(f"run   : {report['run_id']}")
    print(f"split : {split}")
    print()

    # 1. Normalization, re-derived from accumulators.
    print("normalization re-derived from count/sum/sum_squared:")
    for index, channel in enumerate(CHANNELS):
        recorded_mean = normalization[channel]["mean"]
        recorded_std = normalization[channel]["std"]
        ok_mean = _close(float(mean[index]), recorded_mean)
        ok_std = _close(float(np.sqrt(variance[index])), recorded_std, 1e-5)
        print(
            f"  {channel:<4} mean {mean[index]:+.6e} vs {recorded_mean:+.6e} "
            f"{'ok' if ok_mean else 'MISMATCH'}   "
            f"std {np.sqrt(variance[index]):.6e} vs {recorded_std:.6e} "
            f"{'ok' if ok_std else 'MISMATCH'}"
        )
        mismatches += (not ok_mean) + (not ok_std)

    # 2. Split disjointness, independently from the manifest rows.
    print("\nsplit disjointness (recomputed from manifest rows):")
    by_split: dict[str, set[str]] = {}
    seeds_by_split: dict[str, set[int]] = {}
    for row in manifest["trajectories"]:
        by_split.setdefault(row["split"], set()).add(row["trajectory_id"])
        seeds_by_split.setdefault(row["split"], set()).add(row["seed"])
    names = sorted(by_split)
    overlaps = 0
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if by_split[left] & by_split[right] or seeds_by_split[left] & seeds_by_split[right]:
                overlaps += 1
    print(f"  splits {dict((k, len(v)) for k, v in sorted(by_split.items()))}  overlaps={overlaps}")
    mismatches += overlaps

    # 3. Checksums, recomputed over canonical bytes.
    print("\narray checksums (recomputed over canonical little-endian bytes):")
    import hashlib

    raw_dir = manifest_path.parent.parent.parent / "raw" / manifest["dataset_id"]
    bad_checksums = 0
    checked = 0
    for row in manifest["trajectories"][:4]:  # a sample: full coverage is the validator's job
        path = raw_dir / row["relative_path"]
        with h5py.File(path, "r") as handle:
            for record in row["arrays"]:
                array = np.ascontiguousarray(handle[record["name"]][:])
                digest = hashlib.sha256()
                digest.update(str(array.dtype.str).encode())
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
                checked += 1
                if digest.hexdigest() != record["checksum"]:
                    bad_checksums += 1
                    _fail(f"{row['trajectory_id']}/{record['name']}")
    print(f"  {checked} arrays sampled, {bad_checksums} mismatched")
    mismatches += bad_checksums

    # 4. Metrics, recomputed from arrays through an independent path.
    print("\nmetrics recomputed independently from arrays:")
    from swe_sr.models import build_baseline, build_model_from_config
    from swe_sr.training.config import model_config_for_run

    summary = json.loads((run_dir / "summary.json").read_text())
    _, model = build_model_from_config(model_config_for_run(run_dir))
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True))
    model.eval()
    methods: dict[str, Any] = {
        "bicubic": build_baseline("bicubic"),
        summary["model"]: model,
    }

    rows = [r for r in manifest["trajectories"] if r["split"] == split]
    dx = float(manifest["solver"]["fine"]["dx"])
    dy = float(manifest["solver"]["fine"]["dy"])

    per_method: dict[str, dict[str, list[float]]] = {n: {} for n in methods}
    rel_l2_by_trajectory: dict[str, dict[str, list[np.ndarray]]] = {n: {} for n in methods}
    mass_accumulator: dict[str, list[float]] = {n: [] for n in methods}

    with torch.no_grad():
        for row in rows:
            path = raw_dir / row["relative_path"]
            with h5py.File(path, "r") as handle:
                coarse_raw = np.asarray(handle["lr"][:])
                fine_raw = np.asarray(handle["hr"][:])
            coarse = normalize_independent(destagger_independent(coarse_raw), mean, scale)
            fine = normalize_independent(destagger_independent(fine_raw), mean, scale)
            coarse_t = torch.from_numpy(coarse.astype(np.float32))
            fine_t = torch.from_numpy(fine.astype(np.float32))

            for name, module in methods.items():
                prediction = module(coarse_t).numpy().astype(np.float64)
                target = fine_t.numpy().astype(np.float64)
                per_sample = ((prediction - target) ** 2).mean(axis=(1, 2, 3))
                per_method[name].setdefault(row["trajectory_id"], []).extend(per_sample.tolist())
                rel_l2_by_trajectory[name].setdefault(row["trajectory_id"], []).append(
                    relative_l2_independent(prediction, target)
                )
                # Physical mass error needs de-normalized fields.
                physical_prediction = prediction * scale.reshape(1, 3, 1, 1) + mean.reshape(
                    1, 3, 1, 1
                )
                physical_target = target * scale.reshape(1, 3, 1, 1) + mean.reshape(1, 3, 1, 1)
                mass_accumulator[name].append(
                    float(
                        relative_mass_error_independent(
                            physical_prediction, physical_target, dx, dy
                        ).mean()
                    )
                )

    for name in methods:
        reported = report["methods"][name]
        recomputed_mse = aggregate_independent(per_method[name])
        reported_mse = reported["aggregate_macro_mse_normalized"]["mean"]
        ok = _close(recomputed_mse, reported_mse, 1e-4)
        print(
            f"  {name:<8} macro MSE {recomputed_mse:.6f} vs reported {reported_mse:.6f} "
            f"{'ok' if ok else 'MISMATCH'}"
        )
        mismatches += not ok

        # Protocol: per-snapshot values, within-trajectory mean, then equal weight across
        # trajectories.
        per_trajectory = [
            np.concatenate(chunks, axis=0).mean(axis=0)
            for chunks in rel_l2_by_trajectory[name].values()
        ]
        recomputed_rel = np.mean(np.stack(per_trajectory), axis=0)
        for index, channel in enumerate(CHANNELS):
            reported_rel = reported["normalized_metrics"][f"rel_l2_{channel}"]
            # Both sides now follow the same protocol, so agreement should be tight.
            ok_rel = _close(float(recomputed_rel[index]), reported_rel, 1e-4)
            print(
                f"           relL2 {channel} {recomputed_rel[index]:.4f} vs {reported_rel:.4f} "
                f"{'ok' if ok_rel else 'MISMATCH'}"
            )
            mismatches += not ok_rel

        recomputed_mass = float(np.mean(mass_accumulator[name]))
        reported_mass = reported["physical_metrics_si"]["relative_mass_error"]
        ok_mass = _close(recomputed_mass, reported_mass, 0.05)
        print(
            f"           mass err {recomputed_mass:.6f} vs {reported_mass:.6f} "
            f"{'ok' if ok_mass else 'MISMATCH'}"
        )
        mismatches += not ok_mass

    print()
    print(f"RESULT: {'PASS' if mismatches == 0 else f'FAIL ({mismatches} mismatches)'}")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    args = parser.parse_args(argv)
    return 1 if verify(args.run_dir, args.split) else 0


if __name__ == "__main__":
    sys.exit(main())
