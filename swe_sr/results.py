"""Build `docs/results/index.json`, the committed source of every number the prose cites (D024).

    python -m swe_sr.results --write

`runs/` is gitignored, so a clone has no evaluation artifacts and neither CI nor a reader can
re-derive a number that a document states. Before this module the consequence was visible:
`0.0400` appeared in eight files and `0.0830` in nine, each transcribed by hand, so a re-run
meant editing every one of them consistently and nothing detected a missed edit.

This module is the promotion step. It reads `docs/results/runs.yaml` -- the one place a run ID
is written by hand -- loads each registered run's `evaluation_*.json`, and writes a small
committed index. `swe_sr/docgen.py` renders the documentation's tables from that index, so:

- a number reaches prose by being rendered, never by being typed;
- CI can verify every rendered table without `runs/` present;
- re-running an arm produces one reviewable diff in `index.json` plus rendered consequences,
  rather than a hunt through nine files.

Nothing is recomputed here. Every value is copied from an artifact that `swe_sr.evaluate`
already wrote, or derived from such values by arithmetic stated at the point of derivation.
The two exceptions to "copied" are both marked in the output: `gap` and `lead_time_bands` are
reductions of values in the artifact, and `paired_vs_reference` is a bootstrap over
per-trajectory means that the artifact carries but does not itself pair.

`seconds_per_frame` is deliberately **excluded**. It is wall-clock time on whatever host
evaluated the run, at whatever load, so it is not comparable across arms and has already
misled once (`swe_sr/report.py` documents the case). Only reproducible numbers belong in a
file whose purpose is to be cited.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "docs" / "results" / "runs.yaml"
INDEX_PATH = REPO_ROOT / "docs" / "results" / "index.json"
RUNS_ROOT = REPO_ROOT / "runs"

BASELINES = ("nearest", "bicubic")

# Baselines are identical work in every run by construction, so the same baseline evaluated in
# two runs must agree exactly rather than approximately. A mismatch means the runs did not share
# a manifest or normalization, which would invalidate every cross-arm comparison built on them,
# so it is an error and not a warning.
BASELINE_TOLERANCE = 1e-12


class RegistryError(RuntimeError):
    """The registry is inconsistent with the artifacts it names."""


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry: dict[str, Any] = yaml.safe_load(path.read_text())
    ids = [arm["id"] for arm in registry["arms"]]
    duplicates = {name for name in ids if ids.count(name) > 1}
    if duplicates:
        raise RegistryError(f"duplicate arm ids in {path.name}: {sorted(duplicates)}")
    run_ids = [arm["run_id"] for arm in registry["arms"]]
    repeated = {run for run in run_ids if run_ids.count(run) > 1}
    if repeated:
        raise RegistryError(f"the same run_id is registered under two ids: {sorted(repeated)}")
    for arm in registry["arms"]:
        reference = arm.get("reference")
        if reference is not None and reference not in ids:
            raise RegistryError(f"arm {arm['id']} references unknown arm {reference}")
    return registry


def registered_run_ids(path: Path = REGISTRY_PATH) -> set[str]:
    """Run IDs the documentation is allowed to cite. Used by the docs lint."""
    return {arm["run_id"] for arm in load_registry(path)["arms"]}


def frozen_run_ids(path: Path = REGISTRY_PATH) -> set[str]:
    return {arm["run_id"] for arm in load_registry(path)["arms"] if arm.get("frozen")}


def _band_means(
    curve: dict[str, float], bands: list[dict[str, Any]], *, context: str
) -> dict[str, float]:
    """Mean of the lead-time curve within each band, over `min <= t < max`.

    Asserts no lead time sits exactly on a boundary. If one does, the band an artifact's
    snapshot lands in would depend on an inclusivity convention that the band *labels* in
    `runs.yaml` do not state, so the honest response is to fail and make the convention
    explicit rather than to pick one silently.
    """
    grouped: dict[str, list[float]] = {band["label"]: [] for band in bands}
    for key, value in curve.items():
        hours = float(key)
        for band in bands:
            low = float(band.get("min", 0.0))
            high = float(band.get("max", math.inf))
            for boundary in (low, high):
                if math.isfinite(boundary) and hours == boundary:
                    raise RegistryError(
                        f"{context}: lead time {hours} h falls exactly on band boundary "
                        f"{boundary} of {band['label']!r}; the band labels in runs.yaml do not "
                        "state an inclusivity convention, so fix the boundary or the labels"
                    )
            if low <= hours < high:
                grouped[band["label"]].append(value)
    missing = [label for label, values in grouped.items() if not values]
    if missing:
        raise RegistryError(f"{context}: no snapshots fall in band(s) {missing}")
    return {label: sum(values) / len(values) for label, values in grouped.items()}


def _method_block(
    report: dict[str, Any], method: str, bands: list[dict[str, Any]], *, context: str
) -> dict[str, Any]:
    block = report["methods"][method]
    aggregate = block["aggregate_macro_mse_normalized"]
    normalized = block["normalized_metrics"]
    physical = block["physical_metrics_si"]
    curve = block["normalized_macro_mse_by_lead_time_hours"]
    return {
        "params": block["trainable_parameters"],
        "normmse": {
            "mean": aggregate["mean"],
            "ci_low": aggregate["ci_low"],
            "ci_high": aggregate["ci_high"],
            "confidence": aggregate["confidence"],
        },
        "rel_l2": {
            "eta": normalized["rel_l2_eta"],
            "u": normalized["rel_l2_u"],
            "v": normalized["rel_l2_v"],
        },
        "relative_mass_error": physical["relative_mass_error"],
        "relative_energy_error": physical["relative_energy_error"],
        "lead_time_bands": _band_means(curve, bands, context=context),
        "lead_time_curve": curve,
        "trajectory_means": block.get("trajectory_means_macro_mse_normalized"),
    }


def _training_block(summary: dict[str, Any], *, context: str) -> dict[str, Any]:
    """Training dynamics the curve comparison cites, all read from `summary.json`.

    `gap` is the selection metric over the training loss **at the same epoch**, not at the end
    of the run, so it measures generalization at the checkpoint that was actually selected. A
    run that early-stops has a final training loss from an epoch whose weights were discarded.
    """
    epochs: list[dict[str, Any]] = summary["epochs"]
    best_epoch = summary["best_epoch"]
    at_best = next((record for record in epochs if record["epoch"] == best_epoch), None)
    if at_best is None:
        raise RegistryError(
            f"{context}: summary.json has no epoch record for best_epoch {best_epoch}"
        )
    best_validation = summary["best_validation_mse_normalized_macro"]
    return {
        "best_validation_mse": best_validation,
        "best_epoch": best_epoch,
        "epochs": len(epochs),
        "train_mse_at_best_epoch": at_best["train_mse"],
        "final_train_mse": epochs[-1]["train_mse"],
        "gap": best_validation / at_best["train_mse"],
        "steps_completed": summary["steps_completed"],
        "stopped_reason": summary["stopped_reason"],
        "mean_samples_per_second": summary["projection"]["mean_samples_per_second"],
        "config_hash": summary["config_hash"],
        "seed": summary["seed"],
        "resolved_precision": summary["resolved_precision"],
    }


def _paired_vs_reference(
    arm_id: str, arm: dict[str, Any], index_arms: dict[str, Any], *, seed: int
) -> dict[str, Any] | None:
    """Paired bootstrap of an arm against the arm it varies, on shared trajectories.

    Paired because both arms are evaluated on the same trajectories, and on this split the
    trajectory-to-trajectory spread (CIs of roughly +/-0.03) is an order of magnitude larger
    than the effects under test (about 0.002). Comparing two independent intervals is a
    strictly weaker test on the same data.

    Returns None, with a stated reason, when either arm's artifact predates D021 and so carries
    no per-trajectory means. `scripts/compare_ablation.py` can recompute those from the
    checkpoint; this module does not, because it must run from artifacts alone.
    """
    from swe_sr.metrics.aggregate import TrajectoryAggregate, paired_bootstrap_difference

    reference_id = arm["reference"]
    left = index_arms[arm_id]["test"]["trajectory_means"]
    right = index_arms[reference_id]["test"]["trajectory_means"]
    if left is None or right is None:
        absent = [name for name, means in ((arm_id, left), (reference_id, right)) if means is None]
        return {
            "reference": reference_id,
            "unavailable": (
                f"no per-trajectory means in the artifact(s) for {', '.join(absent)}; those runs "
                "predate D021 and are not re-evaluated, because re-running evaluation on a "
                "frozen run would rewrite its host-dependent fields. Use "
                "scripts/compare_ablation.py, which recomputes them in memory."
            ),
        }

    def aggregates(means: dict[str, float]) -> list[TrajectoryAggregate]:
        # `frames` is unused by the paired bootstrap: step 3 of the protocol weights
        # trajectories equally, so a per-trajectory frame count carries no weight.
        return [
            TrajectoryAggregate(trajectory_id=key, mean=value, frames=0)
            for key, value in sorted(means.items())
        ]

    paired = paired_bootstrap_difference(aggregates(left), aggregates(right), seed=seed)
    better = sum(1 for key in left if left[key] < right[key])
    return {
        "reference": reference_id,
        "seed": seed,
        # Sign convention: arm minus reference, so negative favours the ARM.
        "mean_difference": paired["mean_difference"],
        "ci_low": paired["ci_low"],
        "ci_high": paired["ci_high"],
        "confidence": paired["confidence"],
        "excludes_zero": paired["ci_high"] < 0 or paired["ci_low"] > 0,
        "arm_better_on": f"{better} of {len(left)}",
    }


def _load(run_dir: Path, filename: str) -> dict[str, Any]:
    path = run_dir / filename
    if not path.is_file():
        raise RegistryError(
            f"{path} does not exist. Registered runs must have been evaluated; run "
            f"`python -m swe_sr.evaluate --run-dir {run_dir}` or remove the arm from "
            "docs/results/runs.yaml."
        )
    artifact: dict[str, Any] = json.loads(path.read_text())
    return artifact


def build_index(
    registry: dict[str, Any] | None = None,
    *,
    runs_root: Path = RUNS_ROOT,
    split: str = "test",
    seed: int = 0,
) -> dict[str, Any]:
    registry = registry if registry is not None else load_registry()
    bands = registry["lead_time_bands"]

    arms: dict[str, Any] = {}
    baselines: dict[str, Any] = {}
    provenance: dict[str, Any] | None = None

    for arm in registry["arms"]:
        arm_id = arm["id"]
        run_dir = runs_root / arm["run_id"]
        report = _load(run_dir, f"evaluation_{split}.json")
        summary = _load(run_dir, "summary.json")
        model = report["model"]
        if model not in report["methods"]:
            raise RegistryError(f"{arm_id}: evaluation artifact has no methods entry for {model!r}")

        context = f"{arm_id} ({arm['run_id']})"
        arms[arm_id] = {
            "label": arm["label"],
            "short": arm.get("short", arm["label"]),
            "run_id": arm["run_id"],
            "role": arm["role"],
            "experiment": arm["experiment"],
            "frozen": bool(arm.get("frozen", False)),
            "model": model,
            "checkpoint": report["checkpoint"],
            "trained_at_commit": report["trained_at_commit"],
            "test": _method_block(report, model, bands, context=context),
            "paired_vs_bicubic": report["paired_bootstrap_vs_bicubic"][f"{model}_minus_bicubic"],
            "training": _training_block(summary, context=context),
        }

        # Every arm must sit on the same dataset, split, and snapshot count, or the tables that
        # place them in one row-set are comparing different data. Check rather than assume.
        current = {
            "dataset_id": report["dataset_id"],
            "pair_id": report["pair_id"],
            "ic_registry_hash": report["ic_registry_hash"],
            "split": report["split"],
            "snapshots": report["snapshots"],
            "trajectories": report["trajectories"],
        }
        if provenance is None:
            provenance = current
        elif current != provenance:
            differing = {k: (provenance[k], v) for k, v in current.items() if provenance[k] != v}
            raise RegistryError(
                f"{context}: evaluated on different data than earlier arms {differing}; arms in "
                "one table must share a manifest and split"
            )

        for name in BASELINES:
            if name not in report["methods"]:
                continue
            block = _method_block(report, name, bands, context=f"{context}/{name}")
            if name not in baselines:
                baselines[name] = block
            elif abs(baselines[name]["normmse"]["mean"] - block["normmse"]["mean"]) > (
                BASELINE_TOLERANCE
            ):
                raise RegistryError(
                    f"{context}: {name} scores {block['normmse']['mean']:.12f} here but "
                    f"{baselines[name]['normmse']['mean']:.12f} in an earlier arm. The baselines "
                    "are identical work in every run, so this means the runs do not share a "
                    "manifest or normalization."
                )

        transfer_name = f"evaluation_{split}__"
        for path in sorted(run_dir.glob(f"{transfer_name}*.json")):
            transfer = json.loads(path.read_text())
            arms[arm_id].setdefault("transfer", {})[transfer["pair_id"]] = {
                "snapshots": transfer["snapshots"],
                "trajectories": transfer["trajectories"],
                "model": _method_block(
                    transfer, model, bands, context=f"{context}/{transfer['pair_id']}"
                ),
                "bicubic": _method_block(
                    transfer, "bicubic", bands, context=f"{context}/{transfer['pair_id']}/bicubic"
                ),
                "paired_vs_bicubic": transfer["paired_bootstrap_vs_bicubic"][
                    f"{model}_minus_bicubic"
                ],
            }

    paired: dict[str, Any] = {}
    for arm in registry["arms"]:
        if arm.get("reference"):
            comparison = _paired_vs_reference(arm["id"], arm, arms, seed=seed)
            if comparison is not None:
                paired[arm["id"]] = comparison

    return {
        "schema": 1,
        "generated_by": "python -m swe_sr.results --write",
        "source": "docs/results/runs.yaml plus each run's evaluation and summary artifacts",
        # Band labels are carried as an ordered list because the index is serialized with sorted
        # keys, which would otherwise put "16-24 h" before "<= 12 h" and silently reorder every
        # rendered column. The registry's order is the meaningful one.
        "lead_time_band_labels": [band["label"] for band in bands],
        "provenance": provenance,
        "baselines": baselines,
        "arms": arms,
        "paired_vs_reference": paired,
    }


def render_index(index: dict[str, Any]) -> str:
    """Serialize deterministically, so an unchanged experiment produces an unchanged file."""
    return json.dumps(index, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help=f"write {INDEX_PATH.name} in place")
    parser.add_argument("--out", type=Path, default=INDEX_PATH)
    parser.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--seed", type=int, default=0, help="paired bootstrap seed")
    args = parser.parse_args(argv)

    text = render_index(build_index(runs_root=args.runs_root, split=args.split, seed=args.seed))
    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out} ({len(text.splitlines())} lines)")
        print("now run `python -m swe_sr.docgen render` to update the documents that cite it")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
