"""Held-out evaluation of a trained checkpoint against both baselines (E-01).

    python -m swe_sr.evaluate --run-dir runs/<run-id>

Evaluates the run's best checkpoint on every frame of the held-out **test** split, alongside
the nearest and bicubic baselines on identical states, and follows the aggregation protocol in
`docs/VALIDATION.md`: per snapshot, then within trajectory, then equal weight across
trajectories, with a trajectory-level bootstrap interval.

Three things this deliberately refuses to do, each because it would produce a
plausible-looking but wrong number:

- **Fit anything.** Normalization comes from the run's own frozen manifest, so the test split
  never influences a statistic.
- **Pool snapshots.** Trajectories carry equal weight, so a trajectory cannot dominate.
- **Report a single aggregate without the lead-time breakdown.** The pilot showed the models
  far worse than bicubic at short lead times and far better at long ones, which a single mean
  hides entirely.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from swe_sr.data.dataset import PairedSnapshotDataset
from swe_sr.data.manifest import load_manifest
from swe_sr.data.normalization import Normalization
from swe_sr.data.storage import resolve_array_dir
from swe_sr.metrics.aggregate import (
    SnapshotMetric,
    aggregate_by_trajectory,
    paired_bootstrap_difference,
)
from swe_sr.metrics.field import CHANNEL_NAMES, per_sample_relative_l2
from swe_sr.metrics.physics import (
    PhysicalParameters,
    physics_metrics,
    relative_mass_error,
)
from swe_sr.models import build_baseline, build_model_from_config, resource_summary
from swe_sr.training.config import REPO_ROOT, git_commit

# Methods evaluated on identical states, so any difference is the method (docs/EXPERIMENT_PLAN.md).
BASELINE_NAMES = ("nearest", "bicubic")


@dataclass
class MethodResult:
    """Everything reported for one method on one split."""

    name: str
    parameters: int
    normalized_metrics: dict[str, float]
    physical_metrics: dict[str, float]
    aggregate: dict[str, Any]
    by_lead_time: dict[float, float]
    trajectory_means: list[Any]
    seconds_per_frame: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "trainable_parameters": self.parameters,
            # Every metric name states normalization or units; CLAUDE.md forbids bare numbers.
            "normalized_metrics": self.normalized_metrics,
            "physical_metrics_si": self.physical_metrics,
            "aggregate_macro_mse_normalized": self.aggregate,
            "normalized_macro_mse_by_lead_time_hours": {
                f"{hours:.3f}": value for hours, value in sorted(self.by_lead_time.items())
            },
            "seconds_per_frame": self.seconds_per_frame,
        }


def _load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = json.loads((run_dir / "summary.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    return summary, config


@torch.no_grad()
def _evaluate_method(
    name: str,
    model: torch.nn.Module,
    dataset: PairedSnapshotDataset,
    indices: list[int],
    parameters: PhysicalParameters,
    normalization: Normalization,
    *,
    batch_size: int,
    seed: int,
) -> MethodResult:
    """Run one method over a whole split and reduce it per the documented protocol."""
    import time

    model.eval()
    snapshots: list[SnapshotMetric] = []
    lead_time_values: dict[float, list[float]] = {}
    physical_totals: dict[str, list[float]] = {}
    mse_totals = torch.zeros(3, dtype=torch.float64)
    max_abs = torch.zeros(3, dtype=torch.float64)
    mse_count = 0
    rel_l2_by_trajectory: dict[str, dict[str, list[float]]] = {}
    mass_by_trajectory: dict[str, list[float]] = {}
    elapsed = 0.0
    frames = 0

    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        samples = [dataset[i] for i in chunk]
        coarse = torch.stack([s["coarse"] for s in samples])
        fine = torch.stack([s["fine"] for s in samples])

        began = time.perf_counter()
        prediction = model(coarse)
        elapsed += time.perf_counter() - began
        frames += coarse.shape[0]

        # Per-snapshot macro MSE feeds the aggregation protocol; a batch mean would destroy
        # the per-trajectory structure the protocol depends on.
        per_sample = ((prediction - fine) ** 2).mean(dim=(1, 2, 3))
        for offset, sample in enumerate(samples):
            snapshots.append(
                SnapshotMetric(
                    trajectory_id=str(sample["trajectory_id"]),
                    frame=int(sample["frame"]),
                    value=float(per_sample[offset]),
                )
            )
            hours = float(sample["time"]) / 3600.0
            lead_time_values.setdefault(hours, []).append(float(per_sample[offset]))

        # Per-channel MSE accumulates as a sample-weighted sum, which is exact regardless of
        # batching. Ratio metrics must NOT be averaged per batch: relL2 is a norm ratio, so a
        # mean over batches depends on batch size, which is an implementation detail rather
        # than a property of the data. Independent recomputation caught exactly that, with the
        # reported bicubic relL2 off by 7 percent. Ratios are therefore accumulated per
        # snapshot and reduced by the documented aggregation protocol below.
        channel_mse = ((prediction - fine) ** 2).mean(dim=(0, 2, 3)).double()
        mse_totals += channel_mse * coarse.shape[0]
        mse_count += coarse.shape[0]
        worst = (prediction - fine).abs().amax(dim=(0, 2, 3)).double()
        max_abs = torch.maximum(max_abs, worst)

        sample_rel_l2 = per_sample_relative_l2(prediction, fine).double()
        for offset, sample in enumerate(samples):
            trajectory = str(sample["trajectory_id"])
            for index, channel in enumerate(CHANNEL_NAMES):
                rel_l2_by_trajectory.setdefault(channel, {}).setdefault(trajectory, []).append(
                    float(sample_rel_l2[offset, index])
                )
        # Physical diagnostics are computed after de-normalization (docs/ARCHITECTURE.md).
        physical_prediction = torch.from_numpy(
            normalization.invert(prediction.numpy()).astype(np.float32)
        )
        physical_fine = torch.from_numpy(normalization.invert(fine.numpy()).astype(np.float32))
        for key, value in physics_metrics(physical_prediction, physical_fine, parameters).items():
            physical_totals.setdefault(key, []).append(value)
        # Mass error is a per-sample ratio, so it follows the protocol too.
        sample_mass = relative_mass_error(physical_prediction, physical_fine, parameters).double()
        for offset, sample in enumerate(samples):
            mass_by_trajectory.setdefault(str(sample["trajectory_id"]), []).append(
                float(sample_mass[offset])
            )

    aggregate = aggregate_by_trajectory(snapshots, metric="macro_mse_normalized", seed=seed)

    # Reduce ratio metrics by the documented protocol: within-trajectory mean, then equal
    # weight across trajectories. Batch-size independent by construction.
    def _protocol(by_trajectory: dict[str, list[float]]) -> float:
        return float(np.mean([float(np.mean(v)) for v in by_trajectory.values()]))

    per_channel_mse_values = (mse_totals / max(mse_count, 1)).numpy()
    normalized = {"mse_macro": float(per_channel_mse_values.mean())}
    for index, channel in enumerate(CHANNEL_NAMES):
        normalized[f"mse_{channel}"] = float(per_channel_mse_values[index])
        normalized[f"rmse_{channel}"] = float(np.sqrt(per_channel_mse_values[index]))
        normalized[f"rel_l2_{channel}"] = _protocol(rel_l2_by_trajectory.get(channel, {}))
        normalized[f"max_abs_{channel}"] = float(max_abs[index])
    physical = {k: float(np.mean(v)) for k, v in physical_totals.items()}
    physical["relative_mass_error"] = _protocol(mass_by_trajectory)

    return MethodResult(
        name=name,
        parameters=int(resource_summary(model)["trainable_parameters"]),
        normalized_metrics=normalized,
        physical_metrics=physical,
        aggregate=aggregate.to_dict(),
        by_lead_time={h: float(np.mean(v)) for h, v in lead_time_values.items()},
        trajectory_means=aggregate.trajectory_means,
        seconds_per_frame=elapsed / max(frames, 1),
    )


def evaluate_run(
    run_dir: Path,
    *,
    split: str = "test",
    manifest_override: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a run's best checkpoint and both baselines on a held-out split."""
    run_dir = Path(run_dir)
    summary, config = _load_run(run_dir)

    # The run's own frozen manifest by default: evaluating against a different dataset than the
    # one trained on is a silent way to produce a meaningless number.
    manifest_path = Path(manifest_override or REPO_ROOT / str(config["manifest"]))
    manifest = load_manifest(manifest_path)

    # A cross-pair evaluation must never land on the canonical filename. The frozen runs'
    # `evaluation_test.json` is what docs/RESULTS.md, docs/EXPERIMENT_FREEZE.md and
    # scripts/verify_independent.py all read, and an override that reused that name would
    # silently replace a frozen result with a number from a different dataset. Deriving the
    # name from the pair makes that impossible rather than relying on the caller.
    trained_manifest_path = REPO_ROOT / str(config["manifest"])
    trained_pair_id = (
        manifest.pair_id
        if manifest_override is None
        else str(load_manifest(trained_manifest_path).pair_id)
    )
    transfer = manifest.pair_id != trained_pair_id
    output_name = (
        f"evaluation_{split}__{manifest.pair_id}.json" if transfer else f"evaluation_{split}.json"
    )
    normalization = Normalization.from_dict(manifest.normalization)
    dataset = PairedSnapshotDataset(
        manifest,
        resolve_array_dir(manifest_path, manifest.dataset_id),
        split=split,
        normalization=normalization,
    )
    indices = list(range(len(dataset)))
    parameters = PhysicalParameters.from_manifest(manifest)

    model_name, model = build_model_from_config(
        REPO_ROOT / f"configs/model/{summary['model']}_x4.yaml"
    )
    checkpoint = run_dir / "checkpoints" / "best.pt"
    model.load_state_dict(torch.load(checkpoint, weights_only=True))

    batch_size = int(config.get("batch_size", 8))
    # Typed as the common base: baselines are parameter-free Interpolation modules while the
    # model is a ResidualSuperResolution, and evaluation treats them identically on purpose.
    methods: dict[str, nn.Module] = {name: build_baseline(name) for name in BASELINE_NAMES}
    methods[model_name] = model

    results = {
        name: _evaluate_method(
            name,
            module,
            dataset,
            indices,
            parameters,
            normalization,
            batch_size=batch_size,
            seed=seed,
        )
        for name, module in methods.items()
    }

    # Paired against bicubic on the same trajectories: the pairing removes the
    # trajectory-to-trajectory spread that would otherwise swamp the comparison.
    comparisons = {}
    for name, result in results.items():
        if name == "bicubic":
            continue
        comparisons[f"{name}_minus_bicubic"] = paired_bootstrap_difference(
            result.trajectory_means, results["bicubic"].trajectory_means, seed=seed
        )

    report = {
        "run_id": summary["run_id"],
        "stage": summary.get("stage"),
        "model": model_name,
        "split": split,
        "artifact": output_name,
        "checkpoint": str(checkpoint.relative_to(run_dir)),
        "checkpoint_selection_rule": summary.get("checkpoint_selection_rule"),
        "dataset_id": manifest.dataset_id,
        "pair_id": manifest.pair_id,
        # Stated on every report, not only transfer ones, so a reader never has to infer
        # whether the evaluated pair is the trained one.
        "trained_pair_id": trained_pair_id,
        "resolution_transfer": transfer,
        "ic_registry_hash": manifest.ic_registry_hash,
        "trained_at_commit": summary.get("git_commit"),
        "evaluated_at_commit": git_commit(),
        "seed": seed,
        "trajectories": len(dataset.trajectory_ids),
        "snapshots": len(dataset),
        "methods": {name: result.to_dict() for name, result in results.items()},
        "paired_bootstrap_vs_bicubic": comparisons,
        # Stated explicitly so a reader cannot mistake a mean near 1.0 for a bug.
        "reference_notes": {
            "mean_predictor_normalized_mse": 1.0,
            "note": (
                "normalized channels have unit variance, so predicting the channel mean scores "
                "1.0; a method scoring near 1.0 carries no information about the target"
            ),
        },
    }
    if transfer:
        report["transfer_notes"] = {
            "trained_on": trained_pair_id,
            "evaluated_on": manifest.pair_id,
            "normalization_pair_id": manifest.normalization.get("pair_id"),
            "note": (
                "the checkpoint was trained on a different resolution pair; normalization is "
                "the evaluated pair's own train-split statistics, so amplitude is "
                "re-standardized and what this measures is transfer of the learned operator "
                "across absolute grid spacing at a fixed x4 factor"
            ),
        }
    (run_dir / output_name).write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def render_table(report: dict[str, Any]) -> str:
    """The comparison table from docs/VALIDATION.md, with units and aggregation stated."""
    lines = [
        f"run      : {report['run_id']}  (stage={report['stage']})",
        f"split    : {report['split']}  "
        f"{report['snapshots']} snapshots over {report['trajectories']} trajectories",
        f"dataset  : {report['dataset_id']}",
        "",
        "aggregation: per snapshot, then within trajectory, then equal weight across "
        "trajectories; 95% trajectory bootstrap",
        "reference  : predicting the channel mean scores 1.0 normalized MSE",
        "",
        f"{'method':<10} {'params':>10} {'normMSE':>9} {'95% CI':>17} "
        f"{'eta relL2':>10} {'u relL2':>9} {'v relL2':>9} {'mass err':>10} {'s/frame':>9}",
    ]
    for name, method in report["methods"].items():
        aggregate = method["aggregate_macro_mse_normalized"]
        normalized = method["normalized_metrics"]
        physical = method["physical_metrics_si"]
        interval = f"[{aggregate['ci_low']:.4f},{aggregate['ci_high']:.4f}]"
        lines.append(
            f"{name:<10} {method['trainable_parameters']:>10,} {aggregate['mean']:>9.4f} "
            f"{interval:>17} {normalized['rel_l2_eta']:>10.4f} {normalized['rel_l2_u']:>9.4f} "
            f"{normalized['rel_l2_v']:>9.4f} {physical['relative_mass_error']:>10.4f} "
            f"{method['seconds_per_frame']:>9.4f}"
        )

    lines += ["", "paired bootstrap against bicubic (negative favours the method):"]
    for key, comparison in report["paired_bootstrap_vs_bicubic"].items():
        excludes_zero = comparison["ci_high"] < 0 or comparison["ci_low"] > 0
        significant = "excludes 0" if excludes_zero else "includes 0"
        lines.append(
            f"  {key:<24} {comparison['mean_difference']:+.4f} "
            f"[{comparison['ci_low']:+.4f},{comparison['ci_high']:+.4f}]  {significant}"
        )

    lines += ["", "normalized macro MSE by lead time (hours):", f"  {'t (h)':>7}"]
    names = list(report["methods"])
    header = f"  {'t (h)':>7}" + "".join(f"{n:>11}" for n in names)
    lines[-1] = header
    any_method = report["methods"][names[0]]["normalized_macro_mse_by_lead_time_hours"]
    keys = sorted(any_method, key=float)
    for key in keys[:: max(1, len(keys) // 12)]:
        row = f"  {float(key):>7.2f}"
        for name in names:
            breakdown = report["methods"][name]["normalized_macro_mse_by_lead_time_hours"]
            row += f"{breakdown[key]:>11.4f}"
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = evaluate_run(
        args.run_dir, split=args.split, manifest_override=args.manifest, seed=args.seed
    )
    print(render_table(report))
    print(f"\nwrote {args.run_dir / report['artifact']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
