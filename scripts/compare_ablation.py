"""Paired comparison of the D022 ablation arms against their frozen counterparts.

    python scripts/compare_ablation.py --residual runs/<edsr> runs/<unet> \
        --direct runs/<edsr_direct> runs/<unet_direct>

Reads stored `evaluation_<split>.json` artifacts, so every headline number it prints is traceable
to an evaluation that already happened.

One exception, and it is deliberate. The frozen T-03 artifacts predate
`trajectory_means_macro_mse_normalized`, which D021 added *without* regenerating them, because
re-running evaluation on a frozen run would also rewrite its host-load-dependent
`seconds_per_frame` inside a frozen record. So for a run missing that field this script recomputes
the per-trajectory means from the checkpoint **in memory and writes nothing**. The recomputation is
self-checking: its equal-weight mean must reproduce the aggregate the frozen artifact already
reports, and the script fails if it does not.

The comparison that matters is **paired**. The two arms of each architecture are evaluated in
separate processes against separate run directories, so asking whether their independent
confidence intervals overlap is a strictly weaker test on the same data. Pairing on trajectory
removes the trajectory-to-trajectory spread, which on this split is far larger than the effect
under test: the aggregate CIs span roughly +/-0.03 while the arms differ by about 0.002.

A sign test over the 8 test trajectories is reported alongside, because with 8 pairs a
bootstrap interval is itself estimated from little data and agreement between the two is worth
more than either alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import swe_sr  # noqa: F401  # numpy-before-torch load-order guard; see swe_sr/__init__
from swe_sr.metrics.aggregate import TrajectoryAggregate, paired_bootstrap_difference


def _load(run_dir: Path, split: str) -> dict[str, Any]:
    path = run_dir / f"evaluation_{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; run `python -m swe_sr.evaluate` first")
    return json.loads(path.read_text())


def _aggregates(
    report: dict[str, Any], method: str, run_dir: Path, split: str
) -> list[TrajectoryAggregate]:
    """The protocol's step-2 output: one equal-weighted mean per trajectory."""
    block = report["methods"][method]
    if "trajectory_means_macro_mse_normalized" in block:
        means = block["trajectory_means_macro_mse_normalized"]
        # `frames` is not serialized, and the paired bootstrap does not read it: step 3 of the
        # protocol weights trajectories equally, so a per-trajectory frame count carries no weight.
        frames = report["snapshots"] // max(len(means), 1)
        return [
            TrajectoryAggregate(trajectory_id=key, mean=value, frames=frames)
            for key, value in sorted(means.items())
        ]
    return _recompute_trajectory_means(run_dir, split, report, method)


def _recompute_trajectory_means(
    run_dir: Path, split: str, report: dict[str, Any], method: str
) -> list[TrajectoryAggregate]:
    """Recompute step 2 for a run written before D021 serialized it. Writes nothing."""
    import torch
    import yaml

    from swe_sr.data.dataset import PairedSnapshotDataset
    from swe_sr.data.manifest import load_manifest
    from swe_sr.data.normalization import Normalization
    from swe_sr.data.storage import resolve_array_dir
    from swe_sr.metrics.aggregate import SnapshotMetric, aggregate_by_trajectory
    from swe_sr.models import build_model_from_config
    from swe_sr.training.config import REPO_ROOT, model_config_for_run

    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    manifest_path = REPO_ROOT / str(config["manifest"])
    manifest = load_manifest(manifest_path)
    dataset = PairedSnapshotDataset(
        manifest,
        resolve_array_dir(manifest_path, manifest.dataset_id),
        split=split,
        normalization=Normalization.from_dict(manifest.normalization),
    )
    _, model = build_model_from_config(model_config_for_run(run_dir))
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True))
    model.eval()

    batch_size = int(config.get("batch_size", 8))
    snapshots: list[SnapshotMetric] = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            samples = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            coarse = torch.stack([s["coarse"] for s in samples])
            fine = torch.stack([s["fine"] for s in samples])
            per_sample = ((model(coarse) - fine) ** 2).mean(dim=(1, 2, 3))
            for offset, sample in enumerate(samples):
                snapshots.append(
                    SnapshotMetric(
                        trajectory_id=str(sample["trajectory_id"]),
                        frame=int(sample["frame"]),
                        value=float(per_sample[offset]),
                    )
                )

    aggregate = aggregate_by_trajectory(snapshots, metric="macro_mse_normalized", seed=0)
    # Self-check: this must reproduce what the frozen artifact already reports, or the
    # recomputation is not measuring the same thing and no comparison built on it is valid.
    reported = report["methods"][method]["aggregate_macro_mse_normalized"]["mean"]
    if abs(aggregate.mean - reported) > 1e-6:
        raise SystemExit(
            f"{run_dir.name}: recomputed aggregate {aggregate.mean:.8f} does not match the "
            f"artifact's {reported:.8f}; refusing to build a paired comparison on it"
        )
    print(
        f"  recomputed per-trajectory means for {run_dir.name} "
        f"(pre-D021 artifact, nothing written); aggregate reproduces {reported:.6f}"
    )
    return list(aggregate.trajectory_means)


def compare(residual: Path, direct: Path, *, split: str, seed: int) -> dict[str, Any]:
    """One architecture: its residual arm against its direct arm, paired on trajectory."""
    left, right = _load(direct, split), _load(residual, split)
    direct_name, residual_name = left["model"], right["model"]
    direct_means = _aggregates(left, direct_name, direct, split)
    residual_means = _aggregates(right, residual_name, residual, split)

    paired = paired_bootstrap_difference(direct_means, residual_means, seed=seed)
    by_id = {a.trajectory_id: a.mean for a in direct_means}
    other = {a.trajectory_id: a.mean for a in residual_means}
    direct_wins = sum(1 for k in by_id if by_id[k] < other[k])

    return {
        "architecture": residual_name,
        "residual_run": residual.name,
        "direct_run": direct.name,
        "residual_normmse": right["methods"][residual_name]["aggregate_macro_mse_normalized"],
        "direct_normmse": left["methods"][direct_name]["aggregate_macro_mse_normalized"],
        "residual_mass_error": right["methods"][residual_name]["physical_metrics_si"][
            "relative_mass_error"
        ],
        "direct_mass_error": left["methods"][direct_name]["physical_metrics_si"][
            "relative_mass_error"
        ],
        # Sign convention: direct minus residual, so negative favours DIRECT.
        "paired_direct_minus_residual": paired,
        "direct_wins_on_trajectories": f"{direct_wins} of {len(by_id)}",
        "bicubic_normmse": right["methods"]["bicubic"]["aggregate_macro_mse_normalized"]["mean"],
        "lead_time": {
            "bicubic": right["methods"]["bicubic"]["normalized_macro_mse_by_lead_time_hours"],
            "residual": right["methods"][residual_name]["normalized_macro_mse_by_lead_time_hours"],
            "direct": left["methods"][direct_name]["normalized_macro_mse_by_lead_time_hours"],
        },
    }


def render(results: list[dict[str, Any]]) -> str:
    lines = [
        "D022 ablation: outer bicubic residual against direct prediction",
        "",
        "Held-out test split, normalized macro-averaged MSE, aggregation per snapshot then within",
        "trajectory then equal weight across trajectories. Negative difference favours DIRECT.",
        "",
        f"{'arch':<6} {'residual':>10} {'direct':>10} {'paired diff':>12} {'95% CI':>22} "
        f"{'excl. 0':>8} {'direct wins':>12}",
    ]
    for result in results:
        paired = result["paired_direct_minus_residual"]
        excludes = paired["ci_high"] < 0 or paired["ci_low"] > 0
        lines.append(
            f"{result['architecture']:<6} {result['residual_normmse']['mean']:>10.4f} "
            f"{result['direct_normmse']['mean']:>10.4f} "
            f"{paired['mean_difference']:>+12.5f} "
            f"{f'[{paired["ci_low"]:+.5f},{paired["ci_high"]:+.5f}]':>22} "
            f"{'yes' if excludes else 'no':>8} "
            f"{result['direct_wins_on_trajectories']:>12}"
        )

    lines += ["", "relative mass error, de-normalized SI (lower is better):"]
    for result in results:
        lines.append(
            f"  {result['architecture']:<6} residual {result['residual_mass_error']:.4f}  "
            f"direct {result['direct_mass_error']:.4f}  "
            f"(bicubic 0.0392)"
        )

    lines += ["", "normalized macro MSE by lead time:", ""]
    first = results[0]["lead_time"]["bicubic"]
    keys = sorted(first, key=float)
    header = f"  {'t (h)':>7} {'bicubic':>9}"
    for result in results:
        header += f" {result['architecture'] + ' res':>13} {result['architecture'] + ' dir':>13}"
    lines.append(header)
    for key in keys[:: max(1, len(keys) // 12)]:
        row = f"  {float(key):>7.2f} {first[key]:>9.4f}"
        for result in results:
            row += (
                f" {result['lead_time']['residual'][key]:>13.4f}"
                f" {result['lead_time']['direct'][key]:>13.4f}"
            )
        lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual", type=Path, nargs="+", required=True)
    parser.add_argument("--direct", type=Path, nargs="+", required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="write the raw JSON here")
    args = parser.parse_args(argv)

    if len(args.residual) != len(args.direct):
        raise SystemExit("--residual and --direct must name the same number of runs, in order")

    results = [
        compare(residual, direct, split=args.split, seed=args.seed)
        for residual, direct in zip(args.residual, args.direct, strict=True)
    ]
    print(render(results))
    if args.out:
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
