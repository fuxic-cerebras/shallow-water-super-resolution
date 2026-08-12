"""Plot pilot training curves and the coarse/fine decorrelation finding.

    python scripts/plot_pilot.py --runs runs/<edsr_run> runs/<unet_run>

A reporting client, like `scripts/visualize.py`: it imports matplotlib and therefore lives
outside `swe_sr/`, so nothing on the training or generation path depends on plotting.

The decorrelation panel is the important one. Bicubic scoring ~1.0 normalized MSE is easy to
misread as a bug, so the plot shows the reference line at 1.0 -- what predicting the channel
mean scores -- and the growth of baseline error with lead time that explains it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import swe_sr  # noqa: F401  # numpy-before-torch load-order guard; see swe_sr/__init__
from swe_sr.data.dataset import PairedSnapshotDataset
from swe_sr.data.manifest import load_manifest
from swe_sr.data.normalization import Normalization
from swe_sr.data.storage import resolve_array_dir
from swe_sr.metrics.field import per_channel_mse
from swe_sr.models import build_baseline, build_model_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]
# Normalized channels have unit variance, so predicting the channel mean scores exactly 1.0.
# Every normalized-MSE plot carries this line; without it a value near 1.0 looks unremarkable.
MEAN_PREDICTOR_MSE = 1.0


def _load_run(run_dir: Path) -> dict[str, object]:
    summary = json.loads((run_dir / "summary.json").read_text())
    with (run_dir / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return {"summary": summary, "rows": rows, "dir": run_dir}


def plot_curves(runs: list[dict[str, object]], out_path: Path) -> None:
    """Training and validation curves for every run, plus the baseline reference."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")
    colors = {"edsr": "#1f77b4", "unet": "#d62728"}

    for run in runs:
        summary = run["summary"]  # type: ignore[index]
        rows = run["rows"]  # type: ignore[index]
        name = str(summary["model"])  # type: ignore[index]
        color = colors.get(name, "#555555")
        steps = [int(r["step"]) for r in rows]  # type: ignore[union-attr]
        axes[0].plot(
            steps, [float(r["train_mse"]) for r in rows], "--", color=color, label=f"{name} train"
        )
        axes[0].plot(
            steps,
            [float(r["validation_mse"]) for r in rows],
            "-o",
            ms=3,
            color=color,
            label=f"{name} validation",
        )
        axes[1].plot(
            [float(r["elapsed_seconds"]) / 60 for r in rows],
            [float(r["validation_mse"]) for r in rows],
            "-o",
            ms=3,
            color=color,
            label=name,
        )
        for channel, style in (("eta", "-"), ("u", "--"), ("v", ":")):
            axes[2].plot(
                steps,
                [float(r[f"validation_mse_{channel}"]) for r in rows],
                style,
                color=color,
                label=f"{name} {channel}",
            )

    for axis, (title, xlabel) in zip(
        axes,
        (
            ("loss vs optimizer step", "optimizer step"),
            ("validation vs wall time", "elapsed (minutes)"),
            ("per-channel validation MSE", "optimizer step"),
        ),
        strict=True,
    ):
        axis.axhline(
            MEAN_PREDICTOR_MSE,
            color="grey",
            lw=1,
            ls="-.",
            label="predicting the channel mean",
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("normalized MSE")
        axis.set_yscale("log")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)

    figure.suptitle(
        "Pilot runs: 8 train / 2 validation trajectories, 1,970 steps, 16 Slurm threads, BF16",
        fontsize=11,
    )
    figure.savefig(out_path, dpi=120)
    plt.close(figure)


def plot_decorrelation(
    manifest_path: Path, run_dir: Path | None, out_path: Path, stride: int = 10
) -> dict[str, list[float]]:
    """Baseline and model error against lead time -- the finding that reframes every result."""
    manifest = load_manifest(manifest_path)
    dataset = PairedSnapshotDataset(
        manifest,
        resolve_array_dir(manifest_path, manifest.dataset_id),
        split="validation",
        normalization=Normalization.from_dict(manifest.normalization),
    )
    baselines = {"nearest": build_baseline("nearest"), "bicubic": build_baseline("bicubic")}
    model = None
    model_name = "model"
    if run_dir is not None:
        summary = json.loads((run_dir / "summary.json").read_text())
        model_name = str(summary["model"])
        model_name_path = REPO_ROOT / f"configs/model/{model_name}_x4.yaml"
        model_name, model = build_model_from_config(model_name_path)
        model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True))
        model.eval()

    frames: dict[int, dict[str, list[float]]] = {}
    times: dict[int, float] = {}
    with torch.no_grad():
        for index, sample in enumerate(dataset.sample_index):
            if sample.frame % stride:
                continue
            item = dataset[index]
            coarse = item["coarse"].unsqueeze(0)
            fine = item["fine"].unsqueeze(0)
            bucket = frames.setdefault(sample.frame, {})
            times[sample.frame] = sample.time
            for name, baseline in baselines.items():
                bucket.setdefault(name, []).append(
                    float(per_channel_mse(baseline(coarse), fine).mean())
                )
            if model is not None:
                bucket.setdefault(model_name, []).append(
                    float(per_channel_mse(model(coarse), fine).mean())
                )

    ordered = sorted(frames)
    hours = [times[f] / 3600 for f in ordered]
    series = {
        name: [float(np.mean(frames[f][name])) for f in ordered] for name in frames[ordered[0]]
    }

    figure, axis = plt.subplots(figsize=(8.5, 5.2), layout="constrained")
    styles = {
        "nearest": ("#7f7f7f", "-s"),
        "bicubic": ("#2ca02c", "-o"),
        model_name: ("#d62728", "-^"),
    }
    for name, values in series.items():
        color, marker = styles.get(name, ("#1f77b4", "-o"))
        axis.plot(hours, values, marker, ms=4, color=color, label=name)

    axis.axhline(
        MEAN_PREDICTOR_MSE, color="black", lw=1.4, ls="-.", label="predicting the channel mean"
    )
    axis.set_xlabel("lead time since initial condition (hours)")
    axis.set_ylabel("normalized macro MSE on the validation split")
    axis.set_title(
        "Coarse and fine solves decorrelate with lead time\n"
        "independent integrations (D002) drift apart; late frames are progressively "
        "less recoverable",
        fontsize=11,
    )
    axis.grid(alpha=0.3)
    axis.legend()
    axis.annotate(
        "at 2 h the coarse state\nnearly determines the fine one",
        xy=(hours[0], series["bicubic"][0]),
        xytext=(hours[0] + 3, 0.02),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    axis.annotate(
        "only beyond ~30 h does bicubic reach\nmean-predictor level; its split mean is 0.47",
        xy=(hours[-1], series["bicubic"][-1]),
        xytext=(hours[-1] - 17, 1.45),
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.8},
    )
    figure.savefig(out_path, dpi=120)
    plt.close(figure)
    return {"hours": hours, **series}


def plot_qualitative(
    manifest_path: Path, run_dir: Path, out_path: Path, frames: tuple[int, ...] = (0, 90, 180)
) -> None:
    """LR input, bicubic, prediction, HR target, and signed error, at several lead times.

    Shared symmetric colour limits per row, as `docs/EXPERIMENT_PLAN.md` requires, so panels are
    directly comparable rather than each autoscaled to look equally good.
    """
    manifest = load_manifest(manifest_path)
    dataset = PairedSnapshotDataset(
        manifest,
        resolve_array_dir(manifest_path, manifest.dataset_id),
        split="validation",
        normalization=Normalization.from_dict(manifest.normalization),
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    model_name, model = build_model_from_config(
        REPO_ROOT / f"configs/model/{summary['model']}_x4.yaml"
    )
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True))
    model.eval()
    bicubic = build_baseline("bicubic")

    first_trajectory = dataset.trajectory_ids[0]
    lookup = {
        sample.frame: index
        for index, sample in enumerate(dataset.sample_index)
        if sample.trajectory_id == first_trajectory
    }

    figure, axes = plt.subplots(
        len(frames), 5, figsize=(15, 3.1 * len(frames)), layout="constrained"
    )
    with torch.no_grad():
        for row, frame in enumerate(frames):
            item = dataset[lookup[frame]]
            coarse = item["coarse"].unsqueeze(0)
            fine = item["fine"].unsqueeze(0)
            interpolated = bicubic(coarse)
            predicted = model(coarse)
            error = predicted - fine

            limit = float(fine[0, 0].abs().max())
            panels = (
                ("coarse 32x32 (input)", coarse[0, 0], limit, "RdBu_r"),
                ("bicubic x4", interpolated[0, 0], limit, "RdBu_r"),
                (f"{model_name} prediction", predicted[0, 0], limit, "RdBu_r"),
                ("fine 128x128 (target)", fine[0, 0], limit, "RdBu_r"),
                ("prediction - target", error[0, 0], float(error.abs().max()), "PuOr_r"),
            )
            for column, (title, data, vlimit, cmap) in enumerate(panels):
                axis = axes[row, column] if len(frames) > 1 else axes[column]
                image = axis.imshow(
                    data.numpy(), origin="lower", vmin=-vlimit, vmax=vlimit, cmap=cmap
                )
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(title, fontsize=10)
                if column == 0:
                    axis.set_ylabel(f"t = {item['time'] / 3600:.1f} h", fontsize=10)
                figure.colorbar(image, ax=axis, fraction=0.046)

    figure.suptitle(
        f"eta channel, normalized units, {model_name} pilot best checkpoint. "
        "Error grows with lead time as the two solves decorrelate.",
        fontsize=11,
    )
    figure.savefig(out_path, dpi=110)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/staging/processed/swe_gaussian_32x128_v1/manifest.json",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "viz")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    runs = [_load_run(path) for path in args.runs]
    plot_curves(runs, args.out / "pilot_curves.png")
    print(f"wrote {args.out / 'pilot_curves.png'}")

    best = min(runs, key=lambda r: r["summary"]["best_validation_mse_normalized_macro"])  # type: ignore[index,call-overload]
    series = plot_decorrelation(args.manifest, best["dir"], args.out / "decorrelation.png")  # type: ignore[arg-type]
    print(f"wrote {args.out / 'decorrelation.png'}")
    for hours, bicubic_value in zip(series["hours"], series["bicubic"], strict=False):
        print(f"    t={hours:6.2f} h  bicubic={bicubic_value:.4f}")

    plot_qualitative(args.manifest, best["dir"], args.out / "qualitative.png")  # type: ignore[arg-type]
    print(f"wrote {args.out / 'qualitative.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
