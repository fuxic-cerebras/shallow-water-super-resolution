"""Final result figures for the frozen T-03 experiment.

    python scripts/plot_final.py --runs runs/<edsr-run> runs/<unet-run>

Produces three artifacts in `viz/`:

- `final_curves.png` — training and validation loss for both models, with the interpolation
  bicubic baseline as a horizontal reference level. Baselines have no curve because they have
  nothing to train; drawing bicubic as a level is what makes "did the model beat the baseline, and
  when" readable off the same axes. Nearest is omitted because it scores within 0.0006 of bicubic
  and its line would simply overlay it. The mean-predictor level at 1.0 is omitted too: nothing on
  this figure comes near it, so it would only compress the axis away from where the models
  separate. It is kept on the lead-time trace in the animation, where bicubic does climb past it.
- `qualitative.png` — bicubic and both models side by side with the coarse input and fine target,
  at several lead times. Colour limits are shared per row and taken from the target, so no panel is
  flattered by its own autoscale.
- `qualitative_error.png` — the signed errors for the same frames, as a separate figure so each
  carries one row label and one message rather than interleaving lead times with MSE values.
Nearest-neighbour is omitted from the qualitative figures: it is a 4x4 block replication of the
coarse input, so its panel duplicates the input column. It remains in `final_curves.png` as a
baseline level, where its score still carries information.

- `qualitative.gif` — the same comparison animated over lead time, with a running per-method error
  readout. The animation is the clearest way to see the finding that matters: the baselines
  degrade steadily as the coarse and fine solves decorrelate while the models stay close to target.

A reporting client, like `scripts/visualize.py`: it imports matplotlib and lives outside `swe_sr/`,
so nothing on the training or generation path depends on plotting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

import swe_sr  # noqa: F401  # numpy-before-torch load-order guard; see swe_sr/__init__
from swe_sr.data.dataset import PairedSnapshotDataset
from swe_sr.data.manifest import load_manifest
from swe_sr.data.normalization import Normalization
from swe_sr.data.storage import resolve_array_dir
from swe_sr.models import build_baseline, build_model_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]
# Normalized channels have unit variance, so predicting the channel mean scores exactly 1.0.
MEAN_PREDICTOR = 1.0
COLORS = {"nearest": "#7f7f7f", "bicubic": "#2ca02c", "edsr": "#1f77b4", "unet": "#d62728"}


def _load_methods(run_dirs: list[Path], *, include_nearest: bool = True) -> dict[str, Any]:
    """Baselines plus every supplied run's best checkpoint, in display order.

    `include_nearest` is off for the qualitative figures: nearest is a 4x4 block replication of
    the coarse input, so its panel is visually near-identical to the input column and only
    crowds the comparison. It remains in `final_curves.png` as a baseline level, where its score
    still carries information.
    """
    methods: dict[str, Any] = {}
    if include_nearest:
        methods["nearest"] = build_baseline("nearest")
    methods["bicubic"] = build_baseline("bicubic")
    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        name, module = build_model_from_config(
            REPO_ROOT / f"configs/model/{summary['model']}_x4.yaml"
        )
        module.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True))
        module.eval()
        methods[name] = module
    return methods


def _baseline_levels(run_dirs: list[Path], split: str = "test") -> dict[str, float]:
    """Baseline scores from the evaluation artifacts, so the figure cites measured values."""
    for run_dir in run_dirs:
        path = run_dir / f"evaluation_{split}.json"
        if path.is_file():
            report = json.loads(path.read_text())
            # Bicubic only. Nearest scores 0.4301 against bicubic's 0.4295, so its line would
            # sit on top of bicubic's and add a legend entry without adding information. Its
            # score is reported in docs/RESULTS.md where the numbers can be compared directly.
            return {
                name: report["methods"][name]["aggregate_macro_mse_normalized"]["mean"]
                for name in ("bicubic",)
                if name in report["methods"]
            }
    return {}


def plot_final_curves(run_dirs: list[Path], out_path: Path, split: str = "test") -> None:
    """Loss curves for the trained models against the baselines as levels."""
    levels = _baseline_levels(run_dirs, split)
    figure, axes = plt.subplots(1, 3, figsize=(19, 5), layout="constrained")

    for run_dir in run_dirs:
        summary = json.loads((run_dir / "summary.json").read_text())
        name = str(summary["model"])
        color = COLORS.get(name, "#555555")
        with (run_dir / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        steps = [int(r["step"]) for r in rows]
        minutes = [float(r["elapsed_seconds"]) / 60 for r in rows]
        train = [float(r["train_mse"]) for r in rows]
        validation = [float(r["validation_mse"]) for r in rows]

        axes[0].plot(steps, train, "--", color=color, lw=1.2, alpha=0.8, label=f"{name} train")
        axes[0].plot(steps, validation, "-", color=color, lw=1.8, label=f"{name} validation")
        best = summary["best_validation_mse_normalized_macro"]
        axes[0].plot(
            steps[validation.index(min(validation))],
            best,
            "*",
            color=color,
            ms=15,
            markeredgecolor="black",
            markeredgewidth=0.5,
            label=f"{name} best {best:.4f}",
        )
        axes[1].plot(minutes, validation, "-", color=color, lw=1.8, label=f"{name} validation")

        # Per channel: a macro average cannot show which channel limits the score, and
        # docs/PROJECT_SPEC.md research question 3 asks whether gains are consistent across them.
        for channel, style in (("eta", "-"), ("u", "--"), ("v", ":")):
            axes[2].plot(
                steps,
                [float(r[f"validation_mse_{channel}"]) for r in rows],
                style,
                color=color,
                lw=1.5,
                label=f"{name} {channel}",
            )

    for axis in axes:
        for name, level in levels.items():
            axis.axhline(
                level,
                color=COLORS.get(name, "grey"),
                ls=":",
                lw=1.8,
                label=f"{name} baseline {level:.4f}",
            )
        axis.set_ylabel("normalized macro-averaged MSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=7, loc="upper right", ncol=2)

    axes[0].set_xlabel("optimizer step")
    axes[0].set_title("training and validation loss")
    axes[1].set_xlabel("wall-clock (minutes, 16 Slurm threads, BF16)")
    axes[1].set_title("validation loss vs cost")
    axes[2].set_xlabel("optimizer step")
    axes[2].set_title("per-channel validation MSE")
    figure.suptitle(
        "Frozen T-03: 30,000 steps per model. Baselines are levels, not curves -- they have "
        "nothing to train.",
        fontsize=11,
    )
    figure.savefig(out_path, dpi=130)
    plt.close(figure)


def _dataset(manifest_path: Path, split: str) -> tuple[PairedSnapshotDataset, Any]:
    manifest = load_manifest(manifest_path)
    dataset = PairedSnapshotDataset(
        manifest,
        resolve_array_dir(manifest_path, manifest.dataset_id),
        split=split,
        normalization=Normalization.from_dict(manifest.normalization),
    )
    return dataset, manifest


@torch.no_grad()
def plot_qualitative(
    manifest_path: Path,
    run_dirs: list[Path],
    fields_path: Path,
    errors_path: Path,
    *,
    split: str = "test",
    frames: tuple[int, ...] = (0, 65, 130, 196),
    channel: int = 0,
) -> None:
    """Two figures rather than one: fields, and signed errors.

    Previously both were stacked in a single tall figure, which interleaved lead-time-labelled
    field rows with MSE-labelled error rows and made neither easy to scan. Splitting them lets
    each carry one row label and one message.

    Field colour limits are shared across a row and taken from the *target*, so a smoother
    prediction cannot look better merely by being autoscaled to its own narrower range. Error
    limits are shared across a row too, so panels are comparable within a lead time.
    """
    dataset, _ = _dataset(manifest_path, split)
    methods = _load_methods(run_dirs, include_nearest=False)
    trajectory = dataset.trajectory_ids[0]
    lookup = {
        sample.frame: index
        for index, sample in enumerate(dataset.sample_index)
        if sample.trajectory_id == trajectory
    }
    frames = tuple(f for f in frames if f in lookup)

    field_columns = ["coarse input", *methods.keys(), "fine target"]
    field_figure, field_axes = plt.subplots(
        len(frames),
        len(field_columns),
        figsize=(2.6 * len(field_columns), 2.8 * len(frames)),
        layout="constrained",
    )
    error_columns = list(methods.keys())
    error_figure, error_axes = plt.subplots(
        len(frames),
        len(error_columns),
        figsize=(3.0 * len(error_columns), 3.0 * len(frames)),
        layout="constrained",
    )

    for row, frame in enumerate(frames):
        item = dataset[lookup[frame]]
        coarse = item["coarse"].unsqueeze(0)
        target = item["fine"].unsqueeze(0)
        predictions = {name: module(coarse) for name, module in methods.items()}
        hours = item["time"] / 3600

        limit = float(target[0, channel].abs().max())
        errors = {n: (p - target)[0, channel].numpy() for n, p in predictions.items()}
        error_limit = max(float(np.abs(e).max()) for e in errors.values()) or 1.0

        panels = [("coarse input", coarse[0, channel].numpy())]
        panels += [(n, p[0, channel].numpy()) for n, p in predictions.items()]
        panels += [("fine target", target[0, channel].numpy())]
        for column, (title, data) in enumerate(panels):
            axis = field_axes[row, column]
            image = axis.imshow(data, origin="lower", vmin=-limit, vmax=limit, cmap="RdBu_r")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=11)
            if column == 0:
                axis.set_ylabel(f"t = {hours:.1f} h", fontsize=11)
            if column == len(panels) - 1:
                field_figure.colorbar(image, ax=axis, fraction=0.046, label="eta (normalized)")

        for column, name in enumerate(error_columns):
            axis = error_axes[row, column]
            image = axis.imshow(
                errors[name], origin="lower", vmin=-error_limit, vmax=error_limit, cmap="PuOr_r"
            )
            mse = float(((predictions[name] - target) ** 2).mean())
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{name}   MSE {mse:.4f}", fontsize=11)
            if column == 0:
                axis.set_ylabel(f"t = {hours:.1f} h", fontsize=11)
            if column == len(error_columns) - 1:
                error_figure.colorbar(image, ax=axis, fraction=0.046, label="signed error")

    field_figure.suptitle(
        "eta, normalized units. Colour limits shared per row, taken from the target. "
        "Frozen T-03 best checkpoints.",
        fontsize=12,
    )
    field_figure.savefig(fields_path, dpi=115)
    plt.close(field_figure)

    error_figure.suptitle(
        "Signed error, prediction minus target. Colour limits shared per row so panels are "
        "comparable within a lead time.",
        fontsize=12,
    )
    error_figure.savefig(errors_path, dpi=115)
    plt.close(error_figure)


@torch.no_grad()
def animate_comparison(
    manifest_path: Path,
    run_dirs: list[Path],
    out_path: Path,
    *,
    split: str = "test",
    stride: int = 3,
    channel: int = 0,
    fps: int = 8,
) -> None:
    """Animate coarse, both baselines, both models, and the target over lead time.

    Predictions are computed up front rather than inside the animation callback: matplotlib may
    call a frame function more than once, and recomputing a network forward pass per call would
    make the render both slow and non-deterministic in timing.
    """
    dataset, _ = _dataset(manifest_path, split)
    methods = _load_methods(run_dirs, include_nearest=False)
    trajectory = dataset.trajectory_ids[0]
    indices = [
        (sample.frame, index, sample.time)
        for index, sample in enumerate(dataset.sample_index)
        if sample.trajectory_id == trajectory
    ]
    indices = indices[::stride]

    targets, coarses, times = [], [], []
    predictions: dict[str, list[np.ndarray]] = {name: [] for name in methods}
    per_frame_mse: dict[str, list[float]] = {name: [] for name in methods}
    for _, index, time in indices:
        item = dataset[index]
        coarse = item["coarse"].unsqueeze(0)
        target = item["fine"].unsqueeze(0)
        coarses.append(coarse[0, channel].numpy())
        targets.append(target[0, channel].numpy())
        times.append(time / 3600)
        for name, module in methods.items():
            prediction = module(coarse)
            predictions[name].append(prediction[0, channel].numpy())
            per_frame_mse[name].append(float(((prediction - target) ** 2).mean()))

    limit = float(np.abs(np.stack(targets)).max())
    names = list(methods)
    columns = 1 + len(names) + 1

    figure = plt.figure(figsize=(2.5 * columns, 6.2), layout="constrained")
    grid = figure.add_gridspec(2, columns, height_ratios=[3, 2])
    image_axes = [figure.add_subplot(grid[0, i]) for i in range(columns)]
    trace_axis = figure.add_subplot(grid[1, :])

    titles = ["coarse input", *names, "fine target"]
    stacks = [coarses, *[predictions[n] for n in names], targets]
    images = []
    for axis, title, stack in zip(image_axes, titles, stacks, strict=True):
        image = axis.imshow(stack[0], origin="lower", vmin=-limit, vmax=limit, cmap="RdBu_r")
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
        images.append(image)

    for name in names:
        trace_axis.plot(times, per_frame_mse[name], "-", color=COLORS.get(name, "grey"), label=name)
    trace_axis.axhline(MEAN_PREDICTOR, color="black", ls="-.", lw=1, label="channel mean")
    marker = trace_axis.axvline(times[0], color="black", lw=1.2)
    trace_axis.set_xlabel("lead time (hours)")
    trace_axis.set_ylabel("normalized MSE")
    trace_axis.set_yscale("log")
    trace_axis.grid(alpha=0.3)
    trace_axis.legend(fontsize=8, ncol=len(names) + 1, loc="upper left")
    title = figure.suptitle("", fontsize=12)

    def update(frame: int) -> list[Any]:
        for image, stack in zip(images, stacks, strict=True):
            image.set_data(stack[frame])
        marker.set_xdata([times[frame], times[frame]])
        scores = "   ".join(f"{n} {per_frame_mse[n][frame]:.4f}" for n in names)
        title.set_text(f"t = {times[frame]:5.2f} h      {scores}")
        return [*images, marker, title]

    anim = animation.FuncAnimation(figure, update, frames=len(indices), interval=1000 // fps)
    anim.save(str(out_path), writer=animation.PillowWriter(fps=fps))
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/processed/swe_gaussian_32x128_v1/manifest.json",
    )
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "viz")
    parser.add_argument("--gif-stride", type=int, default=3)
    parser.add_argument("--skip-gif", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    plot_final_curves(args.runs, args.out / "final_curves.png", split=args.split)
    print(f"wrote {args.out / 'final_curves.png'}")

    plot_qualitative(
        args.manifest,
        args.runs,
        args.out / "qualitative.png",
        args.out / "qualitative_error.png",
        split=args.split,
    )
    print(f"wrote {args.out / 'qualitative.png'}")
    print(f"wrote {args.out / 'qualitative_error.png'}")

    if not args.skip_gif:
        animate_comparison(
            args.manifest,
            args.runs,
            args.out / "qualitative.gif",
            split=args.split,
            stride=args.gif_stride,
        )
        path = args.out / "qualitative.gif"
        print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
