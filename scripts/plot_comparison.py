"""Training-loss and validation-MSE comparison figures.

Two figures, because they answer two different questions:

- `compare_architectures.png` -- U-Net against EDSR against ConvMixer as published (D023).
- `compare_convmixer_variants.png` -- the ConvMixer arms against each other (A-03, A-05).

Both plot training loss and validation MSE side by side against epoch rather than optimizer
step. Every run uses the same 788-step epoch on the frozen manifest, so the two axes are
equivalent and epochs read more directly.

Colour is assigned from a fixed categorical order that was checked for colour-vision
separation rather than chosen by eye; `docs/RESULTS.md` records the substitution. The palette
the repository used before this (matplotlib's tab10) failed: ConvMixer's purple against EDSR's
blue measured dE 1.7 under protanopia and 14.2 even with normal vision, and that is the single
most important pair in the first figure. An entity keeps its colour across both figures, so
ConvMixer is aqua in each.

The unnormalized arm is drawn in muted grey rather than taking a categorical slot. That is not
a workaround for running out of colours: it is the one variant that changes the *normalization
design* rather than a regularization setting, so it belongs to a different class than the
arms it is shown against, and it scores far enough outside their range that a categorical hue
would imply a peer comparison the numbers do not support.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

# Chart surface and ink, light mode.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"

# Fixed categorical order, validated at --pairs all for both figures. Keyed by series, not by
# rank, so a series never changes colour between figures.
SERIES = {
    "unet": ("U-Net", "#2a78d6", "-"),
    "edsr": ("EDSR", "#eb6834", "-"),
    "convmixer": ("ConvMixer", "#1baf7a", "-"),
    "convmixer_droppath": ("ConvMixer + stochastic depth", "#2a78d6", "-"),
    "convmixer_wd": ("ConvMixer + weight decay 1e-2", "#eb6834", "-"),
    "convmixer_d12": ("ConvMixer-256/12", "#4a3aa7", "-"),
    # Different design class, so grey rather than a categorical slot. See the module docstring.
    "convmixer_nonorm": ("ConvMixer, no normalization", MUTED, "--"),
}

# Normalized channels have unit variance, so predicting the channel mean scores exactly 1.0.
# Split-independent and exact, which is why it is the reference line rather than a bicubic
# value measured on one particular split.
MEAN_PREDICTOR = 1.0


def _series_key(run_dir: Path) -> str:
    """Which series a run belongs to.

    The weight-decay arm records `model: convmixer`, because weight decay lives in the
    experiment config rather than the model config, so it would otherwise collide with the
    published run. Disambiguated on the recorded value.
    """
    import yaml

    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    model = str(summary["model"])
    if model == "convmixer" and float(config.get("weight_decay", 0.0)) > 1e-5:
        return "convmixer_wd"
    return model


def _load(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text())
    evaluation = run_dir / "evaluation_test.json"
    test = None
    if evaluation.is_file():
        payload = json.loads(evaluation.read_text())
        test = payload["methods"][payload["model"]]["aggregate_macro_mse_normalized"]["mean"]
    return {
        "key": _series_key(run_dir),
        "epochs": [e["epoch"] for e in summary["epochs"]],
        "train": [e["train_mse"] for e in summary["epochs"]],
        "validation": [e["validation_mse"] for e in summary["epochs"]],
        "best": summary["best_validation_mse_normalized_macro"],
        "best_epoch": summary.get("best_epoch"),
        "test": test,
        "parameters": summary.get("resources", {}).get("trainable_parameters"),
    }


def _style(axis: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_yscale("log")
    axis.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    axis.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
    axis.set_title(title, color=INK, fontsize=10.5, pad=8)
    axis.grid(True, which="major", color=MUTED, alpha=0.22, lw=0.6)
    axis.grid(True, which="minor", color=MUTED, alpha=0.10, lw=0.5)
    axis.tick_params(colors=INK_SECONDARY, labelsize=8.5, which="both")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)
        axis.spines[side].set_linewidth(0.8)


def figure(runs: list[dict[str, Any]], title: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    for axis, field, panel in (
        (axes[0], "train", "Training loss"),
        (axes[1], "validation", "Validation MSE"),
    ):
        for run in runs:
            label, colour, dash = SERIES[run["key"]]
            # The legend carries each series' held-out test score, which doubles as the
            # visible-label relief the palette check requires for the low-contrast aqua.
            # Labelled once, on the validation panel, because the two panels share a single
            # legend outside the axes. With five series an in-axes legend covered exactly the
            # band where the arms separate.
            legend = "_nolegend_" if field == "train" else f"{label} — test {run['test']:.4f}"
            axis.plot(
                run["epochs"],
                run[field],
                color=colour,
                ls=dash,
                lw=2.0,
                marker="o",
                ms=4.5,
                markevery=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.2,
                label=legend,
                zorder=3,
            )
        if field == "validation":
            # No reference line for the channel-mean predictor (1.0) or bicubic (0.4295) here.
            # Every curve sits 5-25x below both, so drawing either would spend most of the
            # vertical range on empty space and flatten the 0.04-0.08 band that is the actual
            # comparison. Both levels are stated in the table instead.
            axis.annotate(
                "★ selected checkpoint",
                xy=(0.985, 0.97),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=8,
                color=MUTED,
            )
            # Mark each run's selected checkpoint: the reported number comes from there, not
            # from the last epoch, and several arms early-stop well before the budget.
            for run in runs:
                _, colour, _ = SERIES[run["key"]]
                axis.plot(
                    run["best_epoch"],
                    run["best"],
                    marker="*",
                    ms=13,
                    color=colour,
                    markeredgecolor=SURFACE,
                    markeredgewidth=1.0,
                    zorder=4,
                )
        _style(axis, "epoch", "normalized macro-averaged MSE", panel)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=min(3, len(labels)),
        fontsize=8.5,
        frameon=False,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(title, color=INK, fontsize=12, x=0.008, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def table(runs: list[dict[str, Any]], caption: str) -> None:
    """The table view the palette check obliges for a low-contrast series."""
    print(f"\n{caption}\n")
    print(
        f"For scale: bicubic scores 0.4295 on the test split and predicting the channel mean "
        f"scores exactly {MEAN_PREDICTOR:.1f}. Neither is drawn in the figure; every curve sits "
        f"5-25x below both, so a reference line would flatten the band being compared.\n"
    )
    print("| series | best val | best epoch | epochs | final train | gap | test |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for run in sorted(runs, key=lambda r: r["test"] if r["test"] is not None else 1e9):
        label = SERIES[run["key"]][0]
        gap = run["best"] / run["train"][run["best_epoch"] - 1]
        print(
            f"| {label} | {run['best']:.4f} | {run['best_epoch']} | {len(run['epochs'])} "
            f"| {run['train'][-1]:.4f} | {gap:.2f}x | {run['test']:.4f} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "viz")
    parser.add_argument("--title", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()

    runs = [_load(r) for r in args.runs]
    unknown = [r["key"] for r in runs if r["key"] not in SERIES]
    if unknown:
        raise SystemExit(f"no colour assigned for series {unknown}; add it to SERIES")
    figure(runs, args.title, args.out / f"{args.name}.png")
    table(runs, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
