"""Figures for the cross-resolution transfer experiment (64->256 tested with 32->128 models).

Two figures, from the stored artifacts only -- no inference here, so these are exactly the
numbers `docs/TRANSFER.md` quotes:

  * `transfer_leadtime.png` -- normalized macro MSE against lead time, one panel per pair, same
    y-axis so the two are directly comparable. The point of the figure is the *crossover*: both
    models lose to bicubic at short lead times and win at long ones, and transfer moves the
    crossover later.
  * `transfer_decomposition.png` -- the r/c decomposition. Left, the correction magnitude ratio
    `r` against the alignment `c` that would make it optimal: distance above the c-curve is
    over-correction. Right, the measured MSE ratio against the floor `1 - c^2` that optimal
    rescaling would reach.

A horizontal line at 1.0 is drawn on the ratio axes and it is load-bearing here, unlike in
`final_curves.png` where nothing came near it: ratio 1.0 *is* bicubic, so the line is the
win/lose boundary and the crossings are the result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
COLORS = {"bicubic": "#2ca02c", "edsr": "#1f77b4", "unet": "#d62728"}
PAIR_LABELS = {
    "swe_gaussian_32x128_v1": "32->128  (trained on this)",
    "swe_gaussian_64x256_v1": "64->256  (transfer, never trained)",
}


def _evaluation(run_dir: Path, pair_id: str, trained_pair: str, split: str) -> dict[str, Any]:
    name = (
        f"evaluation_{split}.json"
        if pair_id == trained_pair
        else f"evaluation_{split}__{pair_id}.json"
    )
    return json.loads((run_dir / name).read_text())


def _lead_curve(method: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    breakdown = method["normalized_macro_mse_by_lead_time_hours"]
    hours = np.array(sorted(float(k) for k in breakdown))
    values = np.array([breakdown[f"{h:.3f}"] for h in hours])
    return hours, values


def plot_leadtime(
    run_dirs: list[Path], pairs: list[str], out_path: Path, *, trained_pair: str, split: str
) -> None:
    figure, axes = plt.subplots(
        1, len(pairs), figsize=(6.2 * len(pairs), 4.6), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for axis, pair_id in zip(axes, pairs, strict=True):
        drew_baseline = False
        for run_dir in run_dirs:
            report = _evaluation(run_dir, pair_id, trained_pair, split)
            if not drew_baseline:
                hours, values = _lead_curve(report["methods"]["bicubic"])
                axis.plot(hours, values, color=COLORS["bicubic"], lw=2.0, label="bicubic")
                drew_baseline = True
            model = report["model"]
            hours, values = _lead_curve(report["methods"][model])
            axis.plot(hours, values, color=COLORS[model], lw=2.0, label=model)
        axis.set_yscale("log")
        axis.set_xlabel("lead time (hours)")
        axis.set_title(PAIR_LABELS.get(pair_id, pair_id), fontsize=11)
        axis.grid(alpha=0.3, which="both")
        axis.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("normalized macro-averaged MSE")
    # Shared limits: the comparison between panels is the whole point, and independent
    # autoscaling would hide that the transfer task is intrinsically the easier one.
    low = min(min(a.get_ylim()) for a in axes)
    high = max(max(a.get_ylim()) for a in axes)
    for axis in axes:
        axis.set_ylim(low, high)
    figure.suptitle(
        "Where a learned model beats bicubic, and where it does not\n"
        "test split, 1,576 snapshots over 8 trajectories, both models trained only on 32->128",
        fontsize=12,
    )
    figure.savefig(out_path, dpi=130)
    plt.close(figure)


def plot_decomposition(run_dirs: list[Path], pairs: list[str], out_path: Path, split: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), layout="constrained")
    styles = {pairs[0]: "-", pairs[-1]: "--"}
    for run_dir in run_dirs:
        for pair_id in pairs:
            path = run_dir / f"decomposition_{split}__{pair_id}.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text())
            rows = result["rows"]
            model = result["model"]
            hours = np.array([row["hours"] for row in rows])
            style = styles.get(pair_id, ":")
            label = f"{model}  {pair_id.split('_')[2]}"

            axes[0].plot(
                hours,
                [row["magnitude_ratio_r"] for row in rows],
                color=COLORS[model],
                ls=style,
                lw=1.9,
                label=f"r  {label}",
            )
            axes[0].plot(
                hours,
                [row["alignment_c"] for row in rows],
                color=COLORS[model],
                ls=style,
                lw=1.0,
                alpha=0.45,
                label=f"c  {label}",
            )
            axes[1].plot(
                hours,
                [row["mse_ratio_vs_bicubic"] for row in rows],
                color=COLORS[model],
                ls=style,
                lw=1.9,
                label=f"measured  {label}",
            )
            axes[1].plot(
                hours,
                [row["best_possible_ratio_if_rescaled"] for row in rows],
                color=COLORS[model],
                ls=style,
                lw=1.0,
                alpha=0.45,
                label=f"floor 1-c^2  {label}",
            )

    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("lead time (hours)")
        axis.grid(alpha=0.3, which="both")
        # Unlike the mean-predictor line in final_curves.png, this 1.0 is the result: it is
        # bicubic itself, so every crossing is a change of verdict.
        axis.axhline(1.0, color="0.25", lw=1.0, ls=":", zorder=0)
        axis.legend(fontsize=6.5, ncols=2, loc="best")
    axes[0].set_ylabel("correction magnitude ratio r,  alignment c")
    axes[0].set_title("r above c means over-correcting; r = c is optimal", fontsize=11)
    axes[1].set_ylabel("MSE relative to bicubic")
    axes[1].set_title("measured against what rescaling alone could reach", fontsize=11)
    figure.suptitle(
        "Why the models lose at short lead time: magnitude, not direction\n"
        "solid = 32->128 (trained), dashed = 64->256 (transfer);  1.0 = bicubic",
        fontsize=12,
    )
    figure.savefig(out_path, dpi=130)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["swe_gaussian_32x128_v1", "swe_gaussian_64x256_v1"],
    )
    parser.add_argument("--trained-pair", default="swe_gaussian_32x128_v1")
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "viz")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    plot_leadtime(
        args.runs,
        args.pairs,
        args.out / "transfer_leadtime.png",
        trained_pair=args.trained_pair,
        split=args.split,
    )
    print(f"wrote {args.out / 'transfer_leadtime.png'}")

    plot_decomposition(args.runs, args.pairs, args.out / "transfer_decomposition.png", args.split)
    print(f"wrote {args.out / 'transfer_decomposition.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
