"""Decompose why a super-resolution model beats or loses to bicubic, per lead time.

Both models score *worse* than bicubic at short lead times, on the pair they were trained on
and more so on the transfer pair. Aggregate MSE cannot say why. This script separates the two
distinguishable failures using an identity rather than an estimate.

Write the correction bicubic would need, and the one the model actually applied:

    need    = y - bicubic(x)
    applied = y_hat - bicubic(x)

With `r = ||applied|| / ||need||` and `c = cos(applied, need)`, expanding
`||y_hat - y||^2 = ||applied - need||^2` gives **exactly**

    MSE_model / MSE_bicubic = r^2 - 2*r*c + 1

which is quadratic in `r` and minimized at `r = c`, where it equals `1 - c^2`. So:

  * `c` alone bounds what the model could achieve if its correction were rescaled optimally --
    it measures whether the correction points the right way, which no rescaling can fix;
  * the gap between the measured ratio and `1 - c^2` is pure magnitude miscalibration, which a
    single scalar per lead time would remove.

That distinction is the whole question for transfer: a model that merely over-corrects is
recoverable, one whose correction points the wrong way is not.

The identity is checked numerically per snapshot rather than assumed, because it is the entire
basis of the conclusion. Ratios are reduced by the project's aggregation protocol -- per
snapshot, then within trajectory, then equal weight across trajectories (docs/VALIDATION.md) --
never averaged per batch, which is the defect V-05 found.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import swe_sr  # noqa: F401  # numpy-before-torch load-order guard; see swe_sr/__init__
from swe_sr.data.dataset import PairedSnapshotDataset
from swe_sr.data.manifest import load_manifest
from swe_sr.data.normalization import Normalization
from swe_sr.data.storage import resolve_array_dir
from swe_sr.models import build_baseline, build_model_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def _protocol(per_trajectory: dict[str, list[float]]) -> float:
    """Within-trajectory mean, then equal weight across trajectories."""
    if not per_trajectory:
        return float("nan")
    return float(np.mean([float(np.mean(values)) for values in per_trajectory.values()]))


def analyze(
    manifest_path: Path,
    run_dir: Path,
    *,
    split: str = "test",
    stride: int = 4,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    normalization = Normalization.from_dict(manifest.normalization)
    dataset = PairedSnapshotDataset(
        manifest,
        resolve_array_dir(manifest_path, manifest.dataset_id),
        split=split,
        normalization=normalization,
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    name, model = build_model_from_config(REPO_ROOT / f"configs/model/{summary['model']}_x4.yaml")
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True))
    model.eval()
    bicubic = build_baseline("bicubic")

    saved_times = manifest.saved_times
    # Subsampled on purpose, and reported: at 256x256 a full sweep of both models over both
    # pairs is hours of CPU. A stride keeps every trajectory and spans the whole lead-time
    # range, which is what the stratification needs. docs/AGENT_WORKFLOW.md forbids a silent cap.
    selected = [i for i, s in enumerate(dataset.sample_index) if s.frame % stride == 0]

    by_lead: dict[float, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    worst_identity_error = 0.0

    with torch.no_grad():
        for index in selected:
            sample = dataset[index]
            coarse = sample["coarse"].unsqueeze(0)
            fine = sample["fine"].unsqueeze(0)
            meta = dataset.sample_index[index]

            base = bicubic(coarse)
            prediction = model(coarse)

            need = (fine - base).flatten().double()
            applied = (prediction - base).flatten().double()

            need_norm = float(torch.linalg.vector_norm(need))
            applied_norm = float(torch.linalg.vector_norm(applied))
            if need_norm == 0.0:
                continue
            ratio_r = applied_norm / need_norm
            cosine = (
                float(torch.dot(applied, need)) / (applied_norm * need_norm)
                if applied_norm > 0.0
                else 0.0
            )

            mse_bicubic = float(torch.mean((base - fine).double() ** 2))
            mse_model = float(torch.mean((prediction - fine).double() ** 2))
            measured = mse_model / mse_bicubic
            predicted = ratio_r**2 - 2.0 * ratio_r * cosine + 1.0
            worst_identity_error = max(worst_identity_error, abs(measured - predicted))

            hours = saved_times[meta.frame] / 3600.0
            bucket = by_lead[round(hours, 3)]
            bucket["r"][meta.trajectory_id].append(ratio_r)
            bucket["c"][meta.trajectory_id].append(cosine)
            bucket["ratio"][meta.trajectory_id].append(measured)
            # 1 - c^2 is the best this correction could do if rescaled optimally per snapshot.
            bucket["floor"][meta.trajectory_id].append(1.0 - cosine**2)

    rows = []
    for hours in sorted(by_lead):
        bucket = by_lead[hours]
        rows.append(
            {
                "hours": hours,
                "magnitude_ratio_r": _protocol(bucket["r"]),
                "alignment_c": _protocol(bucket["c"]),
                "mse_ratio_vs_bicubic": _protocol(bucket["ratio"]),
                "best_possible_ratio_if_rescaled": _protocol(bucket["floor"]),
            }
        )

    return {
        "model": name,
        "run_id": summary["run_id"],
        "pair_id": manifest.pair_id,
        "split": split,
        "snapshots_analyzed": len(selected),
        "snapshots_available": len(dataset),
        "frame_stride": stride,
        "trajectories": len(dataset.trajectory_ids),
        "worst_per_snapshot_identity_error": worst_identity_error,
        "identity": "mse_model / mse_bicubic == r^2 - 2*r*c + 1, exactly",
        "rows": rows,
    }


def render(result: dict[str, Any], *, every: int = 4) -> str:
    lines = [
        f"{result['model']} on {result['pair_id']}  (split={result['split']})",
        f"  {result['snapshots_analyzed']} of {result['snapshots_available']} snapshots "
        f"(every {result['frame_stride']}th frame), {result['trajectories']} trajectories",
        f"  worst per-snapshot identity error: {result['worst_per_snapshot_identity_error']:.2e}"
        "  (exact identity, so this bounds float error only)",
        "",
        "  ratios reduced per-snapshot, then within-trajectory, then equal weight across",
        "  trajectories. r<c means under-correcting, r>c over-correcting; ratio<1 beats bicubic.",
        "",
        f"  {'t (h)':>7} {'r':>7} {'c':>7} {'MSE ratio':>10} {'floor 1-c^2':>12} {'verdict':>14}",
    ]
    for row in result["rows"][::every]:
        ratio = row["mse_ratio_vs_bicubic"]
        r_value = row["magnitude_ratio_r"]
        c_value = row["alignment_c"]
        if ratio > 1.0:
            verdict = "loses" if r_value <= c_value * 1.2 else "over-corrects"
        else:
            verdict = "beats bicubic"
        lines.append(
            f"  {row['hours']:>7.2f} {r_value:>7.3f} {c_value:>7.3f} {ratio:>10.4f} "
            f"{row['best_possible_ratio_if_rescaled']:>12.4f} {verdict:>14}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    result = analyze(args.manifest, args.run_dir, split=args.split, stride=args.stride)
    print(render(result))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
