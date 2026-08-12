"""Evaluate a frozen checkpoint on a fresh post-training workload (E-02, E-03).

    python -m swe_sr.evaluate_fresh --scenario ring_ood --run-dir runs/<run-id>

`docs/VALIDATION.md` fixes the protocol: freeze normalization, architecture, weights, and
model-selection decisions; generate the workload with new manifest IDs; run both models and the
interpolation baselines without fine-tuning; report separately from the original test split.
And critically: "If out-of-distribution results fail, preserve and report the failure rather
than changing the workload after inspection."

Two safeguards enforce that rather than relying on discipline:

- the workload is generated on demand from its scenario name and checked disjoint from the
  training registry, so it cannot share an initial condition with training;
- normalization comes from the *training* manifest, never refitted on the fresh data, so a
  fresh score cannot be flattered by statistics fitted to it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from swe_sr.data.fresh import SCENARIOS, assert_disjoint_from_registry, build_fresh_workload
from swe_sr.data.manifest import load_manifest
from swe_sr.data.normalization import Normalization
from swe_sr.data.processing import destagger
from swe_sr.metrics.aggregate import SnapshotMetric, aggregate_by_trajectory
from swe_sr.metrics.field import field_metrics
from swe_sr.metrics.physics import PhysicalParameters, physics_metrics
from swe_sr.models import build_baseline, build_model_from_config
from swe_sr.solver.config import ResolutionPair
from swe_sr.solver.diagnostics import assert_admissible
from swe_sr.solver.runner import sample_schedule, solve
from swe_sr.training.config import REPO_ROOT, git_commit

BASELINE_NAMES = ("nearest", "bicubic")


def _solve_pair(
    pair: ResolutionPair, initial_condition: Any, steps: np.ndarray, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two independent solves of one analytic IC, exactly as the training data was made (D002).

    Generating the fresh workload by the same route matters: if it were produced differently --
    say by downsampling the fine run -- a degradation could be blamed on the workload rather
    than on the model.
    """
    del stride
    coarse_config = pair.coarse_config()
    fine_config = pair.fine_config()
    coarse = solve(coarse_config, initial_condition, sample_steps=steps, diagnostic_stride=8)
    fine = solve(fine_config, initial_condition, sample_steps=steps, diagnostic_stride=8)
    assert_admissible(coarse.diagnostics)
    assert_admissible(fine.diagnostics)
    if not np.array_equal(coarse.times, fine.times):
        raise AssertionError("fresh workload saved times differ between resolutions")
    return coarse.fields, fine.fields


@torch.no_grad()
def evaluate_fresh(
    run_dir: Path,
    scenario: str,
    *,
    count: int = 4,
    snapshot_count: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate a fresh workload and evaluate the run's best checkpoint plus baselines on it."""
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())

    # Normalization from the TRAINING manifest, never refitted on fresh data.
    manifest_path = REPO_ROOT / str(config["manifest"])
    manifest = load_manifest(manifest_path)
    normalization = Normalization.from_dict(manifest.normalization)
    parameters = PhysicalParameters.from_manifest(manifest)

    pair = ResolutionPair(
        pair_id=manifest.pair_id,
        coarse_nodes=manifest.coarse_nodes,
        fine_nodes=manifest.fine_nodes,
    )
    # Same cadence as training, subsampled in count only, so lead times are comparable.
    stride = int(manifest.sample_steps[1] - manifest.sample_steps[0])
    discard = int(manifest.sample_steps[0])
    # Default to the training manifest's own snapshot count so the fresh workload spans the SAME
    # lead-time range as the held-out test split. Restricting it silently biases the comparison:
    # with 12 snapshots the workload covers only 2.0-3.9 h, the regime where this model is
    # weakest, so a fresh score would look like out-of-distribution degradation when it is
    # really a lead-time mismatch.
    if snapshot_count is None:
        snapshot_count = len(manifest.sample_steps)
    steps = sample_schedule(discard, stride, snapshot_count)

    workload = build_fresh_workload(scenario, count=count)
    assert_disjoint_from_registry(workload)

    model_name, model = build_model_from_config(
        REPO_ROOT / f"configs/model/{summary['model']}_x4.yaml"
    )
    model.load_state_dict(torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True))
    model.eval()
    methods: dict[str, nn.Module] = {n: build_baseline(n) for n in BASELINE_NAMES}
    methods[model_name] = model

    snapshots: dict[str, list[SnapshotMetric]] = {name: [] for name in methods}
    lead_times: dict[str, dict[float, list[float]]] = {name: {} for name in methods}
    normalized: dict[str, dict[str, list[float]]] = {name: {} for name in methods}
    physical: dict[str, dict[str, list[float]]] = {name: {} for name in methods}
    timings: dict[str, float] = dict.fromkeys(methods, 0.0)
    frames_seen = 0

    for _, trajectory_id, initial_condition in workload.trajectories:
        coarse_raw, fine_raw = _solve_pair(pair, initial_condition, steps, stride)
        # Destagger then normalize, exactly as the training loader does (D011, D019).
        coarse = torch.from_numpy(normalization.apply(destagger(coarse_raw)).astype(np.float32))
        fine = torch.from_numpy(normalization.apply(destagger(fine_raw)).astype(np.float32))
        times = steps * manifest.shared_time_step
        frames_seen += coarse.shape[0]

        for name, module in methods.items():
            began = time.perf_counter()
            prediction = module(coarse)
            timings[name] += time.perf_counter() - began

            per_sample = ((prediction - fine) ** 2).mean(dim=(1, 2, 3))
            for index in range(coarse.shape[0]):
                snapshots[name].append(
                    SnapshotMetric(
                        trajectory_id=trajectory_id,
                        frame=index,
                        value=float(per_sample[index]),
                    )
                )
                hours = float(times[index]) / 3600.0
                lead_times[name].setdefault(hours, []).append(float(per_sample[index]))

            for key, value in field_metrics(prediction, fine).items():
                normalized[name].setdefault(key, []).append(value)
            physical_prediction = torch.from_numpy(
                normalization.invert(prediction.numpy()).astype(np.float32)
            )
            physical_fine = torch.from_numpy(normalization.invert(fine.numpy()).astype(np.float32))
            for key, value in physics_metrics(
                physical_prediction, physical_fine, parameters
            ).items():
                physical[name].setdefault(key, []).append(value)

    report: dict[str, Any] = {
        "run_id": summary["run_id"],
        "stage": summary.get("stage"),
        "model": model_name,
        "scenario": scenario,
        "evaluation_only": True,
        "fine_tuned": False,
        "workload": workload.to_dict(),
        "training_dataset_id": manifest.dataset_id,
        "normalization_source": "training manifest, not refitted on fresh data",
        "trained_at_commit": summary.get("git_commit"),
        "evaluated_at_commit": git_commit(),
        "trajectories": len(workload.trajectories),
        "snapshots_per_trajectory": snapshot_count,
        # Stated so a fresh score is never compared against a differently-spanned test score.
        "lead_time_hours": {
            "first": float(steps[0] * manifest.shared_time_step / 3600.0),
            "last": float(steps[-1] * manifest.shared_time_step / 3600.0),
            "matches_training_range": snapshot_count == len(manifest.sample_steps),
        },
        "seed": seed,
        "methods": {},
        "reference_notes": {
            "mean_predictor_normalized_mse": 1.0,
            "ring_diversity_caveat": (
                "ring_ood trajectories differ only by a small centre offset, so they are near "
                "duplicates and the trajectory bootstrap interval is correspondingly tight; "
                "read it as within-workload precision, not as generalization spread"
            )
            if scenario == "ring_ood"
            else None,
            "reported_separately": (
                "fresh results are never mixed into the held-out test score "
                "(docs/DATASET.md); a failure here is preserved and reported, not tuned away"
            ),
        },
    }
    for name in methods:
        aggregate = aggregate_by_trajectory(
            snapshots[name], metric="macro_mse_normalized", seed=seed
        )
        report["methods"][name] = {
            "aggregate_macro_mse_normalized": aggregate.to_dict(),
            "normalized_metrics": {k: float(np.mean(v)) for k, v in normalized[name].items()},
            "physical_metrics_si": {k: float(np.mean(v)) for k, v in physical[name].items()},
            "normalized_macro_mse_by_lead_time_hours": {
                f"{h:.3f}": float(np.mean(v)) for h, v in sorted(lead_times[name].items())
            },
            "seconds_per_frame": timings[name] / max(frames_seen, 1),
        }

    (run_dir / f"evaluation_fresh_{scenario}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def render_table(report: dict[str, Any]) -> str:
    seeds = [entry["seed"] for entry in report["workload"]["trajectories"]]
    lead_time = report["lead_time_hours"]
    lines = [
        f"run      : {report['run_id']}  (stage={report['stage']})",
        f"scenario : {report['scenario']}  (evaluation only, no fine-tuning)",
        f"workload : {report['trajectories']} trajectories x "
        f"{report['snapshots_per_trajectory']} snapshots, seeds {seeds}",
        f"lead time: {lead_time['first']:.2f} h to {lead_time['last']:.2f} h "
        f"(matches training range: {lead_time['matches_training_range']})",
        f"normalization: {report['normalization_source']}",
        "",
        "reported separately from the held-out test split (docs/DATASET.md)",
        "reference: predicting the channel mean scores 1.0 normalized MSE",
        "",
        f"{'method':<10} {'normMSE':>9} {'95% CI':>19} {'eta relL2':>10} {'mass err':>10}",
    ]
    for name, method in report["methods"].items():
        aggregate = method["aggregate_macro_mse_normalized"]
        interval = f"[{aggregate['ci_low']:.4f},{aggregate['ci_high']:.4f}]"
        lines.append(
            f"{name:<10} {aggregate['mean']:>9.4f} {interval:>19} "
            f"{method['normalized_metrics']['rel_l2_eta']:>10.4f} "
            f"{method['physical_metrics_si']['relative_mass_error']:>10.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--snapshots",
        type=int,
        default=None,
        help="snapshots per trajectory; defaults to the training manifest's own count so the "
        "lead-time range matches the test split",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = evaluate_fresh(
        args.run_dir,
        args.scenario,
        count=args.count,
        snapshot_count=args.snapshots,
        seed=args.seed,
    )
    print(render_table(report))
    print(f"\nwrote {args.run_dir / f'evaluation_fresh_{args.scenario}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
