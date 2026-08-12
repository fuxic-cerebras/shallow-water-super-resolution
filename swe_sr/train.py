"""Training entry point (T-01).

    python -m swe_sr.train --config configs/model/unet_x4.yaml
    python -m swe_sr.train --config configs/model/edsr_x4.yaml \
        --experiment configs/experiment/smoke.yaml

Implements the schedule in `docs/EXPERIMENT_PLAN.md` and the run-directory contract in
`docs/ARCHITECTURE.md`. Three properties are treated as requirements rather than niceties:

- **Determinism.** Everything is seeded and the data order is a pure function of
  `(seed, epoch)`, so a rerun with the same config reproduces the same curve. `docs/VALIDATION.md`
  requires identical configs to reproduce identical results, and without it no timing or
  accuracy comparison between the two models means anything.
- **Model selection on full validation.** The best checkpoint is chosen by lowest
  macro-averaged normalized MSE over the *entire* validation split, never a sampled subset.
- **Honest provenance.** Every run records its resolved config, environment, dataset manifest,
  and the commit it was trained from, including a `-dirty` marker when the tree was not clean.

No training run is launched without notifying the project owner (D015).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from swe_sr.data.dataset import PairedSnapshotDataset, assert_splits_disjoint
from swe_sr.data.manifest import load_manifest
from swe_sr.data.normalization import Normalization
from swe_sr.data.processing import AugmentationPolicy
from swe_sr.data.storage import resolve_array_dir
from swe_sr.metrics.field import CHANNEL_NAMES, per_channel_mse
from swe_sr.models import build_model_from_config, resource_summary
from swe_sr.models.common import bicubic_baseline
from swe_sr.training.config import (
    REPO_ROOT,
    TrainingConfig,
    environment_summary,
    git_commit,
)


def seed_everything(seed: int) -> None:
    """Seed every source of randomness the run touches."""
    random.seed(seed)
    # Legacy global numpy seeding on purpose: incidental `np.random` use anywhere in the
    # data path must be seeded too, which a local Generator would not cover. Project code
    # uses `default_rng` explicitly.
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    # Deterministic algorithms where torch offers them. Not fatal if a kernel lacks one, so
    # warn_only keeps the run going rather than failing on an unrelated op.
    torch.use_deterministic_algorithms(True, warn_only=True)


def peak_memory_megabytes() -> float:
    """Peak resident set size. On CPU this is the meaningful memory number to report."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes.
    return usage / 1024.0


def learning_rate_at(step: int, config: TrainingConfig) -> float:
    """Linear warmup then cosine decay (docs/EXPERIMENT_PLAN.md).

    `step` is 1-based. Warmup starts above zero so the very first step makes progress rather
    than being a wasted forward and backward pass.
    """
    if step <= config.warmup_steps and config.warmup_steps > 0:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.max_steps - config.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def autocast_context(config: TrainingConfig) -> Any:
    """BF16 autocast when the hardware supports it, otherwise a no-op (D015).

    Falling back silently to FP32 would be worse than useless for a timing projection, so the
    resolved precision is recorded in the run summary.
    """
    if config.precision == "bf16" and torch.cpu._is_avx512_bf16_supported():
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def resolved_precision(config: TrainingConfig) -> str:
    if config.precision == "bf16" and torch.cpu._is_avx512_bf16_supported():
        return "bf16"
    return "fp32"


@dataclass
class EpochRecord:
    """One row of `metrics.csv`."""

    epoch: int
    step: int
    train_mse: float
    validation_mse: float
    per_channel: dict[str, float]
    learning_rate: float
    samples_per_second: float
    elapsed_seconds: float
    peak_memory_mb: float
    by_lead_time: dict[int, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "epoch": self.epoch,
            "step": self.step,
            "train_mse": self.train_mse,
            "validation_mse": self.validation_mse,
            "learning_rate": self.learning_rate,
            "samples_per_second": self.samples_per_second,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mb": self.peak_memory_mb,
        }
        row.update({f"validation_mse_{k}": v for k, v in self.per_channel.items()})
        return row


@dataclass
class TrainingResult:
    """What a finished run reports back."""

    run_dir: Path
    run_id: str
    best_validation_mse: float
    best_epoch: int
    steps_completed: int
    stopped_reason: str
    history: list[EpochRecord] = field(default_factory=list)
    projection: dict[str, Any] = field(default_factory=dict)


def _subset(
    dataset: PairedSnapshotDataset, trajectories: int | None, frames: int | None
) -> list[int]:
    """Indices for a scoped view of a dataset, used by the smoke and pilot stages.

    Selects whole trajectories rather than random samples, so a reduced run still respects the
    trajectory-level split and never mixes frames from a trajectory it is not using.
    """
    keep = dataset.trajectory_ids if trajectories is None else dataset.trajectory_ids[:trajectories]
    allowed = set(keep)
    indices: list[int] = []
    seen: dict[str, int] = {}
    for index, sample in enumerate(dataset.sample_index):
        if sample.trajectory_id not in allowed:
            continue
        count = seen.get(sample.trajectory_id, 0)
        if frames is not None and count >= frames:
            continue
        seen[sample.trajectory_id] = count + 1
        indices.append(index)
    return indices


def _batches(
    dataset: PairedSnapshotDataset,
    indices: list[int],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> Any:
    """Yield `(coarse, fine)` batches.

    Deliberately a plain generator rather than a `DataLoader`: the datasets are HDF5-backed and
    single-process here, and the ordering must be a pure function of `(seed, epoch)` so a rerun
    reproduces the same curve exactly.
    """
    order = list(indices)
    if shuffle:
        # A string key rather than a tuple: `random.Random` rejects tuples, unlike numpy's
        # `default_rng`. Deriving it from (seed, epoch) keeps the order a pure function of
        # both, so a rerun reproduces the same curve and resuming an epoch is well defined.
        rng = random.Random(f"{seed}:{epoch}")
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        samples = [dataset[i] for i in chunk]
        yield (
            torch.stack([s["coarse"] for s in samples]),
            torch.stack([s["fine"] for s in samples]),
        )


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    dataset: PairedSnapshotDataset,
    indices: list[int],
    config: TrainingConfig,
) -> tuple[float, dict[str, float], dict[int, float]]:
    """Macro-averaged, per-channel, and per-lead-time normalized MSE over an entire split.

    Accumulated as a sample-weighted sum rather than a mean of batch means, so a short final
    batch cannot skew the result.
    """
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    count = 0
    # Stratification by lead time comes free: validation already touches every frame, so
    # recording each sample's error costs nothing and is what reveals whether the model is
    # improving on the early, well-posed frames or only on the late ones.
    by_lead_time: dict[int, list[float]] = {}
    position = 0
    for coarse, fine in _batches(
        dataset, indices, config.batch_size, shuffle=False, seed=config.seed, epoch=0
    ):
        with autocast_context(config):
            prediction = model(coarse)
        channel_mse = per_channel_mse(prediction.float(), fine).double()
        totals += channel_mse * coarse.shape[0]
        count += coarse.shape[0]

        per_sample = ((prediction.float() - fine) ** 2).mean(dim=(1, 2, 3)).double()
        for offset in range(coarse.shape[0]):
            frame = dataset.sample_index[indices[position + offset]].frame
            by_lead_time.setdefault(frame, []).append(float(per_sample[offset]))
        position += coarse.shape[0]

    if count == 0:
        raise ValueError("validation split produced no batches")
    per_channel = totals / count
    return (
        float(per_channel.mean()),
        {name: float(per_channel[i]) for i, name in enumerate(CHANNEL_NAMES)},
        {frame: float(np.mean(values)) for frame, values in by_lead_time.items()},
    )


def train(config: TrainingConfig, *, verbose: bool = True) -> TrainingResult:
    """Run one training job end to end and write its run directory."""
    if config.torch_threads > 0:
        torch.set_num_threads(config.torch_threads)
    seed_everything(config.seed)

    model_name, model = build_model_from_config(REPO_ROOT / config.model_config)

    manifest_path = REPO_ROOT / config.manifest
    manifest = load_manifest(manifest_path)
    if not manifest.normalization:
        raise ValueError(
            f"{manifest_path} carries no normalization block; point --manifest at the "
            "processed manifest, not the raw one (D019)"
        )
    normalization = Normalization.from_dict(manifest.normalization)
    augmentation = AugmentationPolicy(names=config.augmentations)
    array_dir = resolve_array_dir(manifest_path, manifest.dataset_id)

    datasets = {
        split: PairedSnapshotDataset(
            manifest,
            array_dir,
            split=split,
            normalization=normalization,
            augmentation=augmentation if split == "train" else AugmentationPolicy(),
            seed=config.seed,
        )
        for split in ("train", "validation")
    }
    # Cheap, and it catches the one error that would invalidate every number the run produces.
    assert_splits_disjoint(*datasets.values())

    train_indices = _subset(
        datasets["train"], config.max_train_trajectories, config.max_frames_per_trajectory
    )
    validation_indices = _subset(
        datasets["validation"],
        config.max_validation_trajectories,
        config.max_frames_per_trajectory,
    )
    if not train_indices or not validation_indices:
        raise ValueError("training or validation subset is empty; check the scope limits")

    # The bicubic baseline stratified by lead time, computed once. Every stratified plot is
    # read against it, and without it a rising curve cannot be told from a harder regime.
    saved_times = {sample.frame: sample.time for sample in datasets["validation"].sample_index}
    baseline_by_lead_time = _baseline_by_lead_time(
        datasets["validation"], validation_indices, config
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = config.run_id(model_name, timestamp)
    run_dir = REPO_ROOT / config.run_root / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=True))
    environment = environment_summary()
    environment["resolved_precision"] = resolved_precision(config)
    environment["model"] = resource_summary(model)
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True))
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    )

    if verbose:
        print(f"run_id      : {run_id}")
        parameters = int(resource_summary(model)["trainable_parameters"])
        print(f"model       : {model_name}, {parameters:,} params")
        print(f"precision   : {resolved_precision(config)} (requested {config.precision})")
        used_trajectories = {datasets["train"].sample_index[i].trajectory_id for i in train_indices}
        print(
            f"train       : {len(train_indices)} samples over {len(used_trajectories)} trajectories"
        )
        print(f"validation  : {len(validation_indices)} samples")
        print(f"threads     : {torch.get_num_threads()}")
        print()

    history: list[EpochRecord] = []
    best_mse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    step = 0
    started = time.perf_counter()
    stopped_reason = "max_epochs"

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_started = time.perf_counter()

        for coarse, fine in _batches(
            datasets["train"],
            train_indices,
            config.batch_size,
            shuffle=True,
            seed=config.seed,
            epoch=epoch,
        ):
            step += 1
            learning_rate = learning_rate_at(step, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config):
                prediction = model(coarse)
            # Loss in float32 even under autocast: a BF16 reduction over 49k elements loses
            # precision that matters for a metric used to select checkpoints.
            loss = per_channel_mse(prediction.float(), fine).mean()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            optimizer.step()

            epoch_loss += float(loss.detach()) * coarse.shape[0]
            epoch_samples += coarse.shape[0]
            if step >= config.max_steps:
                stopped_reason = "max_steps"
                break

        train_mse = epoch_loss / max(epoch_samples, 1)
        throughput = epoch_samples / max(time.perf_counter() - epoch_started, 1e-9)

        validation_mse, per_channel, by_lead_time = evaluate_split(
            model, datasets["validation"], validation_indices, config
        )
        record = EpochRecord(
            epoch=epoch,
            step=step,
            train_mse=train_mse,
            validation_mse=validation_mse,
            per_channel=per_channel,
            learning_rate=learning_rate_at(max(step, 1), config),
            samples_per_second=throughput,
            elapsed_seconds=time.perf_counter() - started,
            peak_memory_mb=peak_memory_megabytes(),
            by_lead_time=by_lead_time,
        )
        history.append(record)

        # Always write `last`, so a killed run is still evaluable.
        torch.save(model.state_dict(), run_dir / "checkpoints" / "last.pt")
        if validation_mse < best_mse:
            best_mse, best_epoch = validation_mse, epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")
        else:
            epochs_without_improvement += 1

        if verbose:
            marker = " *" if best_epoch == epoch else ""
            print(
                f"epoch {epoch:>4}  step {step:>6}  train {train_mse:.6f}  "
                f"val {validation_mse:.6f}{marker}  lr {record.learning_rate:.2e}  "
                f"{throughput:.1f} samp/s  {record.elapsed_seconds:.1f}s  "
                f"{record.peak_memory_mb:.0f}MB",
                flush=True,
            )

        _write_metrics(run_dir, history)
        # Redrawn every epoch: a long run should be inspectable while it is still going, and a
        # killed run should leave usable figures behind.
        _write_curves(run_dir, history, saved_times=saved_times, baseline=baseline_by_lead_time)

        if step >= config.max_steps:
            stopped_reason = "max_steps"
            break
        if epochs_without_improvement >= config.early_stopping_patience:
            stopped_reason = "early_stopping"
            break

    _write_curves(run_dir, history, saved_times=saved_times, baseline=baseline_by_lead_time)
    projection = _project_full_run(history, config)
    result = TrainingResult(
        run_dir=run_dir,
        run_id=run_id,
        best_validation_mse=best_mse,
        best_epoch=best_epoch,
        steps_completed=step,
        stopped_reason=stopped_reason,
        history=history,
        projection=projection,
    )
    _write_summary(run_dir, config, model_name, manifest, result, environment)
    return result


def _write_metrics(run_dir: Path, history: list[EpochRecord]) -> None:
    """Rewrite `metrics.csv` after every epoch, so a killed run still leaves usable curves."""
    rows = [record.to_row() for record in history]
    if not rows:
        return
    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def _baseline_by_lead_time(
    dataset: PairedSnapshotDataset, indices: list[int], config: TrainingConfig
) -> dict[int, float]:
    """Bicubic macro MSE per lead time, the reference every stratified curve is read against."""
    baseline = bicubic_baseline(scale=4)
    values: dict[int, list[float]] = {}
    position = 0
    for coarse, fine in _batches(
        dataset, indices, config.batch_size, shuffle=False, seed=config.seed, epoch=0
    ):
        errors = ((baseline(coarse) - fine) ** 2).mean(dim=(1, 2, 3)).double()
        for offset in range(coarse.shape[0]):
            frame = dataset.sample_index[indices[position + offset]].frame
            values.setdefault(frame, []).append(float(errors[offset]))
        position += coarse.shape[0]
    return {frame: float(np.mean(v)) for frame, v in values.items()}


def _write_curves(
    run_dir: Path,
    history: list[EpochRecord],
    *,
    saved_times: dict[int, float] | None = None,
    baseline: dict[int, float] | None = None,
) -> None:
    """Write `curves.png`, redrawn after every epoch so a running job stays inspectable.

    Guarded matplotlib import: `docs/ARCHITECTURE.md` lists curves.png in the run directory but
    also insists plotting is a client of the data, never a dependency of it, so a node without
    matplotlib must still be able to train.

    The third panel is the diagnostic one. It shows validation error against lead time as
    training progresses, against the bicubic reference and the unit line that a mean-predictor
    scores. That is what distinguishes a model still learning to sharpen early frames from one
    that has settled on a single smoothing policy.
    """
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    panels = 3 if (saved_times and baseline) else 2
    figure, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4.6), layout="constrained")
    steps = [record.step for record in history]

    axes[0].plot(steps, [r.train_mse for r in history], "--", label="train")
    axes[0].plot(steps, [r.validation_mse for r in history], "-o", ms=3, label="validation")
    axes[0].set_title("loss vs optimizer step")
    axes[0].set_xlabel("optimizer step")

    for name in CHANNEL_NAMES:
        axes[1].plot(steps, [r.per_channel[name] for r in history], "-o", ms=3, label=name)
    axes[1].set_title("per-channel validation MSE")
    axes[1].set_xlabel("optimizer step")

    for axis in axes[:2]:
        axis.axhline(1.0, color="grey", ls="-.", lw=1, label="predicting the channel mean")
        axis.set_ylabel("normalized MSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    if panels == 3 and saved_times and baseline:
        axis = axes[2]
        frames = sorted(baseline)
        hours = [saved_times[f] / 3600 for f in frames]
        axis.plot(
            hours,
            [baseline[f] for f in frames],
            "-",
            color="green",
            lw=2,
            label="bicubic baseline",
        )
        # Early epochs faint, latest epoch bold: the direction of travel is the whole point.
        shown = [*history[:: max(1, len(history) // 6)], history[-1]]
        for index, record in enumerate(shown):
            if not record.by_lead_time:
                continue
            alpha = 0.25 + 0.75 * (index / max(len(shown) - 1, 1))
            axis.plot(
                hours,
                [record.by_lead_time.get(f, float("nan")) for f in frames],
                "-",
                color="crimson",
                alpha=alpha,
                lw=2 if record is history[-1] else 1,
                label=f"epoch {record.epoch}" if record is history[-1] or index == 0 else None,
            )
        axis.axhline(1.0, color="grey", ls="-.", lw=1, label="predicting the channel mean")
        axis.set_title("validation MSE vs lead time (faint = early epochs)")
        axis.set_xlabel("lead time (hours)")
        axis.set_ylabel("normalized macro MSE")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)

    figure.savefig(run_dir / "curves.png", dpi=110)
    plt.close(figure)


def _project_full_run(history: list[EpochRecord], config: TrainingConfig) -> dict[str, Any]:
    """Project a full run's cost from measured throughput (T-02).

    Uses the median epoch time rather than the mean: the first epoch carries import and cache
    warmup, which would inflate a mean and make the projection optimistic in the wrong
    direction.
    """
    if not history:
        return {}
    epoch_times = [
        history[i].elapsed_seconds - (history[i - 1].elapsed_seconds if i else 0.0)
        for i in range(len(history))
    ]
    median_epoch = float(np.median(epoch_times))
    steps_per_epoch = history[0].step if history[0].step else 1
    epochs_for_cap = config.max_steps / steps_per_epoch
    return {
        "median_epoch_seconds": median_epoch,
        "steps_per_epoch": steps_per_epoch,
        "seconds_per_step": median_epoch / steps_per_epoch,
        "epochs_to_reach_max_steps": epochs_for_cap,
        "projected_full_run_hours": median_epoch * epochs_for_cap / 3600.0,
        "peak_memory_mb": max(record.peak_memory_mb for record in history),
        "mean_samples_per_second": float(
            np.mean([record.samples_per_second for record in history])
        ),
    }


def _write_summary(
    run_dir: Path,
    config: TrainingConfig,
    model_name: str,
    manifest: Any,
    result: TrainingResult,
    environment: dict[str, Any],
) -> None:
    summary = {
        "run_id": result.run_id,
        "stage": config.stage,
        "model": model_name,
        "git_commit": git_commit(),
        "config_hash": config.config_hash,
        "dataset_id": manifest.dataset_id,
        "pair_id": manifest.pair_id,
        "ic_registry_hash": manifest.ic_registry_hash,
        "seed": config.seed,
        "resolved_precision": environment.get("resolved_precision"),
        "steps_completed": result.steps_completed,
        "stopped_reason": result.stopped_reason,
        "best_epoch": result.best_epoch,
        # Units stated: this is a dimensionless MSE on normalized channels, macro-averaged.
        "best_validation_mse_normalized_macro": result.best_validation_mse,
        "checkpoint_selection_rule": (
            "lowest macro-averaged normalized MSE on the full validation split"
        ),
        "projection": result.projection,
        "epochs": [record.to_row() for record in result.history],
        # Stratified by lead time so the decorrelation finding lives in the artifact rather
        # than only in a figure that could be regenerated differently later.
        "validation_mse_by_lead_time": [
            {"epoch": record.epoch, "by_frame": record.by_lead_time} for record in result.history
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="model config YAML")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="experiment config YAML (smoke, pilot, full); defaults to the full schedule",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="override the dataset manifest")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = TrainingConfig.from_yaml(args.experiment) if args.experiment else TrainingConfig()
    overrides: dict[str, Any] = {"model_config": str(args.config)}
    if args.manifest is not None:
        overrides["manifest"] = str(args.manifest)
    from dataclasses import replace

    config = replace(config, **overrides)

    result = train(config, verbose=not args.quiet)
    print(f"\nrun_dir                 : {result.run_dir}")
    print(f"stopped                 : {result.stopped_reason} after {result.steps_completed} steps")
    print(
        f"best validation MSE     : {result.best_validation_mse:.6f} (normalized, macro-averaged)"
    )
    print(f"best epoch              : {result.best_epoch}")
    for key, value in result.projection.items():
        print(f"  {key:<28}: {value:.4f}" if isinstance(value, float) else f"  {key:<28}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
