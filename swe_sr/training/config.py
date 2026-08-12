"""Typed training configuration and run identity (T-01).

Defaults are the schedule fixed in `docs/EXPERIMENT_PLAN.md`. Every value that affects a
result is here rather than in code, so `runs/<run_id>/config.yaml` fully describes the run.

The run ID follows `docs/ARCHITECTURE.md`: timestamp, model name, short config hash, and Git
commit. The config hash covers the declared fields only, so two runs that differ merely in
where they wrote output share a hash, while any change to the schedule, seed, or data does not.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# docs/EXPERIMENT_PLAN.md.
PRIMARY_SEED = 20260812


@dataclass(frozen=True)
class TrainingConfig:
    """One training run, fully specified."""

    # -- Identity and data -----------------------------------------------------
    model_config: str = "configs/model/unet_x4.yaml"
    manifest: str = "data/processed/swe_gaussian_32x128_v1/manifest.json"
    run_root: str = "runs"
    stage: str = "full"  # smoke | pilot | full, for labelling only

    # -- Optimizer (docs/EXPERIMENT_PLAN.md) -----------------------------------
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    batch_size: int = 8
    gradient_clip_norm: float = 1.0
    max_epochs: int = 100
    max_steps: int = 30_000
    early_stopping_patience: int = 15
    seed: int = PRIMARY_SEED

    # -- Precision and threading ----------------------------------------------
    # BF16 if supported, otherwise FP32. The Slurm nodes are AMD EPYC 9R14 with
    # avx512_bf16 (D015), so autocast is a real speedup there rather than a no-op.
    precision: str = "bf16"
    num_workers: int = 0
    torch_threads: int = 0  # 0 leaves torch's own default

    # -- Scope limits, used by the smoke and pilot stages ----------------------
    max_train_trajectories: int | None = None
    max_validation_trajectories: int | None = None
    max_frames_per_trajectory: int | None = None
    # Cap validation cost during smoke runs; None validates the full split as required.
    validate_every_epochs: int = 1

    augmentations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.precision not in ("bf16", "fp32"):
            raise ValueError(f"precision must be bf16 or fp32, got {self.precision!r}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps {self.warmup_steps} must be below max_steps {self.max_steps}; "
                "otherwise the cosine schedule never decays"
            )

    @classmethod
    def from_yaml(cls, path: Path) -> TrainingConfig:
        payload = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in field_names(cls)}
        unknown = set(payload) - known
        if unknown:
            # A silently ignored key would produce a run that does not match its own config.
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        if "augmentations" in payload and payload["augmentations"] is not None:
            payload["augmentations"] = tuple(payload["augmentations"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["augmentations"] = list(self.augmentations)
        return payload

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def run_id(self, model_name: str, timestamp: str) -> str:
        """`<timestamp>_<model>_<config8>_<commit8>`, per docs/ARCHITECTURE.md."""
        return f"{timestamp}_{model_name}_{self.config_hash[:8]}_{git_commit()[:8]}"


def field_names(cls: type) -> tuple[Any, ...]:
    from dataclasses import fields

    return fields(cls)


def git_commit() -> str:
    """Current commit, reporting a dirty tree honestly.

    A run trained from uncommitted code is not reproducible from the commit alone, so the
    marker belongs in the run's provenance rather than being quietly dropped.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def environment_summary() -> dict[str, Any]:
    """Everything needed to interpret a timing number later (docs/ARCHITECTURE.md)."""
    import torch

    summary: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit(),
    }
    # CPU model and flags matter here: the whole BF16 case rests on avx512_bf16 (D015).
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                summary["cpu_model"] = line.split(":", 1)[1].strip()
                break
        summary["avx512_bf16"] = "avx512_bf16" in cpuinfo
        summary["amx_bf16"] = "amx_bf16" in cpuinfo
    except OSError:
        pass
    with contextlib.suppress(AttributeError, OSError):
        summary["cpu_count"] = len(os.sched_getaffinity(0))
    for variable in ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURMD_NODENAME"):
        value = os.environ.get(variable)
        if value:
            summary[variable.lower()] = value
    return summary
