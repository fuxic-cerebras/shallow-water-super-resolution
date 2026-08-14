"""Super-resolution models and interpolation baselines.

`build_model` is the single construction point, so training, evaluation, and tests all
instantiate models the same way and a config file is the only thing that varies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch import nn

from swe_sr.models.common import (
    Interpolation,
    ResidualSuperResolution,
    bicubic_baseline,
    count_parameters,
    nearest_baseline,
    upsample,
)
from swe_sr.models.convmixer import ConvMixerConfig, ConvMixerSR, build_convmixer
from swe_sr.models.edsr import EDSR, EDSRConfig, build_edsr
from swe_sr.models.unet import ResidualUNet, UNetConfig, build_unet

__all__ = [
    "EDSR",
    "ConvMixerConfig",
    "ConvMixerSR",
    "EDSRConfig",
    "Interpolation",
    "ResidualSuperResolution",
    "ResidualUNet",
    "UNetConfig",
    "bicubic_baseline",
    "build_baseline",
    "build_convmixer",
    "build_edsr",
    "build_model",
    "build_model_from_config",
    "build_unet",
    "count_parameters",
    "nearest_baseline",
    "resource_summary",
    "upsample",
]

ARCHITECTURES = {"unet": build_unet, "edsr": build_edsr, "convmixer": build_convmixer}
BASELINES = {"nearest": nearest_baseline, "bicubic": bicubic_baseline}


def build_model(architecture: str, **overrides: Any) -> ResidualSuperResolution:
    """Construct a trainable model by architecture name."""
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture {architecture!r}; available: {sorted(ARCHITECTURES)}"
        )
    return ARCHITECTURES[architecture](**overrides)


def build_baseline(name: str, scale: int = 4) -> Interpolation:
    """Construct a parameter-free interpolation baseline (M-01)."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    return BASELINES[name](scale=scale)


def build_model_from_config(path: Path) -> tuple[str, ResidualSuperResolution]:
    """Load `configs/model/*.yaml` and construct the model it names.

    Returns `(name, model)`. Unknown top-level keys are rejected rather than ignored, since a
    silently dropped hyperparameter would yield a model not matching its own config.
    """
    payload = yaml.safe_load(Path(path).read_text())
    known = {"name", "architecture", "model"}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    model = build_model(payload["architecture"], **payload.get("model", {}))
    return str(payload["name"]), model


def resource_summary(module: nn.Module) -> dict[str, Any]:
    """Parameter and buffer counts, reported alongside accuracy per docs/ARCHITECTURE.md."""
    total = sum(p.numel() for p in module.parameters())
    return {
        "trainable_parameters": count_parameters(module),
        "total_parameters": total,
        "buffer_elements": sum(b.numel() for b in module.buffers()),
        # Weights only. Activations dominate at run time and are measured in the pilot.
        "parameter_megabytes": total * 4 / 1e6,
    }
