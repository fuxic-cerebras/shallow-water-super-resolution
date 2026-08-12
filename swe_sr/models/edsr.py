"""EDSR adapted to three normalized physical channels (M-03).

Follows the baseline configuration in `docs/ARCHITECTURE.md`: 16 residual blocks, 64
features, 3x3 convolutions, ReLU, residual scaling 0.1, a global body skip, and two
pixel-shuffle x2 stages for x4 output.

Three deliberate departures from the reference `EDSR-PyTorch`, each required by this project:

- **No `MeanShift`.** The reference subtracts a fixed RGB mean. These are elevation and
  velocity channels in SI units, and dataset normalization (D019) already centres them.
  A hard-coded RGB mean would be actively wrong.
- **Random initialization.** Natural-image weights encode edge statistics of photographs,
  which do not transfer to a rotating shallow-water field, and the input channels do not even
  mean the same thing.
- **An outer bicubic residual.** The reference maps low-resolution features to the output
  directly. Here both models predict a residual over the same endpoint-aligned bicubic
  baseline (D006), so the U-Net/EDSR comparison isolates learned detail.

Nothing is copied from the reference implementation; it is used as documented evidence for
the architecture, per `docs/RESEARCH_MATRIX.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from swe_sr.models.common import (
    CHANNELS,
    PixelShuffleUpsampler,
    ResidualBlock,
    ResidualSuperResolution,
)


@dataclass(frozen=True)
class EDSRConfig:
    """EDSR hyperparameters. Defaults are the values fixed in `docs/ARCHITECTURE.md`."""

    features: int = 64
    blocks: int = 16
    res_scale: float = 0.1
    scale: int = 4
    channels: int = CHANNELS

    def __post_init__(self) -> None:
        if self.scale not in (1, 2, 4, 8):
            raise ValueError(f"scale must be a power of two up to 8, got {self.scale}")
        if self.blocks < 1:
            raise ValueError(f"blocks must be >= 1, got {self.blocks}")
        if self.features < 1:
            raise ValueError(f"features must be >= 1, got {self.features}")


class EDSR(ResidualSuperResolution):
    """Residual-in-residual EDSR body with a sub-pixel decoder.

    Resolution-generic by construction: every layer is a convolution or a pixel shuffle, so
    the same weights accept 32x32 and 64x64 inputs and produce 128x128 and 256x256 outputs.
    D016 requires that, and `tests/models/test_models.py` checks both shapes.
    """

    def __init__(self, config: EDSRConfig | None = None) -> None:
        config = config or EDSRConfig()
        super().__init__(scale=config.scale)
        self.config = config

        self.head = nn.Conv2d(config.channels, config.features, 3, padding=1)
        self.body = nn.Sequential(
            *[
                ResidualBlock(config.features, activation=nn.ReLU, res_scale=config.res_scale)
                for _ in range(config.blocks)
            ],
            # The body's own trailing convolution, before the global skip closes.
            nn.Conv2d(config.features, config.features, 3, padding=1),
        )

        stages = config.scale.bit_length() - 1  # 4 -> two x2 stages
        self.upsampler = nn.Sequential(
            *[PixelShuffleUpsampler(config.features) for _ in range(stages)]
        )
        self.tail = nn.Conv2d(config.features, config.channels, 3, padding=1)

    def residual(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.head(inputs)
        # Global body skip: the identity path lets the stack learn a correction rather than
        # having to reconstruct the features it was given.
        features = features + self.body(features)
        residual: torch.Tensor = self.tail(self.upsampler(features))
        return residual


def build_edsr(**overrides: object) -> EDSR:
    """Construct an EDSR from keyword overrides, as the model config YAML supplies them."""
    return EDSR(EDSRConfig(**overrides))  # type: ignore[arg-type]
