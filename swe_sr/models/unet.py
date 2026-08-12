"""Residual U-Net x4 for three normalized physical channels (M-02).

Follows the recipe in `docs/ARCHITECTURE.md` exactly:

1. bicubic-upsample the normalized input to its x4 output shape;
2. bias-free 3x3 input convolution with 32 features;
3. three encoder stages with 32, 64, 128 features;
4. two residual blocks per stage, SiLU/Swish activation;
5. 2x2 average pooling to downsample;
6. pixel-shuffle x2 decoder blocks, concatenating matching encoder features;
7. bias-free 3x3 output convolution predicting three residual channels.

This preserves the skip connections, residual blocks, average pooling, Swish activations and
sub-pixel decoder of the 2024 dynamic shallow-water super-resolution work, while adapting it
from same-grid correction to true x4 image-to-image reconstruction (`docs/RESEARCH_MATRIX.md`).
Nothing is vendored from the reference repositories.

The structural contrast with EDSR is deliberate and is what the comparison tests: the U-Net
does its work on the *upsampled* grid with multi-scale context, while EDSR does most of its
work on the coarse grid and upsamples at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from swe_sr.models.common import (
    CHANNELS,
    PixelShuffleUpsampler,
    ResidualBlock,
    ResidualSuperResolution,
    upsample,
)


@dataclass(frozen=True)
class UNetConfig:
    """U-Net hyperparameters. Defaults are the values fixed in `docs/ARCHITECTURE.md`."""

    stage_features: tuple[int, ...] = (32, 64, 128)
    blocks_per_stage: int = 2
    scale: int = 4
    channels: int = CHANNELS
    # Bias-free input and output convolutions, per docs/ARCHITECTURE.md. Interior
    # convolutions keep their biases; only the two documented ones are bias-free.
    bias_free_ends: bool = True
    activation_name: str = "silu"
    _activations: dict[str, type[nn.Module]] = field(
        default_factory=lambda: {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU},
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if len(self.stage_features) < 2:
            raise ValueError("need at least two stages for an encoder/decoder")
        if self.scale < 1 or (self.scale & (self.scale - 1)):
            raise ValueError(f"scale must be a power of two, got {self.scale}")
        if self.activation_name not in self._activations:
            raise ValueError(
                f"unknown activation {self.activation_name!r}; "
                f"available: {sorted(self._activations)}"
            )

    @property
    def activation(self) -> type[nn.Module]:
        return self._activations[self.activation_name]


class _Stage(nn.Module):
    """A channel-changing convolution followed by residual blocks at fixed width."""

    def __init__(
        self, in_features: int, out_features: int, blocks: int, activation: type[nn.Module]
    ) -> None:
        super().__init__()
        self.project = (
            nn.Conv2d(in_features, out_features, 3, padding=1)
            if in_features != out_features
            else nn.Identity()
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(out_features, activation=activation) for _ in range(blocks)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.blocks(self.project(inputs))
        return output


class ResidualUNet(ResidualSuperResolution):
    """Multi-scale residual U-Net operating on the bicubic-upsampled grid.

    Resolution-generic: the only size-dependent operations are pooling and pixel shuffle,
    both integer factors, so 32->128 and 64->256 both work with one set of weights. The
    encoder depth does constrain the minimum input size -- three stages with two poolings need
    the upsampled grid to be divisible by 4 -- which `__init__` states and `forward` checks.
    """

    def __init__(self, config: UNetConfig | None = None) -> None:
        config = config or UNetConfig()
        super().__init__(scale=config.scale)
        self.config = config
        activation = config.activation
        features = config.stage_features
        bias = not config.bias_free_ends

        self.head = nn.Conv2d(config.channels, features[0], 3, padding=1, bias=bias)

        self.encoders = nn.ModuleList()
        previous = features[0]
        for width in features:
            self.encoders.append(_Stage(previous, width, config.blocks_per_stage, activation))
            previous = width
        # Average pooling rather than strided convolution, per the 2024 paper: it is a
        # fixed low-pass filter, which suits fields whose coarse-scale content must survive.
        self.pool = nn.AvgPool2d(2)

        # Decode from the deepest stage back up, concatenating the matching encoder output.
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(len(features) - 1, 0, -1):
            deep, shallow = features[level], features[level - 1]
            # PixelShuffle(2) divides channels by 4, so project to 4x the target width first.
            self.upsamplers.append(PixelShuffleUpsampler(deep))
            self.decoders.append(
                _Stage(deep + shallow, shallow, config.blocks_per_stage, activation)
            )

        self.tail = nn.Conv2d(features[0], config.channels, 3, padding=1, bias=bias)
        self._downsamplings = len(features) - 1

    @property
    def size_multiple(self) -> int:
        """The upsampled grid must be divisible by this for the skips to line up."""
        return int(2**self._downsamplings)

    def residual(self, inputs: torch.Tensor) -> torch.Tensor:
        # Step 1: work on the target grid, so the network refines at full resolution.
        upsampled = upsample(inputs, self.scale)
        height, width = upsampled.shape[-2:]
        multiple = self.size_multiple
        if height % multiple or width % multiple:
            raise ValueError(
                f"upsampled size {height}x{width} must be divisible by {multiple} for "
                f"{self._downsamplings} pooling stages; got a remainder"
            )

        features = self.head(upsampled)

        skips: list[torch.Tensor] = []
        for level, encoder in enumerate(self.encoders):
            features = encoder(features)
            if level < len(self.encoders) - 1:
                skips.append(features)
                features = self.pool(features)

        for upsampler, decoder in zip(self.upsamplers, self.decoders, strict=True):
            features = upsampler(features)
            skip = skips.pop()
            features = decoder(torch.cat([features, skip], dim=1))

        residual: torch.Tensor = self.tail(features)
        return residual


def build_unet(**overrides: object) -> ResidualUNet:
    """Construct a U-Net from keyword overrides, as the model config YAML supplies them."""
    if "stage_features" in overrides and isinstance(overrides["stage_features"], list):
        overrides["stage_features"] = tuple(overrides["stage_features"])
    return ResidualUNet(UNetConfig(**overrides))  # type: ignore[arg-type]
