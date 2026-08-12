"""Shared building blocks for the super-resolution models (M-01..M-03).

The single most important thing here is `upsample`. `docs/ARCHITECTURE.md` requires that
bicubic interpolation "aligns the physical domain endpoints; its exact library options are
fixed in config and shared by both models and the baseline". If the baseline and the models'
residual path ever used different interpolation settings, the comparison between them would
be measuring the interpolation difference rather than the learned residual. So there is
exactly one implementation, and both use it.

`align_corners=True` is the correct choice, not a stylistic one: the grids are
`linspace(-L/2, L/2, N)` with both endpoints included (D016), so corner pixel *centers* sit
exactly on the domain corners at every resolution. `align_corners=True` is precisely the
convention that maps corner centers onto corner centers. With `False`, torch would treat
pixels as area samples offset by half a cell, which would reintroduce the same half-cell
error D011 exists to eliminate.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = 3
InterpolationMode = Literal["bicubic", "bilinear", "nearest"]

# Frozen by D016. Changing either value requires a new decision record, because every
# reported number depends on them.
DEFAULT_MODE: InterpolationMode = "bicubic"
ALIGN_CORNERS = True


def upsample(
    inputs: torch.Tensor,
    scale: int,
    *,
    mode: InterpolationMode = DEFAULT_MODE,
    align_corners: bool = ALIGN_CORNERS,
) -> torch.Tensor:
    """Endpoint-aligned spatial upsampling by an integer node factor.

    Args:
        inputs: `[batch, channels, height, width]`.
        scale: node-count factor. 4 for this project's pairs.
        mode: `bicubic` for the residual path and the bicubic baseline, `nearest` for the
            nearest-neighbour baseline.
        align_corners: kept as an argument only so tests can demonstrate what the wrong
            setting does. Production callers must leave it at the default.

    Returns:
        `[batch, channels, height * scale, width * scale]`.

    Note that bicubic interpolation can overshoot beyond the input range. That is left
    unclamped on purpose: `docs/ARCHITECTURE.md` requires the output be unconstrained during
    training, and clamping to an image-style range is explicitly forbidden.
    """
    if inputs.ndim != 4:
        raise ValueError(f"expected [batch, channels, height, width], got {tuple(inputs.shape)}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if scale == 1:
        return inputs

    height, width = inputs.shape[-2:]
    target = (height * scale, width * scale)
    if mode == "nearest":
        # `nearest` takes no align_corners; it is a pure index map.
        return F.interpolate(inputs, size=target, mode="nearest")
    return F.interpolate(inputs, size=target, mode=mode, align_corners=align_corners)


class Interpolation(nn.Module):
    """A parameter-free `nn.Module` baseline wrapping `upsample` (M-01).

    Being a module means the baselines go through exactly the same evaluation path as the
    trained models -- same batching, same device handling, same metric code -- so a
    comparison cannot accidentally differ in anything but the mapping itself.
    """

    def __init__(self, scale: int = 4, mode: InterpolationMode = DEFAULT_MODE) -> None:
        super().__init__()
        self.scale = scale
        self.mode: InterpolationMode = mode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return upsample(inputs, self.scale, mode=self.mode)

    def extra_repr(self) -> str:
        return f"scale={self.scale}, mode={self.mode}, align_corners={ALIGN_CORNERS}"


def nearest_baseline(scale: int = 4) -> Interpolation:
    """Nearest-neighbour x4 baseline (M-01)."""
    return Interpolation(scale=scale, mode="nearest")


def bicubic_baseline(scale: int = 4) -> Interpolation:
    """Endpoint-aligned bicubic x4 baseline (M-01)."""
    return Interpolation(scale=scale, mode="bicubic")


class ResidualSuperResolution(nn.Module):
    """Base class for `y_hat = bicubic(x) + R(x)` (D006).

    Both models share this outer form so the comparison isolates the learned residual. The
    subclass supplies `residual()`; the outer bicubic path is fixed here and identical to the
    M-01 baseline, so a model that learned nothing would exactly reproduce that baseline.
    """

    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        self.scale = scale

    def residual(self, inputs: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.shape[-3] != CHANNELS:
            raise ValueError(
                f"expected {CHANNELS} channels [eta, u, v] on axis -3, got {tuple(inputs.shape)}"
            )
        return upsample(inputs, self.scale) + self.residual(inputs)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a scaled identity skip.

    `res_scale` follows the EDSR practice of shrinking the residual branch before adding it,
    which is what lets a deep stack of these train stably from random initialization. The
    activation is injected rather than fixed, because the two models deliberately differ:
    EDSR uses ReLU as published, while the U-Net uses SiLU/Swish per the 2024 shallow-water
    paper (docs/ARCHITECTURE.md).
    """

    def __init__(
        self,
        features: int,
        *,
        activation: type[nn.Module] = nn.ReLU,
        res_scale: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1, bias=bias),
            activation(),
            nn.Conv2d(features, features, 3, padding=1, bias=bias),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        branch: torch.Tensor = self.body(inputs)
        return inputs + branch * self.res_scale


class PixelShuffleUpsampler(nn.Module):
    """One sub-pixel x2 upsampling stage: conv to 4x channels, then `PixelShuffle(2)`.

    Sub-pixel convolution is the decoder both reference architectures use. Two of these give
    the x4 factor without ever materializing a transposed-convolution checkerboard.
    """

    def __init__(self, features: int, *, activation: type[nn.Module] | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(features, features * 4, 3, padding=1),
            nn.PixelShuffle(2),
        ]
        if activation is not None:
            layers.append(activation())
        self.body = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Typed local: nn.Sequential.__call__ is untyped, so returning it directly is Any.
        output: torch.Tensor = self.body(inputs)
        return output


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
