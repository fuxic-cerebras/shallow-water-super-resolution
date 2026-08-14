"""ConvMixer adapted to x4 super-resolution of three normalized channels (M-05, D023).

Follows Trockman & Kolter, "Patches Are All You Need?" (arXiv:2201.09792). The published
model is a patch-embedding stem followed by `depth` repetitions of a single block that mixes
space with a large-kernel *depthwise* convolution and channels with a *pointwise* one, at
constant resolution throughout:

    z_0     = BN(GELU(Conv_{p, stride p}(x)))
    z'_l    = BN(GELU(DepthwiseConv_k(z_{l-1}))) + z_{l-1}      (Eq. 2)
    z_{l+1} = BN(GELU(PointwiseConv(z'_l)))                     (Eq. 3)

Note the single residual, around the depthwise convolution only. That asymmetry is measured,
not incidental: adding a second one around the pointwise convolution cost 1.10% on CIFAR-10
(paper Table 3), so it is reproduced here exactly.

Why this architecture earns a third slot in the benchmark. The U-Net reaches global context by
pooling down a pyramid, and EDSR never downsamples but sees only ~33 px through its stack of
16 3x3 convolutions -- barely one grid width. ConvMixer is *isotropic*, holding one resolution
end to end, and buys range purely through kernel size. At the defaults below the receptive
field is 129 px at low resolution, wider than both the 32x32 and the 64x64 grid, so a single
unit sees the entire basin without any pooling. Whether that helps in a closed rotating basin,
where gravity waves reflect off the walls, is precisely the question the arm exists to answer.

Four deliberate departures from the reference, each required by this project:

- **A sub-pixel decoder replaces the classifier.** The published model ends in global average
  pooling and a linear layer. Pooling destroys the spatial field, so the head here is the same
  1x1 projection, two `PixelShuffleUpsampler` stages, and 3x3 tail that EDSR uses.
- **A 1x1 projection to `head_features` before that decoder.** Not in the paper, and load
  bearing: `PixelShuffleUpsampler(features)` costs `36h^2 + 4h` parameters, so a full-width
  256 decoder would be 4.7M, larger than the entire body. Projecting to 64 makes the decoder
  byte-identical to EDSR's, which keeps the comparison about the body.
- **An outer bicubic residual** (D006, inherited from `ResidualSuperResolution`), so all three
  architectures share one outer form.
- **`patch_size` defaults to 1.** The stem is then a pointwise lift rather than a true patch
  embedding. The paper's own CIFAR-10 sweep is at 32x32, exactly this project's low
  resolution, and prefers p=1 there: p=2 costs 0.80% and p=4 costs 3.27%. Patching discards
  the high-frequency content super-resolution exists to recover, so all upsampling is deferred
  to the decoder. The parameter is kept so p>1 remains a one-line ablation.

BatchNorm is retained as published, which is the one contested choice here (D023). EDSR omits
normalization deliberately, and this file does not follow it: the paper measures 1.44% against
LayerNorm on CIFAR-10, so the layer is doing real work. The cost is that a training-mode
forward pass couples samples within a batch. Every path in this repository that reports a
number calls `model.eval()` first, which switches BatchNorm to its running statistics and
restores exact per-sample independence; `tests/models/test_models.py` pins both halves of that
statement. Nothing is vendored from the reference implementation; it is used as documented
evidence for the architecture, per `docs/RESEARCH_MATRIX.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from swe_sr.models.common import (
    CHANNELS,
    PixelShuffleUpsampler,
    ResidualSuperResolution,
)

# `batch` is the published block. `none` is the D023 A-03 ablation arm, which tests EDSR's
# claim that normalization harms super-resolution against ConvMixer's that BatchNorm is worth
# 1.44% -- the one point where the two papers this project draws on directly disagree.
NORMALIZATIONS: tuple[str, ...] = ("batch", "none")


def _normalization(name: str, features: int) -> nn.Module:
    return nn.BatchNorm2d(features) if name == "batch" else nn.Identity()


@dataclass(frozen=True)
class ConvMixerConfig:
    """ConvMixer hyperparameters. Defaults are the values fixed in `docs/ARCHITECTURE.md`.

    The paper names its models `ConvMixer-h/d`; these defaults are ConvMixer-256/16 with
    `k=9, p=1`, sized to land between EDSR (1,517,571) and the U-Net (1,930,208) so that no
    result can be attributed to capacity.
    """

    features: int = 256  # h, the patch-embedding dimension held constant through the body
    depth: int = 16  # d, repetitions of the mixer block
    kernel_size: int = 9  # k, the depthwise kernel; the knee of the paper's CIFAR-10 sweep
    patch_size: int = 1  # p, stem stride; see the module docstring for why 1
    head_features: int = 64  # decoder width, matching EDSR's so the two share a decoder
    scale: int = 4
    channels: int = CHANNELS
    # `batch` as published (D023). `none` is the A-03 arm. It is the one variant here that
    # cannot hold capacity exactly fixed, because removing BatchNorm necessarily removes its
    # affine parameters: 1,720,067 against 1,703,171, a 0.98% difference.
    normalization: str = "batch"
    # The next two are stabilizers, both off by default so the published block is unchanged.
    # They exist because BatchNorm cannot simply be deleted from this architecture: the
    # pointwise stage is not residual, so 16 unnormalized non-residual layers compound and the
    # first block's gradient norm measures 1.1e-11 against 9.1e-02 with BatchNorm. Restoring a
    # residual around the pointwise fixes the gradient but lets activations reach std 49;
    # adding EDSR's residual scaling brings them to 0.41 with a first/last gradient ratio of
    # 0.90. Together they are EDSR's recipe -- residual everywhere, scaled, unnormalized --
    # which is what makes an unnormalized arm trainable rather than merely untested.
    pointwise_residual: bool = False
    res_scale: float = 1.0
    # Stochastic depth on the residual branches, ramped linearly from 0 at the first block to
    # this value at the last, as DeiT and timm do. 0.0 reproduces the published model. A-05.
    drop_path: float = 0.0
    # D006 by default. `none` is the D022 ablation arm, which for ConvMixer means the isotropic
    # body must produce the absolute field rather than a correction to bicubic. No config ships
    # for it yet; the flag works because the base class implements it.
    outer_baseline: str = "bicubic"

    def __post_init__(self) -> None:
        if self.scale not in (1, 2, 4, 8):
            raise ValueError(f"scale must be a power of two up to 8, got {self.scale}")
        if self.depth < 1:
            raise ValueError(f"depth must be >= 1, got {self.depth}")
        if self.features < 1:
            raise ValueError(f"features must be >= 1, got {self.features}")
        if self.head_features < 1:
            raise ValueError(f"head_features must be >= 1, got {self.head_features}")
        # Even kernels cannot be centred, and `padding="same"` rejects them for stride-1 convs
        # anyway. Failing here names the actual problem.
        if self.kernel_size < 1 or not self.kernel_size % 2:
            raise ValueError(f"kernel_size must be odd and >= 1, got {self.kernel_size}")
        # The decoder undoes the stem stride with pixel shuffle, so `scale * patch_size` has to
        # be a power of two for the x2 stages to compose to exactly the right factor.
        if self.patch_size < 1 or (self.patch_size & (self.patch_size - 1)):
            raise ValueError(f"patch_size must be a power of two, got {self.patch_size}")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(
                f"unknown normalization {self.normalization!r}; available: {sorted(NORMALIZATIONS)}"
            )
        if not 0.0 < self.res_scale <= 1.0:
            raise ValueError(f"res_scale must be in (0, 1], got {self.res_scale}")
        if not 0.0 <= self.drop_path < 1.0:
            raise ValueError(f"drop_path must be in [0, 1), got {self.drop_path}")

    @property
    def receptive_field(self) -> int:
        """Extent in low-resolution pixels that one output unit of the body can see.

        Each block widens it by `k - 1` internal cells, and each internal cell spans `p` input
        pixels. Reported because it is the property that distinguishes this architecture from
        the other two, and `docs/ARCHITECTURE.md` quotes it.
        """
        return self.patch_size * (1 + self.depth * (self.kernel_size - 1))


class _DropPath(nn.Module):
    """Stochastic depth: drop a whole residual branch, per sample, while training.

    Only ever wraps a *residual* branch, so a dropped sample falls back to the identity path
    rather than losing its signal. Off in eval mode, which keeps the D023 reporting contract
    intact: every path that reports a number calls `model.eval()`, so inference is
    deterministic and batch-independent regardless of this.

    Added for A-05. ConvMixer reaches the U-Net's *final* training loss by epoch 21 (0.0148
    against 0.0145) yet generalizes 34% worse, and its train/validation gap grows to 5.24x
    against the U-Net's 2.75x, so the binding constraint is generalization rather than
    capacity. The ConvMixer paper notes DeiT used stochastic depth while it did not.
    """

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"drop probability must be in [0, 1), got {probability}")
        self.probability = probability

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return inputs
        keep = 1.0 - self.probability
        # One Bernoulli draw per sample, broadcast over channels and space, then rescaled so
        # the expected activation is unchanged and eval needs no correction.
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep)
        return inputs * mask / keep

    def extra_repr(self) -> str:
        return f"probability={self.probability}"


class _MixerBlock(nn.Module):
    """One ConvMixer layer: residual depthwise spatial mixing, then pointwise channel mixing.

    The residual wraps the depthwise branch only, per Eq. 2/Eq. 3. Resolution is unchanged,
    which is what makes the body isotropic and therefore resolution-generic.
    """

    def __init__(
        self,
        features: int,
        kernel_size: int,
        normalization: str = "batch",
        *,
        pointwise_residual: bool = False,
        res_scale: float = 1.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.pointwise_residual = pointwise_residual
        self.res_scale = res_scale
        self.drop_path = _DropPath(drop_path)
        self.depthwise = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, groups=features, padding="same"),
            nn.GELU(),
            _normalization(normalization, features),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(features, features, 1),
            nn.GELU(),
            _normalization(normalization, features),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Typed locals: nn.Sequential.__call__ is untyped, so returning it directly is Any.
        mixed: torch.Tensor = inputs + self.drop_path(self.res_scale * self.depthwise(inputs))
        branch: torch.Tensor = self.pointwise(mixed)
        if self.pointwise_residual:
            dropped: torch.Tensor = self.drop_path(self.res_scale * branch)
            return mixed + dropped
        # Not a residual path, so stochastic depth must not touch it: dropping here would zero
        # the signal outright rather than fall back to an identity.
        return branch


class ConvMixerSR(ResidualSuperResolution):
    """Isotropic large-kernel body with a sub-pixel decoder.

    Resolution-generic by construction: every layer is a convolution or a pixel shuffle and
    nothing depends on the grid size, so one set of weights accepts 32x32 and 64x64 and
    produces 128x128 and 256x256. D016 requires that, and `tests/models/test_models.py`
    checks both shapes. At the default `patch_size=1` there is no divisibility constraint at
    all; at p>1 the input must be divisible by p, which `residual` checks.
    """

    def __init__(self, config: ConvMixerConfig | None = None) -> None:
        config = config or ConvMixerConfig()
        super().__init__(scale=config.scale, outer_baseline=config.outer_baseline)
        self.config = config

        self.stem = nn.Sequential(
            nn.Conv2d(
                config.channels,
                config.features,
                config.patch_size,
                stride=config.patch_size,
            ),
            nn.GELU(),
            _normalization(config.normalization, config.features),
        )
        self.body = nn.Sequential(
            *[
                _MixerBlock(
                    config.features,
                    config.kernel_size,
                    config.normalization,
                    pointwise_residual=config.pointwise_residual,
                    res_scale=config.res_scale,
                    # Linear ramp: early blocks are kept almost always, deep ones dropped most.
                    # `depth - 1` in the denominator so the last block sees exactly `drop_path`;
                    # guarded because a depth-1 body would divide by zero.
                    drop_path=(
                        config.drop_path * index / (config.depth - 1) if config.depth > 1 else 0.0
                    ),
                )
                for index in range(config.depth)
            ]
        )
        # Narrow to the decoder width before pixel shuffle; see the module docstring.
        self.project = nn.Conv2d(config.features, config.head_features, 1)

        # The decoder must undo the stem's stride as well as reach the target scale.
        stages = (config.scale * config.patch_size).bit_length() - 1
        self.upsampler = nn.Sequential(
            *[PixelShuffleUpsampler(config.head_features) for _ in range(stages)]
        )
        self.tail = nn.Conv2d(config.head_features, config.channels, 3, padding=1)

    @property
    def size_multiple(self) -> int:
        """The input grid must be divisible by this for the stem to tile it exactly."""
        return self.config.patch_size

    def residual(self, inputs: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        multiple = self.size_multiple
        if height % multiple or width % multiple:
            raise ValueError(
                f"input size {height}x{width} must be divisible by the patch size {multiple}; "
                f"got a remainder"
            )

        # Unlike the U-Net, nothing is interpolated in here: the body works on the coarse grid
        # and the decoder produces the fine one, as EDSR does.
        features: torch.Tensor = self.body(self.stem(inputs))
        residual: torch.Tensor = self.tail(self.upsampler(self.project(features)))
        return residual


def build_convmixer(**overrides: object) -> ConvMixerSR:
    """Construct a ConvMixer from keyword overrides, as the model config YAML supplies them."""
    return ConvMixerSR(ConvMixerConfig(**overrides))  # type: ignore[arg-type]
