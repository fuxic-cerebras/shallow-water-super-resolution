"""Model contract, gradients, checkpoints, and baselines (M-01..M-04, G4 gate).

`docs/AGENT_WORKFLOW.md` sets the G4 bar: `[B,3,H,W] -> [B,3,4H,4W]` for H = 32 and 64,
finite gradients, and checkpoint reload. `docs/VALIDATION.md` adds that both models return
the two required shapes, produce finite gradients for *all* trainable parameters, and that
reload reproduces inference output.

These run on synthetic fixtures by design: `docs/AGENT_WORKFLOW.md` has ML working against
synthetic data until the real release passes G3, so the model gate never depends on a
generated dataset.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn

from swe_sr.models.common import (
    ALIGN_CORNERS,
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

# The two shapes docs/ARCHITECTURE.md requires. Small feature widths keep the gate fast
# while exercising every code path; the published widths are checked separately.
REQUIRED_SHAPES = [(32, 128), (64, 256)]


def _small_unet() -> ResidualUNet:
    return ResidualUNet(UNetConfig(stage_features=(8, 16, 32), blocks_per_stage=1))


def _small_edsr() -> EDSR:
    return EDSR(EDSRConfig(features=16, blocks=2))


def _small_convmixer() -> ConvMixerSR:
    return ConvMixerSR(ConvMixerConfig(features=16, depth=2, kernel_size=5, head_features=8))


# Keyed by architecture name so the contract suites and the checkpoint tests that need to
# rebuild a matching instance stay in sync. Adding an architecture here opts it into every
# parametrized test below, which is the point.
_FIXTURES: dict[str, Callable[[], ResidualSuperResolution]] = {
    "unet": _small_unet,
    "edsr": _small_edsr,
    "convmixer": _small_convmixer,
}


def _models() -> list[tuple[str, ResidualSuperResolution]]:
    return [(name, build()) for name, build in _FIXTURES.items()]


# -- The frozen batch contract (D016) -------------------------------------------------


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
@pytest.mark.parametrize("low,high", REQUIRED_SHAPES)
def test_models_satisfy_the_x4_shape_contract(
    name: str, model: ResidualSuperResolution, low: int, high: int
) -> None:
    """One set of weights must serve both resolutions: the contract is resolution-generic."""
    output = model(torch.randn(2, 3, low, low))
    assert tuple(output.shape) == (2, 3, high, high), f"{name}: {tuple(output.shape)}"


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_the_same_weights_handle_both_resolutions(
    name: str, model: ResidualSuperResolution
) -> None:
    """Explicitly checks the *same instance* takes both shapes, not two instances.

    A model that hard-coded a size would pass the parametrized shape test above by being
    rebuilt each time; this cannot.
    """
    assert tuple(model(torch.randn(1, 3, 32, 32)).shape) == (1, 3, 128, 128)
    assert tuple(model(torch.randn(1, 3, 64, 64)).shape) == (1, 3, 256, 256)


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_wrong_channel_count_is_rejected(name: str, model: ResidualSuperResolution) -> None:
    """docs/VALIDATION.md negative test: an invalid input contract must fail, not coerce."""
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.randn(2, 4, 32, 32))


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_batch_dimension_is_independent(name: str, model: ResidualSuperResolution) -> None:
    """Per-sample outputs must not depend on batch composition, in eval mode.

    Eval mode is the contract, not train mode, and the distinction became real with ConvMixer
    (D023): its BatchNorm layers genuinely couple samples while training, then switch to fixed
    running statistics under `eval()` and recover exact per-sample independence. The U-Net and
    EDSR have no normalization at all and satisfy this in either mode.

    What makes the weaker contract sufficient is that every path in this repository which
    reports a number calls `model.eval()` first -- `evaluate.py:114`, `train.py:218`,
    `evaluate_fresh.py:112`, `scripts/verify_independent.py:203`, `scripts/analyze_transfer.py:80`,
    `scripts/compare_ablation.py:89`, `scripts/plot_final.py:76`, `scripts/plot_pilot.py:138,223`.
    `test_convmixer_batch_coupling_is_confined_to_training_mode` pins both halves.
    """
    model.eval()
    inputs = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        batched = model(inputs)
        individually = torch.cat([model(inputs[i : i + 1]) for i in range(3)])
    torch.testing.assert_close(batched, individually, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_output_is_unconstrained(name: str, model: ResidualSuperResolution) -> None:
    """docs/ARCHITECTURE.md: output is unconstrained during training, never clipped.

    Feeds a large-magnitude input; a clipped model would saturate.
    """
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32) * 50.0)
    assert output.abs().max() > 10.0, "output looks clamped to an image-style range"


# -- Gradients (M-04) ------------------------------------------------------------------


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_all_trainable_parameters_receive_finite_gradients(
    name: str, model: ResidualSuperResolution
) -> None:
    """docs/VALIDATION.md: finite gradients for *all* trainable parameters.

    Checking every parameter matters: a dead branch, an unused skip, or a detached path shows
    up here as a `None` gradient and nowhere else.
    """
    output = model(torch.randn(2, 3, 32, 32))
    output.pow(2).mean().backward()

    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"{name}: parameters received no gradient: {missing}"

    nonfinite = [
        n
        for n, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    assert not nonfinite, f"{name}: non-finite gradients in: {nonfinite}"

    nonzero = sum(1 for _, p in model.named_parameters() if p.grad is not None and p.grad.any())
    assert nonzero > 0, f"{name}: every gradient was exactly zero"


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_gradients_flow_at_both_resolutions(name: str, model: ResidualSuperResolution) -> None:
    for low in (32, 64):
        model.zero_grad(set_to_none=True)
        model(torch.randn(1, 3, low, low)).pow(2).mean().backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad), (
            f"{name}: missing gradients at {low}"
        )


# -- The residual formulation (D006) ---------------------------------------------------


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_output_equals_bicubic_plus_residual(name: str, model: ResidualSuperResolution) -> None:
    """D006: `y_hat = bicubic(x) + R(x)`, with the bicubic path identical to the baseline.

    Verified by reconstructing the output from its two documented parts, so the outer
    residual cannot silently drift from the M-01 baseline it is supposed to share.
    """
    model.eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        combined = model(inputs)
        baseline = bicubic_baseline(scale=4)(inputs)
        residual = model.residual(inputs)
    torch.testing.assert_close(combined, baseline + residual, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_a_zeroed_residual_reproduces_the_bicubic_baseline_exactly(
    name: str, model: ResidualSuperResolution
) -> None:
    """A model that learned nothing must match the baseline exactly.

    This is what makes the comparison fair: any reported improvement over bicubic is
    attributable to the residual and not to a different interpolation path.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), bicubic_baseline(scale=4)(inputs), rtol=0, atol=0)


# -- Baselines (M-01) ------------------------------------------------------------------


def test_baselines_are_parameter_free() -> None:
    assert count_parameters(nearest_baseline()) == 0
    assert count_parameters(bicubic_baseline()) == 0


@pytest.mark.parametrize("low,high", REQUIRED_SHAPES)
def test_baselines_produce_the_contract_shapes(low: int, high: int) -> None:
    inputs = torch.randn(2, 3, low, low)
    assert tuple(nearest_baseline()(inputs).shape) == (2, 3, high, high)
    assert tuple(bicubic_baseline()(inputs).shape) == (2, 3, high, high)


def test_interpolation_is_endpoint_aligned() -> None:
    """The property that makes bicubic exact for these grids (D011/D016).

    Corner pixel centres sit on the domain corners at both resolutions, so upsampling must
    reproduce the corner values exactly. This is the concrete consequence of
    `align_corners=True`, and it is why that setting is not a style choice.
    """
    assert ALIGN_CORNERS is True
    inputs = torch.randn(1, 3, 32, 32)
    output = upsample(inputs, 4)
    for row, column in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        torch.testing.assert_close(
            output[..., row, column], inputs[..., row, column], rtol=1e-5, atol=1e-6
        )


def test_align_corners_false_would_break_corner_alignment() -> None:
    """The negative control for the choice above.

    Without this, `align_corners=True` would be an untested assertion. With
    `align_corners=False` torch treats pixels as area samples offset by half a cell, which
    reintroduces exactly the half-cell error D011 exists to remove.
    """
    inputs = torch.randn(1, 3, 16, 16)
    misaligned = upsample(inputs, 4, align_corners=False)
    assert not torch.allclose(misaligned[..., 0, 0], inputs[..., 0, 0], rtol=1e-3, atol=1e-3), (
        "align_corners=False unexpectedly preserved the corner value"
    )


def test_bicubic_approximates_a_linear_ramp_with_a_small_measured_error() -> None:
    """Records what torch's bicubic actually does on a linear ramp.

    A first attempt at this test assumed the kernel is exact on linear data. It is not:
    measured maximum absolute error on a ramp spanning 2.0 is 6.0e-3 at 32 nodes, 3.0e-3 at
    64, and 7.3e-4 at 256, so it falls roughly as the grid spacing rather than vanishing.
    That is a property of torch's bicubic kernel, not boundary contamination -- the interior
    error scales the same way.

    Worth pinning because bicubic is the shared baseline *and* the residual base for both
    models (D006): everything reported is measured relative to a baseline with this much
    intrinsic smoothing error, which is part of why a learned residual has room to help.
    """
    for nodes, tolerance in ((32, 7e-3), (64, 4e-3), (256, 1e-3)):
        coordinates = torch.linspace(-1.0, 1.0, nodes)
        field = coordinates.view(1, 1, 1, -1).expand(1, 3, nodes, nodes).contiguous()
        upsampled = upsample(field, 4)[0, 0, 0]
        expected = torch.linspace(-1.0, 1.0, nodes * 4)
        error = (upsampled - expected).abs().max().item()
        assert error < tolerance, f"{nodes} nodes: error {error:.3e} exceeds {tolerance:.0e}"
        # Not exact, so a future change claiming exactness should fail here too.
        assert error > 0.0, f"{nodes} nodes: unexpectedly exact"


def test_nearest_and_bicubic_differ() -> None:
    """Two baselines that produced identical output would make one of them pointless."""
    inputs = torch.randn(1, 3, 16, 16)
    assert not torch.allclose(nearest_baseline()(inputs), bicubic_baseline()(inputs))


def test_interpolation_repr_records_its_settings() -> None:
    """Settings must be inspectable, since every reported number depends on them."""
    assert "align_corners=True" in repr(bicubic_baseline())
    assert "bicubic" in repr(bicubic_baseline())


def test_upsample_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match=r"\[batch, channels, height, width\]"):
        upsample(torch.randn(3, 32, 32), 4)
    with pytest.raises(ValueError, match="scale must be"):
        upsample(torch.randn(1, 3, 8, 8), 0)


def test_scale_one_is_a_passthrough() -> None:
    inputs = torch.randn(1, 3, 8, 8)
    torch.testing.assert_close(upsample(inputs, 1), inputs, rtol=0, atol=0)


# -- Checkpoints (M-04) ----------------------------------------------------------------


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_checkpoint_reload_reproduces_inference_bitwise(
    name: str, model: ResidualSuperResolution, tmp_path: Path
) -> None:
    """docs/VALIDATION.md: checkpoint reload reproduces inference output.

    Asserted bitwise, not approximately: a reload that only nearly matches means some state
    was not saved, and that would quietly invalidate every evaluation of that checkpoint.
    """
    model.eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = model(inputs)

    path = tmp_path / f"{name}.pt"
    torch.save(model.state_dict(), path)

    restored = _FIXTURES[name]()
    restored.load_state_dict(torch.load(path, weights_only=True))
    restored.eval()
    with torch.no_grad():
        after = restored(inputs)

    torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.parametrize("name,model", _models(), ids=lambda v: v if isinstance(v, str) else "")
def test_a_freshly_initialized_model_differs_before_loading(
    name: str, model: ResidualSuperResolution
) -> None:
    """Negative control for the reload test: without loading, outputs must differ.

    Otherwise the bitwise match above could be passing for a trivial reason.
    """
    model.eval()
    other = _FIXTURES[name]()
    other.eval()
    inputs = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        assert not torch.allclose(model(inputs), other(inputs))


# -- Published configurations and resource metrics (M-04) ------------------------------


def test_published_configurations_match_the_architecture_document() -> None:
    """The defaults must be the values docs/ARCHITECTURE.md fixes."""
    edsr = EDSRConfig()
    assert (edsr.blocks, edsr.features, edsr.res_scale, edsr.scale) == (16, 64, 0.1, 4)

    unet = UNetConfig()
    assert unet.stage_features == (32, 64, 128)
    assert unet.blocks_per_stage == 2
    assert unet.activation_name == "silu"
    assert unet.bias_free_ends is True

    # ConvMixer-256/16 with k=9, p=1 (D023).
    convmixer = ConvMixerConfig()
    assert (convmixer.features, convmixer.depth) == (256, 16)
    assert (convmixer.kernel_size, convmixer.patch_size) == (9, 1)
    assert convmixer.head_features == 64, "the decoder must stay EDSR-width"


def test_convmixer_sees_the_whole_basin_at_both_resolutions() -> None:
    """The property that distinguishes this arm, asserted rather than asserted-in-prose.

    Each block widens the receptive field by `k - 1`, so the published depth reaches 129 px
    with no pooling anywhere. Both required input grids are smaller than that, which is what
    lets one unit integrate the entire domain -- the U-Net needs a pyramid to do the same and
    EDSR (~33 px) cannot do it at all.
    """
    assert ConvMixerConfig().receptive_field == 129
    assert all(low <= 129 for low, _ in REQUIRED_SHAPES)


def test_bias_free_ends_on_the_unet() -> None:
    """docs/ARCHITECTURE.md specifies bias-free input and output convolutions."""
    model = build_unet()
    assert model.head.bias is None
    assert model.tail.bias is None


def test_each_architecture_uses_its_published_activation() -> None:
    """The activation differences are deliberate and part of what the comparison tests."""
    edsr_activations = {type(m) for m in build_edsr().modules()}
    unet_activations = {type(m) for m in build_unet().modules()}
    convmixer_activations = {type(m) for m in build_convmixer().modules()}
    assert nn.ReLU in edsr_activations and nn.SiLU not in edsr_activations
    assert nn.SiLU in unet_activations and nn.ReLU not in unet_activations
    # GELU as published. The paper measures it as near-irrelevant (95.88% against 95.51% for
    # ReLU on CIFAR-10), so this pins the choice rather than claiming it matters.
    assert nn.GELU in convmixer_activations and nn.ReLU not in convmixer_activations


def test_no_normalization_in_the_two_reference_architectures() -> None:
    """The U-Net and EDSR use no normalization, and that must not drift.

    Scoped to those two on purpose. EDSR's published finding is that batch normalization
    degrades super-resolution, and the U-Net follows suit, so a normalization layer appearing
    in either would be a regression. ConvMixer is deliberately excluded: it keeps BatchNorm as
    published (D023), which `test_convmixer_keeps_batchnorm_as_published` asserts positively so
    the exclusion cannot quietly become an accident.
    """
    for model in (build_unet(), build_edsr()):
        for module in model.modules():
            assert not isinstance(module, nn.BatchNorm2d | nn.InstanceNorm2d | nn.GroupNorm), (
                f"unexpected normalization layer {type(module).__name__}"
            )


def test_convmixer_keeps_batchnorm_as_published() -> None:
    """The positive control for the exclusion above (D023).

    ConvMixer's BatchNorm is a measured choice, not an oversight: the paper reports 1.44% on
    CIFAR-10 for BatchNorm over LayerNorm (Table 3). It is also the one place this project
    accepts train-mode batch coupling, so removing it silently would change what the arm means
    while leaving every other test green.

    Two per block plus one in the stem.
    """
    model = build_convmixer()
    normalizations = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(normalizations) == 2 * model.config.depth + 1, len(normalizations)


def test_the_nonorm_arm_removes_every_normalization_layer() -> None:
    """The A-03 arm (D023): ConvMixer as EDSR would have designed it."""
    model = build_convmixer(normalization="none", pointwise_residual=True, res_scale=0.1)
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    # Capacity differs by exactly BatchNorm's affine parameters and nothing else; the residual
    # and the scaling are free. 33 layers x 2 x 256 = 16,896.
    assert count_parameters(build_convmixer()) - count_parameters(model) == 16_896


def test_the_nonorm_arm_is_batch_independent_in_both_modes() -> None:
    """With no batch statistics anywhere, the train/eval distinction disappears.

    This is the property the published arm gives up and the reason the D023 caveat exists, so
    it is worth asserting that the alternative genuinely does not have it.
    """
    model = build_convmixer(
        features=16,
        depth=2,
        kernel_size=5,
        head_features=8,
        normalization="none",
        pointwise_residual=True,
        res_scale=0.1,
    )
    inputs = torch.randn(3, 3, 32, 32)
    for mode in ("train", "eval"):
        getattr(model, mode)()
        with torch.no_grad():
            batched = model(inputs)
            individually = torch.cat([model(inputs[i : i + 1]) for i in range(3)])
        torch.testing.assert_close(batched, individually, rtol=1e-5, atol=1e-6)


def test_removing_normalization_alone_would_not_train() -> None:
    """Why the A-03 arm changes three things rather than one, pinned as a measurement.

    ConvMixer's pointwise stage is not residual, so without normalization 16 such layers
    compound and gradient never reaches the early blocks. This is the evidence for that claim,
    and it is a test so that a future simplification to a "clean" one-factor ablation fails
    here instead of silently producing a dead run.
    """

    def first_to_last_gradient_ratio(**overrides: object) -> float:
        model = build_convmixer(features=64, depth=16, kernel_size=5, head_features=16, **overrides)
        model.train()
        model.zero_grad(set_to_none=True)
        ((model(torch.randn(4, 3, 32, 32)) - torch.randn(4, 3, 128, 128)) ** 2).mean().backward()
        norms = [float(b.depthwise[0].weight.grad.norm()) for b in model.body]  # type: ignore[union-attr]
        return norms[0] / norms[-1]

    published = first_to_last_gradient_ratio()
    naive = first_to_last_gradient_ratio(normalization="none")
    stabilized = first_to_last_gradient_ratio(
        normalization="none", pointwise_residual=True, res_scale=0.1
    )
    assert naive < 1e-3, f"expected vanishing gradients without normalization, got {naive:.2e}"
    assert 0.01 < published < 100.0, published
    assert 0.01 < stabilized < 100.0, stabilized


def test_convmixer_batch_coupling_is_confined_to_training_mode() -> None:
    """BatchNorm couples samples while training and stops doing so under `eval()`.

    Both halves matter. The first proves the layers are live rather than inert; the second is
    the property every reported number depends on, since evaluation always runs under `eval()`.
    """
    model = _small_convmixer()
    inputs = torch.randn(3, 3, 32, 32)

    model.train()
    with torch.no_grad():
        batched = model(inputs)
        individually = torch.cat([model(inputs[i : i + 1]) for i in range(3)])
    assert not torch.allclose(batched, individually, rtol=1e-4, atol=1e-5), (
        "train-mode ConvMixer was batch-independent, so its BatchNorm is not doing anything"
    )

    model.eval()
    with torch.no_grad():
        batched = model(inputs)
        individually = torch.cat([model(inputs[i : i + 1]) for i in range(3)])
    torch.testing.assert_close(batched, individually, rtol=1e-5, atol=1e-6)


def test_published_models_are_a_reasonable_size() -> None:
    """Resource metrics must be reported alongside accuracy (docs/ARCHITECTURE.md).

    Bounds are loose; the point is to catch an accidental order-of-magnitude change.
    """
    unet_parameters = count_parameters(build_unet())
    edsr_parameters = count_parameters(build_edsr())
    convmixer_parameters = count_parameters(build_convmixer())
    assert 1.0e6 < unet_parameters < 4.0e6, unet_parameters
    assert 1.0e6 < edsr_parameters < 4.0e6, edsr_parameters
    assert 1.0e6 < convmixer_parameters < 4.0e6, convmixer_parameters


def test_the_three_architectures_have_comparable_capacity() -> None:
    """The comparison is about inductive bias, so capacity must not be the confound.

    ConvMixer is sized to sit between the other two rather than to match either exactly; the
    exact count is pinned so a hyperparameter edit cannot move it without this failing.
    """
    assert count_parameters(build_convmixer()) == 1_720_067
    assert (
        count_parameters(build_edsr())
        < count_parameters(build_convmixer())
        < count_parameters(build_unet())
    )


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError, match="scale must be"):
        EDSRConfig(scale=3)
    with pytest.raises(ValueError, match="blocks must be"):
        EDSRConfig(blocks=0)
    with pytest.raises(ValueError, match="at least two stages"):
        UNetConfig(stage_features=(32,))
    with pytest.raises(ValueError, match="unknown activation"):
        UNetConfig(activation_name="mish")
    with pytest.raises(ValueError, match="depth must be"):
        ConvMixerConfig(depth=0)
    # An even kernel cannot be centred, and `padding="same"` would reject it later with a much
    # less informative message.
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        ConvMixerConfig(kernel_size=8)
    # The decoder composes x2 pixel-shuffle stages, so scale * patch_size must be a power of
    # two or the output would not land on the target grid.
    with pytest.raises(ValueError, match="patch_size must be a power of two"):
        ConvMixerConfig(patch_size=3)
    with pytest.raises(ValueError, match="unknown normalization"):
        ConvMixerConfig(normalization="layer")
    with pytest.raises(ValueError, match="res_scale must be"):
        ConvMixerConfig(res_scale=0.0)


def test_the_shipped_nonorm_config_differs_only_where_it_must() -> None:
    """The A-03 arm must differ from the published config in exactly three documented values.

    Guards the same property the D022 configs guard: if a stray hyperparameter drifted between
    the arms, any measured difference could be that instead of the normalization design.
    """
    import yaml

    from swe_sr.models import build_model_from_config

    root = Path(__file__).resolve().parents[2] / "configs" / "model"
    published = yaml.safe_load((root / "convmixer_x4.yaml").read_text())
    nonorm = yaml.safe_load((root / "convmixer_nonorm_x4.yaml").read_text())
    assert nonorm["architecture"] == published["architecture"]
    assert nonorm["name"] == "convmixer_nonorm", "the arm must not reuse the published name"

    expected = {"normalization": "none", "pointwise_residual": True, "res_scale": 0.1}
    for key, value in expected.items():
        assert nonorm["model"].pop(key) == value
    assert nonorm["model"] == published["model"], (
        "convmixer_nonorm_x4.yaml differs from convmixer_x4.yaml beyond the normalization design"
    )

    name, model = build_model_from_config(root / "convmixer_nonorm_x4.yaml")
    assert name == "convmixer_nonorm"
    assert count_parameters(model) == 1_703_171


def test_convmixer_rejects_a_grid_its_patches_cannot_tile() -> None:
    """Unreachable at the shipped `patch_size=1`, which is why it is worth asserting.

    Nobody should "fix" a guard that looks dead. At p=2 a 33x33 input has a remainder and must
    fail loudly rather than silently truncating a row of the basin.
    """
    assert build_convmixer().size_multiple == 1
    patched = build_convmixer(patch_size=2, features=16, depth=2)
    assert tuple(patched(torch.randn(1, 3, 32, 32)).shape) == (1, 3, 128, 128)
    with pytest.raises(ValueError, match="divisible by the patch size"):
        patched(torch.randn(1, 3, 33, 33))


def test_unet_rejects_a_size_it_cannot_pool() -> None:
    """Three stages need the upsampled grid divisible by 4.

    Note this guard is unreachable at the production scale of 4: the upsampled size is always
    `4n`, hence always divisible by 4. It exists for other scales, so the test uses `scale=1`
    to exercise it. Failing loudly beats a shape mismatch deep inside a skip concatenation.
    """
    assert build_unet().size_multiple == 4
    # At scale 4 no input can trip the guard, which is worth asserting so nobody "fixes" a
    # test that appears to be dead.
    build_unet()(torch.randn(1, 3, 5, 5))  # 5 -> 20, divisible by 4, so this must succeed

    same_grid = build_unet(scale=1)
    with pytest.raises(ValueError, match="divisible"):
        same_grid(torch.randn(1, 3, 6, 6))  # 6 % 4 != 0


def test_interpolation_module_is_scriptable_as_a_plain_module() -> None:
    """The baselines must behave like ordinary modules so evaluation can treat them uniformly."""
    baseline = bicubic_baseline()
    assert isinstance(baseline, nn.Module)
    assert list(baseline.parameters()) == []
    assert isinstance(baseline, Interpolation)


# -- The D022 direct-prediction ablation arm ------------------------------------------
#
# `outer_baseline="none"` drops the additive bicubic path so the branch output IS the
# prediction. These tests pin the two properties the ablation depends on: capacity is
# unchanged, so the comparison varies one factor; and the flag demonstrably changes the
# forward pass, so a run labelled `direct` cannot silently be the residual model.


def _direct_models() -> list[tuple[str, ResidualSuperResolution]]:
    return [
        (
            "unet_direct",
            ResidualUNet(
                UNetConfig(stage_features=(8, 16, 32), blocks_per_stage=1, outer_baseline="none")
            ),
        ),
        ("edsr_direct", EDSR(EDSRConfig(features=16, blocks=2, outer_baseline="none"))),
    ]


@pytest.mark.parametrize(
    "name,model", _direct_models(), ids=lambda v: v if isinstance(v, str) else ""
)
@pytest.mark.parametrize("low,high", REQUIRED_SHAPES)
def test_direct_models_satisfy_the_same_shape_contract(
    name: str, model: ResidualSuperResolution, low: int, high: int
) -> None:
    output = model(torch.randn(2, 3, low, low))
    assert tuple(output.shape) == (2, 3, high, high)


@pytest.mark.parametrize(
    "name,model", _direct_models(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_direct_prediction_omits_the_bicubic_path(
    name: str, model: ResidualSuperResolution
) -> None:
    """The forward pass must be the branch alone, with no baseline added."""
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), model.residual(inputs), rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,model", _direct_models(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_a_zeroed_direct_model_outputs_zero_not_bicubic(
    name: str, model: ResidualSuperResolution
) -> None:
    """The negative control for D006.

    The residual arm's defining property is that a model which learned nothing reproduces the
    bicubic baseline exactly. The direct arm must NOT have it: zeroed weights output zero, which
    scores worse than predicting the channel mean. Asserting the difference is what proves the
    two arms are genuinely different models rather than the same graph behind a flag that does
    nothing.
    """
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(inputs)
        baseline = bicubic_baseline(scale=4)(inputs)
    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    assert not torch.allclose(output, baseline)


def test_the_two_arms_have_identical_capacity() -> None:
    """One factor varies. Equal parameter counts are what make that true."""
    assert count_parameters(build_unet()) == count_parameters(build_unet(outer_baseline="none"))
    assert count_parameters(build_edsr()) == count_parameters(build_edsr(outer_baseline="none"))


@pytest.mark.parametrize(
    "name,model", _direct_models(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_direct_models_produce_finite_gradients_everywhere(
    name: str, model: ResidualSuperResolution
) -> None:
    model.zero_grad(set_to_none=True)
    model(torch.randn(2, 3, 32, 32)).pow(2).mean().backward()
    for parameter_name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"{name}.{parameter_name} received no gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name}.{parameter_name} gradient not finite"


def test_an_unknown_outer_baseline_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown outer_baseline"):
        build_edsr(outer_baseline="linear")
    with pytest.raises(ValueError, match="unknown outer_baseline"):
        build_unet(outer_baseline="linear")


def test_the_shipped_ablation_configs_are_the_published_models_minus_the_baseline() -> None:
    """The two new configs must differ from the frozen ones in exactly one value.

    Guards the whole point of the ablation: if a stray hyperparameter drifted between the arms,
    any measured difference could be that instead of the outer form.
    """
    import yaml

    from swe_sr.models import build_model_from_config

    root = Path(__file__).resolve().parents[2] / "configs" / "model"
    for arch, expected_parameters in (("unet", 1_930_208), ("edsr", 1_517_571)):
        published = yaml.safe_load((root / f"{arch}_x4.yaml").read_text())
        direct = yaml.safe_load((root / f"{arch}_direct_x4.yaml").read_text())
        assert direct["architecture"] == published["architecture"]
        assert direct["name"] == f"{arch}_direct", "the ablation must not reuse the frozen name"
        assert direct["model"].pop("outer_baseline") == "none"
        published["model"].pop("outer_baseline", None)
        assert direct["model"] == published["model"], (
            f"{arch}_direct_x4.yaml differs from {arch}_x4.yaml in more than the outer baseline"
        )
        name, model = build_model_from_config(root / f"{arch}_direct_x4.yaml")
        assert name == f"{arch}_direct"
        assert count_parameters(model) == expected_parameters
        assert model.outer_baseline == "none"
