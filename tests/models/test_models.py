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
from swe_sr.models.edsr import EDSR, EDSRConfig, build_edsr
from swe_sr.models.unet import ResidualUNet, UNetConfig, build_unet

# The two shapes docs/ARCHITECTURE.md requires. Small feature widths keep the gate fast
# while exercising every code path; the published widths are checked separately.
REQUIRED_SHAPES = [(32, 128), (64, 256)]


def _small_unet() -> ResidualUNet:
    return ResidualUNet(UNetConfig(stage_features=(8, 16, 32), blocks_per_stage=1))


def _small_edsr() -> EDSR:
    return EDSR(EDSRConfig(features=16, blocks=2))


def _models() -> list[tuple[str, ResidualSuperResolution]]:
    return [("unet", _small_unet()), ("edsr", _small_edsr())]


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
    """Per-sample outputs must not depend on batch composition.

    Guards against any accidental cross-sample coupling, which batch normalization would
    introduce and which neither architecture should have.
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

    restored = _small_unet() if name == "unet" else _small_edsr()
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
    other = _small_unet() if name == "unet" else _small_edsr()
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


def test_bias_free_ends_on_the_unet() -> None:
    """docs/ARCHITECTURE.md specifies bias-free input and output convolutions."""
    model = build_unet()
    assert model.head.bias is None
    assert model.tail.bias is None


def test_edsr_uses_relu_and_unet_uses_silu() -> None:
    """The activation difference is deliberate and part of what the comparison tests."""
    edsr_activations = {type(m) for m in build_edsr().modules()}
    unet_activations = {type(m) for m in build_unet().modules()}
    assert nn.ReLU in edsr_activations and nn.SiLU not in edsr_activations
    assert nn.SiLU in unet_activations and nn.ReLU not in unet_activations


def test_no_batch_or_instance_normalization_anywhere() -> None:
    """Neither reference architecture uses normalization layers, and batch statistics would
    couple samples and make evaluation depend on batch composition."""
    for model in (build_unet(), build_edsr()):
        for module in model.modules():
            assert not isinstance(module, nn.BatchNorm2d | nn.InstanceNorm2d | nn.GroupNorm), (
                f"unexpected normalization layer {type(module).__name__}"
            )


def test_published_models_are_a_reasonable_size() -> None:
    """Resource metrics must be reported alongside accuracy (docs/ARCHITECTURE.md).

    Bounds are loose; the point is to catch an accidental order-of-magnitude change.
    """
    unet_parameters = count_parameters(build_unet())
    edsr_parameters = count_parameters(build_edsr())
    assert 1.0e6 < unet_parameters < 4.0e6, unet_parameters
    assert 1.0e6 < edsr_parameters < 4.0e6, edsr_parameters


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError, match="scale must be"):
        EDSRConfig(scale=3)
    with pytest.raises(ValueError, match="blocks must be"):
        EDSRConfig(blocks=0)
    with pytest.raises(ValueError, match="at least two stages"):
        UNetConfig(stage_features=(32,))
    with pytest.raises(ValueError, match="unknown activation"):
        UNetConfig(activation_name="mish")


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
