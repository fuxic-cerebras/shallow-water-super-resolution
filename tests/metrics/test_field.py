"""Field metrics against hand-computed examples (M-04, docs/VALIDATION.md).

`docs/VALIDATION.md` requires that "metric implementations match small hand-computed
examples", so the core cases here are worked out by hand rather than compared against another
implementation of the same formula.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from swe_sr.metrics.field import (
    CHANNEL_NAMES,
    field_metrics,
    macro_averaged_mse,
    normalized_mse_loss,
    per_channel_max_absolute_error,
    per_channel_mse,
    per_channel_relative_l2,
    per_channel_rmse,
    per_sample_relative_l2,
)


def test_mse_matches_a_hand_computed_example() -> None:
    """Constant per-channel offsets of 1, 2, 3 give MSEs of 1, 4, 9."""
    target = torch.zeros(2, 3, 4, 4)
    prediction = torch.zeros(2, 3, 4, 4)
    prediction[:, 0] = 1.0
    prediction[:, 1] = 2.0
    prediction[:, 2] = 3.0

    torch.testing.assert_close(per_channel_mse(prediction, target), torch.tensor([1.0, 4.0, 9.0]))
    torch.testing.assert_close(per_channel_rmse(prediction, target), torch.tensor([1.0, 2.0, 3.0]))
    # Macro average of 1, 4, 9 is 14/3.
    assert float(macro_averaged_mse(prediction, target)) == pytest.approx(14.0 / 3.0)


def test_relative_l2_matches_a_hand_computed_example() -> None:
    """With target all ones over 16 cells and an error of 0.5 everywhere, relL2 = 0.5."""
    target = torch.ones(1, 3, 4, 4)
    prediction = target + 0.5
    torch.testing.assert_close(
        per_channel_relative_l2(prediction, target), torch.full((3,), 0.5), rtol=1e-6, atol=1e-6
    )


def test_relative_l2_is_scale_invariant() -> None:
    """Scaling target and error together must leave relL2 unchanged; that is its purpose."""
    target = torch.randn(2, 3, 8, 8)
    prediction = target * 1.1
    baseline = per_channel_relative_l2(prediction, target)
    scaled = per_channel_relative_l2(prediction * 1000, target * 1000)
    torch.testing.assert_close(baseline, scaled, rtol=1e-5, atol=1e-7)


def test_max_absolute_error_finds_a_single_outlier() -> None:
    """RMSE hides a lone spike; the max-abs metric exists to surface it."""
    target = torch.zeros(1, 3, 8, 8)
    prediction = torch.zeros(1, 3, 8, 8)
    prediction[0, 1, 3, 5] = -7.5
    worst = per_channel_max_absolute_error(prediction, target)
    torch.testing.assert_close(worst, torch.tensor([0.0, 7.5, 0.0]))


def test_a_perfect_prediction_scores_zero() -> None:
    target = torch.randn(3, 3, 8, 8)
    assert float(macro_averaged_mse(target, target)) == 0.0
    assert float(per_channel_relative_l2(target, target).max()) == pytest.approx(0.0, abs=1e-12)
    assert float(per_channel_max_absolute_error(target, target).max()) == 0.0


def test_zero_target_does_not_divide_by_zero() -> None:
    """The 1e-12 guard in the relL2 denominator, on the case that needs it."""
    target = torch.zeros(1, 3, 4, 4)
    prediction = torch.ones(1, 3, 4, 4)
    result = per_channel_relative_l2(prediction, target)
    assert torch.isfinite(result).all()


def test_metrics_are_reported_per_channel_never_flattened() -> None:
    """docs/VALIDATION.md forbids flattening raw physical channels into one scalar.

    A field whose channels differ by orders of magnitude must produce distinct per-channel
    numbers; a single flattened scalar would be dominated by `eta`.
    """
    target = torch.zeros(2, 3, 4, 4)
    prediction = torch.zeros(2, 3, 4, 4)
    prediction[:, 0] = 10.0  # metres
    prediction[:, 1] = 0.01  # metres per second
    prediction[:, 2] = 0.02

    metrics = field_metrics(prediction, target)
    assert metrics["rmse_eta"] == pytest.approx(10.0)
    assert metrics["rmse_u"] == pytest.approx(0.01)
    assert metrics["rmse_v"] == pytest.approx(0.02)
    # Every channel appears by name, so a number cannot be misattributed.
    for name in CHANNEL_NAMES:
        assert f"mse_{name}" in metrics
        assert f"rel_l2_{name}" in metrics
        assert f"max_abs_{name}" in metrics


def test_per_sample_relative_l2_supports_median_and_percentile() -> None:
    """docs/VALIDATION.md requires sample median and 95th-percentile relL2.

    Those are distributional and cannot be recovered from a batch-level aggregate, so the
    per-sample form must be genuinely per sample.
    """
    target = torch.ones(4, 3, 4, 4)
    prediction = target.clone()
    # Give each sample a different error magnitude.
    for sample, offset in enumerate((0.1, 0.2, 0.3, 0.4)):
        prediction[sample] += offset

    per_sample = per_sample_relative_l2(prediction, target)
    assert per_sample.shape == (4, 3)
    torch.testing.assert_close(
        per_sample[:, 0], torch.tensor([0.1, 0.2, 0.3, 0.4]), rtol=1e-5, atol=1e-6
    )
    metrics = field_metrics(prediction, target)
    assert metrics["rel_l2_median_eta"] == pytest.approx(0.25, abs=1e-3)
    assert metrics["rel_l2_p95_eta"] == pytest.approx(0.4, abs=0.02)


def test_batch_relative_l2_is_not_the_mean_of_per_sample_ratios() -> None:
    """These are different quantities, and conflating them is an easy reporting error.

    A per-sample mean is dominated by samples with small target norms; the norm ratio is not.
    """
    target = torch.ones(2, 3, 4, 4)
    target[0] *= 0.001  # one sample with a tiny norm
    prediction = target + 0.1

    batch_level = per_channel_relative_l2(prediction, target)[0]
    per_sample_mean = per_sample_relative_l2(prediction, target)[:, 0].mean()
    assert not torch.isclose(batch_level, per_sample_mean, rtol=0.05)


def test_loss_is_the_macro_averaged_normalized_mse() -> None:
    """D005: equal-weight normalized channel MSE is the v1 objective."""
    prediction = torch.randn(2, 3, 8, 8)
    target = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(
        normalized_mse_loss(prediction, target), macro_averaged_mse(prediction, target)
    )


def test_equal_weight_means_a_channel_cannot_dominate_by_scale() -> None:
    """The reason the loss is defined on normalized fields (D005).

    Equal weighting only produces a balanced objective once channels share a scale; on raw
    fields the metric would be dominated by the largest-magnitude channel. Demonstrated by
    showing an eta-only error and a u-only error of the same *normalized* size contribute
    equally.
    """
    target = torch.zeros(1, 3, 4, 4)
    eta_error = torch.zeros(1, 3, 4, 4)
    eta_error[:, 0] = 1.0
    u_error = torch.zeros(1, 3, 4, 4)
    u_error[:, 1] = 1.0
    assert float(macro_averaged_mse(eta_error, target)) == pytest.approx(
        float(macro_averaged_mse(u_error, target))
    )


def test_loss_gradient_flows_to_the_prediction() -> None:
    prediction = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randn(2, 3, 8, 8)
    normalized_mse_loss(prediction, target).backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_metrics_agree_with_numpy_on_random_data() -> None:
    """An independent numpy computation, as a cross-check on the torch reductions."""
    rng = np.random.default_rng(4)
    prediction_np = rng.normal(size=(3, 3, 5, 5))
    target_np = rng.normal(size=(3, 3, 5, 5))
    prediction = torch.from_numpy(prediction_np)
    target = torch.from_numpy(target_np)

    expected_mse = ((prediction_np - target_np) ** 2).mean(axis=(0, 2, 3))
    torch.testing.assert_close(
        per_channel_mse(prediction, target), torch.from_numpy(expected_mse), rtol=1e-12, atol=1e-14
    )

    difference = np.sqrt(((prediction_np - target_np) ** 2).sum(axis=(0, 2, 3)))
    magnitude = np.sqrt((target_np**2).sum(axis=(0, 2, 3)))
    torch.testing.assert_close(
        per_channel_relative_l2(prediction, target),
        torch.from_numpy(difference / (magnitude + 1e-12)),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "prediction_shape,target_shape",
    [((2, 3, 4, 4), (2, 3, 8, 8)), ((2, 3, 4, 4), (3, 3, 4, 4))],
)
def test_shape_mismatch_is_rejected(
    prediction_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        per_channel_mse(torch.zeros(prediction_shape), torch.zeros(target_shape))


def test_wrong_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[batch, channel, y, x\]"):
        per_channel_mse(torch.zeros(3, 4, 4), torch.zeros(3, 4, 4))
