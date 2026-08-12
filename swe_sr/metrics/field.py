"""Per-channel field metrics (M-04 support, E-series reporting).

`docs/VALIDATION.md` fixes both the definitions and the aggregation. Two rules there are easy
to violate by accident and are enforced by the API here:

- **Never flatten raw physical channels into one scalar**, because their units and magnitudes
  differ: `eta` is metres, `u` and `v` are metres per second. Every function returns
  per-channel values, and the only scalar is an explicitly macro-averaged *normalized* one.
- **Aggregate per snapshot, then within trajectory, then equally across trajectories**, so a
  trajectory with more frames cannot dominate. That lives in `aggregate.py`.

All functions take `[batch, channel, y, x]` and reduce over batch and space, returning one
value per channel.
"""

from __future__ import annotations

import torch

CHANNEL_NAMES = ("eta", "u", "v")
# docs/VALIDATION.md: relL2 denominator carries a 1e-12 guard.
RELATIVE_L2_EPSILON = 1e-12


def _check(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
    if prediction.ndim != 4:
        raise ValueError(f"expected [batch, channel, y, x], got {tuple(prediction.shape)}")


def per_channel_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error per channel, reduced over batch and space."""
    _check(prediction, target)
    return ((prediction - target) ** 2).mean(dim=(0, 2, 3))


def per_channel_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error per channel. In physical units when inputs are de-normalized."""
    return per_channel_mse(prediction, target).sqrt()


def per_channel_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 error per channel: `||pred - target|| / (||target|| + eps)`.

    Computed over the whole batch rather than per sample and averaged: `docs/VALIDATION.md`
    defines relL2 as a norm ratio, and averaging per-sample ratios is a different quantity
    that would be dominated by samples with small target norms.
    """
    _check(prediction, target)
    difference = ((prediction - target) ** 2).sum(dim=(0, 2, 3)).sqrt()
    magnitude = (target**2).sum(dim=(0, 2, 3)).sqrt()
    return difference / (magnitude + RELATIVE_L2_EPSILON)


def per_channel_max_absolute_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Worst-case error per channel, which an RMSE can hide entirely."""
    _check(prediction, target)
    return (prediction - target).abs().amax(dim=(0, 2, 3))


def per_sample_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 per sample and channel, shaped `[batch, channel]`.

    Needed for the median and 95th-percentile relative L2 that `docs/VALIDATION.md` requires,
    which are distributional and cannot be recovered from a batch-level aggregate.
    """
    _check(prediction, target)
    difference = ((prediction - target) ** 2).sum(dim=(2, 3)).sqrt()
    magnitude = (target**2).sum(dim=(2, 3)).sqrt()
    return difference / (magnitude + RELATIVE_L2_EPSILON)


def macro_averaged_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equal-weight mean of the per-channel MSEs -- the version 1 training objective (D005).

    Only meaningful on *normalized* fields. Applied to raw physical channels it would silently
    let whichever channel has the largest numerical scale dominate, which is exactly what
    `docs/VALIDATION.md` forbids and what channel normalization exists to prevent.
    """
    return per_channel_mse(prediction, target).mean()


def normalized_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The training loss: equal-weight normalized channel MSE (docs/EXPERIMENT_PLAN.md).

    A thin alias for `macro_averaged_mse`, kept separate so the loss used for optimization is
    named distinctly from the metric used for reporting even though they coincide in v1.
    """
    return macro_averaged_mse(prediction, target)


def field_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Every per-channel field metric `docs/VALIDATION.md` requires, as a flat dict.

    Keys carry the channel name, never a bare index, so a reported number cannot be
    misattributed to the wrong physical field.
    """
    mse = per_channel_mse(prediction, target)
    rmse = per_channel_rmse(prediction, target)
    relative = per_channel_relative_l2(prediction, target)
    worst = per_channel_max_absolute_error(prediction, target)
    per_sample = per_sample_relative_l2(prediction, target)

    metrics: dict[str, float] = {f"{prefix}mse_macro": float(mse.mean())}
    for index, name in enumerate(CHANNEL_NAMES):
        metrics[f"{prefix}mse_{name}"] = float(mse[index])
        metrics[f"{prefix}rmse_{name}"] = float(rmse[index])
        metrics[f"{prefix}rel_l2_{name}"] = float(relative[index])
        metrics[f"{prefix}max_abs_{name}"] = float(worst[index])
        # Both the median and the 95th percentile go through `quantile`, deliberately.
        # `torch.median` returns the *lower* of the two middle values for an even sample
        # count, unlike `numpy.median` and unlike `torch.quantile(0.5)`, which interpolate.
        # Mixing the two would make the reported median and p95 follow different conventions.
        samples = per_sample[:, index].double()
        metrics[f"{prefix}rel_l2_median_{name}"] = float(torch.quantile(samples, 0.5))
        metrics[f"{prefix}rel_l2_p95_{name}"] = float(torch.quantile(samples, 0.95))
    return metrics
