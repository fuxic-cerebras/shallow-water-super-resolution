"""Physical diagnostics on de-normalized predictions (E-series reporting).

`docs/VALIDATION.md` is explicit that these are "diagnostics, not optimization claims": the
version 1 objective is MSE only (D005), and nothing here is ever optimized against. They exist
to answer research question 4 -- whether lower pixel MSE also improves mass, energy, and
boundary errors -- which a pixel metric alone cannot address.

Every quantity is computed **after de-normalization**, in SI units, per `docs/ARCHITECTURE.md`.
Conventions follow D014, and inputs are the cell-centered processed fields (D011).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Defaults match the solver's physical parameters; a caller should pass the manifest's values.
DEFAULT_DEPTH = 100.0
DEFAULT_GRAVITY = 9.81


@dataclass(frozen=True)
class PhysicalParameters:
    """The parameters a diagnostic needs, taken from the dataset manifest rather than assumed."""

    depth: float = DEFAULT_DEPTH
    gravity: float = DEFAULT_GRAVITY
    dx: float = 1.0
    dy: float = 1.0

    @classmethod
    def from_manifest(cls, manifest: Any) -> PhysicalParameters:
        fine = manifest.fine_config
        return cls(
            depth=float(fine["depth"]),
            gravity=float(fine["gravity"]),
            dx=float(fine["dx"]),
            dy=float(fine["dy"]),
        )


def total_mass(fields: torch.Tensor, parameters: PhysicalParameters) -> torch.Tensor:
    """Domain-integrated elevation anomaly per sample (D014), shaped `[batch]`."""
    return fields[:, 0].sum(dim=(-2, -1)) * parameters.dx * parameters.dy


def total_energy(fields: torch.Tensor, parameters: PhysicalParameters) -> torch.Tensor:
    """Domain-integrated energy per sample under the D014 convention, shaped `[batch]`."""
    eta, u, v = fields[:, 0], fields[:, 1], fields[:, 2]
    kinetic = 0.5 * (parameters.depth + eta) * (u**2 + v**2)
    potential = 0.5 * parameters.gravity * eta**2
    return (kinetic + potential).sum(dim=(-2, -1)) * parameters.dx * parameters.dy


def relative_mass_error(
    prediction: torch.Tensor, target: torch.Tensor, parameters: PhysicalParameters
) -> torch.Tensor:
    """Relative error in total mass, per sample.

    Normalized by the target's own magnitude with a floor, because a nearly mass-neutral state
    (bumps that cancel) would otherwise produce a meaningless ratio.
    """
    predicted = total_mass(prediction, parameters)
    actual = total_mass(target, parameters)
    scale = actual.abs().clamp_min(1e-8)
    return (predicted - actual).abs() / scale


def relative_energy_error(
    prediction: torch.Tensor, target: torch.Tensor, parameters: PhysicalParameters
) -> torch.Tensor:
    predicted = total_energy(prediction, parameters)
    actual = total_energy(target, parameters)
    return (predicted - actual).abs() / actual.abs().clamp_min(1e-8)


def wall_normal_velocity_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Wall-normal velocity error at each of the four walls, per sample.

    D012 governs the interpretation. On the *processed* cell-centered fields these are not
    identically zero -- destaggering makes the wall-adjacent value half the adjacent face value --
    so this compares predicted against true rather than asserting a hard zero. The exact-zero
    check belongs on the raw staggered arrays and lives in the data validator.

    In storage order `[batch, channel, y, x]`: u is normal to the east and west walls (first and
    last column), v to the north and south (first and last row).
    """
    return {
        "west": (prediction[:, 1, :, 0] - target[:, 1, :, 0]).abs().amax(dim=-1),
        "east": (prediction[:, 1, :, -1] - target[:, 1, :, -1]).abs().amax(dim=-1),
        "south": (prediction[:, 2, 0, :] - target[:, 2, 0, :]).abs().amax(dim=-1),
        "north": (prediction[:, 2, -1, :] - target[:, 2, -1, :]).abs().amax(dim=-1),
    }


def negative_depth_count(fields: torch.Tensor, parameters: PhysicalParameters) -> torch.Tensor:
    """Cells where the predicted total depth is non-positive, per sample.

    The model output is unconstrained by design (`docs/ARCHITECTURE.md`), so nothing prevents a
    prediction from drying the basin. Counting it is how that shows up in a report rather than
    being hidden by an RMSE.
    """
    total_depth = parameters.depth + fields[:, 0]
    return (total_depth <= 0.0).sum(dim=(-2, -1))


def minimum_depth(fields: torch.Tensor, parameters: PhysicalParameters) -> torch.Tensor:
    return (parameters.depth + fields[:, 0]).amin(dim=(-2, -1))


def gradient_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-channel RMSE of first differences, shaped `[channel]`.

    An optional diagnostic in `docs/VALIDATION.md`, and the one most sensitive to over-smoothing:
    a blurred prediction can score well on pixel RMSE while losing the gradients that carry the
    fine structure the model is meant to recover.
    """
    prediction_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_x = target[..., :, 1:] - target[..., :, :-1]
    prediction_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_y = target[..., 1:, :] - target[..., :-1, :]
    error_x = ((prediction_x - target_x) ** 2).mean(dim=(0, 2, 3))
    error_y = ((prediction_y - target_y) ** 2).mean(dim=(0, 2, 3))
    return ((error_x + error_y) / 2.0).sqrt()


def physics_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    parameters: PhysicalParameters,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Every physical diagnostic, batch-averaged, in SI units.

    Keys name the quantity and its units where they are not dimensionless, so a reported number
    cannot be misread -- `CLAUDE.md` forbids a metric without its units.
    """
    walls = wall_normal_velocity_error(prediction, target)
    gradients = gradient_rmse(prediction, target)
    metrics = {
        f"{prefix}relative_mass_error": float(
            relative_mass_error(prediction, target, parameters).mean()
        ),
        f"{prefix}relative_energy_error": float(
            relative_energy_error(prediction, target, parameters).mean()
        ),
        f"{prefix}negative_depth_cells": float(
            negative_depth_count(prediction, parameters).float().mean()
        ),
        f"{prefix}min_predicted_depth_m": float(minimum_depth(prediction, parameters).min()),
        f"{prefix}wall_velocity_error_max_m_per_s": float(torch.stack(list(walls.values())).max()),
    }
    for wall, values in walls.items():
        metrics[f"{prefix}wall_{wall}_velocity_error_m_per_s"] = float(values.mean())
    for index, name in enumerate(("eta", "u", "v")):
        metrics[f"{prefix}gradient_rmse_{name}"] = float(gradients[index])
    return metrics
