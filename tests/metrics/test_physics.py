"""Physical diagnostics (E-series reporting, docs/VALIDATION.md).

These are diagnostics, never optimization targets (D005). The tests check they measure the
right physics and, importantly, that they *fire* on the failure modes they exist to catch --
a diagnostic that stays quiet when the basin dries is worse than none.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from swe_sr.metrics.physics import (
    PhysicalParameters,
    gradient_rmse,
    minimum_depth,
    negative_depth_count,
    physics_metrics,
    relative_energy_error,
    relative_mass_error,
    total_energy,
    total_mass,
    wall_normal_velocity_error,
)

PARAMETERS = PhysicalParameters(depth=100.0, gravity=9.81, dx=7874.0, dy=7874.0)


def test_total_mass_matches_a_hand_computed_example() -> None:
    """Mass is the area-weighted elevation integral (D014)."""
    fields = torch.zeros(1, 3, 4, 4)
    fields[0, 0] = 0.5
    expected = 0.5 * 16 * PARAMETERS.dx * PARAMETERS.dy
    assert float(total_mass(fields, PARAMETERS)[0]) == pytest.approx(expected)


def test_total_energy_on_a_state_at_rest_is_pure_potential() -> None:
    fields = torch.zeros(1, 3, 4, 4)
    fields[0, 0] = 0.25
    expected = 0.5 * PARAMETERS.gravity * 0.25**2 * 16 * PARAMETERS.dx * PARAMETERS.dy
    assert float(total_energy(fields, PARAMETERS)[0]) == pytest.approx(expected)


def test_motion_adds_kinetic_energy() -> None:
    eta = torch.zeros(1, 3, 8, 8)
    moving = eta.clone()
    moving[0, 1] = 0.5
    assert float(total_energy(moving, PARAMETERS)[0]) > float(total_energy(eta, PARAMETERS)[0])


def test_a_perfect_prediction_has_zero_physical_error() -> None:
    target = torch.randn(2, 3, 16, 16) * 0.2
    assert float(relative_mass_error(target, target, PARAMETERS).max()) == 0.0
    assert float(relative_energy_error(target, target, PARAMETERS).max()) == 0.0
    assert float(gradient_rmse(target, target).max()) == 0.0
    assert float(max(v.max() for v in wall_normal_velocity_error(target, target).values())) == 0.0


def test_mass_error_survives_a_nearly_mass_neutral_state() -> None:
    """Bumps that cancel give a near-zero total mass, where a bare ratio would blow up."""
    target = torch.zeros(1, 3, 8, 8)
    target[0, 0, :4] = 1.0
    target[0, 0, 4:] = -1.0  # sums to zero
    prediction = target + 0.01
    error = relative_mass_error(prediction, target, PARAMETERS)
    assert torch.isfinite(error).all()


def test_negative_depth_is_counted_not_hidden() -> None:
    """The model output is unconstrained by design, so a dried basin must be visible.

    An RMSE would absorb a single catastrophic cell; this is how it reaches a report.
    """
    fields = torch.zeros(1, 3, 8, 8)
    assert int(negative_depth_count(fields, PARAMETERS)[0]) == 0

    fields[0, 0, 2, 3] = -PARAMETERS.depth - 1.0
    assert int(negative_depth_count(fields, PARAMETERS)[0]) == 1
    assert float(minimum_depth(fields, PARAMETERS)[0]) < 0.0


def test_wall_error_is_reported_for_all_four_walls() -> None:
    """D012: on processed cell-centered fields these compare predicted against true.

    They are not asserted to be zero, because destaggering makes the wall-adjacent value half
    the adjacent face value; the exact-zero check belongs on the raw arrays.
    """
    target = torch.zeros(2, 3, 8, 8)
    prediction = target.clone()
    prediction[:, 1, :, -1] = 0.3  # east wall, u
    prediction[:, 2, 0, :] = 0.2  # south wall, v

    walls = wall_normal_velocity_error(prediction, target)
    assert set(walls) == {"west", "east", "south", "north"}
    assert float(walls["east"].mean()) == pytest.approx(0.3)
    assert float(walls["south"].mean()) == pytest.approx(0.2)
    assert float(walls["west"].mean()) == 0.0
    assert float(walls["north"].mean()) == 0.0


def test_wall_errors_distinguish_u_from_v() -> None:
    """A u/v mix-up would attribute a wall error to the wrong pair of walls."""
    target = torch.zeros(1, 3, 8, 8)
    u_only = target.clone()
    u_only[0, 1] = 0.5  # u everywhere: affects east and west, not north and south
    walls = wall_normal_velocity_error(u_only, target)
    assert float(walls["east"].mean()) == pytest.approx(0.5)
    assert float(walls["west"].mean()) == pytest.approx(0.5)
    assert float(walls["north"].mean()) == 0.0
    assert float(walls["south"].mean()) == 0.0


def test_gradient_rmse_is_relatively_more_sensitive_to_over_smoothing() -> None:
    """The diagnostic most sensitive to the failure mode an MSE-trained model is prone to.

    Compared *relative to each measure's own scale*, not against each other: a gradient is a
    per-cell difference while a pixel value is a level, so comparing their absolute magnitudes
    is meaningless. An earlier version of this test did exactly that and asserted the wrong
    inequality.

    The real claim is that smoothing degrades gradients by a larger fraction than it degrades
    values, which is why the diagnostic is worth reporting alongside RMSE.
    """
    coordinates = torch.linspace(0, 8 * torch.pi, 32)
    target = torch.sin(coordinates).view(1, 1, 1, -1).expand(1, 3, 32, 32).contiguous()
    smoothed = torch.nn.functional.avg_pool2d(target, 4)
    smoothed = torch.nn.functional.interpolate(smoothed, size=(32, 32), mode="bilinear")

    pixel_scale = float((target**2).mean().sqrt())
    pixel_relative = float(((smoothed - target) ** 2).mean().sqrt()) / pixel_scale

    gradient_scale = float(gradient_rmse(torch.zeros_like(target), target).mean())
    gradient_relative = float(gradient_rmse(smoothed, target).mean()) / gradient_scale

    assert gradient_relative > pixel_relative, (
        f"gradients degraded {gradient_relative:.3f} vs values {pixel_relative:.3f}; "
        "smoothing should hurt gradients relatively more"
    )


def test_metrics_carry_their_units_in_the_key_names() -> None:
    """CLAUDE.md forbids reporting a metric without its units."""
    target = torch.randn(2, 3, 16, 16) * 0.2
    metrics = physics_metrics(target + 0.01, target, PARAMETERS)
    assert "min_predicted_depth_m" in metrics
    assert "wall_velocity_error_max_m_per_s" in metrics
    assert "wall_east_velocity_error_m_per_s" in metrics
    # Dimensionless ratios are named as relative, so they cannot be read as absolute.
    assert "relative_mass_error" in metrics
    assert "relative_energy_error" in metrics
    for channel in ("eta", "u", "v"):
        assert f"gradient_rmse_{channel}" in metrics


def test_parameters_come_from_the_manifest_not_a_default() -> None:
    """Assuming H and g would silently produce wrong physics for any other configuration."""

    class _Manifest:
        fine_config: ClassVar[dict[str, float]] = {
            "depth": 250.0,
            "gravity": 9.5,
            "dx": 100.0,
            "dy": 200.0,
        }

    parameters = PhysicalParameters.from_manifest(_Manifest())
    assert parameters.depth == 250.0
    assert parameters.gravity == 9.5
    assert parameters.dx == 100.0
    assert parameters.dy == 200.0


def test_mass_scales_with_cell_area() -> None:
    """A dx/dy mix-up on a non-square cell would go unnoticed without this."""
    fields = torch.ones(1, 3, 4, 4)
    square = PhysicalParameters(dx=10.0, dy=10.0)
    oblong = PhysicalParameters(dx=10.0, dy=20.0)
    assert float(total_mass(fields, oblong)[0]) == pytest.approx(
        2.0 * float(total_mass(fields, square)[0])
    )
