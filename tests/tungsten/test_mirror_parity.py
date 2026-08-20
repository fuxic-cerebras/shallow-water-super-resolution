"""The Tungsten kernel's float32 mirror against the pinned reference solver.

This is the half of the cross-check that needs no Cerebras toolchain, so it belongs in the
normal suite: it pins the arithmetic the `.w` kernel implements, independently of whether a
fabric simulator is available. The fabric half lives in `tungsten/swe32/` and is driven by
`make check` there.

What this guards: `tungsten/swe32/mirror.py` is the *exact* oracle for the kernel, so if it
ever drifts from `swe.py` the kernel's tier-1 comparison becomes meaningless -- it would
still pass while both sides were wrong together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tungsten" / "swe32"))

from mirror import FUSED, Mirror  # noqa: E402
from swe_config import Config  # noqa: E402

from tests.solver.reference_harness import run_reference  # noqa: E402

# Measured envelope, not a guess: see the table in the module docstring of check.py and the
# tolerances it applies. float32 against float64 over 200 steps stays under 1e-6 absolute.
ATOL = 1e-5
RTOL = 1e-4


def _reference(nodes: int, steps: int):
    # swe.py's loop is `while time_step < max_time_step` from 1, so it does one fewer.
    result = run_reference(
        n_x=nodes,
        n_y=nodes,
        max_time_step=steps + 1,
        anim_interval=max(steps, 1),
        sample_interval=max(steps, 1),
    )
    assert result.n_steps == steps
    return result


@pytest.mark.parametrize(
    ("nodes", "steps"),
    [(8, 10), (8, 200), (16, 200), (32, 50), (32, 200)],
)
def test_mirror_tracks_reference_solver(nodes: int, steps: int) -> None:
    """The float32 mirror reproduces swe.py to within float32 roundoff."""
    config = Config(nodes, nodes)
    reference = _reference(nodes, steps)
    assert config.dt == pytest.approx(reference.dt, rel=0, abs=0)
    assert config.dx == pytest.approx(reference.dx, rel=0, abs=0)

    eta, u, v = Mirror(config).run(steps)
    for name, got, want in (
        ("eta", eta, reference.eta),
        ("u", u, reference.u),
        ("v", v, reference.v),
    ):
        np.testing.assert_allclose(
            got, want, atol=ATOL, rtol=RTOL, err_msg=f"{name} at {nodes}x{nodes}, {steps} steps"
        )


def test_grid_matches_the_reference_script_at_32_nodes() -> None:
    """The documented 32x32 constants, so a change to the config is a visible failure."""
    config = Config(32, 32)
    assert config.dx == 32258.064516129034
    assert config.dy == 32258.064516129034
    # swe.py's own CFL bound for this grid, NOT the dataset's shared 25.1398 s (D003).
    assert config.dt == 102.9920736796937


def test_walls_hold_exactly_positive_zero() -> None:
    """The east wall's u and the north wall's v are +0.0, not -0.0.

    The sign matters: -0.0 survives into the next step's `- alpha*u` wherever the other
    operand is also zero, and it breaks a byte comparison against the kernel. This is why
    the kernel masks with `un2 * mu + 0.0` rather than a bare multiply.
    """
    eta, u, v = Mirror(Config(16, 16)).run(20)
    assert not np.any(eta != eta), "eta went non-finite"
    assert (u[-1, :] == 0).all()
    assert (v[:, -1] == 0).all()
    assert not np.signbit(u[-1, :]).any(), "east wall u contains -0.0"
    assert not np.signbit(v[:, -1]).any(), "north wall v contains -0.0"


def test_mass_is_conserved_to_roundoff() -> None:
    """sum(eta) is a reference-free invariant: the scheme is in flux-divergence form with
    zero flux through every wall, so a broken wall shows up here without any oracle."""
    config = Config(32, 32)
    initial = config.fields32()["eta"].astype(np.float64).sum()
    eta, _, _ = Mirror(config).run(200)
    drift = abs(eta.astype(np.float64).sum() - initial)
    assert drift / abs(initial) < 1e-6


def test_fused_table_covers_every_arithmetic_site() -> None:
    """`FUSED` is read off the emitted assembly; a new site must be classified, not defaulted."""
    expected = {
        "pred_x",
        "pred_y",
        "cor_u1",
        "cor_u2",
        "cor_v1",
        "cor_v2",
        "fe",
        "fw",
        "fn",
        "fs",
        "div",
    }
    assert set(FUSED) == expected


@pytest.mark.parametrize("fused_on", [True, False])
def test_tolerance_holds_under_either_fusion_model(fused_on: bool) -> None:
    """Tier 2 must not depend on how the backend fuses multiply-add.

    Tier 1 does depend on it, which is why FUSED is read from the listing. Tier 2 should
    not, so the physical claim survives a compiler change.
    """
    config = Config(32, 32)
    fused = None if fused_on else dict.fromkeys(FUSED, False)
    eta, _, _ = Mirror(config, fused=fused).run(200)
    reference = _reference(32, 200)
    np.testing.assert_allclose(eta, reference.eta, atol=ATOL, rtol=RTOL)


@pytest.mark.slow
def test_full_reference_trajectory() -> None:
    """The reference script's own step count, 4999 updates at 32x32."""
    config = Config(32, 32)
    eta, u, v = Mirror(config).run(4999)
    reference = _reference(32, 4999)
    assert np.isfinite(eta).all()
    for got, want in ((eta, reference.eta), (u, reference.u), (v, reference.v)):
        np.testing.assert_allclose(got, want, atol=ATOL, rtol=RTOL)
