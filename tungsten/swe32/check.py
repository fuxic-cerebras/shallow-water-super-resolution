#!/usr/bin/env python3
"""Compare the kernel's output against both references, and fail loudly.

Two tiers, because bit-exact agreement with `swe.py` is impossible in principle: the kernel
is float32, the script is float64, and the kernel multiplies by precomputed reciprocals
where the script divides.

  tier 1  kernel vs mirror.py  -- float32, op for op, expected EXACT
  tier 2  kernel vs swe.py     -- float64 oracle, tolerance atol/rtol

Tier 1 is the one that localizes bugs, so a failure prints a per-boundary breakdown: which
wall, which cell, and what each side thought the value was.

Run from a test-<label>/ build directory, where sim-{eta,u,v}.bin live.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from mirror import FUSED, Mirror  # noqa: E402
from swe_config import Config  # noqa: E402

FIELDS = ("eta", "u", "v")


def read_sim(name: str, n_x: int, n_y: int, directory: Path | None = None) -> np.ndarray:
    """Read a `miget` dump back into swe.py's `[x, y]` order.

    The file is y-outer/x-inner because that is what `order="y x _word"` writes, so the
    transpose here is the exact inverse of init.py's. `directory` defaults to the working
    directory, which is the build directory when the Makefile runs the gate; `trajectory.py`
    passes it explicitly rather than keeping a second copy of this axis contract.
    """
    path = (directory or Path()) / f"sim-{name}.bin"
    raw = np.fromfile(path, dtype="<f4")
    if raw.size != n_x * n_y:
        raise SystemExit(f"{path}: {raw.size} floats, expected {n_x * n_y}")
    return raw.reshape(n_y, n_x).T


def boundary_report(got: np.ndarray, want: np.ndarray, n_x: int, n_y: int) -> list[str]:
    """Localize a mismatch by region, so a wall bug is named rather than just counted."""
    regions = {
        "west  x=0": (slice(0, 1), slice(None)),
        "east  x=NX-1": (slice(n_x - 1, n_x), slice(None)),
        "south y=0": (slice(None), slice(0, 1)),
        "north y=NY-1": (slice(None), slice(n_y - 1, n_y)),
        "interior": (slice(1, n_x - 1), slice(1, n_y - 1)),
    }
    lines = []
    for label, sel in regions.items():
        diff = np.abs(got[sel].astype(np.float64) - want[sel].astype(np.float64))
        bad = int((diff > 0).sum())
        if bad:
            lines.append(f"      {label:14s} {bad:5d} cells differ, max {diff.max():.6e}")
    return lines


def signed_zero_report(got: np.ndarray, want: np.ndarray) -> str | None:
    """Distinguish a -0.0/+0.0 sign difference from a real numerical one.

    Worth separating: it is invisible to `==` but breaks a byte comparison, and it is
    exactly what multiplying a wall velocity by a zero mask would produce.
    """
    both_zero = (got == 0) & (want == 0)
    sign_differs = both_zero & (np.signbit(got) != np.signbit(want))
    count = int(sign_differs.sum())
    return f"{count} cells differ only in the sign of zero" if count else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--unfused", action="store_true", help="mirror without fused multiply-add (see `make fma`)"
    )
    parser.add_argument(
        "--tier1-tolerance",
        type=float,
        default=0.0,
        help="allow this max abs error against the mirror instead of exactness",
    )
    args = parser.parse_args()

    config = Config(n_x=args.nx, n_y=args.ny)
    sim = {name: read_sim(name, args.nx, args.ny) for name in FIELDS}

    print(f"grid {args.nx}x{args.ny}  steps {args.steps}  dx={config.dx:.6f}  dt={config.dt:.6f}")

    failures: list[str] = []

    # -- tier 1: the float32 mirror, expected exact ----------------------------------
    fused = dict.fromkeys(FUSED, False) if args.unfused else None
    mirror = dict(zip(FIELDS, Mirror(config, fused=fused).run(args.steps), strict=True))
    print(f"\ntier 1  vs mirror.py (float32, fma={'off' if args.unfused else 'on'})")
    for name in FIELDS:
        got, want = sim[name], mirror[name]
        diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
        exact_bytes = got.view(np.uint32) == want.view(np.uint32)
        status = "exact" if exact_bytes.all() else f"max|d|={diff.max():.6e}"
        print(f"  {name:4s} {status}")
        if not exact_bytes.all():
            note = signed_zero_report(got, want)
            if note:
                print(f"      {note}")
            for line in boundary_report(got, want, args.nx, args.ny):
                print(line)
            if diff.max() > args.tier1_tolerance:
                failures.append(f"tier 1 {name}: max|d|={diff.max():.3e}")

    # -- tier 2: swe.py itself, on tolerance ------------------------------------------
    from tests.solver.reference_harness import run_reference

    # swe.py's loop is `while time_step < max_time_step` starting at 1, so it performs
    # max_time_step - 1 updates.
    ref = run_reference(
        n_x=args.nx,
        n_y=args.ny,
        max_time_step=args.steps + 1,
        anim_interval=max(args.steps, 1),
        sample_interval=max(args.steps, 1),
    )
    if ref.n_steps != args.steps:
        raise SystemExit(f"oracle ran {ref.n_steps} steps, expected {args.steps}")
    if abs(ref.dt - config.dt) > 0:
        raise SystemExit(f"oracle dt {ref.dt!r} != config dt {config.dt!r}")

    print(f"\ntier 2  vs swe.py (float64, atol={args.atol:g} rtol={args.rtol:g})")
    for name, truth in zip(FIELDS, (ref.eta, ref.u, ref.v), strict=True):
        got = sim[name].astype(np.float64)
        diff = np.abs(got - truth)
        slack = diff - (args.atol + args.rtol * np.abs(truth))
        worst = np.unravel_index(int(slack.argmax()), slack.shape)
        print(
            f"  {name:4s} max|d|={diff.max():.3e}  worst cell {worst} "
            f"got {got[worst]:+.9g} want {truth[worst]:+.9g}"
        )
        if slack.max() > 0:
            failures.append(f"tier 2 {name}: exceeds tolerance at {worst}")

    # -- reference-free invariant ------------------------------------------------------
    # The scheme is in flux-divergence form with zero fluxes on every wall, and PE i's east
    # face height is the same expression on the same operands as PE i+1's west face height,
    # so the sum telescopes and sum(eta) is conserved to roundoff. A broken wall shows up
    # here without needing any reference at all.
    initial = config.fields32()["eta"].astype(np.float64).sum()
    drift = abs(sim["eta"].astype(np.float64).sum() - initial)
    scale = max(abs(initial), 1e-30)
    print(f"\nmass    sum(eta) drift {drift:.3e} ({drift / scale:.2e} relative to {initial:.6f})")
    if drift / scale > 1e-4:
        failures.append(f"mass: sum(eta) drift {drift:.3e} is too large for roundoff")

    if failures:
        print("\nFAIL")
        for line in failures:
            print(f"  {line}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
