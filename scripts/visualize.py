"""Render solver trajectories as animations, using the reference repo's own viz_tools.

    python scripts/visualize.py --seed 0 --all

`docs/ARCHITECTURE.md` casts `viz_tools.py`, GIF generation, and `plt.show()` as optional
*clients* of the solver, never dependencies of it. This script is one such client: it
imports matplotlib, so it deliberately lives outside `swe_sr/` and nothing in the
generation or training path imports it.

Two things it is useful for:

- Confirming visually that a full-length trajectory is physically sensible -- waves
  radiating from the bumps, reflecting off the closed walls, and turning under rotation.
- Comparing the coarse and fine members of a pair side by side. They are independent
  solves of the same analytic initial condition (D002), so they should agree on the
  large-scale wave pattern while the coarse one visibly loses fine structure. That
  difference is the signal the super-resolution model is being asked to learn.

Arrays are transposed back from storage order `[y, x]` to the reference solver's `[x, y]`
before being handed to `viz_tools`, which expects the latter.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Must be set before pyplot is imported anywhere: these nodes are headless.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "references" / "shallow-water"

if not (REFERENCE_DIR / "viz_tools.py").is_file():
    raise SystemExit(
        f"{REFERENCE_DIR}/viz_tools.py is missing. It lives in the pinned solver "
        "submodule (D010); run `git submodule update --init`."
    )
sys.path.insert(0, str(REFERENCE_DIR))

import matplotlib.pyplot as plt  # noqa: E402
import viz_tools  # noqa: E402

from swe_sr.data.registry import build_registry  # noqa: E402
from swe_sr.solver.config import SolverConfig  # noqa: E402
from swe_sr.solver.diagnostics import destagger_u, destagger_v  # noqa: E402
from swe_sr.solver.runner import solve  # noqa: E402

# Full primary-pair trajectory: 288 discarded, then every 24th step to 3336 (docs/DATASET.md).
FULL_LAST_STEP = 3336


def _run(nodes: int, seed: int, stride: int, shared_dt: float | None) -> tuple:
    """Integrate one full-length trajectory, sampling densely enough to animate."""
    config = SolverConfig(n_x=nodes, n_y=nodes, dt_override=shared_dt)
    entry = build_registry().by_seed(seed)
    steps = np.arange(0, FULL_LAST_STEP + 1, stride, dtype=np.int64)

    started = time.perf_counter()
    result = solve(config, entry.initial_condition, sample_steps=steps, diagnostic_stride=8)
    elapsed = time.perf_counter() - started

    print(
        f"  {nodes:>3}x{nodes:<3} {FULL_LAST_STEP} steps, {len(steps)} frames "
        f"in {elapsed:.2f} s  |  dt={config.dt:.4f}s  "
        f"T={FULL_LAST_STEP * config.dt / 3600:.2f} h  "
        f"mass_drift={result.diagnostics.relative_mass_drift:.2e}  "
        f"h_min={result.diagnostics.min_total_depth:.3f} m"
    )
    return result, config, elapsed


def _to_reference_order(fields: np.ndarray, channel: int) -> list[np.ndarray]:
    """Storage `[time, channel, y, x]` -> a list of `[x, y]` frames, as viz_tools expects."""
    return [np.asarray(frame, dtype=float).T for frame in fields[:, channel]]


def render_reference_animations(
    result, config: SolverConfig, out_dir: Path, label: str, stride: int
) -> list[Path]:
    """Render the same animation styles the reference README shows."""
    mesh_x, mesh_y = config.mesh
    frame_interval = stride * config.dt
    eta_frames = _to_reference_order(result.fields, 0)
    written: list[Path] = []

    # viz_tools writes "<filename>.<filetype>" relative to the working directory.
    previous = Path.cwd()
    os.chdir(out_dir)
    try:
        viz_tools.eta_animation(mesh_x, mesh_y, eta_frames, frame_interval, f"eta_{label}", "gif")
        written.append(out_dir / f"eta_{label}.gif")
        plt.close("all")

        viz_tools.eta_animation3D(
            mesh_x, mesh_y, eta_frames, frame_interval, f"surface_{label}", "gif"
        )
        written.append(out_dir / f"surface_{label}.gif")
        plt.close("all")

        # Velocities are destaggered first (D011): viz_tools draws colocated arrows, and
        # the raw fields live on cell faces, so quivers would be offset by half a cell.
        u_frames = [destagger_u(np.asarray(f, dtype=float).T) for f in result.fields[:, 1]]
        v_frames = [destagger_v(np.asarray(f, dtype=float).T) for f in result.fields[:, 2]]
        viz_tools.velocity_animation(
            mesh_x, mesh_y, u_frames, v_frames, frame_interval, f"velocity_{label}", "gif"
        )
        written.append(out_dir / f"velocity_{label}.gif")
        plt.close("all")
    finally:
        os.chdir(previous)

    return written


def render_pair_comparison(
    coarse_result,
    fine_result,
    coarse_config: SolverConfig,
    fine_config: SolverConfig,
    out_dir: Path,
    seed: int,
    stride: int,
) -> Path:
    """Side-by-side coarse / fine / difference animation for one pair.

    The difference panel is the point: it shows what an independent coarse solve loses
    relative to the fine solve at the same physical time, which is exactly the target the
    super-resolution model must reconstruct.
    """
    import matplotlib.animation as animation

    coarse_eta = coarse_result.fields[:, 0].astype(float)
    fine_eta = fine_result.fields[:, 0].astype(float)
    assert np.array_equal(coarse_result.times, fine_result.times), "paired times must match"

    # Shared symmetric colour limits so the two panels are directly comparable; a
    # per-panel autoscale would make the coarse field look artificially similar.
    limit = float(np.abs(fine_eta).max())

    # Compare on the coarse grid via endpoint-aligned nearest sampling of the fine field.
    # Deliberately not bicubic: this is a sanity view, and any interpolation here would
    # blur the very structure the difference panel is meant to expose.
    index_x = np.round(np.linspace(0, fine_config.n_x - 1, coarse_config.n_x)).astype(int)
    index_y = np.round(np.linspace(0, fine_config.n_y - 1, coarse_config.n_y)).astype(int)
    fine_on_coarse = fine_eta[:, index_y[:, None], index_x[None, :]]
    difference = coarse_eta - fine_on_coarse
    difference_limit = float(np.abs(difference).max()) or 1.0

    extent = [
        fine_config.x[0] / 1000,
        fine_config.x[-1] / 1000,
        fine_config.y[0] / 1000,
        fine_config.y[-1] / 1000,
    ]
    # constrained_layout, and a y label only on the leftmost panel: with three panels each
    # carrying a colorbar, per-panel y labels collide with the neighbouring colorbar.
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor="white", layout="constrained")
    images = []
    for index, (axis, (title, data, vlimit, cmap)) in enumerate(
        zip(
            axes,
            (
                (f"coarse {coarse_config.n_x}x{coarse_config.n_y}", coarse_eta, limit, "RdBu_r"),
                (f"fine {fine_config.n_x}x{fine_config.n_y}", fine_eta, limit, "RdBu_r"),
                ("coarse - fine (on coarse grid)", difference, difference_limit, "PuOr_r"),
            ),
            strict=True,
        )
    ):
        image = axis.imshow(
            data[0],
            origin="lower",
            extent=extent,
            vmin=-vlimit,
            vmax=vlimit,
            cmap=cmap,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("x [km]")
        if index == 0:
            axis.set_ylabel("y [km]")
        else:
            axis.set_yticklabels([])
        figure.colorbar(image, ax=axis, fraction=0.046, label="eta [m]")
        images.append(image)

    suptitle = figure.suptitle("", fontsize=14)

    def update(frame: int):
        images[0].set_data(coarse_eta[frame])
        images[1].set_data(fine_eta[frame])
        images[2].set_data(difference[frame])
        hours = fine_result.times[frame] / 3600
        rms = float(np.sqrt((difference[frame] ** 2).mean()))
        suptitle.set_text(f"seed {seed}   t = {hours:6.2f} h   |coarse - fine| RMS = {rms:.4f} m")
        return [*images, suptitle]

    anim = animation.FuncAnimation(
        figure, update, frames=coarse_eta.shape[0], interval=80, blit=False
    )
    path = out_dir / f"pair_compare_seed{seed}.gif"
    anim.save(str(path), writer=animation.PillowWriter(fps=12))
    plt.close(figure)

    print(
        f"  difference RMS over trajectory: mean {np.sqrt((difference**2).mean()):.4f} m, "
        f"peak {np.abs(difference).max():.4f} m"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="registry seed to render")
    parser.add_argument(
        "--stride",
        type=int,
        default=24,
        help="steps between animation frames (24 matches the dataset cadence)",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "viz")
    parser.add_argument(
        "--all",
        action="store_true",
        help="also render the reference README animation styles for the fine grid",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    entry = build_registry().by_seed(args.seed)
    bumps = entry.initial_condition.bumps
    print(f"seed {args.seed}  ({entry.split})  trajectory {entry.trajectory_id}")
    print(f"  {len(bumps)} bump(s):")
    for bump in bumps:
        print(
            f"    A={bump.amplitude:+.3f} m  sigma={bump.width / 1000:.1f} km  "
            f"center=({bump.center_x / 1000:+.1f}, {bump.center_y / 1000:+.1f}) km"
        )

    # Fine grid first: the pair's shared time step comes from it (D003).
    fine_result, fine_config, fine_seconds = _run(128, args.seed, args.stride, None)
    coarse_result, coarse_config, coarse_seconds = _run(32, args.seed, args.stride, fine_config.dt)
    print(f"  solver total: {fine_seconds + coarse_seconds:.2f} s for both resolutions")

    written = [
        render_pair_comparison(
            coarse_result,
            fine_result,
            coarse_config,
            fine_config,
            args.out,
            args.seed,
            args.stride,
        )
    ]
    if args.all:
        written += render_reference_animations(
            fine_result, fine_config, args.out, f"hr128_seed{args.seed}", args.stride
        )

    print("\nwrote:")
    for path in written:
        print(f"  {path.relative_to(REPO_ROOT)}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
