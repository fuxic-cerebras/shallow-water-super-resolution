"""Animate the Tungsten kernel's trajectory against `mirror.py`, using viz_tools.

    python3 tungsten/swe32/trajectory.py --nx 32 --ny 32 --frames 100 --stride 50
    python scripts/visualize_tungsten.py --all

`tungsten/swe32/check.py` proves bit-exactness at one final state and prints "exact"; this
renders the same claim over a whole trajectory, which is what makes it legible: the wave
radiates, reflects off the closed walls and turns under rotation in both panels, and the
difference panel stays identically zero while it happens.

Two outputs, both from the reference repo's own `viz_tools.py` (`docs/ARCHITECTURE.md` casts
it as an optional *client* of the solver, so this script lives outside `swe_sr/`):

- `tungsten_vs_mirror.gif` -- kernel, mirror, their difference, and the running max of that
  difference. The difference panel is scaled to one float32 ULP of the field, not
  autoscaled: with a bit-exact kernel the data are all zero, and an autoscale would blow
  roundoff-free noise up into a full-contrast image of nothing.
- `--all` additionally re-renders the kernel's own output through
  `viz_tools.eta_animation`, `eta_animation3D` and `velocity_animation`, so the `.w`
  kernel's result appears in exactly the styles the reference README shows.

Arrays in the `.npz` are already in `swe.py`'s `[x, y]` order (`check.py:read_sim` transposes
`miget`'s y-outer dump back), which is the order `viz_tools` wants, so nothing is transposed
here except for `imshow`, which indexes `[row, column]`.
"""

from __future__ import annotations

import argparse
import os
import sys
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
sys.path.insert(0, str(REPO_ROOT / "tungsten" / "swe32"))

import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import viz_tools  # noqa: E402
from swe_config import Config  # noqa: E402

from swe_sr.solver.diagnostics import destagger_u, destagger_v  # noqa: E402

FIELDS = ("eta", "u", "v")


def relative(path: Path) -> Path:
    """Repo-relative for readability, absolute when the path is outside the repo."""
    resolved = path.resolve()
    return resolved.relative_to(REPO_ROOT) if resolved.is_relative_to(REPO_ROOT) else resolved


def load(path: Path) -> dict:
    """Read `trajectory.py`'s `.npz` and re-derive the grid from the recorded size.

    `swe_config.Config` is the single source of truth for `dt` and `dx` (its docstring says
    so), so the stored scalars are checked against a freshly built Config rather than
    trusted: a mismatch means the npz predates a constant change and the time axis of the
    animation would be silently wrong.
    """
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    config = Config(n_x=int(payload["n_x"]), n_y=int(payload["n_y"]))
    for name, stored in (("dt", float(payload["dt"])), ("dx", float(payload["dx"]))):
        if getattr(config, name) != stored:
            raise SystemExit(
                f"{path}: recorded {name}={stored!r} but swe_config gives "
                f"{getattr(config, name)!r}; re-run trajectory.py"
            )
    payload["config"] = config
    return payload


def render_comparison(payload: dict, out_dir: Path, fps: int) -> Path:
    """Kernel, mirror, difference, and the running max difference, as one GIF."""
    config: Config = payload["config"]
    kernel = payload["kernel_eta"].astype(np.float64)
    mirror = payload["mirror_eta"].astype(np.float64)
    times = payload["times"]
    difference = kernel - mirror

    # Per-frame worst case over all three fields, not just eta: u and v are what the walls
    # act on, so a boundary bug could be exact in eta for a while and not in v.
    per_frame = np.zeros(len(times))
    for name in FIELDS:
        delta = np.abs(
            payload[f"kernel_{name}"].astype(np.float64)
            - payload[f"mirror_{name}"].astype(np.float64)
        )
        per_frame = np.maximum(per_frame, delta.reshape(len(times), -1).max(axis=1))

    # Symmetric limits from the mid-trajectory field, the heuristic viz_tools.eta_animation
    # uses, so this GIF and the reference-style one share a colour scale.
    limit = float(np.abs(kernel[len(kernel) // 2]).max()) or 1.0

    # One ULP at the field's own magnitude: the smallest difference float32 could even
    # represent here, and so the honest floor for the difference panel's scale. Autoscaling
    # an all-zero array would render roundoff-free agreement as a full-contrast image of
    # nothing.
    ulp = float(np.spacing(np.float32(limit)))
    difference_limit = max(float(np.abs(difference).max()), ulp)

    extent = [config.x[0] / 1000, config.x[-1] / 1000, config.y[0] / 1000, config.y[-1] / 1000]
    # Three equal-aspect maps in a row with the trace spanning underneath. A 2x2 grid puts
    # the trace beside a map, and constrained_layout then drops the left column's y label
    # off the canvas; this is the shape scripts/visualize.py already uses successfully.
    figure = plt.figure(figsize=(16, 8.6), facecolor="white", layout="constrained")
    grid = figure.add_gridspec(2, 3, height_ratios=[1.0, 0.45])
    panels = (
        ("Tungsten kernel  (swe.w, float32)", kernel, limit, "RdBu_r", "eta [m]"),
        ("mirror.py  (float32, op for op)", mirror, limit, "RdBu_r", "eta [m]"),
        ("kernel - mirror", difference, difference_limit, "PuOr_r", "difference [m]"),
    )
    images = []
    for column, (title, data, vlimit, cmap, label) in enumerate(panels):
        axis = figure.add_subplot(grid[0, column])
        image = axis.imshow(
            data[0].T,
            origin="lower",
            extent=extent,
            vmin=-vlimit,
            vmax=vlimit,
            cmap=cmap,
            interpolation="nearest",
        )
        # viz_tools switches matplotlib to the seaborn style, which draws its grid *over*
        # the artists; on a colour-mesh panel that is a set of white lines across the field.
        axis.grid(False)
        axis.set_title(title, fontsize=13)
        axis.set_xlabel("x [km]")
        if column == 0:
            axis.set_ylabel("y [km]")
        else:
            axis.set_yticklabels([])
        figure.colorbar(image, ax=axis, fraction=0.046, label=label)
        images.append(image)

    # The trace is the point of the exercise: one number per frame, flat on zero for a
    # bit-exact kernel. Linear axis, because log has nowhere to put a zero.
    trace_axis = figure.add_subplot(grid[1, :])
    trace_axis.plot(times / 3600, per_frame, color="#1f77b4", linewidth=1.5)
    trace_axis.axhline(ulp, color="#d62728", linestyle="--", linewidth=1, label="1 float32 ULP")
    (marker,) = trace_axis.plot([], [], "o", color="#1f77b4", markersize=7)
    trace_axis.set_xlim(0, times[-1] / 3600 or 1)
    trace_axis.set_ylim(-0.05 * ulp, 1.35 * ulp)
    trace_axis.set_xlabel("t [hours]")
    trace_axis.set_ylabel(r"max $|\Delta|$ [m]")
    trace_axis.set_title(r"worst cell over $\eta$, $u$, $v$", fontsize=13)
    trace_axis.legend(loc="upper right", fontsize=10)

    suptitle = figure.suptitle("", fontsize=15)

    def update(frame: int):
        images[0].set_data(kernel[frame].T)
        images[1].set_data(mirror[frame].T)
        images[2].set_data(difference[frame].T)
        marker.set_data([times[frame] / 3600], [per_frame[frame]])
        verdict = "bit-exact" if per_frame[frame] == 0 else f"max|d| = {per_frame[frame]:.3e} m"
        suptitle.set_text(
            f"{config.n_x}x{config.n_y}   step {payload['steps'][frame]:5d}   "
            f"t = {times[frame] / 3600:6.2f} h   {verdict}"
        )
        return [*images, marker, suptitle]

    anim = animation.FuncAnimation(figure, update, frames=len(times), interval=80, blit=False)
    path = out_dir / "tungsten_vs_mirror.gif"
    anim.save(str(path), writer=viz_tools.animation_writer("gif", fps))
    plt.close(figure)

    exact = int((per_frame == 0).sum())
    print(
        f"  {exact}/{len(per_frame)} frames bit-exact; worst frame "
        f"max|kernel - mirror| = {per_frame.max():.3e} m (1 ULP = {ulp:.3e} m)"
    )
    return path


def render_reference_styles(payload: dict, out_dir: Path) -> list[Path]:
    """Re-render the kernel's own output in the three styles the reference README shows."""
    config: Config = payload["config"]
    mesh_x, mesh_y = config.mesh
    frame_interval = float(payload["stride"]) * config.dt
    eta_frames = [frame.astype(float) for frame in payload["kernel_eta"]]
    written: list[Path] = []

    # viz_tools writes "<filename>.<filetype>" relative to the working directory.
    previous = Path.cwd()
    os.chdir(out_dir)
    try:
        viz_tools.eta_animation(mesh_x, mesh_y, eta_frames, frame_interval, "tungsten_eta", "gif")
        written.append(out_dir / "tungsten_eta.gif")
        plt.close("all")

        viz_tools.eta_animation3D(
            mesh_x, mesh_y, eta_frames, frame_interval, "tungsten_surface", "gif"
        )
        written.append(out_dir / "tungsten_surface.gif")
        plt.close("all")

        # Destaggered first (D011): the kernel's u lives on the cell's east face and v on
        # its north face, while viz_tools draws colocated arrows.
        u_frames = [destagger_u(frame.astype(float)) for frame in payload["kernel_u"]]
        v_frames = [destagger_v(frame.astype(float)) for frame in payload["kernel_v"]]
        viz_tools.velocity_animation(
            mesh_x, mesh_y, u_frames, v_frames, frame_interval, "tungsten_velocity", "gif"
        )
        written.append(out_dir / "tungsten_velocity.gif")
        plt.close("all")
    finally:
        os.chdir(previous)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        default=REPO_ROOT / "tungsten" / "swe32" / "test-traj" / "trajectory.npz",
        help="output of tungsten/swe32/trajectory.py",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "viz")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--all", action="store_true", help="also render the reference README styles"
    )
    args = parser.parse_args(argv)

    if not args.npz.is_file():
        raise SystemExit(
            f"{args.npz} not found. Record a trajectory first:\n"
            "  python3 tungsten/swe32/trajectory.py --nx 32 --ny 32 --frames 100 --stride 50"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    payload = load(args.npz)
    config: Config = payload["config"]
    steps = payload["steps"]
    print(
        f"{relative(args.npz)}: {config.n_x}x{config.n_y}, {len(steps)} frames, "
        f"{steps[-1]} kernel steps, {payload['times'][-1] / 3600:.2f} h at dt={config.dt:.4f} s"
    )

    written = [render_comparison(payload, args.out, args.fps)]
    if args.all:
        written += render_reference_styles(payload, args.out)

    print("\nwrote:")
    for path in written:
        print(f"  {relative(path)}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
