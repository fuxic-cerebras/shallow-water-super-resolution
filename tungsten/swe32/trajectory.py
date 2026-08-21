#!/usr/bin/env python3
"""Record a kernel *trajectory* by checkpoint-restart, next to `mirror.py`'s own.

`check.py` compares one final state, because that is all `miget` can see: the kernel writes
`eta`, `u`, `v` back at the end of its `j in [0, NSTEP)` loop and the core dump holds only
that. An animation needs many states, so this driver runs the kernel repeatedly with
`NSTEP = --stride`, and after each run `miset`s the state it just read back in as the next
run's initial condition.

That is exact, not an approximation: the scheme is a one-step recurrence whose entire state
is `(eta, u, v)`, every constant is recomputed from `swe_config.py` at load time, and the
probe taps are write-only. So `--frames F --stride S` costs the same F*S kernel steps a
single uninterrupted run of F*S steps would, and the restart is *verified* rather than
assumed -- `mirror.py` is advanced continuously, one step at a time, and every frame is
compared against it. A restart that lost or duplicated a step would break bit-exactness at
the first frame.

Usage, from this directory (see README.md -- the long rungs want cluster cores):

    python3 trajectory.py --nx 32 --ny 32 --frames 100 --stride 50

Writes `<dir>/trajectory.npz`, which `scripts/visualize_tungsten.py` renders.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from check import read_sim  # noqa: E402
from init import write_sp  # noqa: E402
from mirror import Mirror  # noqa: E402
from swe_config import Config  # noqa: E402

FIELDS = ("eta", "u", "v")

# The Makefile's default; overridable the same way, so a moved checkout needs one edit here
# and one there rather than a guess at run time.
DEFAULT_ATG = Path("/net/fuxic-vm.cerebras.aws/srv/nfs/fuxic-data/ws/atg")


def toolchain_env(atg: Path) -> dict[str, str]:
    """`os.environ` with the Tungsten sysroot on PATH, as `flow/devenv.sh` would put it."""
    sysroot_bin = atg / "sysroot-x86" / "bin"
    if not (sysroot_bin / "simfabric").is_file():
        raise SystemExit(f"{sysroot_bin}/simfabric not found; set --atg to the atg checkout")
    env = dict(os.environ)
    env["PATH"] = f"{sysroot_bin}:{env.get('PATH', '')}"
    return env


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    """Run one toolchain command, failing loudly with its own output."""
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(f"{command[0]} failed ({result.returncode}): {' '.join(command)}")


def build(directory: Path, config: Config, stride: int, arch: str, env: dict[str, str]) -> None:
    """Paint and link once, at `NSTEP = stride`, and load the constants and initial state.

    This is `test.mk`'s own recipe -- `make -f ../Makefile -C test-<label> ... VPATH=..` --
    invoked directly because the label is not a `test.rc` rung. Command-line variables beat
    the `NX`/`NY`/`NSTEP` that `test.mk` evaluates out of `test.rc`'s first line, so the
    grid below is the one that gets painted.
    """
    if directory.resolve().parent != HERE:
        raise SystemExit(
            f"--dir must be a subdirectory of {HERE}: paint resolves code('swe) through "
            f"test.mk's VPATH=.. contract, so the build directory has to sit one level "
            f"below the sources. Got {directory}."
        )
    directory.mkdir(parents=True, exist_ok=True)
    run(
        [
            "make",
            "-f",
            "../Makefile",
            "-C",
            directory.name,
            "bench.img",
            "VPATH=..",
            f"NX={config.n_x}",
            f"NY={config.n_y}",
            f"NSTEP={stride}",
            f"ARCH={arch}",
            "MODE=sim",
        ],
        cwd=directory.parent,
        env=env,
    )


def advance(
    directory: Path,
    config: Config,
    state: dict[str, np.ndarray],
    cycles: int,
    threads: int,
    env: dict[str, str],
) -> tuple[dict[str, np.ndarray], str]:
    """Load `state`, run the kernel for its painted `NSTEP`, and read the result back.

    `miset` writes `bench.img` in place, exactly as the Makefile does, so only the three
    state symbols are touched and every constant keeps the value `init.py` gave it. The
    intermediates are removed first: `simfabric` and `elfmi` will happily leave a previous
    frame's `bench-sim.core` behind on failure, and reading that would silently animate a
    stalled trajectory.
    """
    for name, field in state.items():
        write_sp(directory / f"state-{name}.bin", field)
    for stale in (
        "bench.elf",
        "bench.elf-core",
        "bench-sim.core",
        *(f"sim-{f}.bin" for f in FIELDS),
    ):
        (directory / stale).unlink(missing_ok=True)

    for name in FIELDS:
        run(
            [
                "miset",
                "image=bench.img",
                "out=bench.img",
                "map=bench.map",
                f"symbol=grid.{name}",
                "order=y x _word",
                f"data=state-{name}.bin",
            ],
            cwd=directory,
            env=env,
        )
    run(["mielf", "image=bench.img", "elf=bench.elf"], cwd=directory, env=env)

    # simfabric's own stdout is the cycle report; the Makefile redirects it to a log and so
    # do we, because `make warn` greps that file for orphan wavelets.
    with (
        (directory / "bench-sim.log").open("w") as log,
        (directory / "bench-sim-err.log").open("w") as err,
    ):
        result = subprocess.run(
            [
                "simfabric",
                "elf=bench.elf",
                "core=bench.elf-core",
                "json=0",
                f"cycles={cycles}",
                f"threads={threads}",
            ],
            cwd=directory,
            env=env,
            stdout=log,
            stderr=err,
        )
    if result.returncode != 0:
        sys.stderr.write((directory / "bench-sim-err.log").read_text())
        raise SystemExit(f"simfabric failed ({result.returncode})")

    run(["elfmi", "image=bench-sim.core", "elf=bench.elf-core"], cwd=directory, env=env)
    for name in FIELDS:
        run(
            [
                "miget",
                "image=bench-sim.core",
                "map=bench.map",
                f"symbol=grid.{name}",
                "order=y x _word",
                f"data=sim-{name}.bin",
            ],
            cwd=directory,
            env=env,
        )

    fields = {n: read_sim(n, config.n_x, config.n_y, directory) for n in FIELDS}
    return fields, health(directory / "bench-sim.log")


def health(log: Path) -> str:
    """The two things `make warn` looks for, per frame rather than once at the end."""
    text = log.read_text(errors="replace")
    clean = "All router state is clean" in text
    orphans = sum(1 for line in text.splitlines() if "orphan" in line.lower())
    if clean and orphans == 0:
        return "clean"
    return f"{'clean' if clean else 'ROUTER STATE DIRTY'}, {orphans} orphan reports"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--frames", type=int, default=100, help="kernel restarts to perform")
    parser.add_argument("--stride", type=int, default=50, help="kernel steps between frames")
    parser.add_argument("--dir", type=Path, default=HERE / "test-traj")
    parser.add_argument("--arch", default="sdr")
    parser.add_argument("--cycles", type=int, default=100000, help="per-frame simfabric budget")
    parser.add_argument("--threads", type=int, default=0, help="simfabric threads; 0 = all cores")
    parser.add_argument("--atg", type=Path, default=DEFAULT_ATG)
    parser.add_argument("--out", type=Path, default=None, help="default <dir>/trajectory.npz")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="record a frame that is not bit-exact instead of stopping there",
    )
    args = parser.parse_args()

    threads = args.threads or len(os.sched_getaffinity(0))
    out = args.out or args.dir / "trajectory.npz"
    config = Config(n_x=args.nx, n_y=args.ny)
    env = toolchain_env(args.atg)
    if shutil.which("make") is None:
        raise SystemExit("make not found")

    total = args.frames * args.stride
    print(
        f"grid {args.nx}x{args.ny}  {args.frames} frames x {args.stride} steps = {total} steps\n"
        f"dt={config.dt:.6f} s  horizon={total * config.dt / 3600:.2f} h  "
        f"threads={threads}  dir={args.dir}"
    )

    build(args.dir, config, args.stride, args.arch, env)

    # Frame 0 is the initial condition itself, shared by construction, and it anchors the
    # animation at t=0 rather than at t=stride.
    initial = config.fields32()
    state = {name: initial[name].copy() for name in FIELDS}
    mirror = Mirror(config)
    kernel_frames = [dict(state)]
    mirror_frames = [{"eta": mirror.eta.copy(), "u": mirror.u.copy(), "v": mirror.v.copy()}]
    steps = [0]
    worst = 0.0
    started = time.perf_counter()

    for frame in range(1, args.frames + 1):
        state, status = advance(args.dir, config, state, args.cycles, threads, env)
        for _ in range(args.stride):
            mirror.step()

        # Bit-exactness, not closeness: this is check.py's tier 1 applied every frame.
        exact = all(
            (state[n].view(np.uint32) == getattr(mirror, n).view(np.uint32)).all() for n in FIELDS
        )
        deltas = {
            n: float(
                np.abs(state[n].astype(np.float64) - getattr(mirror, n).astype(np.float64)).max()
            )
            for n in FIELDS
        }
        worst = max(worst, *deltas.values())
        elapsed = time.perf_counter() - started
        print(
            f"  frame {frame:4d}/{args.frames}  step {frame * args.stride:6d}  "
            f"{'exact' if exact else 'MISMATCH ' + repr(deltas)}  {status}  "
            f"[{elapsed:.0f}s, {elapsed / frame:.1f}s/frame]"
        )
        if not exact and not args.keep_going:
            raise SystemExit(
                "frame is not bit-exact against mirror.py; pass --keep-going to record it"
            )

        kernel_frames.append(dict(state))
        mirror_frames.append({n: getattr(mirror, n).copy() for n in FIELDS})
        steps.append(frame * args.stride)

    payload = {f"kernel_{n}": np.stack([f[n] for f in kernel_frames]) for n in FIELDS}
    payload.update({f"mirror_{n}": np.stack([f[n] for f in mirror_frames]) for n in FIELDS})
    payload["steps"] = np.asarray(steps, dtype=np.int64)
    payload["times"] = np.asarray(steps, dtype=np.float64) * config.dt
    payload["n_x"] = np.asarray(config.n_x)
    payload["n_y"] = np.asarray(config.n_y)
    payload["dt"] = np.asarray(config.dt)
    payload["dx"] = np.asarray(config.dx)
    payload["stride"] = np.asarray(args.stride)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)

    print(
        f"\n{len(steps)} frames, worst |kernel - mirror| over the whole trajectory {worst:.3e}\n"
        f"wrote {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
