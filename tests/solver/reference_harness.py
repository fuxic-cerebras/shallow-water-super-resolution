"""Execute the pinned reference solver under controlled parameters.

`references/shallow-water/swe.py` is a module-level script, not a library: it imports
matplotlib and `viz_tools` at import time, writes `param_output.txt` into the working
directory, runs a 5000-step 150x150 simulation, and then renders two GIFs and calls
`plt.show()`. None of that can be imported or parameterized directly.

This harness makes it usable as a regression oracle without editing the submodule, which
`CLAUDE.md` forbids. It:

1. reads the pinned source as text;
2. truncates it immediately after the main time loop, so no plotting or GIF work runs;
3. rewrites specific top-level assignments (grid size, step count, sampling intervals)
   with asserted single-match substitutions, so a change in the upstream source is a
   loud failure rather than a silently skipped override;
4. stubs `matplotlib.pyplot` and `viz_tools` in `sys.modules`, so the fast test suite
   needs no plotting dependency at all;
5. executes the result in a private namespace and a temporary working directory.

The returned arrays are the reference solver's own `[x, y]`-indexed output, untransposed.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "references" / "shallow-water"
REFERENCE_SWE = REFERENCE_DIR / "swe.py"

# Everything from this banner onward is reporting and visualization.
_TRUNCATE_MARKER = "# ============================= Main time loop done ="

# The shipped initial condition (swe.py:160), matched by its distinctive `np.exp` form so
# it cannot collide with the `eta_n = np.zeros(...)` allocation earlier in the file.
_IC_PATTERN = re.compile(r"^eta_n = np\.exp\(.*$", re.MULTILINE)


class ReferenceSourceChanged(RuntimeError):
    """The pinned reference source no longer has the shape this harness assumes."""


@dataclass(frozen=True)
class ReferenceResult:
    """Final state and metadata from a reference run, in the solver's own `[x, y]` order."""

    eta: np.ndarray
    u: np.ndarray
    v: np.ndarray
    eta_initial: np.ndarray
    n_steps: int
    dt: float
    dx: float
    dy: float
    n_x: int
    n_y: int
    # Snapshots the script itself accumulated, at `anim_interval` spacing.
    eta_frames: list[np.ndarray]
    u_frames: list[np.ndarray]
    v_frames: list[np.ndarray]


def _require_reference_source() -> str:
    if not REFERENCE_SWE.is_file():
        raise ReferenceSourceChanged(
            f"{REFERENCE_SWE} is missing. The solver reference is a submodule pinned by "
            "decision D010; run `git submodule update --init`."
        )
    return REFERENCE_SWE.read_text()


def _substitute_assignment(source: str, name: str, value_source: str) -> str:
    """Replace a single top-level `name = ...` assignment, preserving any trailing comment.

    Raises if the assignment is not found exactly once, so an upstream rename surfaces as
    an explicit failure instead of an override that quietly does nothing.
    """
    pattern = re.compile(rf"^{re.escape(name)}([ \t]*=[ \t]*)[^\n#]*", re.MULTILINE)
    found = pattern.findall(source)
    if len(found) != 1:
        raise ReferenceSourceChanged(
            f"expected exactly one top-level assignment to {name!r} in "
            f"{REFERENCE_SWE.name}, found {len(found)}"
        )
    return pattern.sub(f"{name} = {value_source}", source, count=1)


def _truncate_before_visualization(source: str) -> str:
    index = source.find(_TRUNCATE_MARKER)
    if index < 0:
        raise ReferenceSourceChanged(
            f"could not find the end-of-time-loop banner in {REFERENCE_SWE.name}; "
            "the harness cannot tell where the visualization section begins"
        )
    if source.count(_TRUNCATE_MARKER) != 1:
        raise ReferenceSourceChanged("end-of-time-loop banner is ambiguous")
    return source[:index]


def _stub_modules() -> dict[str, types.ModuleType | None]:
    """Install import stubs for the plotting modules, returning the previous entries."""
    saved: dict[str, types.ModuleType | None] = {}
    for name in ("matplotlib", "matplotlib.pyplot", "viz_tools"):
        saved[name] = sys.modules.get(name)
        stub = types.ModuleType(name)
        # The truncated source only imports these; it never calls into them.
        stub.__getattr__ = lambda attr: None  # type: ignore[method-assign]
        sys.modules[name] = stub
    # `import matplotlib.pyplot as plt` needs the attribute on the parent package.
    sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]  # type: ignore[attr-defined]
    return saved


def _restore_modules(saved: dict[str, types.ModuleType | None]) -> None:
    for name, module in saved.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def run_reference(
    *,
    n_x: int = 150,
    n_y: int = 150,
    max_time_step: int = 5000,
    anim_interval: int = 20,
    sample_interval: int = 1000,
    eta_initial: np.ndarray | None = None,
) -> ReferenceResult:
    """Run the pinned reference solver and return its final state.

    Argument names mirror the script's own variables. Note the script's loop is
    `while time_step < max_time_step` with `time_step` starting at 1, so it performs
    ``max_time_step - 1`` updates, not ``max_time_step``. `ReferenceResult.n_steps`
    reports the actual count.

    If `eta_initial` is given it replaces the shipped Gaussian bump; it must be shaped
    `(n_x, n_y)` in the solver's own `[x, y]` order.
    """
    source = _require_reference_source()
    source = _truncate_before_visualization(source)
    source = _substitute_assignment(source, "N_x", str(int(n_x)))
    source = _substitute_assignment(source, "N_y", str(int(n_y)))
    source = _substitute_assignment(source, "max_time_step", str(int(max_time_step)))
    source = _substitute_assignment(source, "anim_interval", str(int(anim_interval)))
    source = _substitute_assignment(source, "sample_interval", str(int(sample_interval)))

    namespace: dict[str, Any] = {"__name__": "_swe_reference", "__file__": str(REFERENCE_SWE)}

    if eta_initial is not None:
        expected = (int(n_x), int(n_y))
        if eta_initial.shape != expected:
            raise ValueError(
                f"eta_initial must be shaped {expected} in [x, y] order, got {eta_initial.shape}"
            )
        if _IC_PATTERN.search(source) is None:
            raise ReferenceSourceChanged(
                f"could not find the shipped `eta_n = np.exp(...)` initial condition in "
                f"{REFERENCE_SWE.name}"
            )
        namespace["_INJECTED_ETA0"] = np.array(eta_initial, dtype=float)
        source = _IC_PATTERN.sub("eta_n = _INJECTED_ETA0.copy()", source, count=1)

    saved_modules = _stub_modules()
    saved_path = list(sys.path)
    saved_cwd = Path.cwd()
    try:
        # The script writes param_output.txt relative to the working directory.
        with tempfile.TemporaryDirectory(prefix="swe_reference_") as scratch:
            import os

            os.chdir(scratch)
            sys.path.insert(0, str(REFERENCE_DIR))
            # The script prints its parameter block and a progress line per sample; that
            # is noise here, and the harness may be called many times per test session.
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, str(REFERENCE_SWE), "exec"), namespace)
    finally:
        import os

        os.chdir(saved_cwd)
        sys.path[:] = saved_path
        _restore_modules(saved_modules)

    return ReferenceResult(
        eta=np.array(namespace["eta_n"]),
        u=np.array(namespace["u_n"]),
        v=np.array(namespace["v_n"]),
        eta_initial=np.array(namespace["_INJECTED_ETA0"])
        if eta_initial is not None
        else _shipped_initial_eta(namespace),
        n_steps=int(namespace["time_step"]) - 1,
        dt=float(namespace["dt"]),
        dx=float(namespace["dx"]),
        dy=float(namespace["dy"]),
        n_x=int(namespace["N_x"]),
        n_y=int(namespace["N_y"]),
        eta_frames=[np.array(a) for a in namespace["eta_list"]],
        u_frames=[np.array(a) for a in namespace["u_list"]],
        v_frames=[np.array(a) for a in namespace["v_list"]],
    )


def _shipped_initial_eta(namespace: dict[str, Any]) -> np.ndarray:
    """Recompute the shipped initial condition from the namespace's own grid."""
    x_grid = namespace["X"]
    y_grid = namespace["Y"]
    length_x = namespace["L_x"]
    length_y = namespace["L_y"]
    sigma = 0.05e6
    return np.asarray(
        np.exp(
            -(
                (x_grid - length_x / 2.7) ** 2 / (2 * sigma**2)
                + (y_grid - length_y / 4) ** 2 / (2 * sigma**2)
            )
        )
    )
