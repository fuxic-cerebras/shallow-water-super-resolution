"""Single source of truth for the Tungsten shallow-water kernel's constants.

Every number the kernel sees comes from here: `init.py` writes them into the machine
image with `miset`, and `mirror.py` reads the same values back, so the kernel and its
reference cannot disagree about a bit pattern. Tutorials 27 and 28 instead hard-code
physics as `:sp` literals mirrored by hand in `ref.py`; nothing checks that the two copies
agree, and here every constant depends on the grid size anyway.

The configuration is `references/shallow-water/swe.py`'s own, evaluated at an arbitrary
node count. That means the coarse dataset leg is *not* what this reproduces: it overrides
`dt` with the fine grid's value (D003) and uses a multi-bump initial condition. The oracle
here is the reference script itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

F32 = np.float32

# swe.py:34-47. Friction, wind, source, and sink are all off in the shipped script, so no
# term for them appears anywhere in this port.
LENGTH_X = 1.0e6
LENGTH_Y = 1.0e6
GRAVITY = 9.81
DEPTH = 100.0
CORIOLIS_F0 = 1.0e-4
CORIOLIS_BETA = 2.0e-11
CFL_FACTOR = 0.1

# The shipped initial condition, swe.py:160.
IC_AMPLITUDE = 1.0
IC_CENTER_X = LENGTH_X / 2.7
IC_CENTER_Y = LENGTH_Y / 4
IC_WIDTH = 0.05e6


@dataclass(frozen=True)
class Config:
    """Resolved grid and its derived constants, in `[x, y]` index order like swe.py."""

    n_x: int
    n_y: int

    @property
    def dx(self) -> float:
        """swe.py:57. Both endpoints are included, hence `n_x - 1`."""
        return LENGTH_X / (self.n_x - 1)

    @property
    def dy(self) -> float:
        return LENGTH_Y / (self.n_y - 1)

    @property
    def gravity_wave_speed(self) -> float:
        return math.sqrt(GRAVITY * DEPTH)

    @property
    def dt(self) -> float:
        """swe.py:59 -- the CFL bound for *this* grid, not a shared pair time step."""
        return CFL_FACTOR * min(self.dx, self.dy) / self.gravity_wave_speed

    @property
    def x(self) -> np.ndarray:
        return np.linspace(-LENGTH_X / 2, LENGTH_X / 2, self.n_x)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(-LENGTH_Y / 2, LENGTH_Y / 2, self.n_y)

    @cached_property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """`[x, y]`-indexed meshgrid, matching swe.py's transpose at swe.py:64-66."""
        mesh_x, mesh_y = np.meshgrid(self.x, self.y)
        return mesh_x.T, mesh_y.T

    @property
    def coriolis(self) -> np.ndarray:
        """`f = f_0 + beta*y`, shaped `(n_y,)` -- it varies along the SECOND index only."""
        return CORIOLIS_F0 + CORIOLIS_BETA * self.y

    @property
    def rotation_alpha(self) -> np.ndarray:
        """`dt*f`, swe.py:101."""
        return self.dt * self.coriolis

    @property
    def rotation_beta_c(self) -> np.ndarray:
        """`alpha**2/4`, swe.py:102."""
        return self.rotation_alpha**2 / 4

    def initial_eta(self) -> np.ndarray:
        """The shipped Gaussian bump, swe.py:160, in `[x, y]` order.

        Each axis term is divided by `2*sigma**2` separately and only then summed. That is
        algebraically the same as summing first, but not bit-identical, and this is the
        form the reference script uses.
        """
        mesh_x, mesh_y = self.mesh
        denominator = 2 * IC_WIDTH**2
        exponent_x = (mesh_x - IC_CENTER_X) ** 2 / denominator
        exponent_y = (mesh_y - IC_CENTER_Y) ** 2 / denominator
        return np.asarray(IC_AMPLITUDE * np.exp(-(exponent_x + exponent_y)))

    # -- The float32 values the kernel actually operates on --------------------------
    #
    # Cast once, here, from float64. The kernel has no divide (Tungsten lowers `/` to a
    # Newton-Raphson reciprocal, not an IEEE divide) and no `exp`, so every reciprocal and
    # the Gaussian are evaluated on the host.

    def fields32(self) -> dict[str, np.ndarray]:
        """Every `miset`-loaded symbol, as `(n_x, n_y)` float32 arrays in `[x, y]` order."""
        shape = (self.n_x, self.n_y)
        ones = np.ones(shape)

        # alpha and beta_c are (n_y,) and broadcast along the second axis.
        alpha = ones * self.rotation_alpha
        beta_c = ones * self.rotation_beta_c

        # The east wall is the last x index, the north wall the last y index (swe.py:203-204).
        mask_u = np.ones(shape)
        mask_u[-1, :] = 0.0
        mask_v = np.ones(shape)
        mask_v[:, -1] = 0.0

        return {
            "eta": self.initial_eta().astype(F32),
            "u": np.zeros(shape, F32),
            "v": np.zeros(shape, F32),
            "alpha": alpha.astype(F32),
            "beta_c": beta_c.astype(F32),
            "inv1pb": (1.0 / (1.0 + beta_c)).astype(F32),
            "mask_u": mask_u.astype(F32),
            "mask_v": mask_v.astype(F32),
            "gdtdx": (ones * (GRAVITY * self.dt / self.dx)).astype(F32),
            "gdtdy": (ones * (GRAVITY * self.dt / self.dy)).astype(F32),
            "dtdx": (ones * (self.dt / self.dx)).astype(F32),
            "dtdy": (ones * (self.dt / self.dy)).astype(F32),
            "depth": (ones * DEPTH).astype(F32),
        }


SYMBOLS = (
    "eta",
    "u",
    "v",
    "alpha",
    "beta_c",
    "inv1pb",
    "mask_u",
    "mask_v",
    "gdtdx",
    "gdtdy",
    "dtdx",
    "dtdy",
    "depth",
)
