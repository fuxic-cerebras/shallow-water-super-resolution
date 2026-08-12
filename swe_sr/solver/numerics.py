"""The shallow-water time-stepping kernel.

A faithful transcription of the main loop in `references/shallow-water/swe.py` (D010).
`CLAUDE.md` requires preserving the original behavior while extracting reusable code, so
this module changes nothing about the arithmetic or its ordering. Regression tests in
`tests/solver/test_parity.py` compare it against the pinned script directly.

Scheme (see `references/shallow-water/MODEL_NOTES.md` section 4):

- momentum is forward-in-time, centered-in-space on an Arakawa C-grid, with the
  linearized momentum equations;
- rotation is added by a semi-implicit corrector rather than in the predictor;
- continuity is solved in nonlinear form with upwind fluxes, which is what makes total
  mass conserved to roundoff in a closed basin.

Arrays are indexed `[x, y]`, matching the reference script: `A[i, j]` is the value at
`(x_i, y_j)`. Staggering is expressed only through slicing -- `u[i, j]` is
`u_{i+1/2, j}` at an east face and `v[i, j]` is `v_{i, j+1/2}` at a north face.

Three details below are load-bearing for bit-exact parity and must not be "cleaned up":

1. `u_next`/`v_next` are reused across steps and only partially assigned each step. The
   rotation corrector then reads the whole array, including the last row/column, whose
   value is the zero written by the previous step's boundary condition.
2. The corrector runs before the boundary conditions, which then overwrite the last
   row/column with zero again.
3. The upwind fluxes use `u_next`/`v_next` (the new velocities), not `u`/`v`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from swe_sr.solver.config import SolverConfig


@dataclass
class SolverState:
    """Mutable working state for one run, in `[x, y]` order.

    `eta` is surface elevation, not total height; total height is `H + eta` and must not
    be conflated with it (`CLAUDE.md`).
    """

    eta: np.ndarray
    u: np.ndarray
    v: np.ndarray

    @classmethod
    def from_initial_eta(cls, eta_initial: np.ndarray) -> SolverState:
        """Build a state at rest, matching the reference script's initial conditions.

        The script sets `u = v = 0` everywhere (swe.py:152-155), which already satisfies
        the wall conditions.
        """
        eta = np.array(eta_initial, dtype=float)
        return cls(eta=eta, u=np.zeros_like(eta), v=np.zeros_like(eta))

    def copy(self) -> SolverState:
        return SolverState(eta=self.eta.copy(), u=self.u.copy(), v=self.v.copy())


class _Workspace:
    """Preallocated scratch arrays, allocated once per run as the reference script does."""

    __slots__ = ("h_e", "h_n", "h_s", "h_w", "u_next", "uhwe", "v_next", "vhns")

    def __init__(self, shape: tuple[int, int]) -> None:
        self.u_next = np.zeros(shape)
        self.v_next = np.zeros(shape)
        self.h_e = np.zeros(shape)
        self.h_w = np.zeros(shape)
        self.h_n = np.zeros(shape)
        self.h_s = np.zeros(shape)
        self.uhwe = np.zeros(shape)
        self.vhns = np.zeros(shape)


class ShallowWaterKernel:
    """Advances a `SolverState` in time under a fixed `SolverConfig`."""

    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        shape = (config.n_x, config.n_y)
        self._workspace = _Workspace(shape)

        # Hoist everything that does not change between steps.
        self._dt = config.dt
        self._dx = config.dx
        self._dy = config.dy
        self._g = config.gravity
        self._depth = config.depth
        self._alpha = config.rotation_alpha
        self._beta_c = config.rotation_beta_c

        self._kappa: np.ndarray | None = None
        if config.use_friction:
            self._kappa = np.ones(shape) * config.friction_coefficient

        self._tau_x: np.ndarray | None = None
        self._tau_y: np.ndarray | None = None
        if config.use_wind:
            # swe.py:89-90. tau_x varies with y; tau_y is identically zero.
            self._tau_x = -config.wind_stress_amplitude * np.cos(np.pi * config.y / config.length_y)
            self._tau_y = np.zeros(config.n_y)

        self._sigma: np.ndarray | None = None
        self._sink: np.ndarray | None = None
        if config.use_source:
            mesh_x, mesh_y = config.mesh
            # swe.py:117.
            self._sigma = 0.0001 * np.exp(
                -(
                    (mesh_x - config.length_x / 2) ** 2 / (2 * (1e5) ** 2)
                    + (mesh_y - config.length_y / 2) ** 2 / (2 * (1e5) ** 2)
                )
            )
            if config.use_sink:
                # Uniform sink sized to remove exactly what the source adds (swe.py:124).
                self._sink = np.ones(shape) * float(self._sigma.sum()) / (config.n_x * config.n_y)

    def step(self, state: SolverState) -> None:
        """Advance `state` by one time step, in place."""
        work = self._workspace
        u_next, v_next = work.u_next, work.v_next
        eta, u, v = state.eta, state.u, state.v
        dt, dx, dy, g, depth = self._dt, self._dx, self._dy, self._g, self._depth

        # -- Momentum predictor: forward in time, centered in space (swe.py:185-186) ----
        u_next[:-1, :] = u[:-1, :] - g * dt / dx * (eta[1:, :] - eta[:-1, :])
        v_next[:, :-1] = v[:, :-1] - g * dt / dy * (eta[:, 1:] - eta[:, :-1])

        if self._kappa is not None:
            u_next[:-1, :] -= dt * self._kappa[:-1, :] * u[:-1, :]
            v_next[:, :-1] -= dt * self._kappa[:, :-1] * v[:, :-1]

        if self._tau_x is not None:
            assert self._tau_y is not None
            u_next[:-1, :] += dt * self._tau_x / (self.config.density * depth)
            v_next[:, :-1] += dt * self._tau_y[:-1] / (self.config.density * depth)

        # -- Rotation corrector (swe.py:200-201) ---------------------------------------
        # Deliberately operates on the full array, including the last row/column that the
        # previous step's boundary condition zeroed. See module docstring, note 1.
        if self.config.use_coriolis:
            alpha, beta_c = self._alpha, self._beta_c
            u_next[:, :] = (u_next - beta_c * u + alpha * v) / (1 + beta_c)
            v_next[:, :] = (v_next - beta_c * v - alpha * u) / (1 + beta_c)

        # -- Closed basin: no flow through the north and east walls (swe.py:203-204) ----
        # The west and south walls are enforced implicitly by the one-sided fluxes below.
        v_next[:, -1] = 0.0
        u_next[-1, :] = 0.0

        # -- Upwind face heights for the nonlinear continuity equation (swe.py:208-218) -
        h_e, h_w, h_n, h_s = work.h_e, work.h_w, work.h_n, work.h_s
        h_e[:-1, :] = np.where(u_next[:-1, :] > 0, eta[:-1, :] + depth, eta[1:, :] + depth)
        h_e[-1, :] = eta[-1, :] + depth

        h_w[0, :] = eta[0, :] + depth
        h_w[1:, :] = np.where(u_next[:-1, :] > 0, eta[:-1, :] + depth, eta[1:, :] + depth)

        h_n[:, :-1] = np.where(v_next[:, :-1] > 0, eta[:, :-1] + depth, eta[:, 1:] + depth)
        h_n[:, -1] = eta[:, -1] + depth

        h_s[:, 0] = eta[:, 0] + depth
        h_s[:, 1:] = np.where(v_next[:, :-1] > 0, eta[:, :-1] + depth, eta[:, 1:] + depth)

        # -- Flux divergence (swe.py:220-224) ------------------------------------------
        # Row/column 0 takes a one-sided form: the west and south wall fluxes are zero,
        # which is exactly what makes mass conservation exact.
        uhwe, vhns = work.uhwe, work.vhns
        uhwe[0, :] = u_next[0, :] * h_e[0, :]
        uhwe[1:, :] = u_next[1:, :] * h_e[1:, :] - u_next[:-1, :] * h_w[1:, :]

        vhns[:, 0] = v_next[:, 0] * h_n[:, 0]
        vhns[:, 1:] = v_next[:, 1:] * h_n[:, 1:] - v_next[:, :-1] * h_s[:, 1:]

        # -- Continuity (swe.py:228-236) -----------------------------------------------
        eta_next = eta - dt * (uhwe / dx + vhns / dy)
        if self._sigma is not None:
            eta_next += dt * self._sigma
        if self._sink is not None:
            eta_next -= dt * self._sink

        # -- Commit (swe.py:239-241) ---------------------------------------------------
        state.u = u_next.copy()
        state.v = v_next.copy()
        state.eta = eta_next
