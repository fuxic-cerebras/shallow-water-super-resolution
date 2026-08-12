"""Typed solver configuration and derived grid quantities.

Mirrors the module-level constants of `references/shallow-water/swe.py` (D010) so a
`SolverConfig` can reproduce the reference solver exactly, while making every value
explicit instead of a mutable global. `docs/ARCHITECTURE.md` requires that a run be
reproducible "without relying on mutable global constants in `swe.py`".

All quantities are SI. Arrays here follow the reference solver's own `[x, y]` indexing;
the transpose to storage order `[y, x]` happens once, in `runner.solve`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from functools import cached_property

import numpy as np

# swe.py:59 -- dt = 0.1 * min(dx, dy) / sqrt(g * H)
DEFAULT_CFL_FACTOR = 0.1


@dataclass(frozen=True)
class SolverConfig:
    """A complete, hashable specification of one solver run.

    Defaults reproduce the reference script's physical setup: rotating beta-plane,
    closed basin, no friction, wind, source, or sink.
    """

    n_x: int
    n_y: int

    # Physical parameters (swe.py:34-47).
    length_x: float = 1.0e6
    length_y: float = 1.0e6
    gravity: float = 9.81
    depth: float = 100.0
    coriolis_f0: float = 1.0e-4
    coriolis_beta: float = 2.0e-11
    density: float = 1024.0
    wind_stress_amplitude: float = 0.1
    friction_coefficient: float = 1.0 / (5 * 24 * 3600)

    use_coriolis: bool = True
    use_beta: bool = True
    use_friction: bool = False
    use_wind: bool = False
    use_source: bool = False
    use_sink: bool = False

    cfl_factor: float = DEFAULT_CFL_FACTOR
    # Set to share one pair's fine-grid time step across both of its resolutions (D003).
    # None derives dt from this grid's own CFL bound.
    dt_override: float | None = None

    def __post_init__(self) -> None:
        if self.n_x < 3 or self.n_y < 3:
            raise ValueError(f"grid must be at least 3x3, got {self.n_x}x{self.n_y}")
        if self.depth <= 0:
            raise ValueError(f"resting depth must be positive, got {self.depth}")
        if self.use_sink and not self.use_source:
            # swe.py:122 -- the sink is sized to balance the source.
            raise ValueError("use_sink requires use_source: the sink balances the source")
        if self.dt_override is not None and self.dt_override <= 0:
            raise ValueError(f"dt_override must be positive, got {self.dt_override}")

    # -- Grid -------------------------------------------------------------------

    @property
    def dx(self) -> float:
        """Grid spacing in x. Both endpoints are included, hence `n_x - 1`."""
        return self.length_x / (self.n_x - 1)

    @property
    def dy(self) -> float:
        return self.length_y / (self.n_y - 1)

    @property
    def x(self) -> np.ndarray:
        """Cell-center x coordinates, endpoints included (swe.py:62)."""
        return np.linspace(-self.length_x / 2, self.length_x / 2, self.n_x)

    @property
    def y(self) -> np.ndarray:
        return np.linspace(-self.length_y / 2, self.length_y / 2, self.n_y)

    @cached_property
    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """`(X, Y)` in the solver's `[x, y]` order, matching swe.py:64-66."""
        mesh_x, mesh_y = np.meshgrid(self.x, self.y)
        return np.transpose(mesh_x), np.transpose(mesh_y)

    # -- Time step --------------------------------------------------------------

    @property
    def gravity_wave_speed(self) -> float:
        """Long-wave phase speed sqrt(g*H), the fastest signal in the linearized system."""
        return float(np.sqrt(self.gravity * self.depth))

    @property
    def cfl_time_step(self) -> float:
        """The time step this grid would choose on its own (swe.py:59)."""
        return self.cfl_factor * min(self.dx, self.dy) / self.gravity_wave_speed

    @property
    def dt(self) -> float:
        return self.cfl_time_step if self.dt_override is None else self.dt_override

    @property
    def gravity_cfl(self) -> float:
        """Courant number for gravity waves. Must stay below 1 for stability."""
        return self.gravity_wave_speed * self.dt / min(self.dx, self.dy)

    # -- Rotation ---------------------------------------------------------------

    @property
    def coriolis(self) -> np.ndarray:
        """f(y), shaped `(n_y,)`. Uniform when `use_beta` is False (swe.py:94-98)."""
        if not self.use_coriolis:
            return np.zeros(self.n_y)
        if self.use_beta:
            return self.coriolis_f0 + self.coriolis_beta * self.y
        return self.coriolis_f0 * np.ones(self.n_y)

    @property
    def rotation_alpha(self) -> np.ndarray:
        """`dt * f`, the semi-implicit rotation parameter (swe.py:101)."""
        return self.dt * self.coriolis

    @property
    def rotation_beta_c(self) -> np.ndarray:
        """`alpha**2 / 4` (swe.py:102). The scheme needs `alpha << 1`."""
        return self.rotation_alpha**2 / 4

    @property
    def rossby_radius(self) -> float:
        """sqrt(g*H)/f_0, the scale on which rotation balances pressure gradients."""
        if self.coriolis_f0 == 0:
            return float("inf")
        return self.gravity_wave_speed / self.coriolis_f0

    # -- Provenance -------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Plain-data form for manifests."""
        payload = asdict(self)
        payload.update(
            {
                "dx": self.dx,
                "dy": self.dy,
                "dt": self.dt,
                "cfl_time_step": self.cfl_time_step,
                "gravity_cfl": self.gravity_cfl,
                "gravity_wave_speed": self.gravity_wave_speed,
                "rossby_radius": self.rossby_radius,
            }
        )
        return payload

    @property
    def config_hash(self) -> str:
        """Stable hash of the declared fields, for manifest provenance.

        Covers only the dataclass fields, not derived values, so it is invariant to
        changes in how derived quantities are reported.
        """
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class ResolutionPair:
    """A coarse/fine pair that shares one initial-condition family and one time step.

    Per D003 both members integrate with the fine grid's stable time step, which makes
    their saved-time arrays bit-identical. Per D002 neither member is derived from the
    other by resizing.
    """

    pair_id: str
    coarse_nodes: int
    fine_nodes: int
    physical: dict[str, float] = field(default_factory=dict)

    @property
    def shape_factor(self) -> int:
        """Node-count ratio. 4 for both project pairs."""
        ratio, remainder = divmod(self.fine_nodes, self.coarse_nodes)
        if remainder or ratio < 2:
            raise ValueError(
                f"{self.pair_id}: fine nodes {self.fine_nodes} must be an integer "
                f"multiple (>=2) of coarse nodes {self.coarse_nodes}"
            )
        return ratio

    @property
    def spacing_ratio(self) -> float:
        """Physical spacing ratio, `(fine-1)/(coarse-1)`.

        Deliberately distinct from `shape_factor`: because both endpoints are included,
        these grids are x4 in node count but 4.0968 (32/128) or 4.0476 (64/256) in
        spacing. `docs/DATASET.md` forbids describing them as exact fourfold meshes.
        """
        return (self.fine_nodes - 1) / (self.coarse_nodes - 1)

    def fine_config(self, **overrides: object) -> SolverConfig:
        return SolverConfig(n_x=self.fine_nodes, n_y=self.fine_nodes, **self.physical, **overrides)  # type: ignore[arg-type]

    def coarse_config(self, **overrides: object) -> SolverConfig:
        """Coarse config sharing the fine grid's time step (D003)."""
        shared_dt = self.fine_config().dt
        return SolverConfig(
            n_x=self.coarse_nodes,
            n_y=self.coarse_nodes,
            dt_override=shared_dt,
            **self.physical,  # type: ignore[arg-type]
            **overrides,  # type: ignore[arg-type]
        )
