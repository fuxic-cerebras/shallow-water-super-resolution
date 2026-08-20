#!/usr/bin/env python3
"""Operation-for-operation float32 mirror of `swe.w`.

This is the *bit-exact* reference. It mirrors the kernel, not `swe.py`: the kernel
multiplies by precomputed reciprocals (`inv1pb`, `dtdx`, `dtdy`) because Tungsten has no
IEEE divide, and it groups the flux divergence differently from `swe.py`'s
`dt*(uhwe/dx + vhns/dy)`. Those two deviations are algebraically equivalent and
numerically distinct, so `swe.py` is a *tolerance* oracle and this file is the exact one.

Fused multiply-add is the subtlety. The Tungsten backend forms `fmacs`/`fmss` from the AST
shape `dst <- a +/- b*c` and rounds the result **once**; there is no pragma to disable it
(the pragma set is uthread/socket/fifo/timer/teardown/dot/math/vectorize, and `math.fast`
only *lowers* precision). A binary32 FMA evaluated in binary64 and rounded once is exact,
because 53 >= 2*24+2, so `f32(f64(a) + f64(b)*f64(c))` reproduces the hardware exactly.

`FUSED` records, per arithmetic site, whether the compiler actually fused it. Confirm each
entry against `bench-listing.json` (`make fma`) rather than trusting the default.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from swe_config import Config

F32 = np.float32

# Every `a +/- b*c` shape in the kernel body, keyed by site, with whether the backend
# actually fused it. These are READ OFF the emitted assembly, not assumed: run `make fma`
# and match the opcodes against the swe.w line numbers. The backend fuses far less than
# the `a +/- b*c` shape alone would suggest -- of the nine candidate sites only two become
# `fmss`, and the rest emit a separate `fmuls` followed by `fadds`/`fsubs`.
#
# Observed on arch=sdr, tungsten 5867f711d:
#   swe.w:93  fsubs, fmuls, fsubs        -> pred_x unfused
#   swe.w:94  fsubs, fmuls, fsubs        -> pred_y unfused
#   swe.w:99  fmss, fmuls, fadds, fmuls  -> cor_u1 FUSED, cor_u2 unfused
#   swe.w:100 fmss, fmuls, fsubs, fmuls  -> cor_v1 FUSED, cor_v2 unfused
#   swe.w:144 fmuls, fmuls, fsubs        -> flux_x unfused
#   swe.w:145 fmuls, fmuls, fsubs        -> flux_y unfused
#   swe.w:146 fmuls, fmuls, fadds, fsubs -> div    unfused
#
# The two `fmss` are attributed to `swe.w:1` in the listing rather than to 99/100; the
# line number is wrong but their position in the instruction stream is not.
FUSED = {
    "pred_x": False,  # un  <- uu - gx*(eE - e)
    "pred_y": False,  # vn  <- vv - gy*(eN - e)
    "cor_u1": True,  # t   <- un - bc*uu
    "cor_u2": False,  # t   <- t  + al*vv
    "cor_v1": True,  # t   <- vn - bc*vv
    "cor_v2": False,  # t   <- t  - al*uu
    "fe": False,  # fe  <- uep*ec  + uen*eEc
    "fw": False,  # fw  <- uwp*eWc + uwn*ec
    "fn": False,  # fn  <- vnp*ec  + vnn*eNc
    "fs": False,  # fs  <- vsp*eSc + vsn*ec
    "div": False,  # t   <- tx*(fe-fw) + ty*(fn-fs)
}

TINY32 = np.finfo(np.float32).tiny


class Mirror:
    """Float32 mirror of one PE's program, evaluated over the whole grid at once.

    Vectorizing is safe: NumPy rounds every float32 elementwise op independently, which is
    what the per-PE scalar code does. Only the fused sites need special handling.
    """

    def __init__(
        self, config: Config, *, fused: dict[str, bool] | None = None, ftz: bool = False
    ) -> None:
        self.config = config
        self.fused = dict(FUSED if fused is None else fused)
        self.ftz = ftz
        fields = config.fields32()
        self.eta = fields["eta"].copy()
        self.u = fields["u"].copy()
        self.v = fields["v"].copy()
        for name in (
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
        ):
            setattr(self, name, fields[name])

    # -- primitives ------------------------------------------------------------------

    def _r(self, value: np.ndarray) -> np.ndarray:
        """Round to float32, optionally flushing subnormals as the hardware might."""
        out = value.astype(F32)
        if self.ftz:
            out = np.where(np.abs(out) < TINY32, np.zeros_like(out), out)
        return out

    def fma(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, site: str) -> np.ndarray:
        """`a + b*c`, one rounding if the site is fused, two if not."""
        if self.fused[site]:
            return self._r(a.astype(np.float64) + b.astype(np.float64) * c.astype(np.float64))
        return self._r(a + self._r(b * c))

    def fms(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, site: str) -> np.ndarray:
        """`a - b*c`, one rounding if the site is fused, two if not."""
        if self.fused[site]:
            return self._r(a.astype(np.float64) - b.astype(np.float64) * c.astype(np.float64))
        return self._r(a - self._r(b * c))

    # -- one time step ---------------------------------------------------------------

    def step(self) -> None:
        eta, u, v, dep = self.eta, self.u, self.v, self.depth
        zx = np.zeros((1, self.config.n_y), F32)
        zy = np.zeros((self.config.n_x, 1), F32)

        # Phase 1: eta from all four neighbours. The moat replies 0.
        eta_e = np.concatenate([eta[1:, :], zx], axis=0)
        eta_w = np.concatenate([zx, eta[:-1, :]], axis=0)
        eta_n = np.concatenate([eta[:, 1:], zy], axis=1)
        eta_s = np.concatenate([zy, eta[:, :-1]], axis=1)

        # Momentum predictor. At the east wall eta_e is the moat's 0, making `un` garbage
        # there; the wall mask below discards it, exactly as swe.py overwrites u_np1[-1,:].
        un = self.fms(u, self.gdtdx, self._r(eta_e - eta), "pred_x")
        vn = self.fms(v, self.gdtdy, self._r(eta_n - eta), "pred_y")

        # Rotation corrector: reads the OLD u and v, so both new velocities are formed
        # before either is committed.
        un2 = self._r(
            self.fma(self.fms(un, self.beta_c, u, "cor_u1"), self.alpha, v, "cor_u2") * self.inv1pb
        )
        vn2 = self._r(
            self.fms(self.fms(vn, self.beta_c, v, "cor_v1"), self.alpha, u, "cor_v2") * self.inv1pb
        )

        # Walls, arithmetic and exactly +0.0, matching the kernel's `un2 * mu + 0.0`.
        un2 = self._r(self._r(un2 * self.mask_u) + F32(0.0))
        vn2 = self._r(self._r(vn2 * self.mask_v) + F32(0.0))

        # Phase 2: new u westward, new v southward; the moat replies 0.
        u_w = np.concatenate([zx, un2[:-1, :]], axis=0)
        v_s = np.concatenate([zy, vn2[:, :-1]], axis=1)

        # Total column heights.
        e_c = self._r(eta + dep)
        e_ec = self._r(eta_e + dep)
        e_wc = self._r(eta_w + dep)
        e_nc = self._r(eta_n + dep)
        e_sc = self._r(eta_s + dep)

        # Branch-free upwind flux split, mirroring the kernel exactly:
        #   u * h_upwind == max(u,0)*h_here + min(u,0)*h_downwind
        zero = F32(0.0)
        f_e = self.fma(self._r(np.maximum(un2, zero) * e_c), np.minimum(un2, zero), e_ec, "fe")
        f_w = self.fma(self._r(np.maximum(u_w, zero) * e_wc), np.minimum(u_w, zero), e_c, "fw")
        f_n = self.fma(self._r(np.maximum(vn2, zero) * e_c), np.minimum(vn2, zero), e_nc, "fn")
        f_s = self.fma(self._r(np.maximum(v_s, zero) * e_sc), np.minimum(v_s, zero), e_c, "fs")

        divergence = self.fma(
            self._r(self.dtdx * self._r(f_e - f_w)), self.dtdy, self._r(f_n - f_s), "div"
        )

        self.u, self.v, self.eta = un2, vn2, self._r(eta - divergence)

    def run(self, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(steps):
            self.step()
        return self.eta, self.u, self.v
