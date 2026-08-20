#!/usr/bin/env python3
"""Write the kernel's initial state and constants as `miset`-loadable binaries.

Axis contract, which is the easiest thing to get wrong here: `miset order="y x _word"`
walks y outermost, then x, then the two 16-bit words of an `sp`. Arrays in `swe.py` are
indexed `[x, y]`. So everything is transposed on the way out, and `check.py` transposes
`miget` output back before comparing. `_word` must appear in every `order=` string because
an `sp` occupies two hardware words.

Usage: init.py --nx N --ny N [--outdir .]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from swe_config import SYMBOLS, Config


def write_sp(path: Path, field_xy: np.ndarray) -> None:
    """Write an `[x, y]`-indexed float32 array as little-endian, y outer and x inner."""
    if field_xy.dtype != np.float32:
        raise TypeError(f"{path.name}: expected float32, got {field_xy.dtype}")
    np.ascontiguousarray(field_xy.T).astype("<f4").tofile(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--outdir", type=Path, default=Path())
    args = parser.parse_args()

    config = Config(n_x=args.nx, n_y=args.ny)
    fields = config.fields32()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"grid {args.nx}x{args.ny}  dx={config.dx!r}  dt={config.dt!r}")
    for name in SYMBOLS:
        field = fields[name]
        path = args.outdir / f"init-{name}.bin"
        write_sp(path, field)
        # A checksum per symbol, so a drifted constant is visible in the make log rather
        # than only as a downstream numerical mismatch.
        checksum = int(np.bitwise_xor.reduce(field.view(np.uint32).ravel()))
        print(
            f"  {name:8s} {field.shape} min={field.min():+.9g} "
            f"max={field.max():+.9g} xor=0x{checksum:08x} -> {path.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
