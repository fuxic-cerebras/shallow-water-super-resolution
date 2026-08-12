"""Chunked HDF5 storage, one file per trajectory (D013).

Layout, following `docs/DATASET.md`:

    raw/<dataset_id>/
      trajectories/<trajectory_id>.h5     # lr, hr, time, coordinates, metadata
      manifest.json

Each field array is `[time, channel, y, x]` with channels `[eta, u, v]`, stored float32 in
SI units. `CLAUDE.md` requires that stored raw data keep physical units and that
normalization happen only in the data loader, so nothing here scales values.

Arrays are chunked along time so a loader can stream single frames without reading a whole
trajectory, which matters because full-frame training touches frames in shuffled order.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from swe_sr.data.manifest import ArrayRecord, array_checksum

STORAGE_DTYPE = np.float32
COARSE_ARRAY = "lr"
FINE_ARRAY = "hr"


def _chunk_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """One frame per chunk: `(1, channels, y, x)`.

    Frame-sized chunks match the access pattern (random single frames during training) and
    keep chunk count equal to frame count, which is cheap on NFS. Chunking across time
    would force a multi-frame read for every single-frame access.
    """
    return (1, *shape[1:])


@contextmanager
def open_trajectory(path: Path, mode: str = "r") -> Iterator[h5py.File]:
    with h5py.File(path, mode) as handle:
        yield handle


def write_trajectory(
    path: Path,
    *,
    coarse_fields: np.ndarray,
    fine_fields: np.ndarray,
    times: np.ndarray,
    sample_steps: np.ndarray,
    coarse_x: np.ndarray,
    coarse_y: np.ndarray,
    fine_x: np.ndarray,
    fine_y: np.ndarray,
    metadata: dict[str, Any],
) -> list[ArrayRecord]:
    """Write one paired trajectory and return provenance for each stored array.

    Refuses to overwrite: datasets are immutable release artifacts, and silently replacing
    a trajectory would invalidate the manifest checksum that references it.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already exists. Trajectories are immutable; delete the dataset "
            "directory explicitly or release under a new dataset_id."
        )
    if coarse_fields.shape[0] != fine_fields.shape[0]:
        raise ValueError(
            f"coarse and fine frame counts differ: {coarse_fields.shape[0]} vs "
            f"{fine_fields.shape[0]}"
        )
    if times.shape[0] != coarse_fields.shape[0]:
        raise ValueError("times must have one entry per saved frame")

    coarse = np.ascontiguousarray(coarse_fields, dtype=STORAGE_DTYPE)
    fine = np.ascontiguousarray(fine_fields, dtype=STORAGE_DTYPE)

    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[ArrayRecord] = []

    with h5py.File(path, "w") as handle:
        for name, array in ((COARSE_ARRAY, coarse), (FINE_ARRAY, fine)):
            chunks = _chunk_shape(array.shape)
            handle.create_dataset(
                name,
                data=array,
                chunks=chunks,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
            )
            records.append(
                ArrayRecord(
                    name=name,
                    shape=tuple(array.shape),
                    dtype=str(array.dtype),
                    # Checksum the values, not the container: HDF5 layout and compression
                    # must not affect the integrity signal.
                    checksum=array_checksum(array),
                    chunks=chunks,
                )
            )

        # Times and coordinates stay float64: they define physical position and alignment,
        # and float32 would lose the exact-equality property the pair contract depends on.
        for name, array in (
            ("time", np.asarray(times, dtype=np.float64)),
            ("sample_steps", np.asarray(sample_steps, dtype=np.int64)),
            ("coarse_x", np.asarray(coarse_x, dtype=np.float64)),
            ("coarse_y", np.asarray(coarse_y, dtype=np.float64)),
            ("fine_x", np.asarray(fine_x, dtype=np.float64)),
            ("fine_y", np.asarray(fine_y, dtype=np.float64)),
        ):
            handle.create_dataset(name, data=array)
            records.append(
                ArrayRecord(
                    name=name,
                    shape=tuple(array.shape),
                    dtype=str(array.dtype),
                    checksum=array_checksum(array),
                )
            )

        for key, value in metadata.items():
            handle.attrs[key] = value

    return records


def read_fields(path: Path, array_name: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[array_name][:])


def read_frame(path: Path, array_name: str, frame: int) -> np.ndarray:
    """Read a single frame without materializing the whole trajectory."""
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[array_name][frame])


def read_times(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["time"][:])


def read_coordinates(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {
            "coarse_x": np.asarray(handle["coarse_x"][:]),
            "coarse_y": np.asarray(handle["coarse_y"][:]),
            "fine_x": np.asarray(handle["fine_x"][:]),
            "fine_y": np.asarray(handle["fine_y"][:]),
        }


def read_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        return dict(handle.attrs)


def verify_trajectory_checksums(path: Path, records: list[ArrayRecord]) -> list[str]:
    """Recompute every array checksum and return the names that do not match.

    Used by the independent audit (V-01/V-02), which must recompute from arrays rather
    than trust the generator's logs.
    """
    mismatches: list[str] = []
    with h5py.File(path, "r") as handle:
        for record in records:
            if record.name not in handle:
                mismatches.append(record.name)
                continue
            stored = np.asarray(handle[record.name][:])
            if array_checksum(stored) != record.checksum:
                mismatches.append(record.name)
    return mismatches
