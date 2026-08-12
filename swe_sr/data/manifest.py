"""Immutable dataset manifests with per-array checksums (task D-03).

`docs/DATASET.md` enumerates what every manifest must record. The design goals here are
that a manifest is (a) sufficient to reproduce the dataset, (b) sufficient to detect any
later corruption or tampering, and (c) immutable once written.

Checksums are taken over the raw little-endian bytes of each stored array, so they are
independent of HDF5 chunking, compression, and library version. A checksum over the
container file would change if h5py changed its layout, which would make it useless as a
data-integrity signal.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_SCHEMA_VERSION = 1


def array_checksum(array: np.ndarray) -> str:
    """SHA-256 over the array's canonical little-endian C-contiguous bytes.

    Normalizing byte order and contiguity first means the checksum describes the *values*,
    not an accident of how they happen to be laid out in this process.
    """
    canonical = np.ascontiguousarray(array)
    if canonical.dtype.byteorder == ">":
        canonical = canonical.astype(canonical.dtype.newbyteorder("<"))
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype.str).encode())
    digest.update(str(canonical.shape).encode())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def coordinate_hash(*coordinate_arrays: np.ndarray) -> str:
    """Hash of the physical coordinate arrays.

    `docs/DATASET.md` requires storing coordinate hashes so a mismatched grid is detected
    rather than silently interpolated across. Coordinates are promoted to float64 first,
    since they define physical position and must not depend on storage dtype.
    """
    digest = hashlib.sha256()
    for array in coordinate_arrays:
        digest.update(array_checksum(np.asarray(array, dtype=np.float64)).encode())
    return digest.hexdigest()


def git_commit() -> str:
    """Current commit, for manifest provenance. Reports the dirty state honestly."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[2],
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    # A dataset generated from uncommitted code is not reproducible from the commit alone,
    # so say so in the manifest rather than implying clean provenance.
    return f"{commit}-dirty" if dirty else commit


@dataclass
class ArrayRecord:
    """Provenance for one stored array."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    checksum: str
    chunks: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "checksum": self.checksum,
            "chunks": list(self.chunks) if self.chunks else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArrayRecord:
        return cls(
            name=str(payload["name"]),
            shape=tuple(payload["shape"]),
            dtype=str(payload["dtype"]),
            checksum=str(payload["checksum"]),
            chunks=tuple(payload["chunks"]) if payload.get("chunks") else None,
        )


@dataclass
class TrajectoryRecord:
    """Provenance for one trajectory: its identity, split, arrays, and diagnostics."""

    trajectory_id: str
    seed: int
    split: str
    initial_condition: dict[str, Any]
    arrays: list[ArrayRecord] = field(default_factory=list)
    diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict)
    resolvability: dict[str, dict[str, float]] = field(default_factory=dict)
    relative_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "seed": self.seed,
            "split": self.split,
            "initial_condition": self.initial_condition,
            "arrays": [record.to_dict() for record in self.arrays],
            "diagnostics": self.diagnostics,
            "resolvability": self.resolvability,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrajectoryRecord:
        return cls(
            trajectory_id=str(payload["trajectory_id"]),
            seed=int(payload["seed"]),
            split=str(payload["split"]),
            initial_condition=dict(payload["initial_condition"]),
            arrays=[ArrayRecord.from_dict(a) for a in payload["arrays"]],
            diagnostics=dict(payload.get("diagnostics", {})),
            resolvability=dict(payload.get("resolvability", {})),
            relative_path=str(payload.get("relative_path", "")),
        )

    def checksum_for(self, array_name: str) -> str:
        for record in self.arrays:
            if record.name == array_name:
                return record.checksum
        raise KeyError(f"trajectory {self.trajectory_id} has no array named {array_name!r}")


class ManifestError(ValueError):
    """A manifest is malformed, inconsistent, or would be overwritten."""


@dataclass
class DatasetManifest:
    """Everything needed to reproduce and verify one dataset release.

    Field set follows the enumeration in `docs/DATASET.md`: schema version, commit and
    config hash, physical parameters and boundary conditions, coordinates, shared time
    step and saved times, resolution-family and pair identity, registry hash, endpoint
    convention, coordinate hashes, node counts, shape factor, spacing, split membership,
    per-array dtype/shape/chunking/checksum, and train-split normalization.
    """

    dataset_id: str
    pair_id: str
    resolution_family: str
    coarse_nodes: int
    fine_nodes: int
    shape_factor: int
    spacing_ratio: float
    endpoint_convention: str
    coarse_config: dict[str, Any]
    fine_config: dict[str, Any]
    shared_time_step: float
    sample_steps: list[int]
    saved_times: list[float]
    coarse_coordinate_hash: str
    fine_coordinate_hash: str
    ic_registry_version: str
    ic_registry_hash: str
    git_commit: str
    trajectories: list[TrajectoryRecord] = field(default_factory=list)
    normalization: dict[str, Any] = field(default_factory=dict)
    schema_version: int = MANIFEST_SCHEMA_VERSION

    # -- Access ----------------------------------------------------------------

    def by_split(self, split: str) -> list[TrajectoryRecord]:
        return [record for record in self.trajectories if record.split == split]

    def by_trajectory_id(self, trajectory_id: str) -> TrajectoryRecord:
        for record in self.trajectories:
            if record.trajectory_id == trajectory_id:
                return record
        raise KeyError(f"manifest has no trajectory {trajectory_id}")

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "pair_id": self.pair_id,
            "resolution_family": self.resolution_family,
            "grid": {
                "coarse_nodes": self.coarse_nodes,
                "fine_nodes": self.fine_nodes,
                "shape_factor": self.shape_factor,
                "spacing_ratio": self.spacing_ratio,
                "endpoint_convention": self.endpoint_convention,
                "coarse_coordinate_hash": self.coarse_coordinate_hash,
                "fine_coordinate_hash": self.fine_coordinate_hash,
            },
            "time": {
                "shared_time_step": self.shared_time_step,
                "sample_steps": list(self.sample_steps),
                "saved_times": list(self.saved_times),
            },
            "solver": {"coarse": self.coarse_config, "fine": self.fine_config},
            "initial_conditions": {
                "registry_version": self.ic_registry_version,
                "registry_hash": self.ic_registry_hash,
            },
            "provenance": {"git_commit": self.git_commit},
            "normalization": self.normalization,
            "trajectories": [record.to_dict() for record in self.trajectories],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def manifest_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DatasetManifest:
        grid = payload["grid"]
        time_block = payload["time"]
        return cls(
            dataset_id=str(payload["dataset_id"]),
            pair_id=str(payload["pair_id"]),
            resolution_family=str(payload["resolution_family"]),
            coarse_nodes=int(grid["coarse_nodes"]),
            fine_nodes=int(grid["fine_nodes"]),
            shape_factor=int(grid["shape_factor"]),
            spacing_ratio=float(grid["spacing_ratio"]),
            endpoint_convention=str(grid["endpoint_convention"]),
            coarse_coordinate_hash=str(grid["coarse_coordinate_hash"]),
            fine_coordinate_hash=str(grid["fine_coordinate_hash"]),
            shared_time_step=float(time_block["shared_time_step"]),
            sample_steps=[int(s) for s in time_block["sample_steps"]],
            saved_times=[float(t) for t in time_block["saved_times"]],
            coarse_config=dict(payload["solver"]["coarse"]),
            fine_config=dict(payload["solver"]["fine"]),
            ic_registry_version=str(payload["initial_conditions"]["registry_version"]),
            ic_registry_hash=str(payload["initial_conditions"]["registry_hash"]),
            git_commit=str(payload["provenance"]["git_commit"]),
            trajectories=[TrajectoryRecord.from_dict(r) for r in payload["trajectories"]],
            normalization=dict(payload.get("normalization", {})),
            schema_version=int(payload["schema_version"]),
        )

    def write(self, path: Path) -> str:
        """Write the manifest, refusing to replace a different one.

        Manifests are immutable release artifacts (`docs/AGENT_WORKFLOW.md`): overwriting
        one would break the link between a checkpoint and the data it was trained on.
        """
        if path.exists():
            existing = DatasetManifest.from_dict(json.loads(path.read_text()))
            if existing.manifest_hash == self.manifest_hash:
                return self.manifest_hash
            raise ManifestError(
                f"{path} already exists with a different hash "
                f"({existing.manifest_hash[:12]} != {self.manifest_hash[:12]}). "
                "Manifests are immutable; release under a new dataset_id."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return self.manifest_hash


def load_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.from_dict(json.loads(path.read_text()))
