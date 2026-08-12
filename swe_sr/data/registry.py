"""The immutable initial-condition registry (task D-01).

`docs/DATASET.md` requires one `ic_registry_v1.json` holding 48 analytic IC
specifications, seeds, stable trajectory UUIDs, and split membership, consumed by *both*
resolution pairs. D008 makes this the single shared artifact between the primary and
backup releases: they share IC identities and splits, but nothing else.

Two properties matter most and are enforced here rather than left to convention:

- **The split is fixed before simulation** (D004). Splits are assigned from seed ID, not
  drawn at load time, so no shuffling can leak a trajectory across a boundary.
- **Trajectory IDs are stable and content-derived.** A UUID5 over the registry version and
  seed means the same seed always yields the same ID, on any machine, without persisting a
  counter. Reordering the file cannot renumber a trajectory.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from swe_sr.solver.config import SolverConfig
from swe_sr.solver.initial_conditions import (
    GaussianBumpsIC,
    initial_condition_from_dict,
    sample_gaussian_bumps_ic,
)

REGISTRY_VERSION = "ic_registry_v1"
REGISTRY_SCHEMA_VERSION = 1

Split = Literal["train", "validation", "test"]

# docs/DATASET.md: 32 train, 8 validation, 8 test, assigned by contiguous seed ID.
SPLIT_LAYOUT: tuple[tuple[Split, int, int], ...] = (
    ("train", 0, 32),
    ("validation", 32, 40),
    ("test", 40, 48),
)
TOTAL_TRAJECTORIES = 48

# A fixed namespace, so trajectory UUIDs are reproducible across machines and runs.
# Deriving it from the registry version keeps a future v2 registry from colliding with v1.
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"swe-sr/{REGISTRY_VERSION}")


def split_for_seed(seed: int) -> Split:
    """Map a seed ID to its split. Pure function of the seed, so it cannot drift."""
    for split, start, stop in SPLIT_LAYOUT:
        if start <= seed < stop:
            return split
    raise ValueError(f"seed {seed} is outside the registry range 0-{TOTAL_TRAJECTORIES - 1}")


def trajectory_id_for_seed(seed: int) -> str:
    """Stable, content-derived trajectory UUID."""
    return str(uuid.uuid5(_UUID_NAMESPACE, f"trajectory/{seed}"))


@dataclass(frozen=True)
class TrajectoryEntry:
    """One registry row: a seed, its stable ID, its split, and its analytic IC."""

    seed: int
    trajectory_id: str
    split: Split
    initial_condition: GaussianBumpsIC

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "trajectory_id": self.trajectory_id,
            "split": self.split,
            "initial_condition": self.initial_condition.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TrajectoryEntry:
        initial_condition = initial_condition_from_dict(dict(payload["initial_condition"]))
        if not isinstance(initial_condition, GaussianBumpsIC):
            raise ValueError(
                "registry trajectories must be from the gaussian_bumps family; "
                f"got {type(initial_condition).__name__}"
            )
        return cls(
            seed=int(payload["seed"]),
            trajectory_id=str(payload["trajectory_id"]),
            split=payload["split"],
            initial_condition=initial_condition,
        )


class RegistryValidationError(ValueError):
    """The registry violates an invariant that must hold before any data is generated."""


@dataclass(frozen=True)
class InitialConditionRegistry:
    """An immutable, ordered collection of trajectory specifications."""

    entries: tuple[TrajectoryEntry, ...]
    version: str = REGISTRY_VERSION
    schema_version: int = REGISTRY_SCHEMA_VERSION

    def __iter__(self) -> Iterator[TrajectoryEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def by_split(self, split: Split) -> tuple[TrajectoryEntry, ...]:
        return tuple(entry for entry in self.entries if entry.split == split)

    def by_seed(self, seed: int) -> TrajectoryEntry:
        for entry in self.entries:
            if entry.seed == seed:
                return entry
        raise KeyError(f"no registry entry for seed {seed}")

    # -- Serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "schema_version": self.schema_version,
            "total_trajectories": len(self.entries),
            "split_layout": {
                split: {"start_seed": start, "stop_seed": stop, "count": stop - start}
                for split, start, stop in SPLIT_LAYOUT
            },
            "trajectories": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self) -> str:
        """Canonical JSON: sorted keys and fixed separators, so the hash is stable."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def registry_hash(self) -> str:
        """SHA-256 of the canonical form, recorded in every dataset manifest."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def write(self, path: Path) -> str:
        """Write the registry, refusing to overwrite an existing one.

        `docs/AGENT_WORKFLOW.md` requires immutable artifacts: "Never overwrite a dataset
        or run; use immutable unique IDs." Re-generating a registry in place would silently
        invalidate every manifest that references its hash.
        """
        if path.exists():
            existing = load_registry(path)
            if existing.registry_hash == self.registry_hash:
                return self.registry_hash
            raise RegistryValidationError(
                f"{path} already exists with a different hash "
                f"({existing.registry_hash[:12]} != {self.registry_hash[:12]}). "
                "The IC registry is immutable; write a new version instead."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        # Human-readable on disk; the hash is always computed from the canonical form.
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return self.registry_hash

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InitialConditionRegistry:
        return cls(
            entries=tuple(TrajectoryEntry.from_dict(row) for row in payload["trajectories"]),
            version=str(payload["version"]),
            schema_version=int(payload["schema_version"]),
        )


def load_registry(path: Path) -> InitialConditionRegistry:
    registry = InitialConditionRegistry.from_dict(json.loads(path.read_text()))
    validate_registry(registry)
    return registry


def build_registry(
    reference_config: SolverConfig | None = None,
    *,
    total: int = TOTAL_TRAJECTORIES,
) -> InitialConditionRegistry:
    """Draw all trajectory specifications deterministically from their seed IDs.

    `reference_config` supplies only the domain and resting depth used for admissibility
    checks; the accepted specifications are analytic and are later evaluated on each grid
    independently (D002). It defaults to the primary pair's fine grid.
    """
    config = reference_config or SolverConfig(n_x=128, n_y=128)
    entries = tuple(
        TrajectoryEntry(
            seed=seed,
            trajectory_id=trajectory_id_for_seed(seed),
            split=split_for_seed(seed),
            initial_condition=sample_gaussian_bumps_ic(seed, config),
        )
        for seed in range(total)
    )
    registry = InitialConditionRegistry(entries=entries)
    validate_registry(registry)
    return registry


def validate_registry(registry: InitialConditionRegistry) -> None:
    """Enforce every invariant that must hold before a single trajectory is simulated.

    Runs on both build and load, so a hand-edited registry cannot slip through.
    """
    problems: list[str] = []

    if len(registry) != TOTAL_TRAJECTORIES:
        problems.append(f"expected {TOTAL_TRAJECTORIES} trajectories, found {len(registry)}")

    seeds = [entry.seed for entry in registry]
    if len(set(seeds)) != len(seeds):
        problems.append("seed IDs are not unique")

    ids = [entry.trajectory_id for entry in registry]
    if len(set(ids)) != len(ids):
        problems.append("trajectory IDs are not unique")

    for entry in registry:
        expected_id = trajectory_id_for_seed(entry.seed)
        if entry.trajectory_id != expected_id:
            problems.append(
                f"seed {entry.seed} has trajectory ID {entry.trajectory_id}, "
                f"expected the content-derived {expected_id}"
            )
        expected_split = split_for_seed(entry.seed)
        if entry.split != expected_split:
            problems.append(
                f"seed {entry.seed} is labelled {entry.split}, expected {expected_split}"
            )

    # The property that matters most: no trajectory and no seed may appear in more than
    # one split (D004, docs/VALIDATION.md).
    for split, start, stop in SPLIT_LAYOUT:
        members = registry.by_split(split)
        if len(members) != stop - start:
            problems.append(
                f"split {split} has {len(members)} trajectories, expected {stop - start}"
            )

    split_ids: dict[Split, set[str]] = {
        split: {entry.trajectory_id for entry in registry.by_split(split)}
        for split, _, _ in SPLIT_LAYOUT
    }
    splits = [split for split, _, _ in SPLIT_LAYOUT]
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                problems.append(f"splits {left} and {right} share {len(overlap)} trajectory ID(s)")

    if problems:
        raise RegistryValidationError("; ".join(problems))
