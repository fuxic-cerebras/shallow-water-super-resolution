"""Paired coarse/fine dataset generation (tasks D-02, D-03).

    python -m swe_sr.data.generate --config configs/data/primary_32x128.yaml

The central rule this module implements is D002/`CLAUDE.md`: the coarse and fine states are
produced by two independent solver runs of the *same analytic initial condition* over the
*same physical domain*, with only the spatial resolution differing. No coarse array is ever
obtained by resizing a fine array. Within a pair both runs use the fine grid's time step
(D003), so their saved-time arrays are bit-identical.

Nothing here imports matplotlib; generation is headless by contract.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from swe_sr.data.manifest import (
    DatasetManifest,
    TrajectoryRecord,
    coordinate_hash,
    git_commit,
)
from swe_sr.data.registry import (
    InitialConditionRegistry,
    build_registry,
    load_registry,
)
from swe_sr.data.storage import write_trajectory
from swe_sr.solver.config import ResolutionPair, SolverConfig
from swe_sr.solver.diagnostics import assert_admissible
from swe_sr.solver.runner import sample_schedule, solve

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GenerationConfig:
    """One dataset release specification, loaded from `configs/data/*.yaml`."""

    dataset_id: str
    pair_id: str
    resolution_family: str
    coarse_nodes: int
    fine_nodes: int
    discard_steps: int
    sample_stride: int
    snapshot_count: int
    output_root: str = "data"
    registry_path: str = "data/registries/ic_registry_v1.json"
    diagnostic_stride: int = 8
    # Restricting to a few trajectories makes a smoke dataset; None means the full release.
    trajectory_limit: int | None = None
    physical: dict[str, float] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> GenerationConfig:
        payload = yaml.safe_load(path.read_text())
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(payload) - known
        if unknown:
            # Fail loudly: a typo'd key that is silently ignored would produce a dataset
            # that does not match its own config.
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        return cls(**payload)

    @property
    def pair(self) -> ResolutionPair:
        return ResolutionPair(
            pair_id=self.pair_id,
            coarse_nodes=self.coarse_nodes,
            fine_nodes=self.fine_nodes,
            physical=dict(self.physical),
        )

    @property
    def sample_steps(self) -> np.ndarray:
        return sample_schedule(self.discard_steps, self.sample_stride, self.snapshot_count)


def _configs_for(config: GenerationConfig) -> tuple[SolverConfig, SolverConfig]:
    """Resolve the coarse and fine solver configs, sharing the fine time step (D003)."""
    pair = config.pair
    fine = pair.fine_config()
    coarse = pair.coarse_config()
    if coarse.dt != fine.dt:
        raise AssertionError("coarse and fine time steps must be identical within a pair")
    return coarse, fine


def generate_dataset(
    config: GenerationConfig,
    *,
    registry: InitialConditionRegistry | None = None,
    output_root: Path | None = None,
    verbose: bool = True,
) -> DatasetManifest:
    """Generate a full paired dataset and write its immutable manifest."""
    root = output_root or (REPO_ROOT / config.output_root)
    raw_dir = root / "raw" / config.dataset_id
    trajectories_dir = raw_dir / "trajectories"

    if registry is None:
        registry_path = REPO_ROOT / config.registry_path
        registry = load_registry(registry_path) if registry_path.exists() else build_registry()

    coarse_config, fine_config = _configs_for(config)
    steps = config.sample_steps
    entries = list(registry)
    if config.trajectory_limit is not None:
        # Take from the head of each split so a smoke dataset still exercises all three.
        selected: list[Any] = []
        for split in ("train", "validation", "test"):
            members = [e for e in entries if e.split == split]
            selected.extend(members[: config.trajectory_limit])
        entries = selected

    records: list[TrajectoryRecord] = []
    started = time.perf_counter()

    for index, entry in enumerate(entries, start=1):
        # Two independent integrations of one analytic specification (D002).
        coarse_result = solve(
            coarse_config,
            entry.initial_condition,
            sample_steps=steps,
            diagnostic_stride=config.diagnostic_stride,
        )
        fine_result = solve(
            fine_config,
            entry.initial_condition,
            sample_steps=steps,
            diagnostic_stride=config.diagnostic_stride,
        )

        # Fail before writing: an inadmissible trajectory must not enter a release.
        assert_admissible(coarse_result.diagnostics)
        assert_admissible(fine_result.diagnostics)
        if not np.array_equal(coarse_result.times, fine_result.times):
            raise AssertionError(
                f"trajectory {entry.trajectory_id}: saved times differ between resolutions"
            )

        path = trajectories_dir / f"{entry.trajectory_id}.h5"
        array_records = write_trajectory(
            path,
            coarse_fields=coarse_result.fields,
            fine_fields=fine_result.fields,
            times=fine_result.times,
            sample_steps=steps,
            coarse_x=coarse_config.x,
            coarse_y=coarse_config.y,
            fine_x=fine_config.x,
            fine_y=fine_config.y,
            metadata={
                "trajectory_id": entry.trajectory_id,
                "seed": entry.seed,
                "split": entry.split,
                "dataset_id": config.dataset_id,
                "pair_id": config.pair_id,
                "shared_time_step": fine_config.dt,
                "depth": fine_config.depth,
                "gravity": fine_config.gravity,
                "channel_order": "eta,u,v",
                "axis_order": "time,channel,y,x",
                "staggering": "arakawa_c_raw",
                "initial_condition": json.dumps(entry.initial_condition.to_dict()),
            },
        )

        records.append(
            TrajectoryRecord(
                trajectory_id=entry.trajectory_id,
                seed=entry.seed,
                split=entry.split,
                initial_condition=entry.initial_condition.to_dict(),
                arrays=array_records,
                diagnostics={
                    "coarse": coarse_result.diagnostics.to_dict(),
                    "fine": fine_result.diagnostics.to_dict(),
                },
                resolvability={
                    "coarse": entry.initial_condition.resolvability(coarse_config),
                    "fine": entry.initial_condition.resolvability(fine_config),
                },
                relative_path=str(path.relative_to(raw_dir)),
            )
        )

        if verbose:
            elapsed = time.perf_counter() - started
            print(
                f"[{index}/{len(entries)}] {entry.split:<10} seed={entry.seed:<3} "
                f"mass_drift={coarse_result.diagnostics.relative_mass_drift:.2e}/"
                f"{fine_result.diagnostics.relative_mass_drift:.2e} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    pair = config.pair
    manifest = DatasetManifest(
        dataset_id=config.dataset_id,
        pair_id=config.pair_id,
        resolution_family=config.resolution_family,
        coarse_nodes=config.coarse_nodes,
        fine_nodes=config.fine_nodes,
        shape_factor=pair.shape_factor,
        spacing_ratio=pair.spacing_ratio,
        # Both endpoints are included, so these are x4 in node count but not in spacing.
        endpoint_convention="both_endpoints_included",
        coarse_config=coarse_config.to_dict(),
        fine_config=fine_config.to_dict(),
        shared_time_step=fine_config.dt,
        sample_steps=[int(s) for s in steps],
        saved_times=[float(t) for t in (steps * fine_config.dt)],
        coarse_coordinate_hash=coordinate_hash(coarse_config.x, coarse_config.y),
        fine_coordinate_hash=coordinate_hash(fine_config.x, fine_config.y),
        ic_registry_version=registry.version,
        ic_registry_hash=registry.registry_hash,
        git_commit=git_commit(),
        trajectories=records,
    )
    manifest.write(raw_dir / "manifest.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="path to a data config YAML")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="override the dataset root (defaults to the config's output_root)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = GenerationConfig.from_yaml(args.config)
    manifest = generate_dataset(
        config, output_root=args.output_root, verbose=not args.quiet
    )
    print(f"\ndataset_id     : {manifest.dataset_id}")
    print(f"trajectories   : {len(manifest.trajectories)}")
    print(f"registry_hash  : {manifest.ic_registry_hash}")
    print(f"manifest_hash  : {manifest.manifest_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
