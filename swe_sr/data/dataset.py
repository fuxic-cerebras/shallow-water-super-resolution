"""The processed layer: paired full-frame samples ready for training (D-04).

Turns a released raw dataset into normalized, cell-centered `(coarse, fine)` tensor pairs.
This is the only place normalization is applied -- `CLAUDE.md` requires stored raw data to
keep physical units and normalization to happen in the loader.

Each sample is one snapshot pair, drawn as an independent full-frame example. Time is not a
model input (D016): the frozen contract is `[B, 3, H, W] -> [B, 3, 4H, 4W]` at a single
instant. Time metadata is carried on the sample only for pairing, aggregation, and
provenance.

`docs/DATASET.md` requires full-frame pairs rather than patches, so there is deliberately no
cropping here. If patch training is ever added it must map patch bounds by physical
coordinate and be recorded as a new decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from swe_sr.data.manifest import DatasetManifest, load_manifest
from swe_sr.data.normalization import Normalization, check_pair_id
from swe_sr.data.processing import AugmentationPolicy, destagger
from swe_sr.data.storage import COARSE_ARRAY, FINE_ARRAY, read_frame, read_times

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SampleIndex:
    """Locates one snapshot pair: which trajectory file, and which frame within it."""

    trajectory_id: str
    seed: int
    split: str
    path: Path
    frame: int
    time: float


class PairedSnapshotDataset:
    """Paired coarse/fine snapshots from one released dataset and split.

    Implements the `torch.utils.data.Dataset` protocol (`__len__`, `__getitem__`) without
    importing torch at module scope, so the data layer stays usable -- and testable -- in a
    plain numpy environment.

    Args:
        manifest: the released dataset manifest.
        raw_dir: directory containing that manifest and its `trajectories/`.
        split: which split to expose. Never mixes splits; the trajectory-level split is the
            project's primary defence against leakage (D004).
        normalization: statistics fitted on the *training* split of this same pair. Required,
            and checked against the manifest's pair ID, because silently training on
            mis-scaled inputs is a failure that produces plausible-looking numbers.
        augmentation: transforms to apply. Empty by default (D018): the documented
            reflections and transpose are geometrically exact on cell-centered fields but are
            not symmetries of this rotating solver.
        seed: seeds the per-sample augmentation draw. Augmentation is keyed on the sample
            index too, so a given index yields the same transform in every epoch and the
            dataset stays reproducible.
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        raw_dir: Path,
        *,
        split: str,
        normalization: Normalization,
        augmentation: AugmentationPolicy | None = None,
        seed: int = 0,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        check_pair_id(normalization, manifest.pair_id)

        self.manifest = manifest
        self.raw_dir = Path(raw_dir)
        self.split = split
        self.normalization = normalization
        self.augmentation = augmentation or AugmentationPolicy()
        self.seed = seed

        records = manifest.by_split(split)
        if not records:
            raise ValueError(f"manifest {manifest.dataset_id} has no {split} trajectories")

        self._samples: list[SampleIndex] = []
        for record in records:
            path = self.raw_dir / record.relative_path
            times = read_times(path)
            for frame, time in enumerate(times):
                self._samples.append(
                    SampleIndex(
                        trajectory_id=record.trajectory_id,
                        seed=record.seed,
                        split=split,
                        path=path,
                        frame=frame,
                        time=float(time),
                    )
                )

    # -- Dataset protocol ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]

        # Single-frame chunked reads, so an epoch never materializes a whole trajectory.
        coarse_raw = read_frame(sample.path, COARSE_ARRAY, sample.frame)
        fine_raw = read_frame(sample.path, FINE_ARRAY, sample.frame)

        coarse = self.normalization.apply(destagger(coarse_raw))
        fine = self.normalization.apply(destagger(fine_raw))

        transform = "identity"
        if self.augmentation.enabled:
            # Keyed on (seed, index) rather than a shared stream: the transform is then a
            # pure function of the index, so shuffling, multi-worker loading, and resuming
            # cannot change which transform a sample receives.
            rng = np.random.default_rng((self.seed, index))
            transform = self.augmentation.draw(rng)
            coarse, fine = self.augmentation.apply(transform, coarse, fine)

        return {
            "coarse": self._to_tensor(coarse),
            "fine": self._to_tensor(fine),
            "trajectory_id": sample.trajectory_id,
            "seed": sample.seed,
            "frame": sample.frame,
            "time": sample.time,
            "augmentation": transform,
        }

    @staticmethod
    def _to_tensor(array: np.ndarray) -> torch.Tensor:
        import torch

        # float32 is the training dtype; statistics were accumulated in float64.
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    # -- Introspection ---------------------------------------------------------

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.trajectory_id for s in self._samples))

    @property
    def sample_index(self) -> tuple[SampleIndex, ...]:
        return tuple(self._samples)

    def frames_per_trajectory(self) -> int:
        counts = {
            trajectory: sum(1 for s in self._samples if s.trajectory_id == trajectory)
            for trajectory in self.trajectory_ids
        }
        unique = set(counts.values())
        if len(unique) != 1:
            raise ValueError(f"trajectories have differing frame counts: {sorted(unique)}")
        return unique.pop()


def load_split(
    manifest_path: Path,
    *,
    split: str,
    normalization: Normalization,
    augmentation: AugmentationPolicy | None = None,
    seed: int = 0,
) -> PairedSnapshotDataset:
    """Convenience loader from a manifest path, as the training entry point uses."""
    manifest_path = Path(manifest_path)
    manifest = load_manifest(manifest_path)
    return PairedSnapshotDataset(
        manifest,
        manifest_path.parent,
        split=split,
        normalization=normalization,
        augmentation=augmentation,
        seed=seed,
    )


def assert_splits_disjoint(*datasets: PairedSnapshotDataset) -> None:
    """Fail if any trajectory appears in more than one of these datasets.

    Cheap enough to call at the start of every training run, and it catches the one error
    that would invalidate every number the run produces.
    """
    seen: dict[str, str] = {}
    for dataset in datasets:
        for trajectory in dataset.trajectory_ids:
            if trajectory in seen and seen[trajectory] != dataset.split:
                raise ValueError(
                    f"trajectory {trajectory} appears in both {seen[trajectory]!r} and "
                    f"{dataset.split!r}; the split must be trajectory-level (D004)"
                )
            seen[trajectory] = dataset.split
