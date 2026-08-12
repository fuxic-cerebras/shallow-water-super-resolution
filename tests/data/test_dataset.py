"""The processed-layer dataset: paired, destaggered, normalized samples (D-04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swe_sr.data.dataset import PairedSnapshotDataset, assert_splits_disjoint, load_split
from swe_sr.data.generate import GenerationConfig, generate_dataset
from swe_sr.data.manifest import DatasetManifest
from swe_sr.data.normalization import Normalization, NormalizationError, fit_from_manifest
from swe_sr.data.processing import AugmentationPolicy, destagger, reflect_x
from swe_sr.data.registry import build_registry
from swe_sr.data.storage import COARSE_ARRAY, FINE_ARRAY, read_fields

CONFIG = GenerationConfig(
    dataset_id="test_ds",
    pair_id="swe_gaussian_32x128_v1",
    resolution_family="swe_gaussian_x4",
    coarse_nodes=32,
    fine_nodes=128,
    discard_steps=12,
    sample_stride=6,
    snapshot_count=4,
    diagnostic_stride=6,
    trajectory_limit=2,
)


@pytest.fixture(scope="module")
def released(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[DatasetManifest, Path, Normalization]:
    root = tmp_path_factory.mktemp("ds")
    manifest = generate_dataset(CONFIG, registry=build_registry(), output_root=root, verbose=False)
    raw_dir = root / "raw" / CONFIG.dataset_id
    return manifest, raw_dir, fit_from_manifest(manifest, raw_dir, split="train")


def _dataset(
    released: tuple[DatasetManifest, Path, Normalization],
    split: str = "train",
    augmentation: AugmentationPolicy | None = None,
) -> PairedSnapshotDataset:
    manifest, raw_dir, normalization = released
    return PairedSnapshotDataset(
        manifest, raw_dir, split=split, normalization=normalization, augmentation=augmentation
    )


# -- Shapes and the frozen contract ---------------------------------------------------


def test_samples_match_the_frozen_batch_contract(released) -> None:  # type: ignore[no-untyped-def]
    """D016: `[3, 32, 32]` in, `[3, 128, 128]` out, channels `[eta, u, v]`."""
    dataset = _dataset(released)
    sample = dataset[0]
    assert tuple(sample["coarse"].shape) == (3, 32, 32)
    assert tuple(sample["fine"].shape) == (3, 128, 128)
    assert sample["coarse"].dtype.is_floating_point
    assert sample["fine"].dtype.is_floating_point


def test_length_is_trajectories_times_frames(released) -> None:  # type: ignore[no-untyped-def]
    dataset = _dataset(released)
    assert len(dataset) == len(dataset.trajectory_ids) * CONFIG.snapshot_count
    assert dataset.frames_per_trajectory() == CONFIG.snapshot_count


def test_each_snapshot_is_an_independent_sample(released) -> None:  # type: ignore[no-untyped-def]
    """Time is not a model input (D016); snapshots are separate samples carrying time only
    as metadata for pairing, aggregation, and provenance."""
    dataset = _dataset(released)
    first, second = dataset[0], dataset[1]
    assert first["trajectory_id"] == second["trajectory_id"]
    assert first["frame"] != second["frame"]
    assert first["time"] != second["time"]
    assert not np.array_equal(first["fine"].numpy(), second["fine"].numpy())


# -- Destaggering and normalization are actually applied ------------------------------


def test_samples_are_destaggered_then_normalized(released) -> None:  # type: ignore[no-untyped-def]
    """Reproduce one sample by hand from the raw arrays, to pin the exact pipeline order."""
    normalization = released[2]
    dataset = _dataset(released)
    sample = dataset[0]
    index = dataset.sample_index[0]

    raw_fine = read_fields(index.path, FINE_ARRAY)[index.frame]
    expected = normalization.apply(destagger(raw_fine))
    np.testing.assert_allclose(sample["fine"].numpy(), expected.astype(np.float32), rtol=0, atol=0)

    raw_coarse = read_fields(index.path, COARSE_ARRAY)[index.frame]
    expected_coarse = normalization.apply(destagger(raw_coarse))
    np.testing.assert_allclose(
        sample["coarse"].numpy(), expected_coarse.astype(np.float32), rtol=0, atol=0
    )


def test_the_same_statistics_apply_to_both_members(released) -> None:  # type: ignore[no-untyped-def]
    """docs/DATASET.md: a pair's statistics apply to both its LR and HR fields.

    Evidence: normalized coarse and fine channel means land close together, which would not
    happen if each member were standardized separately.
    """
    dataset = _dataset(released)
    sample = dataset[0]
    coarse_means = sample["coarse"].numpy().mean(axis=(1, 2))
    fine_means = sample["fine"].numpy().mean(axis=(1, 2))
    np.testing.assert_allclose(coarse_means, fine_means, atol=0.35)


def test_raw_data_on_disk_stays_in_physical_units(released) -> None:  # type: ignore[no-untyped-def]
    """CLAUDE.md: stored raw data keeps physical units; normalization happens in the loader."""
    manifest, raw_dir, _ = released
    raw = read_fields(raw_dir / manifest.trajectories[0].relative_path, FINE_ARRAY)
    # eta is a metres-scale elevation, not a standardized quantity.
    assert 0.01 < float(np.abs(raw[:, 0]).max()) < 10.0


# -- Split integrity -------------------------------------------------------------------


def test_a_dataset_exposes_only_its_own_split(released) -> None:  # type: ignore[no-untyped-def]
    manifest, _, _ = released
    for split in ("train", "validation", "test"):
        dataset = _dataset(released, split=split)
        expected = {r.trajectory_id for r in manifest.by_split(split)}
        assert set(dataset.trajectory_ids) == expected
        for index in dataset.sample_index:
            assert index.split == split


def test_splits_are_disjoint_across_datasets(released) -> None:  # type: ignore[no-untyped-def]
    """D004, checked at the loader level so a training run can assert it cheaply."""
    train = _dataset(released, "train")
    validation = _dataset(released, "validation")
    test = _dataset(released, "test")
    assert_splits_disjoint(train, validation, test)

    train_ids = set(train.trajectory_ids)
    assert not train_ids & set(validation.trajectory_ids)
    assert not train_ids & set(test.trajectory_ids)


def test_overlapping_splits_are_detected(released) -> None:  # type: ignore[no-untyped-def]
    """The negative case: the guard must actually fire, not just pass on good input."""
    train = _dataset(released, "train")
    duplicate = _dataset(released, "train")
    duplicate.split = "validation"  # pretend a train trajectory was labelled validation
    with pytest.raises(ValueError, match="trajectory-level"):
        assert_splits_disjoint(train, duplicate)


def test_unknown_split_is_rejected(released) -> None:  # type: ignore[no-untyped-def]
    manifest, raw_dir, normalization = released
    with pytest.raises(ValueError, match="split must be one of"):
        PairedSnapshotDataset(manifest, raw_dir, split="holdout", normalization=normalization)


def test_wrong_pair_normalization_is_refused_at_construction(released) -> None:  # type: ignore[no-untyped-def]
    """Fail fast: catching this at load time beats discovering mis-scaled training later."""
    manifest, raw_dir, normalization = released
    from dataclasses import replace

    foreign = replace(normalization, pair_id="swe_gaussian_64x256_v1")
    with pytest.raises(NormalizationError, match="never shared across pair IDs"):
        PairedSnapshotDataset(manifest, raw_dir, split="train", normalization=foreign)


# -- Augmentation ----------------------------------------------------------------------


def test_augmentation_is_off_by_default(released) -> None:  # type: ignore[no-untyped-def]
    """D018: the documented transforms are not symmetries of this rotating solver."""
    dataset = _dataset(released)
    assert not dataset.augmentation.enabled
    assert all(dataset[i]["augmentation"] == "identity" for i in range(len(dataset)))


def test_augmentation_applies_the_same_transform_to_both_members(released) -> None:  # type: ignore[no-untyped-def]
    """A pair must stay consistent; different transforms per member would destroy it."""
    manifest, raw_dir, normalization = released
    dataset = PairedSnapshotDataset(
        manifest,
        raw_dir,
        split="train",
        normalization=normalization,
        augmentation=AugmentationPolicy(names=("reflect_x",)),
        seed=1,
    )
    plain = _dataset(released)

    reflected = [i for i in range(len(dataset)) if dataset[i]["augmentation"] == "reflect_x"]
    assert reflected, "expected at least one reflected sample"
    index = reflected[0]
    np.testing.assert_allclose(
        dataset[index]["coarse"].numpy(), reflect_x(plain[index]["coarse"].numpy()), atol=1e-6
    )
    np.testing.assert_allclose(
        dataset[index]["fine"].numpy(), reflect_x(plain[index]["fine"].numpy()), atol=1e-6
    )


def test_augmentation_choice_is_reproducible_per_index(released) -> None:  # type: ignore[no-untyped-def]
    """Keying the draw on (seed, index) keeps shuffling and multi-worker loading reproducible."""
    manifest, raw_dir, normalization = released
    policy = AugmentationPolicy(names=("reflect_x", "reflect_y", "transpose"))
    first = PairedSnapshotDataset(
        manifest, raw_dir, split="train", normalization=normalization, augmentation=policy, seed=42
    )
    second = PairedSnapshotDataset(
        manifest, raw_dir, split="train", normalization=normalization, augmentation=policy, seed=42
    )
    chosen_first = [first[i]["augmentation"] for i in range(len(first))]
    chosen_second = [second[i]["augmentation"] for i in range(len(second))]
    assert chosen_first == chosen_second
    # Re-reading one index twice must also agree.
    assert first[3]["augmentation"] == first[3]["augmentation"]


def test_transpose_augmentation_keeps_square_shapes(released) -> None:  # type: ignore[no-untyped-def]
    """The transpose is only shape-safe because both grids are square."""
    manifest, raw_dir, normalization = released
    dataset = PairedSnapshotDataset(
        manifest,
        raw_dir,
        split="train",
        normalization=normalization,
        augmentation=AugmentationPolicy(names=("transpose",)),
        seed=3,
    )
    for index in range(len(dataset)):
        sample = dataset[index]
        assert tuple(sample["coarse"].shape) == (3, 32, 32)
        assert tuple(sample["fine"].shape) == (3, 128, 128)


# -- Loader entry point ----------------------------------------------------------------


def test_load_split_reads_from_a_manifest_path(released) -> None:  # type: ignore[no-untyped-def]
    manifest, raw_dir, normalization = released
    dataset = load_split(raw_dir / "manifest.json", split="test", normalization=normalization)
    assert dataset.split == "test"
    assert len(dataset) > 0
    assert set(dataset.trajectory_ids) == {r.trajectory_id for r in manifest.by_split("test")}
