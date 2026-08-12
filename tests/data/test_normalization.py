"""Train-split-only normalization (D-04).

`CLAUDE.md` requires fitting statistics on the training split only, persisting them, and
reusing them everywhere else. `docs/VALIDATION.md` requires that recomputed statistics match
the manifest, and names "wrong normalization pair ID" as a negative test. Both are here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swe_sr.data.generate import GenerationConfig, generate_dataset
from swe_sr.data.manifest import DatasetManifest
from swe_sr.data.normalization import (
    SIGMA_FLOOR,
    Normalization,
    NormalizationError,
    check_pair_id,
    fit_from_manifest,
    fit_normalization,
)
from swe_sr.data.processing import destagger
from swe_sr.data.registry import build_registry
from swe_sr.data.storage import FINE_ARRAY, read_fields

CONFIG = GenerationConfig(
    dataset_id="test_norm",
    pair_id="swe_gaussian_32x128_v1",
    resolution_family="swe_gaussian_x4",
    coarse_nodes=32,
    fine_nodes=128,
    discard_steps=12,
    sample_stride=6,
    snapshot_count=3,
    diagnostic_stride=6,
    trajectory_limit=2,
)


@pytest.fixture(scope="module")
def released(tmp_path_factory: pytest.TempPathFactory) -> tuple[DatasetManifest, Path]:
    root = tmp_path_factory.mktemp("norm")
    manifest = generate_dataset(CONFIG, registry=build_registry(), output_root=root, verbose=False)
    return manifest, root / "raw" / CONFIG.dataset_id


# -- Correctness of the statistics -----------------------------------------------------


def test_statistics_match_a_direct_computation() -> None:
    """Streaming accumulation must agree with a straightforward numpy computation."""
    rng = np.random.default_rng(3)
    batches = [
        rng.normal(loc=[1.0, -2.0, 0.5], scale=1.0, size=(4, 8, 8, 3)).transpose(0, 3, 1, 2)
        for _ in range(3)
    ]
    normalization = fit_normalization(batches, pair_id="p", already_destaggered=True)

    stacked = np.concatenate(batches, axis=0)
    for channel in range(3):
        values = stacked[:, channel].ravel()
        assert normalization.channels[channel].mean == pytest.approx(values.mean(), rel=1e-12)
        assert normalization.channels[channel].std == pytest.approx(values.std(), rel=1e-10)
        assert normalization.channels[channel].count == values.size


def test_streaming_in_batches_matches_one_big_batch() -> None:
    """Chunking is what keeps memory flat over a multi-GiB release; it must not change results."""
    rng = np.random.default_rng(11)
    trajectory = rng.normal(size=(12, 3, 8, 8))
    whole = fit_normalization([trajectory], pair_id="p", already_destaggered=True)
    chunked = fit_normalization(
        [trajectory[0:5], trajectory[5:9], trajectory[9:12]], pair_id="p", already_destaggered=True
    )
    np.testing.assert_allclose(whole.mean, chunked.mean, rtol=1e-12)
    np.testing.assert_allclose(whole.scale, chunked.scale, rtol=1e-12)


def test_normalization_round_trips_each_channel() -> None:
    """docs/VALIDATION.md: normalization round-trips each channel."""
    rng = np.random.default_rng(5)
    fields = rng.normal(loc=[3.0, -1.0, 0.25], scale=[2.0, 0.5, 0.1], size=(6, 16, 16, 3))
    fields = fields.transpose(0, 3, 1, 2)
    normalization = fit_normalization([fields], pair_id="p", already_destaggered=True)

    normalized = normalization.apply(fields)
    np.testing.assert_allclose(normalization.invert(normalized), fields, rtol=1e-12, atol=1e-12)


def test_normalized_channels_are_standardized() -> None:
    """Applied to the data it was fitted on, each channel should be zero-mean unit-variance."""
    rng = np.random.default_rng(9)
    # Generate with the channel axis last so per-channel loc/scale broadcast, then move it
    # into storage position.
    fields = rng.normal(
        loc=[10.0, -4.0, 2.0], scale=[3.0, 1.0, 0.5], size=(8, 12, 12, 3)
    ).transpose(0, 3, 1, 2)
    normalization = fit_normalization([fields], pair_id="p", already_destaggered=True)
    normalized = normalization.apply(fields)
    for channel in range(3):
        assert normalized[:, channel].mean() == pytest.approx(0.0, abs=1e-10)
        assert normalized[:, channel].std() == pytest.approx(1.0, rel=1e-10)


def test_constant_channel_uses_the_sigma_floor() -> None:
    """A constant channel would divide by zero without the documented floor."""
    fields = np.zeros((4, 3, 8, 8))
    fields[:, 0] = 5.0  # eta constant, u and v identically zero
    normalization = fit_normalization([fields], pair_id="p", already_destaggered=True)
    assert normalization.channels[0].std == pytest.approx(0.0, abs=1e-12)
    assert normalization.channels[0].scale == SIGMA_FLOOR
    assert np.all(np.isfinite(normalization.apply(fields)))


def test_fitting_on_no_samples_is_an_explicit_error() -> None:
    with pytest.raises(NormalizationError, match="no train samples"):
        fit_normalization([], pair_id="p")


# -- The train-only rule ---------------------------------------------------------------


def test_statistics_are_fitted_on_the_train_split_only(
    released: tuple[DatasetManifest, Path],
) -> None:
    """The central leakage guard: validation and test data must not influence statistics."""
    manifest, raw_dir = released
    train_only = fit_from_manifest(manifest, raw_dir, split="train")
    assert train_only.split == "train"

    # Fitting on a different split must give different numbers; if it did not, the split
    # would not actually be restricting anything.
    test_only = fit_from_manifest(manifest, raw_dir, split="test")
    assert not np.allclose(train_only.mean, test_only.mean)


def test_statistics_come_from_the_fine_grid_destaggered_fields(
    released: tuple[DatasetManifest, Path],
) -> None:
    """Fitted on the processed representation the model consumes, not the raw staggered one.

    Destaggering averages adjacent faces, which reduces velocity variance, so fitting on raw
    arrays would leave the model's inputs systematically mis-scaled.
    """
    manifest, raw_dir = released
    fitted = fit_from_manifest(manifest, raw_dir, split="train")
    assert fitted.source == "fine_grid_destaggered"

    batches = [
        read_fields(raw_dir / record.relative_path, FINE_ARRAY)
        for record in manifest.by_split("train")
    ]
    expected = fit_normalization(
        [destagger(b) for b in batches], pair_id=manifest.pair_id, already_destaggered=True
    )
    np.testing.assert_allclose(fitted.mean, expected.mean, rtol=1e-12)
    np.testing.assert_allclose(fitted.scale, expected.scale, rtol=1e-12)

    # And confirm it genuinely differs from the raw-array fit, so the choice is not moot.
    raw_fit = fit_normalization(batches, pair_id=manifest.pair_id, already_destaggered=True)
    assert not np.allclose(fitted.scale, raw_fit.scale)


def test_recomputed_statistics_match_the_persisted_ones(
    released: tuple[DatasetManifest, Path], tmp_path: Path
) -> None:
    """docs/VALIDATION.md: recomputed training normalization must match the manifest."""
    manifest, raw_dir = released
    fitted = fit_from_manifest(manifest, raw_dir, split="train")

    path = tmp_path / "normalization.json"
    fitted.write(path)
    import json

    reloaded = Normalization.from_dict(json.loads(path.read_text()))

    np.testing.assert_allclose(reloaded.mean, fitted.mean, rtol=0, atol=0)
    np.testing.assert_allclose(reloaded.scale, fitted.scale, rtol=0, atol=0)
    # The manifest must carry the raw accumulators, so statistics can be re-derived and
    # audited rather than taken on trust.
    for channel in reloaded.channels:
        assert channel.count > 0
        assert np.isfinite(channel.total)
        assert np.isfinite(channel.total_squared)


def test_persisted_form_records_counts_sums_and_squared_sums(
    released: tuple[DatasetManifest, Path],
) -> None:
    manifest, raw_dir = released
    payload = fit_from_manifest(manifest, raw_dir, split="train").to_dict()
    assert payload["fitted_on_split"] == "train"
    assert payload["sigma_floor"] == SIGMA_FLOOR
    assert payload["channel_order"] == ["eta", "u", "v"]
    for name in ("eta", "u", "v"):
        assert set(payload["channels"][name]) >= {"count", "sum", "sum_squared", "mean", "std"}


# -- Negative test: wrong pair ID ------------------------------------------------------


def test_applying_another_pairs_statistics_is_refused(
    released: tuple[DatasetManifest, Path],
) -> None:
    """docs/VALIDATION.md names this negative test, and the failure it prevents is silent:
    the model would train happily on mis-scaled inputs and produce plausible numbers."""
    manifest, raw_dir = released
    fitted = fit_from_manifest(manifest, raw_dir, split="train")
    check_pair_id(fitted, manifest.pair_id)  # must not raise

    with pytest.raises(NormalizationError, match="never shared across pair IDs"):
        check_pair_id(fitted, "swe_gaussian_64x256_v1")


@pytest.mark.backup
def test_statistics_differ_between_pair_ids(tmp_path: Path) -> None:
    """D008 forbids sharing statistics; confirm the two pairs really do differ."""
    registry = build_registry()
    primary = generate_dataset(CONFIG, registry=registry, output_root=tmp_path / "a", verbose=False)
    backup_config = GenerationConfig(
        dataset_id="test_norm_backup",
        pair_id="swe_gaussian_64x256_v1",
        resolution_family="swe_gaussian_x4",
        coarse_nodes=64,
        fine_nodes=256,
        discard_steps=12,
        sample_stride=6,
        snapshot_count=2,
        diagnostic_stride=6,
        trajectory_limit=2,
    )
    backup = generate_dataset(
        backup_config, registry=registry, output_root=tmp_path / "b", verbose=False
    )

    primary_norm = fit_from_manifest(primary, tmp_path / "a" / "raw" / primary.dataset_id)
    backup_norm = fit_from_manifest(backup, tmp_path / "b" / "raw" / backup.dataset_id)
    assert primary_norm.pair_id != backup_norm.pair_id
    assert not np.allclose(primary_norm.scale, backup_norm.scale)
