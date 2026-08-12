"""Paired generation, storage, and manifests (tasks D-02, D-03).

These encode the G2 acceptance criterion from `docs/AGENT_WORKFLOW.md`: both pairs
independently integrate matching initial conditions over matching domains with exact
within-pair saved times. Every check recomputes from the stored arrays rather than trusting
what the generator reported, which is the standard the independent audit must also meet.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swe_sr.data.generate import GenerationConfig, generate_dataset
from swe_sr.data.manifest import (
    DatasetManifest,
    ManifestError,
    array_checksum,
    coordinate_hash,
    load_manifest,
)
from swe_sr.data.registry import build_registry
from swe_sr.data.storage import (
    read_coordinates,
    read_fields,
    read_frame,
    read_metadata,
    read_times,
    verify_trajectory_checksums,
    write_trajectory,
)

# Small but structurally complete: real cadence shape, few steps, one trajectory per split.
FAST_CONFIG = GenerationConfig(
    dataset_id="test_32x128",
    pair_id="swe_gaussian_32x128_v1",
    resolution_family="swe_gaussian_x4",
    coarse_nodes=32,
    fine_nodes=128,
    discard_steps=12,
    sample_stride=6,
    snapshot_count=4,
    diagnostic_stride=4,
    trajectory_limit=1,
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[DatasetManifest, Path]:
    root = tmp_path_factory.mktemp("dataset")
    manifest = generate_dataset(
        FAST_CONFIG, registry=build_registry(), output_root=root, verbose=False
    )
    return manifest, root / "raw" / FAST_CONFIG.dataset_id


# -- The core pair contract -----------------------------------------------------------


def test_within_pair_saved_times_are_bit_identical(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """D003 exists to make this exact, not approximate (docs/VALIDATION.md)."""
    manifest, raw_dir = generated
    expected = np.asarray(manifest.saved_times)
    for record in manifest.trajectories:
        stored = read_times(raw_dir / record.relative_path)
        np.testing.assert_array_equal(stored, expected)
        # Times must be exact integer multiples of the shared step.
        steps = np.asarray(manifest.sample_steps)
        np.testing.assert_array_equal(stored, steps * manifest.shared_time_step)


def test_coarse_and_fine_cover_the_same_physical_domain(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """docs/VALIDATION.md: LR and HR coordinates must cover the same physical domain.

    Endpoints must coincide *exactly*, since that is what makes endpoint-aligned
    interpolation well defined (D011).
    """
    manifest, raw_dir = generated
    for record in manifest.trajectories:
        coords = read_coordinates(raw_dir / record.relative_path)
        assert coords["coarse_x"][0] == coords["fine_x"][0]
        assert coords["coarse_x"][-1] == coords["fine_x"][-1]
        assert coords["coarse_y"][0] == coords["fine_y"][0]
        assert coords["coarse_y"][-1] == coords["fine_y"][-1]
        assert coords["coarse_x"].size == manifest.coarse_nodes
        assert coords["fine_x"].size == manifest.fine_nodes


def test_stored_coordinate_hashes_match_the_manifest(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """A coordinate-hash mismatch is one of the negative tests docs/VALIDATION.md names."""
    manifest, raw_dir = generated
    for record in manifest.trajectories:
        coords = read_coordinates(raw_dir / record.relative_path)
        assert (
            coordinate_hash(coords["coarse_x"], coords["coarse_y"])
            == manifest.coarse_coordinate_hash
        )
        assert coordinate_hash(coords["fine_x"], coords["fine_y"]) == manifest.fine_coordinate_hash


def test_shape_factor_is_four_but_spacing_ratio_is_not(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """docs/DATASET.md forbids describing these grids as exact fourfold meshes."""
    manifest, _ = generated
    assert manifest.shape_factor == 4
    assert manifest.spacing_ratio == pytest.approx(127 / 31)
    assert manifest.spacing_ratio != 4.0
    assert manifest.endpoint_convention == "both_endpoints_included"


def test_stored_shapes_follow_the_batch_contract(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """`[time, channel, y, x]` with three channels, at both resolutions (D016)."""
    manifest, raw_dir = generated
    frames = len(manifest.saved_times)
    for record in manifest.trajectories:
        path = raw_dir / record.relative_path
        assert read_fields(path, "lr").shape == (frames, 3, 32, 32)
        assert read_fields(path, "hr").shape == (frames, 3, 128, 128)
        assert read_metadata(path)["channel_order"] == "eta,u,v"
        assert read_metadata(path)["axis_order"] == "time,channel,y,x"


def test_coarse_initial_state_is_an_analytic_evaluation_not_a_resize(tmp_path: Path) -> None:
    """D002, as a decisive check rather than a suggestive one.

    With no spin-up discard, frame 0 is the initial state. An independently evaluated
    coarse field equals the analytic specification on coarse coordinates to the last bit;
    any resampling of the fine field would carry the fine grid's discretization and could
    not. Bit equality is the whole point, so this is `assert_array_equal`, not `allclose`.

    Uses its own zero-discard dataset because the shipped configs deliberately discard
    spin-up, which would leave frame 0 already evolved.
    """
    from swe_sr.solver.config import SolverConfig
    from swe_sr.solver.initial_conditions import initial_condition_from_dict

    config = GenerationConfig(
        dataset_id="test_t0",
        pair_id="swe_gaussian_32x128_v1",
        resolution_family="swe_gaussian_x4",
        coarse_nodes=32,
        fine_nodes=128,
        discard_steps=0,
        sample_stride=4,
        snapshot_count=2,
        diagnostic_stride=2,
        trajectory_limit=1,
    )
    manifest = generate_dataset(
        config, registry=build_registry(), output_root=tmp_path, verbose=False
    )
    raw_dir = tmp_path / "raw" / config.dataset_id
    record = manifest.trajectories[0]
    specification = initial_condition_from_dict(record.initial_condition)

    coarse_config = SolverConfig(n_x=32, n_y=32, dt_override=manifest.shared_time_step)
    fine_config = SolverConfig(n_x=128, n_y=128)

    lr = read_fields(raw_dir / record.relative_path, "lr")
    hr = read_fields(raw_dir / record.relative_path, "hr")

    # float32 storage of a float64 analytic evaluation, transposed [x, y] -> [y, x].
    np.testing.assert_array_equal(
        lr[0, 0], specification.evaluate(coarse_config).T.astype(np.float32)
    )
    np.testing.assert_array_equal(
        hr[0, 0], specification.evaluate(fine_config).T.astype(np.float32)
    )
    # And both start at rest, as the reference solver does.
    assert np.all(lr[0, 1:] == 0.0)
    assert np.all(hr[0, 1:] == 0.0)


def test_coarse_and_fine_diverge_where_the_signal_actually_is(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """The two independent solves must differ substantially, not just at roundoff.

    Measured as a relative RMS over each field so the comparison is scale-aware. An
    absolute tolerance would be meaningless: most of the domain is quiet this early, and an
    earlier version of this test compared a domain corner holding values of order 1e-23,
    where any absolute tolerance passes trivially.
    """
    manifest, raw_dir = generated
    path = raw_dir / manifest.trajectories[0].relative_path
    lr = read_fields(path, "lr")
    hr = read_fields(path, "hr")

    for channel in range(3):
        lr_rms = float(np.sqrt((lr[:, channel].astype(np.float64) ** 2).mean()))
        hr_rms = float(np.sqrt((hr[:, channel].astype(np.float64) ** 2).mean()))
        assert lr_rms > 0.0 and hr_rms > 0.0
        # Relative discrepancy between the two solves, well above float32 roundoff.
        assert abs(lr_rms - hr_rms) / max(lr_rms, hr_rms) > 1e-4


def test_coarse_solve_loses_structure_relative_to_fine(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """The task must be non-trivial: the coarse solve should not already equal the target."""
    manifest, raw_dir = generated
    path = raw_dir / manifest.trajectories[0].relative_path
    lr_std = float(read_fields(path, "lr")[:, 0].std())
    hr_std = float(read_fields(path, "hr")[:, 0].std())
    assert lr_std > 0.0 and hr_std > 0.0
    assert lr_std != hr_std


# -- Physical admissibility of stored data --------------------------------------------


def test_stored_states_are_finite_with_positive_depth(
    generated: tuple[DatasetManifest, Path],
) -> None:
    manifest, raw_dir = generated
    for record in manifest.trajectories:
        path = raw_dir / record.relative_path
        depth = float(read_metadata(path)["depth"])
        for name in ("lr", "hr"):
            fields = read_fields(path, name)
            assert np.isfinite(fields).all()
            assert float((depth + fields[:, 0]).min()) > 0.0


def test_raw_wall_velocities_are_exactly_zero(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """D012: the hard zero check belongs on the raw staggered arrays.

    In `[y, x]` order, east-wall u is the last column and north-wall v the last row.
    """
    manifest, raw_dir = generated
    for record in manifest.trajectories:
        path = raw_dir / record.relative_path
        for name in ("lr", "hr"):
            fields = read_fields(path, name)
            assert np.all(fields[:, 1, :, -1] == 0.0), f"{name}: east wall u must be zero"
            assert np.all(fields[:, 2, -1, :] == 0.0), f"{name}: north wall v must be zero"


def test_manifest_records_diagnostics_for_both_resolutions(
    generated: tuple[DatasetManifest, Path],
) -> None:
    manifest, _ = generated
    for record in manifest.trajectories:
        for resolution in ("coarse", "fine"):
            diagnostics = record.diagnostics[resolution]
            assert diagnostics["relative_mass_drift"] < 1e-12
            assert diagnostics["min_total_depth"] > 0.0
            assert diagnostics["max_wall_normal_velocity"] == 0.0
            assert not diagnostics["non_finite_steps"]
        assert record.resolvability["coarse"]["min_sigma_over_dx"] > 0.0


# -- Splits ---------------------------------------------------------------------------


def test_splits_are_disjoint_in_the_manifest(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """docs/VALIDATION.md: trajectory IDs and seeds disjoint across splits."""
    manifest, _ = generated
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for split in ("train", "validation", "test"):
        records = manifest.by_split(split)
        assert records, f"{split} is empty"
        ids = {r.trajectory_id for r in records}
        seeds = {r.seed for r in records}
        assert not (ids & seen_ids)
        assert not (seeds & seen_seeds)
        seen_ids |= ids
        seen_seeds |= seeds


# -- Checksums and integrity ----------------------------------------------------------


def test_recomputed_checksums_match_the_manifest(
    generated: tuple[DatasetManifest, Path],
) -> None:
    manifest, raw_dir = generated
    for record in manifest.trajectories:
        mismatches = verify_trajectory_checksums(raw_dir / record.relative_path, record.arrays)
        assert mismatches == []


def test_checksum_detects_a_tampered_array(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """A checksum that cannot catch corruption is not an integrity signal.

    Verified by mutating a copy of the array in memory, rather than the released file,
    which must stay immutable.
    """
    manifest, raw_dir = generated
    record = manifest.trajectories[0]
    fields = read_fields(raw_dir / record.relative_path, "hr")
    assert array_checksum(fields) == record.checksum_for("hr")

    tampered = fields.copy()
    tampered[0, 0, 0, 0] = np.float32(tampered[0, 0, 0, 0] + np.float32(1e-3))
    assert array_checksum(tampered) != record.checksum_for("hr")


def test_checksum_is_independent_of_memory_layout() -> None:
    """The checksum must describe values, not a transient array layout."""
    contiguous = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    fortran = np.asfortranarray(contiguous)
    assert array_checksum(contiguous) == array_checksum(fortran)
    assert array_checksum(contiguous) != array_checksum(contiguous + 1)


def test_chunking_allows_single_frame_reads(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """Arrays are chunked per frame so training can stream frames in shuffled order."""
    manifest, raw_dir = generated
    path = raw_dir / manifest.trajectories[0].relative_path
    frame = read_frame(path, "hr", 2)
    assert frame.shape == (3, 128, 128)
    np.testing.assert_array_equal(frame, read_fields(path, "hr")[2])

    record = manifest.trajectories[0]
    hr_record = next(a for a in record.arrays if a.name == "hr")
    assert hr_record.chunks == (1, 3, 128, 128)


def test_stored_fields_are_float32_but_times_stay_float64(
    generated: tuple[DatasetManifest, Path],
) -> None:
    """Fields are float32 SI (PROJECT_SPEC.md); times stay float64 to keep exact equality."""
    manifest, raw_dir = generated
    record = manifest.trajectories[0]
    path = raw_dir / record.relative_path
    assert read_fields(path, "hr").dtype == np.float32
    assert read_fields(path, "lr").dtype == np.float32
    assert read_times(path).dtype == np.float64
    assert read_coordinates(path)["fine_x"].dtype == np.float64


# -- Immutability ---------------------------------------------------------------------


def test_regenerating_over_an_existing_dataset_is_refused(
    tmp_path: Path,
) -> None:
    """Datasets and manifests are immutable release artifacts (docs/AGENT_WORKFLOW.md)."""
    registry = build_registry()
    generate_dataset(FAST_CONFIG, registry=registry, output_root=tmp_path, verbose=False)
    with pytest.raises(FileExistsError, match="immutable"):
        generate_dataset(FAST_CONFIG, registry=registry, output_root=tmp_path, verbose=False)


def test_manifest_round_trips_and_refuses_conflicting_overwrite(
    generated: tuple[DatasetManifest, Path], tmp_path: Path
) -> None:
    manifest, raw_dir = generated
    reloaded = load_manifest(raw_dir / "manifest.json")
    assert reloaded.manifest_hash == manifest.manifest_hash

    path = tmp_path / "manifest.json"
    manifest.write(path)
    assert manifest.write(path) == manifest.manifest_hash  # idempotent

    conflicting = load_manifest(raw_dir / "manifest.json")
    conflicting.trajectories = conflicting.trajectories[:-1] or conflicting.trajectories
    conflicting.dataset_id = "changed"
    with pytest.raises(ManifestError, match="immutable"):
        conflicting.write(path)


def test_writing_a_trajectory_twice_is_refused(tmp_path: Path) -> None:
    payload = {
        "coarse_fields": np.zeros((2, 3, 4, 4)),
        "fine_fields": np.zeros((2, 3, 8, 8)),
        "times": np.array([0.0, 1.0]),
        "sample_steps": np.array([0, 1]),
        "coarse_x": np.linspace(-1, 1, 4),
        "coarse_y": np.linspace(-1, 1, 4),
        "fine_x": np.linspace(-1, 1, 8),
        "fine_y": np.linspace(-1, 1, 8),
        "metadata": {},
    }
    path = tmp_path / "t.h5"
    write_trajectory(path, **payload)  # type: ignore[arg-type]
    with pytest.raises(FileExistsError):
        write_trajectory(path, **payload)  # type: ignore[arg-type]


def test_mismatched_frame_counts_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frame counts differ"):
        write_trajectory(
            tmp_path / "bad.h5",
            coarse_fields=np.zeros((2, 3, 4, 4)),
            fine_fields=np.zeros((3, 3, 8, 8)),
            times=np.array([0.0, 1.0]),
            sample_steps=np.array([0, 1]),
            coarse_x=np.linspace(-1, 1, 4),
            coarse_y=np.linspace(-1, 1, 4),
            fine_x=np.linspace(-1, 1, 8),
            fine_y=np.linspace(-1, 1, 8),
            metadata={},
        )


# -- Config loading -------------------------------------------------------------------


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    """A typo'd key that is silently ignored yields a dataset not matching its config."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "dataset_id: x\npair_id: y\nresolution_family: z\n"
        "coarse_nodes: 32\nfine_nodes: 128\n"
        "discard_steps: 1\nsample_stride: 1\nsnapshot_count: 1\n"
        "snapshot_kount: 4\n"
    )
    with pytest.raises(ValueError, match="unknown config keys"):
        GenerationConfig.from_yaml(path)


def test_shipped_configs_match_the_documented_cadence() -> None:
    """The repo's real configs must encode the docs/DATASET.md schedule."""
    configs = Path("configs/data")
    primary = GenerationConfig.from_yaml(configs / "primary_32x128.yaml")
    assert (primary.discard_steps, primary.sample_stride, primary.snapshot_count) == (288, 24, 197)
    assert primary.sample_steps[-1] == 4992
    assert primary.pair.fine_config().dt == pytest.approx(25.1398, abs=1e-4)

    backup = GenerationConfig.from_yaml(configs / "backup_64x256.yaml")
    assert (backup.discard_steps, backup.sample_stride, backup.snapshot_count) == (576, 48, 197)
    assert backup.sample_steps[-1] == 9984
    assert backup.pair.fine_config().dt == pytest.approx(12.5206, abs=1e-4)
    assert primary.dataset_id != backup.dataset_id

    # Both releases must yield the same frame count, so a model sees the same number of
    # samples at either resolution (D017).
    assert primary.snapshot_count == backup.snapshot_count == 197


def test_documented_split_sizes_follow_from_the_snapshot_count() -> None:
    """The docs/DATASET.md split table must stay consistent with the shipped configs.

    Stale snapshot counts are exactly the kind of thing that ends up in a published table,
    so these are derived from the config rather than trusted from prose.
    """
    primary = GenerationConfig.from_yaml(Path("configs/data/primary_32x128.yaml"))
    per_trajectory = primary.snapshot_count
    assert 32 * per_trajectory == 6304
    assert 8 * per_trajectory == 1576
    assert 48 * per_trajectory == 9456

    # Raw payload must stay inside the sub-2-GiB target in docs/PROJECT_SPEC.md.
    bytes_per_pair = 3 * 4 * (32**2 + 128**2)
    payload_gib = 48 * per_trajectory * bytes_per_pair / 1024**3
    assert payload_gib == pytest.approx(1.840, abs=0.005)
    assert payload_gib < 2.0


def test_both_pairs_consume_the_same_registry_but_produce_different_manifests(
    tmp_path: Path,
) -> None:
    """D008: shared IC identities and splits, separate manifests and normalization."""
    registry = build_registry()
    primary = generate_dataset(
        FAST_CONFIG, registry=registry, output_root=tmp_path / "a", verbose=False
    )
    backup_config = GenerationConfig(
        dataset_id="test_64x256",
        pair_id="swe_gaussian_64x256_v1",
        resolution_family="swe_gaussian_x4",
        coarse_nodes=64,
        fine_nodes=256,
        discard_steps=12,
        sample_stride=6,
        snapshot_count=2,
        diagnostic_stride=6,
        trajectory_limit=1,
    )
    backup = generate_dataset(
        backup_config, registry=registry, output_root=tmp_path / "b", verbose=False
    )

    assert primary.ic_registry_hash == backup.ic_registry_hash
    assert [r.trajectory_id for r in primary.trajectories] == [
        r.trajectory_id for r in backup.trajectories
    ]
    assert primary.manifest_hash != backup.manifest_hash
    assert primary.shared_time_step != backup.shared_time_step
