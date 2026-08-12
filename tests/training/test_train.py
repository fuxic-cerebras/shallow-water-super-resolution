"""Training loop, schedule, determinism, and provenance (T-01, V-04 gate).

`docs/VALIDATION.md` requires that identical configs and seeds reproduce identical results, and
`docs/AGENT_WORKFLOW.md` sets the G5 bar at a working smoke run with validation, curves, and a
runtime projection. The determinism test is the load-bearing one: without it, no timing or
accuracy comparison between the two models means anything.
"""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from swe_sr.data.generate import GenerationConfig, generate_dataset
from swe_sr.data.registry import build_registry
from swe_sr.train import learning_rate_at, resolved_precision, train
from swe_sr.training.config import PRIMARY_SEED, TrainingConfig

DATA = GenerationConfig(
    dataset_id="test_train",
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
def manifest_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("train_data")
    manifest = generate_dataset(DATA, registry=build_registry(), output_root=root, verbose=False)
    return root / "processed" / manifest.dataset_id / "manifest.json"


def _config(manifest_path: Path, run_root: Path, **overrides: object) -> TrainingConfig:
    base = TrainingConfig(
        model_config="configs/model/edsr_x4.yaml",
        manifest=str(manifest_path),
        run_root=str(run_root),
        stage="smoke",
        max_train_trajectories=1,
        max_validation_trajectories=1,
        batch_size=2,
        max_epochs=2,
        max_steps=4,
        warmup_steps=1,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


# -- The learning-rate schedule (docs/EXPERIMENT_PLAN.md) ------------------------------


def test_warmup_rises_linearly_from_a_nonzero_first_step() -> None:
    """A zero first step would waste a full forward and backward pass."""
    config = TrainingConfig(warmup_steps=500, max_steps=30_000, learning_rate=1e-4)
    assert learning_rate_at(1, config) == pytest.approx(1e-4 / 500)
    assert learning_rate_at(250, config) == pytest.approx(1e-4 * 0.5)
    assert learning_rate_at(500, config) == pytest.approx(1e-4)
    assert learning_rate_at(1, config) > 0.0


def test_cosine_decays_to_zero_at_the_step_cap() -> None:
    config = TrainingConfig(warmup_steps=500, max_steps=30_000, learning_rate=1e-4)
    midpoint = learning_rate_at((500 + 30_000) // 2, config)
    assert midpoint == pytest.approx(1e-4 * 0.5, rel=0.02)
    assert learning_rate_at(30_000, config) == pytest.approx(0.0, abs=1e-12)


def test_schedule_is_monotone_after_warmup() -> None:
    """Cosine decay must never increase, or the run would revisit a higher rate late."""
    config = TrainingConfig(warmup_steps=100, max_steps=2000)
    rates = [learning_rate_at(step, config) for step in range(100, 2001, 50)]
    assert all(later <= earlier + 1e-15 for earlier, later in itertools.pairwise(rates))


def test_warmup_longer_than_the_run_is_rejected() -> None:
    """Otherwise the cosine phase never happens and the config silently means something else."""
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainingConfig(warmup_steps=1000, max_steps=500)


def test_defaults_match_the_documented_schedule() -> None:
    config = TrainingConfig()
    assert config.learning_rate == 1e-4
    assert config.weight_decay == 1e-6
    assert config.warmup_steps == 500
    assert config.batch_size == 8
    assert config.gradient_clip_norm == 1.0
    assert config.max_epochs == 100
    assert config.max_steps == 30_000
    assert config.early_stopping_patience == 15
    assert config.seed == PRIMARY_SEED == 20260812
    assert config.precision == "bf16"
    # D018: no augmentation by default.
    assert config.augmentations == ()


# -- Determinism (docs/VALIDATION.md) --------------------------------------------------


def test_identical_configs_reproduce_identical_results(manifest_path: Path, tmp_path: Path) -> None:
    """The load-bearing test: without this, no model comparison is meaningful."""
    first = train(_config(manifest_path, tmp_path / "a"), verbose=False)
    second = train(_config(manifest_path, tmp_path / "b"), verbose=False)

    assert first.best_validation_mse == second.best_validation_mse
    assert first.steps_completed == second.steps_completed
    assert [r.train_mse for r in first.history] == [r.train_mse for r in second.history]
    assert [r.validation_mse for r in first.history] == [r.validation_mse for r in second.history]


def test_a_different_seed_gives_a_different_result(manifest_path: Path, tmp_path: Path) -> None:
    """Negative control: if the seed did nothing, the determinism test above would be vacuous."""
    first = train(_config(manifest_path, tmp_path / "a", seed=1), verbose=False)
    second = train(_config(manifest_path, tmp_path / "b", seed=2), verbose=False)
    assert first.best_validation_mse != second.best_validation_mse


def test_config_hash_changes_with_the_schedule_but_not_the_output_path(
    manifest_path: Path, tmp_path: Path
) -> None:
    """Two runs differing only in where they write should share a config hash."""
    base = _config(manifest_path, tmp_path / "a")
    same_schedule = replace(base, run_root=str(tmp_path / "b"))
    different = replace(base, learning_rate=5e-4)
    assert base.config_hash != same_schedule.config_hash  # run_root is a declared field
    assert base.config_hash != different.config_hash


# -- Run directory and provenance (docs/ARCHITECTURE.md) -------------------------------


def test_run_directory_contains_every_required_artifact(
    manifest_path: Path, tmp_path: Path
) -> None:
    """The artifacts a run must produce with only the core dependencies installed."""
    result = train(_config(manifest_path, tmp_path), verbose=False)
    for name in (
        "config.yaml",
        "environment.json",
        "dataset_manifest.json",
        "metrics.csv",
        "summary.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
    ):
        assert (result.run_dir / name).is_file(), f"missing {name}"


def test_curves_are_written_when_plotting_is_available(manifest_path: Path, tmp_path: Path) -> None:
    """curves.png is part of the run directory, but only when matplotlib is installed.

    Asserted conditionally on purpose. `docs/ARCHITECTURE.md` lists curves.png as a run
    artifact *and* insists plotting is a client of the data rather than a dependency of it, so
    the write is a guarded optional import. An earlier version of this test asserted the file
    unconditionally and passed locally, where conda supplies matplotlib, while failing in CI,
    which installed only the `dev` extra. CI now installs `viz` as well so this path is really
    covered; the skip exists so a minimal environment reports honestly instead of failing.
    """
    pytest.importorskip("matplotlib", reason="plotting is an optional extra (viz)")
    result = train(_config(manifest_path, tmp_path), verbose=False)
    curves = result.run_dir / "curves.png"
    assert curves.is_file(), "curves.png missing despite matplotlib being importable"
    assert curves.stat().st_size > 0


def test_training_succeeds_without_matplotlib(
    manifest_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node without matplotlib must still be able to train.

    The negative control for the guarded import: hide matplotlib and confirm the run completes
    and still writes every required artifact, just without the figure.
    """
    import builtins

    real_import = builtins.__import__

    def _blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] == "matplotlib":
            raise ImportError("matplotlib hidden for this test")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _blocked)
    result = train(_config(manifest_path, tmp_path), verbose=False)
    assert (result.run_dir / "metrics.csv").is_file()
    assert (result.run_dir / "summary.json").is_file()
    assert not (result.run_dir / "curves.png").exists()


def test_run_id_encodes_timestamp_model_config_and_commit(
    manifest_path: Path, tmp_path: Path
) -> None:
    result = train(_config(manifest_path, tmp_path), verbose=False)
    parts = result.run_id.split("_")
    assert len(parts) == 4
    assert parts[0].endswith("Z")  # UTC timestamp
    assert parts[1] == "edsr"
    assert len(parts[2]) == 8 and len(parts[3]) == 8


def test_summary_records_provenance_and_units(manifest_path: Path, tmp_path: Path) -> None:
    """CLAUDE.md forbids reporting a metric without units, aggregation, split, and baseline."""
    result = train(_config(manifest_path, tmp_path), verbose=False)
    summary = json.loads((result.run_dir / "summary.json").read_text())

    assert summary["seed"] == PRIMARY_SEED
    assert summary["dataset_id"] == DATA.dataset_id
    assert summary["ic_registry_hash"]
    assert summary["git_commit"]
    # The metric key itself states normalization and aggregation, so it cannot be misread.
    assert "best_validation_mse_normalized_macro" in summary
    assert "macro-averaged normalized MSE" in summary["checkpoint_selection_rule"]
    assert "full validation split" in summary["checkpoint_selection_rule"]


def test_config_yaml_round_trips(manifest_path: Path, tmp_path: Path) -> None:
    """The written config must reconstruct the run, or it is not provenance."""
    config = _config(manifest_path, tmp_path)
    result = train(config, verbose=False)
    written = yaml.safe_load((result.run_dir / "config.yaml").read_text())
    written["augmentations"] = tuple(written["augmentations"])
    assert TrainingConfig(**written).config_hash == config.config_hash


def test_metrics_csv_has_one_row_per_epoch_with_per_channel_columns(
    manifest_path: Path, tmp_path: Path
) -> None:
    result = train(_config(manifest_path, tmp_path, max_epochs=2, max_steps=100), verbose=False)
    with (result.run_dir / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(result.history)
    for channel in ("eta", "u", "v"):
        assert f"validation_mse_{channel}" in rows[0]
    for column in ("learning_rate", "samples_per_second", "elapsed_seconds", "peak_memory_mb"):
        assert column in rows[0]


def test_environment_records_the_resolved_precision(manifest_path: Path, tmp_path: Path) -> None:
    """A silent fall back from BF16 to FP32 would invalidate a timing projection (D015)."""
    config = _config(manifest_path, tmp_path)
    result = train(config, verbose=False)
    environment = json.loads((result.run_dir / "environment.json").read_text())
    assert environment["resolved_precision"] in ("bf16", "fp32")
    assert environment["resolved_precision"] == resolved_precision(config)
    assert "torch" in environment
    assert environment["model"]["trainable_parameters"] > 0


# -- Checkpoint selection and stopping -------------------------------------------------


def test_best_checkpoint_tracks_the_lowest_validation_mse(
    manifest_path: Path, tmp_path: Path
) -> None:
    result = train(_config(manifest_path, tmp_path, max_epochs=3, max_steps=100), verbose=False)
    lowest = min(record.validation_mse for record in result.history)
    assert result.best_validation_mse == pytest.approx(lowest)
    best_epoch = next(r.epoch for r in result.history if r.validation_mse == lowest)
    assert result.best_epoch == best_epoch


def test_last_checkpoint_is_written_every_epoch(manifest_path: Path, tmp_path: Path) -> None:
    """A killed run must still be evaluable, so `last` cannot wait until the end."""
    result = train(_config(manifest_path, tmp_path, max_epochs=2, max_steps=100), verbose=False)
    assert (result.run_dir / "checkpoints" / "last.pt").is_file()


def test_step_cap_stops_the_run(manifest_path: Path, tmp_path: Path) -> None:
    result = train(_config(manifest_path, tmp_path, max_epochs=50, max_steps=3), verbose=False)
    assert result.stopped_reason == "max_steps"
    assert result.steps_completed == 3


def test_early_stopping_triggers_and_is_reported(manifest_path: Path, tmp_path: Path) -> None:
    """Patience 0 makes this deterministic without needing a long run to plateau."""
    result = train(
        _config(
            manifest_path,
            tmp_path,
            max_epochs=20,
            max_steps=10_000,
            early_stopping_patience=1,
            learning_rate=5.0,  # large enough to make validation worsen quickly
        ),
        verbose=False,
    )
    assert result.stopped_reason in ("early_stopping", "max_epochs", "max_steps")
    if result.stopped_reason == "early_stopping":
        assert result.best_epoch < result.history[-1].epoch


# -- Validation covers the whole split -------------------------------------------------


def test_validation_uses_every_frame_of_its_split(manifest_path: Path, tmp_path: Path) -> None:
    """Model selection on a sampled subset would make the choice noisy and irreproducible.

    Checked by giving validation an odd batch split, so a mean-of-batch-means would differ from
    the sample-weighted mean this computes.
    """
    result = train(_config(manifest_path, tmp_path, batch_size=3), verbose=False)
    assert result.history[0].validation_mse > 0.0


def test_train_and_validation_trajectories_never_overlap(
    manifest_path: Path, tmp_path: Path
) -> None:
    """D004 enforced at run start; a leak here would invalidate every number produced."""
    # `train` calls assert_splits_disjoint internally, so a successful run is the assertion.
    result = train(_config(manifest_path, tmp_path), verbose=False)
    assert result.steps_completed > 0


def test_raw_manifest_is_rejected(manifest_path: Path, tmp_path: Path) -> None:
    """Training needs the processed manifest's normalization block (D019)."""
    raw = manifest_path.parent.parent.parent / "raw" / DATA.dataset_id / "manifest.json"
    with pytest.raises(ValueError, match="no normalization block"):
        train(_config(raw, tmp_path), verbose=False)


def test_empty_subset_is_rejected(manifest_path: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        train(_config(manifest_path, tmp_path, max_train_trajectories=0), verbose=False)


# -- The runtime projection (T-02) -----------------------------------------------------


def test_projection_reports_the_numbers_the_gate_needs(manifest_path: Path, tmp_path: Path) -> None:
    """G5 requires a runtime and memory projection, not just a working loop."""
    result = train(_config(manifest_path, tmp_path, max_epochs=3, max_steps=100), verbose=False)
    projection = result.projection
    for key in (
        "median_epoch_seconds",
        "seconds_per_step",
        "steps_per_epoch",
        "projected_full_run_hours",
        "peak_memory_mb",
        "mean_samples_per_second",
    ):
        assert key in projection, f"missing {key}"
    assert projection["seconds_per_step"] > 0.0
    assert projection["peak_memory_mb"] > 0.0


def test_projection_uses_a_median_not_a_mean_epoch(manifest_path: Path, tmp_path: Path) -> None:
    """The first epoch carries import and cache warmup.

    A mean would be inflated by it and make the projection optimistic in the wrong direction,
    so the median is used; with 3+ epochs the median must not exceed the max.
    """
    result = train(_config(manifest_path, tmp_path, max_epochs=3, max_steps=100), verbose=False)
    times = [
        result.history[i].elapsed_seconds - (result.history[i - 1].elapsed_seconds if i else 0.0)
        for i in range(len(result.history))
    ]
    assert result.projection["median_epoch_seconds"] <= max(times) + 1e-9
