"""The immutable IC registry (task D-01).

The registry is the one artifact both resolution pairs share (D008), and the split it
encodes is the project's primary defense against leakage (D004). These tests include the
negative cases `docs/VALIDATION.md` requires: a registry with overlapping splits, renamed
trajectory IDs, or a mutated entry must be rejected, not merely noticed.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from swe_sr.data.registry import (
    SPLIT_LAYOUT,
    TOTAL_TRAJECTORIES,
    InitialConditionRegistry,
    RegistryValidationError,
    build_registry,
    load_registry,
    split_for_seed,
    trajectory_id_for_seed,
    validate_registry,
)
from swe_sr.solver.config import SolverConfig


@pytest.fixture(scope="module")
def registry() -> InitialConditionRegistry:
    return build_registry()


# -- Structure ------------------------------------------------------------------------


def test_registry_holds_the_documented_split_sizes(registry: InitialConditionRegistry) -> None:
    assert len(registry) == TOTAL_TRAJECTORIES == 48
    assert len(registry.by_split("train")) == 32
    assert len(registry.by_split("validation")) == 8
    assert len(registry.by_split("test")) == 8


def test_split_membership_is_a_pure_function_of_the_seed() -> None:
    """The split is fixed before simulation (D004), so it must not depend on load order."""
    assert split_for_seed(0) == "train"
    assert split_for_seed(31) == "train"
    assert split_for_seed(32) == "validation"
    assert split_for_seed(39) == "validation"
    assert split_for_seed(40) == "test"
    assert split_for_seed(47) == "test"
    for bad_seed in (-1, 48, 1000):
        with pytest.raises(ValueError, match="outside the registry range"):
            split_for_seed(bad_seed)


def test_no_trajectory_or_seed_appears_in_two_splits(
    registry: InitialConditionRegistry,
) -> None:
    """docs/VALIDATION.md: trajectory IDs and seeds must be disjoint across splits."""
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for split, _, _ in SPLIT_LAYOUT:
        ids = {entry.trajectory_id for entry in registry.by_split(split)}
        seeds = {entry.seed for entry in registry.by_split(split)}
        assert not (ids & seen_ids), f"{split} shares a trajectory ID with an earlier split"
        assert not (seeds & seen_seeds), f"{split} shares a seed with an earlier split"
        seen_ids |= ids
        seen_seeds |= seeds
    assert len(seen_ids) == 48
    assert len(seen_seeds) == 48


def test_trajectory_ids_are_stable_and_content_derived() -> None:
    """The same seed must yield the same UUID on any machine, with no persisted counter."""
    assert trajectory_id_for_seed(5) == trajectory_id_for_seed(5)
    assert trajectory_id_for_seed(5) != trajectory_id_for_seed(6)
    # Pinned to their literal values so a change to the derivation scheme is a deliberate,
    # visible decision: every released manifest references these IDs, and silently
    # renumbering them would orphan the datasets that cite them.
    assert trajectory_id_for_seed(0) == "993f3dad-1558-5f98-8ef6-c8c38071afad"
    assert trajectory_id_for_seed(1) == "57618449-8dcd-5a3f-ac89-839343ec3283"
    assert trajectory_id_for_seed(47) == "8b5bc312-9be9-5ea7-b7f9-8fb5e1f8097e"


def test_registry_is_reproducible_bit_for_bit() -> None:
    """Rebuilding must give an identical hash, since manifests reference it."""
    first = build_registry()
    second = build_registry()
    assert first.registry_hash == second.registry_hash
    assert first.to_json() == second.to_json()
    # Pinned literally: this hash goes into every dataset manifest, so a change to the
    # sampling or serialization must be noticed here rather than in a stale manifest.
    assert first.registry_hash == "976e3a577a25a633c6a2625263f23e60482768965029805a5efd16be97ab7c8c"


def test_registry_hash_changes_when_any_entry_changes(
    registry: InitialConditionRegistry,
) -> None:
    """A content hash that ignores content would make manifest provenance worthless."""
    mutated_entry = replace(registry.entries[0], seed=registry.entries[0].seed)
    unchanged = InitialConditionRegistry(entries=(mutated_entry, *registry.entries[1:]))
    assert unchanged.registry_hash == registry.registry_hash

    tampered_ic = replace(registry.entries[0].initial_condition.bumps[0], amplitude=0.123456)
    tampered = replace(
        registry.entries[0],
        initial_condition=replace(registry.entries[0].initial_condition, bumps=(tampered_ic,)),
    )
    different = InitialConditionRegistry(entries=(tampered, *registry.entries[1:]))
    assert different.registry_hash != registry.registry_hash


def test_every_entry_is_admissible_on_both_pair_resolutions(
    registry: InitialConditionRegistry,
) -> None:
    """The shared registry must be valid for the primary and backup pairs alike (D008)."""
    for nodes in (32, 128, 64, 256):
        config = SolverConfig(n_x=nodes, n_y=nodes)
        for entry in registry:
            eta = entry.initial_condition.evaluate(config)
            assert float((config.depth + eta).min()) > 0.0, f"seed {entry.seed} dries at {nodes}"


def test_resolvability_is_recorded_for_the_coarse_grid(
    registry: InitialConditionRegistry,
) -> None:
    """docs/DATASET.md requires recording sigma/dx; the coarse grid is the binding case."""
    coarse = SolverConfig(n_x=32, n_y=32)
    ratios = [
        entry.initial_condition.resolvability(coarse)["min_sigma_over_dx"] for entry in registry
    ]
    # 65-120 km widths against a 32.26 km coarse spacing.
    assert min(ratios) >= 2.0
    assert max(ratios) <= 3.8


# -- Persistence and immutability -----------------------------------------------------


def test_registry_round_trips_through_json(
    registry: InitialConditionRegistry, tmp_path: Path
) -> None:
    path = tmp_path / "ic_registry_v1.json"
    written_hash = registry.write(path)
    reloaded = load_registry(path)

    assert written_hash == registry.registry_hash
    assert reloaded.registry_hash == registry.registry_hash
    assert reloaded.entries == registry.entries


def test_rewriting_an_identical_registry_is_allowed(
    registry: InitialConditionRegistry, tmp_path: Path
) -> None:
    """Idempotent regeneration must not fail; only a *conflicting* rewrite should."""
    path = tmp_path / "ic_registry_v1.json"
    registry.write(path)
    assert registry.write(path) == registry.registry_hash


def test_overwriting_with_a_different_registry_is_refused(
    registry: InitialConditionRegistry, tmp_path: Path
) -> None:
    """docs/AGENT_WORKFLOW.md: never overwrite an artifact; use immutable IDs.

    Silently replacing a registry would invalidate every manifest referencing its hash.
    """
    path = tmp_path / "ic_registry_v1.json"
    registry.write(path)

    smaller = InitialConditionRegistry(entries=registry.entries[:-1])
    with pytest.raises(RegistryValidationError, match="immutable"):
        smaller.write(path)


# -- Negative tests (docs/VALIDATION.md) ----------------------------------------------


def test_a_registry_with_overlapping_splits_is_rejected(
    registry: InitialConditionRegistry,
) -> None:
    """The leakage negative test: relabel a test trajectory as train and expect rejection."""
    entries = list(registry.entries)
    test_entry = registry.by_split("test")[0]
    index = entries.index(test_entry)
    entries[index] = replace(test_entry, split="train")

    with pytest.raises(RegistryValidationError, match="expected test"):
        validate_registry(InitialConditionRegistry(entries=tuple(entries)))


def test_a_registry_with_a_renamed_trajectory_id_is_rejected(
    registry: InitialConditionRegistry,
) -> None:
    entries = list(registry.entries)
    entries[3] = replace(entries[3], trajectory_id="00000000-0000-0000-0000-000000000000")
    with pytest.raises(RegistryValidationError, match="content-derived"):
        validate_registry(InitialConditionRegistry(entries=tuple(entries)))


def test_a_registry_with_duplicate_seeds_is_rejected(
    registry: InitialConditionRegistry,
) -> None:
    entries = list(registry.entries)
    entries[1] = replace(entries[1], seed=entries[0].seed)
    with pytest.raises(RegistryValidationError):
        validate_registry(InitialConditionRegistry(entries=tuple(entries)))


def test_a_truncated_registry_is_rejected(registry: InitialConditionRegistry) -> None:
    with pytest.raises(RegistryValidationError, match="expected 48"):
        validate_registry(InitialConditionRegistry(entries=registry.entries[:40]))


def test_loading_a_hand_edited_registry_is_rejected(
    registry: InitialConditionRegistry, tmp_path: Path
) -> None:
    """Validation runs on load, so tampering on disk cannot slip through."""
    path = tmp_path / "ic_registry_v1.json"
    registry.write(path)

    payload = json.loads(path.read_text())
    payload["trajectories"][45]["split"] = "train"  # a test trajectory moved into train
    path.write_text(json.dumps(payload))

    with pytest.raises(RegistryValidationError):
        load_registry(path)
