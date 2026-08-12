"""Destaggering and vector-aware augmentation (D-04).

Includes the verification `docs/DATASET.md` requires before the transpose augmentation may
be used, and the negative test for wrong vector reflection named in `docs/VALIDATION.md`.
The symmetry tests record a measured scientific finding (D018), not just code behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from swe_sr.data.processing import (
    AUGMENTATIONS,
    ETA,
    AugmentationPolicy,
    U,
    V,
    augmentation_symmetry_error,
    destagger,
    reflect_x,
    reflect_y,
    transpose_xy,
)
from swe_sr.solver.diagnostics import destagger_u, destagger_v


def _fields(n_y: int = 5, n_x: int = 7) -> np.ndarray:
    """Distinct, non-symmetric values per channel so axis or channel mix-ups show up."""
    rng = np.random.default_rng(7)
    return rng.normal(size=(3, n_y, n_x))


# -- Destaggering (D011) --------------------------------------------------------------


def test_destagger_matches_the_solver_order_implementation() -> None:
    """Storage-order destaggering must agree with the solver-order version under transpose.

    Two implementations of the same operator on different axis conventions is exactly the
    kind of duplication that silently diverges, so this pins them together. Uses a
    non-square grid so an axis swap cannot pass.
    """
    fields = _fields(n_y=5, n_x=7)
    centered = destagger(fields)

    # Solver order is [x, y]; the diagnostics helpers act on axis 0 for u and axis 1 for v.
    np.testing.assert_allclose(centered[U], destagger_u(fields[U].T).T)
    np.testing.assert_allclose(centered[V], destagger_v(fields[V].T).T)


def test_destagger_leaves_eta_untouched() -> None:
    fields = _fields()
    np.testing.assert_array_equal(destagger(fields)[ETA], fields[ETA])


def test_destagger_averages_u_along_x_and_v_along_y() -> None:
    """The axis each component is averaged over is the thing most likely to be wrong."""
    fields = np.zeros((3, 4, 4))
    fields[U] = np.array([[1.0, 3.0, 5.0, 7.0]] * 4)  # varies along x
    fields[V] = np.array([[1.0] * 4, [3.0] * 4, [5.0] * 4, [7.0] * 4])  # varies along y

    centered = destagger(fields)
    np.testing.assert_allclose(centered[U][0], [0.5, 2.0, 4.0, 6.0])
    np.testing.assert_allclose(centered[V][:, 0], [0.5, 2.0, 4.0, 6.0])


def test_destagger_handles_leading_time_axis() -> None:
    """Generation destaggers whole trajectories; batching must not change the result."""
    trajectory = np.stack([_fields() for _ in range(4)])
    batched = destagger(trajectory)
    for frame in range(4):
        np.testing.assert_array_equal(batched[frame], destagger(trajectory[frame]))


def test_destagger_does_not_mutate_its_input() -> None:
    fields = _fields()
    original = fields.copy()
    destagger(fields)
    np.testing.assert_array_equal(fields, original)


def test_destagger_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="3 channels"):
        destagger(np.zeros((2, 4, 4)))


# -- Augmentation geometry ------------------------------------------------------------


def test_reflect_x_flips_x_and_negates_u_only() -> None:
    fields = _fields()
    result = reflect_x(fields)
    np.testing.assert_array_equal(result[ETA], fields[ETA][:, ::-1])
    np.testing.assert_array_equal(result[U], -fields[U][:, ::-1])
    np.testing.assert_array_equal(result[V], fields[V][:, ::-1])


def test_reflect_y_flips_y_and_negates_v_only() -> None:
    fields = _fields()
    result = reflect_y(fields)
    np.testing.assert_array_equal(result[ETA], fields[ETA][::-1, :])
    np.testing.assert_array_equal(result[U], fields[U][::-1, :])
    np.testing.assert_array_equal(result[V], -fields[V][::-1, :])


def test_transpose_swaps_axes_and_swaps_u_with_v() -> None:
    """docs/DATASET.md: the transpose swaps axes *and* swaps u with v.

    This is the unit test that gates the transpose's use. Forgetting the channel swap leaves
    a plausible-looking array in which every velocity vector points the wrong way.
    """
    fields = _fields(n_y=5, n_x=5)
    result = transpose_xy(fields)
    np.testing.assert_array_equal(result[ETA], fields[ETA].T)
    np.testing.assert_array_equal(result[U], fields[V].T)
    np.testing.assert_array_equal(result[V], fields[U].T)


def test_wrong_vector_reflection_is_detectably_different() -> None:
    """The negative test named in docs/VALIDATION.md.

    Reflecting geometry while forgetting the sign flip, or negating the wrong component,
    must not coincide with the correct transform.
    """
    fields = _fields()
    correct = reflect_x(fields)

    geometry_only = fields[:, :, ::-1]
    assert not np.allclose(correct, geometry_only)

    wrong_component = fields[:, :, ::-1].copy()
    wrong_component[V] *= -1.0
    assert not np.allclose(correct, wrong_component)


def test_each_augmentation_is_its_own_involution_or_returns_to_identity() -> None:
    """Applying any of these twice returns the original, a cheap structural check."""
    fields = _fields(n_y=6, n_x=6)
    for name in ("reflect_x", "reflect_y", "transpose"):
        transform = AUGMENTATIONS[name]
        np.testing.assert_allclose(transform(transform(fields)), fields, atol=1e-15)


def test_augmentations_permute_values_without_rescaling_them() -> None:
    """docs/DATASET.md forbids independent channel scaling, so these must only permute.

    Checked per channel against the source channel it draws from: reflections keep each
    channel's own values, while the transpose swaps u and v. Sorted absolute values compare
    the multiset, which is invariant to the reordering and sign flips the transforms apply.
    """
    fields = _fields(n_y=6, n_x=6)
    source_channel = {
        "reflect_x": {ETA: ETA, U: U, V: V},
        "reflect_y": {ETA: ETA, U: U, V: V},
        "transpose": {ETA: ETA, U: V, V: U},
    }
    for name, mapping in source_channel.items():
        result = AUGMENTATIONS[name](fields)
        for target, source in mapping.items():
            np.testing.assert_allclose(
                np.sort(np.abs(result[target]).ravel()),
                np.sort(np.abs(fields[source]).ravel()),
            )


# -- Augmentation policy --------------------------------------------------------------


def test_augmentation_is_disabled_by_default() -> None:
    """D018: none of the documented transforms is a symmetry of this rotating solver."""
    policy = AugmentationPolicy()
    assert not policy.enabled
    assert policy.draw(np.random.default_rng(0)) == "identity"


def test_unknown_augmentation_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown augmentation"):
        AugmentationPolicy(names=("rotate_90",))


def test_policy_applies_one_transform_to_every_array_it_is_given() -> None:
    """A pair must receive the *same* transform, or the correspondence is destroyed."""
    policy = AugmentationPolicy(names=("reflect_x",))
    coarse = _fields(n_y=4, n_x=4)
    fine = _fields(n_y=8, n_x=8)
    got_coarse, got_fine = policy.apply("reflect_x", coarse, fine)
    np.testing.assert_array_equal(got_coarse, reflect_x(coarse))
    np.testing.assert_array_equal(got_fine, reflect_x(fine))


def test_policy_draw_includes_the_identity() -> None:
    """Some samples must stay untransformed, so the model still sees the real orientation."""
    policy = AugmentationPolicy(names=("reflect_x", "reflect_y"))
    drawn = {policy.draw(np.random.default_rng(seed)) for seed in range(60)}
    assert "identity" in drawn
    assert drawn <= {"identity", "reflect_x", "reflect_y"}


# -- The scientific finding behind D018 -----------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("name", ["reflect_x", "reflect_y", "transpose"])
def test_documented_augmentations_are_not_symmetries_of_the_rotating_solver(name: str) -> None:
    """docs/DATASET.md calls these "symmetry-preserving"; for this solver they are not.

    Rotation is chiral: a mirror image of a rotating system rotates the other way, so a
    reflection is a symmetry only if `f -> -f`, and `f` here is positive everywhere. The
    transpose additionally makes `f` depend on x. Measured discrepancies are near 0.9
    relative, so this is a qualitative failure rather than a small correction.

    Asserted as a *lower* bound: if a future change made these symmetries, that would be a
    significant physics change and should fail loudly here.
    """
    error = augmentation_symmetry_error(name, nodes=48, steps=300)
    assert error > 0.1, f"{name} unexpectedly behaves like a symmetry (error {error:.3e})"


@pytest.mark.slow
def test_transpose_is_an_exact_symmetry_without_rotation() -> None:
    """Isolates the cause: with rotation off, the transpose commutes with the dynamics.

    This is what makes the D018 diagnosis specific rather than hand-waving -- the transpose
    failure is attributable to Coriolis alone, not to the discretization.
    """
    error = augmentation_symmetry_error(
        "transpose", nodes=48, steps=300, use_coriolis=False, use_beta=False
    )
    assert error < 1e-12, f"transpose should be exact without rotation, got {error:.3e}"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["reflect_x", "reflect_y"])
def test_reflections_stay_broken_even_without_rotation(name: str) -> None:
    """The second, independent cause: the C-grid staggering is not reflection-symmetric.

    `u_i` sits at `x_{i+1/2}`, offset east, so reflecting maps an east-face variable onto a
    west-face position -- a different index alignment. Small compared to the Coriolis effect
    but real, and it means reflections would remain invalid even on a non-rotating basin.
    """
    error = augmentation_symmetry_error(
        name, nodes=48, steps=300, use_coriolis=False, use_beta=False
    )
    assert 1e-4 < error < 0.5, f"expected a small nonzero staggering asymmetry, got {error:.3e}"
