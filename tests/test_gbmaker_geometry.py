# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.gbmaker.geometry import (
    _box_periodic_basis,
    _cartesian_from_box_coordinates,
    _clip_complete_origins_to_cartesian_box,
    _complete_origin_atom_mask,
    _deduplicate_complete_origins,
    _filter_complete_origins,
    _reduced_box_coordinates,
    _reduced_coordinate_tolerance,
    _scaled_periodic_basis_vector,
    _select_complete_origins_in_box_basis,
    _selection_basis_vectors,
    _x_index_range,
    wrap_reduced_coordinate,
)
from GBOpt.gbmaker.types import GBMakerConstructionValueError

# --------------------------------------------------------------------------------------
# wrap_reduced_coordinate
# --------------------------------------------------------------------------------------


def test_wrap_reduced_coordinate_preserves_exact_thresholds():
    tol = 1e-10
    coords = np.array([tol / 2, tol, 1.0 - tol, 1.0 - tol / 2])
    wrapped = wrap_reduced_coordinate(coords, tol=tol)
    expected = np.array([0.0, tol, 1.0 - tol, 0.0])
    np.testing.assert_allclose(wrapped, expected, atol=1e-15, rtol=0.0)


def test_wrap_reduced_coordinate_scalar_and_0d_inputs():
    for coord in (0.375, np.array(0.375)):
        wrapped = wrap_reduced_coordinate(coord, tol=1e-10)
        assert wrapped.shape == ()
        np.testing.assert_allclose(wrapped, 0.375, atol=1e-15, rtol=0.0)


def test_wrap_reduced_coordinate_preserves_multidimensional_shape():
    coords = np.array([[0.25, 1.2], [-0.2, 2.75]])
    wrapped = wrap_reduced_coordinate(coords, tol=1e-10)
    assert wrapped.shape == coords.shape
    expected = np.array([[0.25, 0.2], [0.8, 0.75]])
    np.testing.assert_allclose(wrapped, expected, atol=1e-15, rtol=0.0)


def test_wrap_reduced_coordinate_wraps_multiple_periods_away():
    coords = np.array([2.2, -3.7, 4.125, -5.875])
    wrapped = wrap_reduced_coordinate(coords, tol=1e-10)
    expected = np.array([0.2, 0.3, 0.125, 0.125])
    np.testing.assert_allclose(wrapped, expected, atol=1e-15, rtol=0.0)


def test_wrap_reduced_coordinate_negative_tolerance_raises_construction_error():
    with pytest.raises(GBMakerConstructionValueError):
        wrap_reduced_coordinate(np.array([0.25]), tol=-1e-10)


@pytest.mark.parametrize("tol", [np.nan, np.inf, -np.inf])
def test_wrap_reduced_coordinate_non_finite_tolerance_raises_construction_error(tol):
    with pytest.raises(GBMakerConstructionValueError):
        wrap_reduced_coordinate(np.array([0.25]), tol=tol)


# --------------------------------------------------------------------------------------
# _reduced_coordinate_tolerance
# --------------------------------------------------------------------------------------


def test_reduced_coordinate_tolerance_scales_with_basis_length():
    basis_vector = np.array([3.0, 4.0, 0.0])

    tol = _reduced_coordinate_tolerance(basis_vector, epsilon=2e-8)

    assert tol == pytest.approx(4e-9, abs=1e-18)


# --------------------------------------------------------------------------------------
# _scaled_periodic_basis_vector
# --------------------------------------------------------------------------------------


def test_scaled_periodic_basis_vector_scales_selected_axis_projection():
    period_vector = np.array([2.0, -1.0, 0.5])

    scaled = _scaled_periodic_basis_vector(period_vector, 10.0, 0)

    np.testing.assert_allclose(
        scaled, np.array([10.0, -5.0, 2.5]), atol=1e-12, rtol=0.0
    )
    np.testing.assert_allclose(
        period_vector, np.array([2.0, -1.0, 0.5]), atol=0.0, rtol=0.0
    )


def test_scaled_periodic_basis_vector_accepts_nonzero_projection_on_nonzero_axis():
    period_vector = np.array([0.0, 1e-10, 0.0])

    scaled = _scaled_periodic_basis_vector(period_vector, 10.0, 1)

    np.testing.assert_allclose(scaled, np.array([0.0, 10.0, 0.0]), atol=1e-12, rtol=0.0)


def test_scaled_periodic_basis_vector_scales_axis_index_two():
    period_vector = np.array([1.5, -0.75, 3.0])

    scaled = _scaled_periodic_basis_vector(period_vector, 12.0, 2)

    np.testing.assert_allclose(
        scaled, np.array([6.0, -3.0, 12.0]), atol=1e-12, rtol=0.0
    )


def test_scaled_periodic_basis_vector_scales_negative_selected_axis_projection():
    period_vector = np.array([1.5, -0.75, -3.0])

    scaled = _scaled_periodic_basis_vector(period_vector, 12.0, 2)

    np.testing.assert_allclose(
        scaled, np.array([-6.0, 3.0, 12.0]), atol=1e-12, rtol=0.0
    )


def test_scaled_periodic_basis_vector_rejects_zero_axis_projection():
    with pytest.raises(GBMakerConstructionValueError):
        _scaled_periodic_basis_vector(np.array([1.0, 2.0, 0.0]), 10.0, 2)


def test_scaled_periodic_basis_vector_accepts_numpy_integer_axis_index():
    scaled = _scaled_periodic_basis_vector(
        np.array([1.0, 2.0, 3.0]), 10.0, np.int64(1)
    )

    np.testing.assert_allclose(
        scaled, np.array([5.0, 10.0, 15.0]), atol=1e-12, rtol=0.0
    )


@pytest.mark.parametrize("box_length", [0.0, -1.0])
def test_scaled_periodic_basis_vector_rejects_non_positive_box_length(box_length):
    with pytest.raises(GBMakerConstructionValueError):
        _scaled_periodic_basis_vector(np.array([1.0, 2.0, 3.0]), box_length, 0)


@pytest.mark.parametrize("box_length", [np.nan, np.inf, -np.inf])
def test_scaled_periodic_basis_vector_rejects_non_finite_box_length(box_length):
    with pytest.raises(GBMakerConstructionValueError):
        _scaled_periodic_basis_vector(np.array([1.0, 2.0, 3.0]), box_length, 0)


def test_scaled_periodic_basis_vector_rejects_nan_in_period_vector():
    with pytest.raises(GBMakerConstructionValueError):
        _scaled_periodic_basis_vector(np.array([np.nan, 1.0, 1.0]), 10.0, 0)


def test_scaled_periodic_basis_vector_rejects_non_finite_scaled_vector():
    with pytest.raises(GBMakerConstructionValueError):
        _scaled_periodic_basis_vector(np.array([1e-308, 1e308, 0.0]), 1e308, 0)


# --------------------------------------------------------------------------------------
# _box_periodic_basis
# --------------------------------------------------------------------------------------


def test_box_periodic_basis_scales_orthogonal_basis_and_zeros_nonperiodic_axis():
    primitive_periods = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 5.0]])

    basis = _box_periodic_basis(
        primitive_periods,
        inplane_periodic=(True, False),
        box_lengths=(12.0, 15.0),
        epsilon=1e-10,
    )

    np.testing.assert_allclose(
        basis,
        np.array([[0.0, 12.0, 0.0], [0.0, 0.0, 0.0]]),
        atol=1e-12,
        rtol=0.0,
    )


def test_box_periodic_basis_preserves_tilted_components_while_matching_axis_lengths():
    primitive_periods = np.array([[2.0, 4.0, -1.0], [-3.0, 1.5, 5.0]])

    basis = _box_periodic_basis(
        primitive_periods,
        inplane_periodic=(True, True),
        box_lengths=(12.0, 15.0),
        epsilon=1e-10,
    )

    np.testing.assert_allclose(
        basis,
        np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]]),
        atol=1e-12,
        rtol=0.0,
    )


def test_box_periodic_basis_rejects_near_zero_selected_axis_projection():
    primitive_periods = np.array([[1.0, 1e-12, 0.0], [0.0, 0.0, 5.0]])

    with pytest.raises(GBMakerConstructionValueError):
        _box_periodic_basis(
            primitive_periods,
            inplane_periodic=(True, True),
            box_lengths=(12.0, 15.0),
            epsilon=1e-10,
        )


# --------------------------------------------------------------------------------------
# _selection_basis_vectors
# --------------------------------------------------------------------------------------


def test_selection_basis_vectors_uses_periodic_box_basis_for_both_axes():
    primitive_periods = np.array([[2.0, 4.0, -1.0], [-3.0, 1.5, 5.0]])

    basis = _selection_basis_vectors(
        primitive_periods,
        inplane_periodic=(True, True),
        box_lengths=(12.0, 15.0),
        epsilon=1e-10,
    )

    np.testing.assert_allclose(
        basis,
        np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]]),
        atol=1e-12,
        rtol=0.0,
    )


def test_selection_basis_vectors_uses_cartesian_unit_vector_for_nonperiodic_y():
    primitive_periods = np.array([[2.0, 4.0, -1.0], [-3.0, 1.5, 5.0]])

    basis = _selection_basis_vectors(
        primitive_periods,
        inplane_periodic=(False, True),
        box_lengths=(12.0, 15.0),
        epsilon=1e-10,
    )

    np.testing.assert_allclose(
        basis,
        np.array([[0.0, 1.0, 0.0], [-9.0, 4.5, 15.0]]),
        atol=1e-12,
        rtol=0.0,
    )


def test_selection_basis_vectors_uses_cartesian_unit_vector_for_nonperiodic_z():
    primitive_periods = np.array([[2.0, 4.0, -1.0], [-3.0, 1.5, 5.0]])

    basis = _selection_basis_vectors(
        primitive_periods,
        inplane_periodic=(True, False),
        box_lengths=(12.0, 15.0),
        epsilon=1e-10,
    )

    np.testing.assert_allclose(
        basis,
        np.array([[6.0, 12.0, -3.0], [0.0, 0.0, 1.0]]),
        atol=1e-12,
        rtol=0.0,
    )


# --------------------------------------------------------------------------------------
# _reduced_box_coordinates / _cartesian_from_box_coordinates
# --------------------------------------------------------------------------------------


def test_reduced_box_coordinates_handles_orthorhombic_basis():
    box_basis = np.array([[0.0, 12.0, 0.0], [0.0, 0.0, 15.0]])
    cartesian = np.array([[1.25, 6.0, 3.75], [4.5, 3.0, 12.0]])

    reduced = _reduced_box_coordinates(cartesian, box_basis, epsilon=1e-10)

    np.testing.assert_allclose(
        reduced,
        np.array([[1.25, 0.5, 0.25], [4.5, 0.25, 0.8]]),
        atol=1e-12,
        rtol=0.0,
    )


def test_cartesian_from_box_coordinates_handles_tilted_basis():
    box_basis = np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]])
    box_coordinates = np.array([[1.5, 0.25, 0.75], [-2.0, 0.5, 0.2]])

    cartesian = _cartesian_from_box_coordinates(box_coordinates, box_basis)

    np.testing.assert_allclose(
        cartesian,
        np.array([[-3.75, 6.375, 10.5], [-0.8, 6.9, 1.5]]),
        atol=1e-12,
        rtol=0.0,
    )


def test_reduced_box_coordinates_and_cartesian_from_box_coordinates_round_trip():
    box_basis = np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]])
    cartesian = np.array(
        [[-3.75, 6.375, 10.5], [-0.8, 6.9, 1.5], [2.25, 0.0, 7.5]]
    )

    reduced = _reduced_box_coordinates(cartesian, box_basis, epsilon=1e-10)
    reconstructed = _cartesian_from_box_coordinates(reduced, box_basis)

    np.testing.assert_allclose(reconstructed, cartesian, atol=1e-12, rtol=0.0)


def test_reduced_box_coordinates_preserves_vectorized_shape():
    box_basis = np.array([[6.0, 12.0, -3.0], [-9.0, 4.5, 15.0]])
    cartesian = np.array(
        [
            [[-3.75, 6.375, 10.5], [-0.8, 6.9, 1.5]],
            [[2.25, 0.0, 7.5], [1.5, 8.25, 0.0]],
        ]
    )

    reduced = _reduced_box_coordinates(cartesian, box_basis, epsilon=1e-10)
    reconstructed = _cartesian_from_box_coordinates(reduced, box_basis)

    assert reduced.shape == cartesian.shape
    np.testing.assert_allclose(reconstructed, cartesian, atol=1e-12, rtol=0.0)


# --------------------------------------------------------------------------------------
# _x_index_range
# --------------------------------------------------------------------------------------


def test_x_index_range_orthogonal_configuration_covers_slab():
    primitive_periods = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    rotated_unit_cell_basis = np.eye(3)
    x_bounds = np.array([0.0, 3.5])

    nx_range = _x_index_range(
        primitive_periods,
        rotated_unit_cell_basis,
        x_bounds,
        inplane_periodic=(True, True),
        box_lengths=(12.0, 15.0),
        epsilon=1e-10,
    )

    assert np.all(np.diff(nx_range) == 1)
    assert 0 in nx_range
    assert 3 in nx_range
    covered_min = nx_range[0]
    covered_max = nx_range[-1] + 1.0
    assert covered_min <= x_bounds[0]
    assert covered_max >= x_bounds[1]


def test_x_index_range_is_contiguous_and_includes_expected_indices_for_tilted_box():
    primitive_periods = np.array([[1.0, 2.0, 0.0], [-1.0, 0.0, 2.0]])
    rotated_unit_cell_basis = np.eye(3)
    x_bounds = np.array([0.0, 8.0])

    nx_range = _x_index_range(
        primitive_periods,
        rotated_unit_cell_basis,
        x_bounds,
        inplane_periodic=(True, True),
        box_lengths=(12.0, 15.0),
        epsilon=1e-10,
    )

    np.testing.assert_array_equal(
        nx_range,
        np.arange(nx_range[0], nx_range[-1] + 1, dtype=int),
    )
    for expected_index in (0, 1, 2):
        assert expected_index in nx_range


def test_x_index_range_raises_when_x_period_direction_has_zero_x_projection():
    primitive_periods = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    rotated_unit_cell_basis = np.eye(3)

    with pytest.raises(GBMakerConstructionValueError):
        _x_index_range(
            primitive_periods,
            rotated_unit_cell_basis,
            np.array([0.0, 5.0]),
            inplane_periodic=(True, True),
            box_lengths=(12.0, 15.0),
            epsilon=1e-10,
        )


# --------------------------------------------------------------------------------------
# _clip_complete_origins_to_cartesian_box
# --------------------------------------------------------------------------------------


def test_clip_complete_origins_to_cartesian_box_epsilon_controls_boundary_inclusion():
    # An atom at x=0.0 with x_min=1e-12 straddles the lower slab boundary. With
    # epsilon=1e-10: 0.0 >= 1e-12 - 1e-10 = -9.9e-11 -> included. With
    # epsilon=1e-13: 0.0 < 1e-12 - 1e-13 = 9e-13 -> excluded.
    boundary_atom = np.array([("Cu", 0.0, 5.0, 5.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([0], dtype=np.int64)
    x_bounds = np.array([1e-12, 10.0])

    result_large, _ = _clip_complete_origins_to_cartesian_box(
        boundary_atom,
        origin_ids,
        x_bounds,
        1,
        inplane_periodic=(False, False),
        box_lengths=(10.0, 10.0),
        epsilon=1e-10,
    )
    assert len(result_large) == 1

    result_small, _ = _clip_complete_origins_to_cartesian_box(
        boundary_atom,
        origin_ids,
        x_bounds,
        1,
        inplane_periodic=(False, False),
        box_lengths=(10.0, 10.0),
        epsilon=1e-13,
    )
    assert len(result_small) == 0


# --------------------------------------------------------------------------------------
# _complete_origin_atom_mask
# --------------------------------------------------------------------------------------


def test_complete_origin_atom_mask_fast_path_keeps_complete_groups_and_drops_incomplete():
    # Contiguous, unique groups of size 2 -> eligible for the grouped fast path.
    origin_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    atom_mask = np.array([True, True, True, False])

    result = _complete_origin_atom_mask(atom_mask, origin_ids, 2)

    np.testing.assert_array_equal(result, [True, True, False, False])


def test_complete_origin_atom_mask_general_path_for_noncontiguous_origin_ids():
    # Interleaved origin IDs are not contiguous groups, so the fast-path reshape
    # check fails and the function falls back to the origin-ID-count path.
    origin_ids = np.array([0, 1, 0, 1], dtype=np.int64)
    atom_mask = np.array([True, False, True, True])

    result = _complete_origin_atom_mask(atom_mask, origin_ids, 2)

    np.testing.assert_array_equal(result, [True, False, True, False])


def test_complete_origin_atom_mask_general_path_triggered_by_duplicate_group_ids():
    # len(atom_mask) % basis_size == 0, but the reshaped groups don't have unique
    # IDs, so grouped_unique is False and the fallback path is used instead.
    origin_ids = np.array([0, 0, 0, 0], dtype=np.int64)
    atom_mask = np.array([True, True, True, True])

    result = _complete_origin_atom_mask(atom_mask, origin_ids, 2)

    # All four atoms share origin 0, so its total count (4) never equals basis_size
    # (2); the whole origin is dropped.
    np.testing.assert_array_equal(result, [False, False, False, False])


def test_complete_origin_atom_mask_basis_size_one_is_pass_through():
    atom_mask = np.array([True, False, True])
    origin_ids = np.array([0, 1, 2], dtype=np.int64)

    result = _complete_origin_atom_mask(atom_mask, origin_ids, 1)

    np.testing.assert_array_equal(result, atom_mask)


def test_complete_origin_atom_mask_empty_input_returns_copy():
    atom_mask = np.array([], dtype=bool)
    origin_ids = np.array([], dtype=np.int64)

    result = _complete_origin_atom_mask(atom_mask, origin_ids, 1)

    assert result.shape == (0,)
    assert result is not atom_mask


def test_complete_origin_atom_mask_rejects_non_1d_atom_mask():
    with pytest.raises(GBMakerConstructionValueError):
        _complete_origin_atom_mask(
            np.array([[True, False]]), np.array([0, 1], dtype=np.int64), 1
        )


def test_complete_origin_atom_mask_rejects_non_bool_atom_mask():
    with pytest.raises(GBMakerConstructionValueError):
        _complete_origin_atom_mask(
            np.array([1, 0]), np.array([0, 1], dtype=np.int64), 1
        )


def test_complete_origin_atom_mask_rejects_non_1d_origin_ids():
    with pytest.raises(GBMakerConstructionValueError):
        _complete_origin_atom_mask(
            np.array([True, False]), np.array([[0], [1]], dtype=np.int64), 1
        )


def test_complete_origin_atom_mask_rejects_non_integer_origin_ids():
    with pytest.raises(GBMakerConstructionValueError):
        _complete_origin_atom_mask(np.array([True, False]), np.array([0.0, 1.0]), 1)


def test_complete_origin_atom_mask_rejects_mismatched_lengths():
    with pytest.raises(GBMakerConstructionValueError):
        _complete_origin_atom_mask(
            np.array([True, False, True]), np.array([0, 1], dtype=np.int64), 1
        )


@pytest.mark.parametrize("basis_size", [True, 2.5, 0, -1, "2"])
def test_complete_origin_atom_mask_rejects_invalid_basis_size(basis_size):
    with pytest.raises(GBMakerConstructionValueError):
        _complete_origin_atom_mask(
            np.array([True, False]), np.array([0, 1], dtype=np.int64), basis_size
        )


# --------------------------------------------------------------------------------------
# _filter_complete_origins
# --------------------------------------------------------------------------------------


def test_filter_complete_origins_keeps_only_complete_groups_and_returns_copies():
    atoms = np.array(
        [
            ("Cu", 0.0, 0.0, 0.0),
            ("Cu", 1.0, 0.0, 0.0),
            ("Fe", 2.0, 0.0, 0.0),
            ("Fe", 3.0, 0.0, 0.0),
        ],
        dtype=Atom.atom_dtype,
    )
    origin_ids = np.array([0, 0, 1, 1], dtype=np.int64)
    atom_mask = np.array([True, True, True, False])

    filtered_atoms, filtered_origin_ids = _filter_complete_origins(
        atoms, origin_ids, atom_mask, 2
    )

    np.testing.assert_array_equal(filtered_atoms["name"], ["Cu", "Cu"])
    np.testing.assert_array_equal(filtered_origin_ids, [0, 0])

    # Returned arrays must be independent copies, not views into the inputs.
    filtered_atoms["x"][0] = 99.0
    assert atoms["x"][0] == 0.0
    filtered_origin_ids[0] = 99
    assert origin_ids[0] == 0


def test_filter_complete_origins_rejects_mismatched_atoms_and_origin_ids_lengths():
    atoms = np.array([("Cu", 0.0, 0.0, 0.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([0, 1], dtype=np.int64)

    with pytest.raises(GBMakerConstructionValueError):
        _filter_complete_origins(atoms, origin_ids, np.array([True]), 1)


def test_filter_complete_origins_propagates_mask_validation_error():
    atoms = np.array([("Cu", 0.0, 0.0, 0.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([0], dtype=np.int64)

    with pytest.raises(GBMakerConstructionValueError):
        _filter_complete_origins(atoms, origin_ids, np.array([True]), 0)


# --------------------------------------------------------------------------------------
# _deduplicate_complete_origins
# --------------------------------------------------------------------------------------


def test_deduplicate_complete_origins_removes_exact_duplicate_groups_keeping_first():
    atoms = np.array(
        [
            ("Cu", 0.0, 0.0, 0.0),
            ("Fe", 1.0, 0.0, 0.0),  # origin 0
            ("Cu", 0.0, 0.0, 0.0),
            ("Fe", 1.0, 0.0, 0.0),  # origin 1, duplicate of origin 0
            ("Cu", 5.0, 0.0, 0.0),
            ("Fe", 6.0, 0.0, 0.0),  # origin 2, distinct
        ],
        dtype=Atom.atom_dtype,
    )
    origin_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)

    deduped_atoms, deduped_origin_ids = _deduplicate_complete_origins(
        atoms, origin_ids, 2, epsilon=1e-6
    )

    np.testing.assert_array_equal(deduped_origin_ids, [0, 0, 2, 2])
    np.testing.assert_allclose(deduped_atoms["x"], [0.0, 1.0, 5.0, 6.0])


def test_deduplicate_complete_origins_epsilon_controls_quantization():
    atoms = np.array(
        [("Cu", 0.0, 0.0, 0.0), ("Cu", 1e-9, 0.0, 0.0)],
        dtype=Atom.atom_dtype,
    )
    origin_ids = np.array([0, 1], dtype=np.int64)

    deduped_loose, _ = _deduplicate_complete_origins(
        atoms, origin_ids, 1, epsilon=1e-6
    )
    assert len(deduped_loose) == 1

    deduped_tight, _ = _deduplicate_complete_origins(
        atoms, origin_ids, 1, epsilon=1e-12
    )
    assert len(deduped_tight) == 2


def test_deduplicate_complete_origins_empty_input_returns_copies():
    atoms = np.array([], dtype=Atom.atom_dtype)
    origin_ids = np.array([], dtype=np.int64)

    deduped_atoms, deduped_origin_ids = _deduplicate_complete_origins(
        atoms, origin_ids, 1, epsilon=1e-6
    )

    assert len(deduped_atoms) == 0
    assert deduped_atoms is not atoms
    assert deduped_origin_ids is not origin_ids


def test_deduplicate_complete_origins_rejects_noncontiguous_origin_groups():
    atoms = np.array(
        [
            ("Cu", 0.0, 0.0, 0.0),
            ("Fe", 1.0, 0.0, 0.0),
            ("Cu", 2.0, 0.0, 0.0),
            ("Fe", 3.0, 0.0, 0.0),
        ],
        dtype=Atom.atom_dtype,
    )
    origin_ids = np.array([0, 1, 1, 0], dtype=np.int64)

    with pytest.raises(GBMakerConstructionValueError):
        _deduplicate_complete_origins(atoms, origin_ids, 2, epsilon=1e-6)


def test_deduplicate_complete_origins_rejects_length_not_divisible_by_basis_size():
    atoms = np.array(
        [("Cu", 0.0, 0.0, 0.0), ("Fe", 1.0, 0.0, 0.0), ("Fe", 2.0, 0.0, 0.0)],
        dtype=Atom.atom_dtype,
    )
    origin_ids = np.array([0, 0, 1], dtype=np.int64)

    with pytest.raises(GBMakerConstructionValueError):
        _deduplicate_complete_origins(atoms, origin_ids, 2, epsilon=1e-6)


def test_deduplicate_complete_origins_rejects_mismatched_lengths():
    atoms = np.array([("Cu", 0.0, 0.0, 0.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([0, 1], dtype=np.int64)

    with pytest.raises(GBMakerConstructionValueError):
        _deduplicate_complete_origins(atoms, origin_ids, 1, epsilon=1e-6)


def test_deduplicate_complete_origins_rejects_non_1d_origin_ids():
    atoms = np.array([("Cu", 0.0, 0.0, 0.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([[0]], dtype=np.int64)

    with pytest.raises(GBMakerConstructionValueError):
        _deduplicate_complete_origins(atoms, origin_ids, 1, epsilon=1e-6)


def test_deduplicate_complete_origins_rejects_non_integer_origin_ids():
    atoms = np.array([("Cu", 0.0, 0.0, 0.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([0.0])

    with pytest.raises(GBMakerConstructionValueError):
        _deduplicate_complete_origins(atoms, origin_ids, 1, epsilon=1e-6)


@pytest.mark.parametrize("basis_size", [True, 2.5, 0, -1])
def test_deduplicate_complete_origins_rejects_invalid_basis_size(basis_size):
    atoms = np.array([("Cu", 0.0, 0.0, 0.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([0], dtype=np.int64)

    with pytest.raises(GBMakerConstructionValueError):
        _deduplicate_complete_origins(atoms, origin_ids, basis_size, epsilon=1e-6)


# --------------------------------------------------------------------------------------
# _select_complete_origins_in_box_basis
# --------------------------------------------------------------------------------------


def test_select_complete_origins_in_box_basis_axis_aligned_fast_path():
    # Both primitive periods lie purely in the y/z plane, so the selection basis
    # has a zero x column and the axis-aligned fast path is used.
    primitive_periods = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])

    atoms = np.array(
        [
            ("Cu", 0.0, 1.0, 1.0),  # clearly interior -> kept unchanged
            ("Cu", 0.0, 3.0 - 1e-12, 1.0),  # just above the y boundary -> snaps to 0
            ("Cu", 0.0, 1.0, -5e-11),  # just below the z boundary -> snaps to 0
            ("Cu", 0.0, -0.5, 1.0),  # well outside the periodic tolerance -> dropped
            ("Cu", 5.0, 1.0, 1.0),  # passes y/z but fails the final x-slab check
        ],
        dtype=Atom.atom_dtype,
    )
    origin_ids = np.arange(len(atoms), dtype=np.int64)

    selected_atoms, selected_origin_ids = _select_complete_origins_in_box_basis(
        atoms,
        origin_ids,
        primitive_periods,
        np.array([-1.0, 1.0]),
        1,
        inplane_periodic=(True, True),
        box_lengths=(3.0, 4.0),
        epsilon=1e-10,
    )

    np.testing.assert_array_equal(selected_origin_ids, [0, 1, 2])
    np.testing.assert_allclose(
        selected_atoms["y"], [1.0, 0.0, 1.0], atol=1e-9, rtol=0.0
    )
    np.testing.assert_allclose(
        selected_atoms["z"], [1.0, 1.0, 0.0], atol=1e-9, rtol=0.0
    )


def test_select_complete_origins_in_box_basis_fast_path_returns_empty_when_nothing_selected():
    atoms = np.array([("Cu", 0.0, -5.0, -5.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([0], dtype=np.int64)
    primitive_periods = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])

    selected_atoms, selected_origin_ids = _select_complete_origins_in_box_basis(
        atoms,
        origin_ids,
        primitive_periods,
        np.array([-1.0, 1.0]),
        1,
        inplane_periodic=(True, True),
        box_lengths=(3.0, 4.0),
        epsilon=1e-10,
    )

    assert len(selected_atoms) == 0
    assert len(selected_origin_ids) == 0


def test_select_complete_origins_in_box_basis_general_path_drops_incomplete_origin():
    # The primitive y period has a nonzero x component, so the selection basis has
    # a nonzero x column and the general mixed-basis path is used.
    primitive_periods = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 4.0]])

    atoms = np.array(
        [
            ("Cu", 0.0, 1.0, 1.0),  # origin 0, both atoms interior -> kept
            ("Fe", 0.2, 1.5, 2.0),  # origin 0
            ("Cu", 0.0, 1.0, 1.0),  # origin 1, interior
            ("Fe", 0.0, 10.0, 1.0),  # origin 1, well outside the y period -> dropped
        ],
        dtype=Atom.atom_dtype,
    )
    origin_ids = np.array([0, 0, 1, 1], dtype=np.int64)

    selected_atoms, selected_origin_ids = _select_complete_origins_in_box_basis(
        atoms,
        origin_ids,
        primitive_periods,
        np.array([-1.0, 1.0]),
        2,
        inplane_periodic=(True, True),
        box_lengths=(3.0, 4.0),
        epsilon=1e-10,
    )

    np.testing.assert_array_equal(selected_origin_ids, [0, 0])
    np.testing.assert_array_equal(selected_atoms["name"], ["Cu", "Fe"])
    np.testing.assert_allclose(selected_atoms["x"], [0.0, 0.2], atol=1e-9, rtol=0.0)


def test_select_complete_origins_in_box_basis_general_path_clips_nonperiodic_axis():
    # y is periodic (tilted into x); z is non-periodic and clipped to box_lengths[1].
    primitive_periods = np.array([[1.0, 3.0, 0.0], [0.0, 0.0, 0.0]])

    atoms = np.array(
        [
            ("Cu", 0.0, 1.0, 2.0),  # interior on both axes -> kept
            ("Fe", 0.0, 1.0, 6.0),  # z outside the non-periodic box extent -> dropped
        ],
        dtype=Atom.atom_dtype,
    )
    origin_ids = np.array([0, 1], dtype=np.int64)

    selected_atoms, selected_origin_ids = _select_complete_origins_in_box_basis(
        atoms,
        origin_ids,
        primitive_periods,
        np.array([-1.0, 1.0]),
        1,
        inplane_periodic=(True, False),
        box_lengths=(3.0, 5.0),
        epsilon=1e-10,
    )

    np.testing.assert_array_equal(selected_origin_ids, [0])
    np.testing.assert_array_equal(selected_atoms["name"], ["Cu"])
    np.testing.assert_allclose(selected_atoms["y"], [1.0], atol=1e-9, rtol=0.0)
    np.testing.assert_allclose(selected_atoms["z"], [2.0], atol=1e-9, rtol=0.0)


def test_select_complete_origins_in_box_basis_raises_for_singular_selection_basis():
    # Both primitive periods point in the same direction, so the y/z columns of the
    # resulting (nonzero-x-column) selection basis are linearly dependent.
    atoms = np.array([("Cu", 0.0, 0.0, 0.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([0], dtype=np.int64)
    primitive_periods = np.array([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])

    with pytest.raises(GBMakerConstructionValueError):
        _select_complete_origins_in_box_basis(
            atoms,
            origin_ids,
            primitive_periods,
            np.array([-1.0, 1.0]),
            1,
            inplane_periodic=(True, True),
            box_lengths=(2.0, 8.0),
            epsilon=1e-10,
        )


@pytest.mark.parametrize(
    "x_bounds",
    [
        np.array([1.0, 1.0]),
        np.array([1.0, 0.0]),
        np.array([np.nan, 1.0]),
        np.array([0.0]),
    ],
)
def test_select_complete_origins_in_box_basis_rejects_invalid_x_bounds(x_bounds):
    atoms = np.array([("Cu", 0.0, 0.0, 0.0)], dtype=Atom.atom_dtype)
    origin_ids = np.array([0], dtype=np.int64)
    primitive_periods = np.array([[0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])

    with pytest.raises(GBMakerConstructionValueError):
        _select_complete_origins_in_box_basis(
            atoms,
            origin_ids,
            primitive_periods,
            x_bounds,
            1,
            inplane_periodic=(True, True),
            box_lengths=(3.0, 4.0),
            epsilon=1e-10,
        )
