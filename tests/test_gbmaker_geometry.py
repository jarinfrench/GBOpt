# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.gbmaker.geometry import (
    _box_periodic_basis,
    _cartesian_from_box_coordinates,
    _clip_complete_origins_to_cartesian_box,
    _reduced_box_coordinates,
    _reduced_coordinate_tolerance,
    _scaled_periodic_basis_vector,
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
