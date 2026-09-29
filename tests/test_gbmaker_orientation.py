# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from GBOpt.BoundarySpec import BoundaryEmbedding
from GBOpt.gbmaker.orientation import (
    _approximate_rotation_matrix_as_int,
    _decompose_misorientation,
    _reduce_integer_row,
    _row_angle_error_deg,
    _x_period,
    resolve_orientation,
)
from GBOpt.gbmaker.types import GBMakerConstructionValueError

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _embedding(**overrides) -> BoundaryEmbedding:
    kwargs = {
        "P": np.eye(3, dtype=int),
        "Q": np.eye(3, dtype=int),
        "R_left": np.eye(3),
        "R_right": np.eye(3),
        "exact": True,
        "coherent": True,
        "source": "pq",
    }
    kwargs.update(overrides)
    return BoundaryEmbedding(**kwargs)


# --------------------------------------------------------------------------------------
# _reduce_integer_row
# --------------------------------------------------------------------------------------


def test_reduce_integer_row_basic():
    result = _reduce_integer_row(np.array([4, 6, 2]))
    np.testing.assert_array_equal(result, np.array([2, 3, 1]))


def test_reduce_integer_row_already_reduced():
    result = _reduce_integer_row(np.array([1, 2, 3]))
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))


def test_reduce_integer_row_all_zeros():
    result = _reduce_integer_row(np.array([0, 0, 0]))
    np.testing.assert_array_equal(result, np.array([0, 0, 0]))


def test_reduce_integer_row_with_negatives():
    result = _reduce_integer_row(np.array([-4, 6, -2]))
    np.testing.assert_array_equal(result, np.array([-2, 3, -1]))


# --------------------------------------------------------------------------------------
# _row_angle_error_deg
# --------------------------------------------------------------------------------------


def test_row_angle_error_parallel():
    err = _row_angle_error_deg(np.array([1.0, 0.0, 0.0]), np.array([5, 0, 0]))
    assert err == pytest.approx(0.0, abs=1e-10)


def test_row_angle_error_perpendicular():
    err = _row_angle_error_deg(np.array([1.0, 0.0, 0.0]), np.array([0, 1, 0]))
    assert err == pytest.approx(90.0, abs=1e-10)


def test_row_angle_error_antiparallel():
    err = _row_angle_error_deg(np.array([1.0, 0.0, 0.0]), np.array([-1, 0, 0]))
    assert err == pytest.approx(180.0, abs=1e-10)


def test_row_angle_error_zero_vector():
    err = _row_angle_error_deg(np.array([1.0, 0.0, 0.0]), np.array([0, 0, 0]))
    assert err == 180.0


# --------------------------------------------------------------------------------------
# _approximate_rotation_matrix_as_int
# --------------------------------------------------------------------------------------


def test_approximate_rotation_matrix_as_int():
    rotation_matrix = np.array(
        [
            [0.70710678, 0.5, 0.5],
            [0.70710678, -0.5, -0.5],
            [0.0, 0.70710678, -0.70710678],
        ]
    )
    approx_matrix = _approximate_rotation_matrix_as_int(rotation_matrix)
    expected_matrix = np.array(
        [
            [7, 5, 5],
            [7, -5, -5],
            [0, 1, -1],
        ]
    )
    np.testing.assert_array_equal(approx_matrix, expected_matrix)


def test_approx_matrix_sigma5_within_tolerance():
    """Sigma5 [001] 36.87 - all rows must be within 0.5 of the float matrix."""
    theta = math.radians(36.869898)
    R = Rotation.from_euler("z", theta).as_matrix()
    approx = _approximate_rotation_matrix_as_int(R)
    for ref_row, approx_row in zip(R, approx.astype(float), strict=True):
        err = _row_angle_error_deg(ref_row, approx_row)
        assert err <= 0.5


# --------------------------------------------------------------------------------------
# _decompose_misorientation
# --------------------------------------------------------------------------------------


def test_decompose_misorientation_zero_angles_gives_identity_rotations():
    mis_angles, incl_angles, R_mis, R_incl = _decompose_misorientation(np.zeros(5))
    np.testing.assert_array_equal(mis_angles, np.zeros(3))
    np.testing.assert_array_equal(incl_angles, np.zeros(2))
    np.testing.assert_allclose(R_mis, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(R_incl, np.eye(3), atol=1e-12)


def test_decompose_misorientation_splits_misorientation_and_inclination():
    theta = math.radians(36.868698)
    misorientation = np.array([theta, 0.1, 0.2, 0.3, 0.4])
    mis_angles, incl_angles, R_mis, R_incl = _decompose_misorientation(misorientation)
    np.testing.assert_array_equal(mis_angles, misorientation[:3])
    np.testing.assert_array_equal(incl_angles, misorientation[3:])
    expected_R_mis = Rotation.from_euler("ZXZ", misorientation[:3]).as_matrix()
    expected_R_incl = (
        Rotation.from_euler("z", misorientation[4])
        * Rotation.from_euler("y", misorientation[3])
    ).as_matrix()
    np.testing.assert_allclose(R_mis, expected_R_mis)
    np.testing.assert_allclose(R_incl, expected_R_incl)


# --------------------------------------------------------------------------------------
# _x_period
# --------------------------------------------------------------------------------------


def test_x_period_uses_row_zero_norm():
    rows = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=object)
    assert _x_period(rows, a0=3.61) == pytest.approx(3.61 * math.sqrt(2))


def test_x_period_rejects_zero_row():
    rows = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=object)
    with pytest.raises(GBMakerConstructionValueError):
        _x_period(rows, a0=3.61)


# --------------------------------------------------------------------------------------
# resolve_orientation
# --------------------------------------------------------------------------------------


def test_resolve_orientation_exact_embedding_uses_pq_rows_directly():
    """Exact P/Q rows must be carried through unchanged, never re-approximated."""
    P = np.array([[3, 1, 1], [-1, 2, 1], [-1, -1, 2]], dtype=int)
    Q = np.array([[3, 1, 1], [-1, 2, 1], [-1, -1, 2]], dtype=int)
    theta = math.radians(36.869898)
    # A rotation far from the exact P/Q rows: if the exact path were bypassed and the
    # rows were re-derived from R_left/R_right, they would not equal P/Q.
    R = Rotation.from_euler("z", theta).as_matrix()
    embedding = _embedding(P=P, Q=Q, R_left=R, R_right=R, exact=True, source="pq")

    state = resolve_orientation(np.zeros(5), embedding=embedding, a0=3.61)

    np.testing.assert_array_equal(state.left_periodic_miller_rows, P)
    np.testing.assert_array_equal(state.right_periodic_miller_rows, Q)
    np.testing.assert_array_equal(state.R_left, R)
    np.testing.assert_array_equal(state.R_right, R)


def test_resolve_orientation_non_exact_embedding_approximates_rows():
    theta = math.radians(36.869898)
    R = Rotation.from_euler("z", theta).as_matrix()
    embedding = _embedding(
        P=None, Q=None, R_left=R, R_right=R, exact=False, source="five_dof"
    )

    state = resolve_orientation(np.zeros(5), embedding=embedding, a0=3.61)

    expected_rows = _approximate_rotation_matrix_as_int(R)
    np.testing.assert_array_equal(
        state.left_periodic_miller_rows.astype(int), expected_rows
    )
    np.testing.assert_array_equal(
        state.right_periodic_miller_rows.astype(int), expected_rows
    )


def test_resolve_orientation_legacy_path_composes_rotations_from_euler_angles():
    theta = math.radians(36.868698)
    misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])

    state = resolve_orientation(misorientation, embedding=None, a0=3.61)

    _, _, R_mis, R_incl = _decompose_misorientation(misorientation)
    np.testing.assert_allclose(state.R_left, R_incl)
    np.testing.assert_allclose(state.R_right, np.dot(R_incl, R_mis))
    assert state.embedding is None


def test_resolve_orientation_coherent_embedding_marks_both_axes_periodic():
    embedding = _embedding(coherent=True, source="pq")
    state = resolve_orientation(np.zeros(5), embedding=embedding, a0=3.61)
    assert state.inplane_periodic == (True, True)


def test_resolve_orientation_incoherent_embedding_marks_both_axes_nonperiodic():
    embedding = _embedding(coherent=False, source="pq")
    state = resolve_orientation(np.zeros(5), embedding=embedding, a0=3.61)
    assert state.inplane_periodic == (False, False)


def test_resolve_orientation_legacy_path_warns_and_marks_axis_nonperiodic_over_threshold():
    theta = math.radians(36.868698)
    misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])

    with pytest.warns(UserWarning, match="non-periodic along"):
        state = resolve_orientation(
            misorientation, embedding=None, a0=3.61, threshold=1e-6
        )

    assert state.inplane_periodic == (False, False)


def test_resolve_orientation_does_not_mutate_embedding():
    embedding = _embedding(coherent=True, source="pq")
    resolve_orientation(np.zeros(5), embedding=embedding, a0=3.61)
    assert embedding.coherent is True
    np.testing.assert_array_equal(embedding.P, np.eye(3, dtype=int))
