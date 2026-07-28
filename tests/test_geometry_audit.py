# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for deterministic periodic bicrystal geometry auditing."""

import numpy as np
import pytest

from GBOpt.geometry_audit import (
    GeometryAuditError,
    GeometryAuditThresholds,
    audit_bicrystal_geometry,
)


BOX = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
CENTRAL_PLANE_X = 5.0


def _flat_bicrystal() -> tuple[np.ndarray, np.ndarray]:
    yz = np.array(
        [
            [2.5, 2.5],
            [2.5, 7.5],
            [7.5, 2.5],
            [7.5, 7.5],
        ]
    )
    left = np.array(
        [[x, y, z] for x in (1.0, 4.0) for y, z in yz],
        dtype=np.float64,
    )
    right = np.array(
        [[x, y, z] for x in (6.0, 9.0) for y, z in yz],
        dtype=np.float64,
    )
    return left, right


def test_flat_interfaces_return_uniform_known_gaps():
    left, right = _flat_bicrystal()

    result = audit_bicrystal_geometry(
        left,
        right,
        BOX,
        central_plane_x=CENTRAL_PLANE_X,
        bins=(2, 2),
    )

    assert result.status == "ok"
    assert result.reasons == ()
    assert result.central_interface.minimum_angstrom == pytest.approx(2.0)
    assert result.central_interface.median_angstrom == pytest.approx(2.0)
    assert result.central_interface.percentile_95_angstrom == pytest.approx(2.0)
    assert result.central_interface.maximum_angstrom == pytest.approx(2.0)
    assert result.central_interface.range_angstrom == pytest.approx(0.0)
    assert result.periodic_interface.minimum_angstrom == pytest.approx(2.0)
    assert result.periodic_interface.maximum_angstrom == pytest.approx(2.0)
    assert result.periodic_interface.range_angstrom == pytest.approx(0.0)
    assert result.central_interface.valid_bins == 4
    assert result.periodic_interface.valid_bins == 4
    assert result.nearest_neighbors.left_internal_min_angstrom == pytest.approx(3.0)
    assert result.nearest_neighbors.right_internal_min_angstrom == pytest.approx(3.0)
    assert result.nearest_neighbors.central_cross_min_angstrom == pytest.approx(2.0)
    assert result.nearest_neighbors.periodic_cross_min_angstrom == pytest.approx(2.0)
    assert result.nearest_neighbors.periodic_duplicate_count == 0


def test_localized_channel_is_classified_as_suspicious():
    left, right = _flat_bicrystal()
    target = (right[:, 0] == 6.0) & (right[:, 1] == 2.5) & (right[:, 2] == 2.5)
    right[target, 0] = 8.0

    result = audit_bicrystal_geometry(
        left,
        right,
        BOX,
        central_plane_x=CENTRAL_PLANE_X,
        bins=(2, 2),
    )

    assert result.status == "suspicious"
    assert "central_interface_heavy_gap_tail" in result.reasons
    assert result.central_interface.minimum_angstrom == pytest.approx(2.0)
    assert result.central_interface.maximum_angstrom == pytest.approx(4.0)


def test_empty_interface_bin_fraction_is_reported_and_classified():
    left, right = _flat_bicrystal()
    keep = ~((right[:, 1] == 2.5) & (right[:, 2] == 2.5))
    right = right[keep]

    result = audit_bicrystal_geometry(
        left,
        right,
        BOX,
        central_plane_x=CENTRAL_PLANE_X,
        bins=(2, 2),
        thresholds=GeometryAuditThresholds(max_empty_bin_fraction=0.20),
    )

    assert result.status == "suspicious"
    assert result.central_interface.empty_right_bin_fraction == pytest.approx(0.25)
    assert result.periodic_interface.empty_right_bin_fraction == pytest.approx(0.25)
    assert "central_interface_excess_empty_right_bins" in result.reasons
    assert "periodic_interface_excess_empty_right_bins" in result.reasons


def test_periodic_duplicate_site_is_counted_once():
    left = np.array(
        [
            [0.0, 2.5, 2.5],
            [4.0, 2.5, 2.5],
            [1.0, 7.5, 7.5],
            [4.0, 7.5, 7.5],
        ]
    )
    right = np.array(
        [
            [6.0, 2.5, 2.5],
            [10.0, 2.5, 2.5],
            [6.0, 7.5, 7.5],
            [9.0, 7.5, 7.5],
        ]
    )

    result = audit_bicrystal_geometry(
        left,
        right,
        BOX,
        central_plane_x=CENTRAL_PLANE_X,
        bins=(1, 2),
    )

    assert result.nearest_neighbors.periodic_duplicate_count == 1
    assert result.status == "suspicious"
    assert "periodic_duplicate_sites" in result.reasons


def test_metrics_are_invariant_to_atom_order_and_periodic_wrapping():
    left, right = _flat_bicrystal()
    reference = audit_bicrystal_geometry(
        left,
        right,
        BOX,
        central_plane_x=CENTRAL_PLANE_X,
        bins=(2, 2),
    )

    wrapped_left = left[[6, 0, 5, 3, 7, 1, 4, 2]].copy()
    wrapped_right = right[[1, 7, 0, 6, 2, 4, 3, 5]].copy()
    wrapped_left[:, 1] += 10.0
    wrapped_right[:, 2] -= 20.0

    result = audit_bicrystal_geometry(
        wrapped_left,
        wrapped_right,
        BOX,
        central_plane_x=CENTRAL_PLANE_X,
        bins=(2, 2),
    )

    assert result.to_dict() == reference.to_dict()


def test_structured_atom_arrays_are_supported():
    left, right = _flat_bicrystal()
    dtype = np.dtype([("name", "U2"), ("x", float), ("y", float), ("z", float)])
    structured_left = np.array(
        [("U", *position) for position in left],
        dtype=dtype,
    )
    structured_right = np.array(
        [("O", *position) for position in right],
        dtype=dtype,
    )

    result = audit_bicrystal_geometry(
        structured_left,
        structured_right,
        BOX,
        central_plane_x=CENTRAL_PLANE_X,
        bins=(2, 2),
    )

    assert result.status == "ok"
    assert result.nearest_neighbors.periodic_duplicate_count == 0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"box_dims": np.eye(3)}, "shape", id="bad-box-shape"),
        pytest.param({"central_plane_x": 0.0}, "inside the x box", id="plane-on-face"),
        pytest.param({"bins": (0, 2)}, "bins_y", id="zero-bin-count"),
        pytest.param(
            {"min_bins_per_axis": 8, "max_bins_per_axis": 4},
            "must not exceed",
            id="reversed-auto-bin-limits",
        ),
    ],
)
def test_invalid_inputs_raise_geometry_audit_error(kwargs, match):
    left, right = _flat_bicrystal()
    arguments = {
        "left_atoms": left,
        "right_atoms": right,
        "box_dims": BOX,
        "central_plane_x": CENTRAL_PLANE_X,
        "bins": (2, 2),
    }
    arguments.update(kwargs)

    with pytest.raises(GeometryAuditError, match=match):
        audit_bicrystal_geometry(**arguments)


def test_roundoff_below_periodic_lower_bound_is_clamped_inside_box():
    left, right = _flat_bicrystal()
    left[0, 1] = -1.0e-16
    right[0, 2] = -1.0e-16

    result = audit_bicrystal_geometry(
        left,
        right,
        BOX,
        central_plane_x=CENTRAL_PLANE_X,
        bins=(2, 2),
    )

    assert result.status in {"ok", "suspicious"}
    assert result.nearest_neighbors.left_internal_min_angstrom is not None
    assert result.nearest_neighbors.right_internal_min_angstrom is not None
