# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import dataclasses

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
from GBOpt.interface.model import InterfaceCandidate
from GBOpt.interface.types import (
    InterfaceCandidateTypeError,
    InterfaceCandidateValueError,
)

_BOX_DIMS = np.asarray([[0.0, 20.0], [0.0, 10.0], [0.0, 10.0]], dtype=float)
_GB_PLANE_X = 10.0
_LEFT_BOUNDS = (0.0, 9.5)
_RIGHT_BOUNDS = (10.5, 20.0)
_TOLERANCE = 1.0e-8


def _atoms() -> np.ndarray:
    return np.asarray(
        [
            ("U", 3.0, 1.0, 1.0),
            ("O", 7.5, 2.0, 3.0),
            ("O", 12.5, 4.0, 5.0),
            ("U", 17.0, 6.0, 7.0),
        ],
        dtype=Atom.atom_dtype,
    )


def _labels() -> np.ndarray:
    return np.asarray(
        [LEFT_GRAIN_LABEL, LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
        dtype=np.int8,
    )


def _make_candidate(**overrides) -> InterfaceCandidate:
    kwargs = {
        "atoms": _atoms(),
        "box_dims": _BOX_DIMS,
        "gb_plane_x": _GB_PLANE_X,
        "left_grain_x_bounds": _LEFT_BOUNDS,
        "right_grain_x_bounds": _RIGHT_BOUNDS,
        "grain_labels": _labels(),
        "inplane_periodic": (True, True),
        "normal_topology": BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        "coordinate_tolerance": _TOLERANCE,
        "interface_separation": 0.25,
    }
    kwargs.update(overrides)
    return InterfaceCandidate(**kwargs)


# --------------------------------------------------------------------------------------
# Valid construction, immutability, and defensive copying
# --------------------------------------------------------------------------------------


def test_valid_construction_preserves_all_geometry_fields():
    candidate = _make_candidate()

    np.testing.assert_array_equal(candidate.atoms, _atoms())
    np.testing.assert_array_equal(candidate.box_dims, _BOX_DIMS)
    assert candidate.gb_plane_x == pytest.approx(_GB_PLANE_X)
    np.testing.assert_array_equal(candidate.left_grain_x_bounds, np.asarray(_LEFT_BOUNDS))
    np.testing.assert_array_equal(
        candidate.right_grain_x_bounds, np.asarray(_RIGHT_BOUNDS)
    )
    np.testing.assert_array_equal(candidate.grain_labels, _labels())
    assert candidate.inplane_periodic == (True, True)
    assert candidate.normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    assert candidate.coordinate_tolerance == pytest.approx(_TOLERANCE)
    assert candidate.interface_separation == pytest.approx(0.25)


def test_periodic_outer_x_interface_matches_topology():
    periodic = _make_candidate(normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL)
    slab = _make_candidate(normal_topology=BoundaryNormalTopology.SINGLE_INTERFACE_SLAB)

    assert periodic.periodic_outer_x_interface is True
    assert slab.periodic_outer_x_interface is False


def test_candidate_is_frozen():
    candidate = _make_candidate()

    with pytest.raises(dataclasses.FrozenInstanceError):
        candidate.gb_plane_x = 5.0  # ty: ignore[invalid-assignment]


def test_atoms_property_is_defensive_and_read_only():
    source = _atoms()
    candidate = InterfaceCandidate(
        atoms=source,
        box_dims=_BOX_DIMS,
        gb_plane_x=_GB_PLANE_X,
        left_grain_x_bounds=_LEFT_BOUNDS,
        right_grain_x_bounds=_RIGHT_BOUNDS,
        grain_labels=_labels(),
        inplane_periodic=(True, True),
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        coordinate_tolerance=_TOLERANCE,
    )

    source["x"][0] = -100.0
    assert candidate.atoms["x"][0] == pytest.approx(3.0)

    returned = candidate.atoms
    assert returned.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        returned["x"][0] = -100.0
    assert candidate.atoms["x"][0] == pytest.approx(3.0)


def test_grain_labels_property_is_defensive_and_read_only():
    candidate = _make_candidate()

    returned = candidate.grain_labels
    assert returned.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        returned[0] = RIGHT_GRAIN_LABEL
    assert candidate.grain_labels[0] == LEFT_GRAIN_LABEL


# --------------------------------------------------------------------------------------
# Validation failures
# --------------------------------------------------------------------------------------


def test_rejects_atoms_missing_required_fields():
    bad_atoms = np.zeros(2, dtype=[("name", "U2"), ("x", float)])

    with pytest.raises(InterfaceCandidateValueError, match="structured array"):
        _make_candidate(atoms=bad_atoms, grain_labels=_labels()[:2])


def test_rejects_grain_labels_length_mismatch():
    with pytest.raises(InterfaceCandidateValueError, match="atom count"):
        _make_candidate(grain_labels=np.asarray([LEFT_GRAIN_LABEL], dtype=np.int8))


def test_rejects_non_integer_grain_labels():
    with pytest.raises(InterfaceCandidateTypeError, match="integer"):
        _make_candidate(grain_labels=np.asarray([0.0, 0.0, 1.0, 1.0]))


def test_rejects_grain_labels_missing_a_grain():
    with pytest.raises(InterfaceCandidateValueError, match="both grains"):
        _make_candidate(
            grain_labels=np.asarray(
                [LEFT_GRAIN_LABEL] * 4, dtype=np.int8
            )
        )


def test_rejects_box_dims_wrong_shape():
    with pytest.raises(InterfaceCandidateValueError, match="shape"):
        _make_candidate(box_dims=np.zeros((2, 2)))


def test_rejects_non_finite_box_dims():
    bad_box = np.array(_BOX_DIMS, copy=True)
    bad_box[0, 1] = np.nan

    with pytest.raises(InterfaceCandidateValueError, match="finite"):
        _make_candidate(box_dims=bad_box)


def test_rejects_boolean_box_dims_entry():
    bad_box = np.array(_BOX_DIMS, dtype=object)
    bad_box[0, 0] = True

    with pytest.raises(InterfaceCandidateTypeError, match="finite"):
        _make_candidate(box_dims=bad_box)


def test_rejects_non_positive_coordinate_tolerance():
    with pytest.raises(InterfaceCandidateValueError, match="positive"):
        _make_candidate(coordinate_tolerance=0.0)


def test_rejects_negative_interface_separation():
    with pytest.raises(InterfaceCandidateValueError, match="nonnegative"):
        _make_candidate(interface_separation=-0.1)


def test_rejects_boolean_gb_plane_x():
    with pytest.raises(InterfaceCandidateTypeError, match="finite"):
        _make_candidate(gb_plane_x=True)


def test_rejects_box_bounds_not_strictly_ordered():
    bad_box = np.array(_BOX_DIMS, copy=True)
    bad_box[1, 0] = bad_box[1, 1]

    with pytest.raises(InterfaceCandidateValueError, match="strictly ordered"):
        _make_candidate(box_dims=bad_box)


def test_rejects_gb_plane_x_outside_box():
    with pytest.raises(InterfaceCandidateValueError, match="strictly inside"):
        _make_candidate(gb_plane_x=25.0)


def test_rejects_unordered_physical_grain_bounds():
    with pytest.raises(InterfaceCandidateValueError, match="strictly ordered"):
        _make_candidate(left_grain_x_bounds=(9.5, 9.5))


def test_rejects_overlapping_physical_grain_bounds():
    with pytest.raises(InterfaceCandidateValueError, match="without overlapping"):
        _make_candidate(right_grain_x_bounds=(9.0, 20.0))


def test_rejects_inplane_periodic_wrong_length():
    with pytest.raises(InterfaceCandidateValueError, match="exactly two"):
        _make_candidate(inplane_periodic=(True,))


def test_rejects_inplane_periodic_wrong_type():
    with pytest.raises(InterfaceCandidateTypeError, match="Boolean"):
        _make_candidate(inplane_periodic="TT")


def test_rejects_inplane_periodic_non_boolean_flag():
    with pytest.raises(InterfaceCandidateTypeError, match="Boolean"):
        _make_candidate(inplane_periodic=(1, True))


def test_rejects_unsupported_normal_topology():
    with pytest.raises(InterfaceCandidateValueError, match="not-a-topology"):
        _make_candidate(normal_topology="not-a-topology")


def test_rejects_atoms_outside_box():
    bad_atoms = _atoms()
    bad_atoms["z"][0] = 50.0

    with pytest.raises(InterfaceCandidateValueError, match="half-open"):
        _make_candidate(atoms=bad_atoms)


def test_rejects_atoms_outside_labeled_grain_bounds():
    bad_atoms = _atoms()
    bad_atoms["x"][0] = 9.8  # labeled left, but outside the left grain bound of 9.5

    with pytest.raises(InterfaceCandidateValueError, match="physical grain bounds"):
        _make_candidate(atoms=bad_atoms)
