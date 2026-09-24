# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.manipulation.translation: RightGrainTranslation."""

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
from GBOpt.interface.model import InterfaceCandidate
from GBOpt.manipulation.translation import RightGrainTranslation
from GBOpt.manipulation.types import (
    ManipulationCapabilityError,
    ManipulationConfigurationError,
    ManipulationContext,
)

_BOX_DIMS = np.asarray([[0.0, 20.0], [0.0, 10.0], [0.0, 10.0]], dtype=float)
_GB_PLANE_X = 10.0
_LEFT_BOUNDS = (0.0, 10.0)
_RIGHT_BOUNDS = (10.0, 20.0)
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


def _make_candidate(*, inplane_periodic=(True, True), **overrides) -> InterfaceCandidate:
    kwargs = {
        "atoms": _atoms(),
        "box_dims": _BOX_DIMS,
        "gb_plane_x": _GB_PLANE_X,
        "left_grain_x_bounds": _LEFT_BOUNDS,
        "right_grain_x_bounds": _RIGHT_BOUNDS,
        "grain_labels": _labels(),
        "inplane_periodic": inplane_periodic,
        "normal_topology": BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        "coordinate_tolerance": _TOLERANCE,
        "interface_separation": 0.0,
    }
    kwargs.update(overrides)
    return InterfaceCandidate(**kwargs)


def _context(parent, **params) -> ManipulationContext:
    return ManipulationContext(
        parents=(parent,), rng=np.random.default_rng(0), params=params
    )


def test_name_and_arity():
    operation = RightGrainTranslation()
    assert operation.name == "right_grain_translation"
    assert operation.arity == 1


def test_translates_right_grain_only():
    parent = _make_candidate()
    result = RightGrainTranslation().execute(_context(parent, dy=1.0, dz=0.5))
    child = result.children[0]
    np.testing.assert_array_equal(child.atoms["name"], parent.atoms["name"])
    # Left grain (rows 0-1) unaffected.
    np.testing.assert_allclose(child.atoms["y"][:2], parent.atoms["y"][:2])
    np.testing.assert_allclose(child.atoms["z"][:2], parent.atoms["z"][:2])
    # Right grain (rows 2-3) translated.
    np.testing.assert_allclose(child.atoms["y"][2:], parent.atoms["y"][2:] + 1.0)
    np.testing.assert_allclose(child.atoms["z"][2:], parent.atoms["z"][2:] + 0.5)


def test_preserves_geometry_fields():
    parent = _make_candidate()
    child = RightGrainTranslation().execute(_context(parent, dy=1.0, dz=0.5)).children[0]
    np.testing.assert_allclose(child.box_dims, parent.box_dims)
    assert child.gb_plane_x == pytest.approx(parent.gb_plane_x)
    np.testing.assert_allclose(child.left_grain_x_bounds, parent.left_grain_x_bounds)
    np.testing.assert_allclose(child.right_grain_x_bounds, parent.right_grain_x_bounds)
    assert child.normal_topology is parent.normal_topology
    assert child.inplane_periodic == parent.inplane_periodic
    assert child.interface_separation == pytest.approx(parent.interface_separation)


def test_records_parameters_and_lineage():
    parent = _make_candidate()
    result = RightGrainTranslation().execute(
        _context(parent, dy=1.0, dz=0.5, dx=0.1)
    )
    assert dict(result.parameters) == {"dx": 0.1, "dy": 1.0, "dz": 0.5}
    assert dict(result.lineage) == {
        "operation": "right_grain_translation",
        "parent_count": 1,
    }


def test_dx_defaults_to_zero():
    parent = _make_candidate()
    result = RightGrainTranslation().execute(_context(parent, dy=0.0, dz=0.0))
    np.testing.assert_allclose(result.children[0].atoms["x"], parent.atoms["x"])


def test_requires_dy_and_dz():
    parent = _make_candidate()
    with pytest.raises(ManipulationConfigurationError):
        RightGrainTranslation().execute(_context(parent, dy=0.0))


def test_nonperiodic_displacement_leaving_box_is_rejected():
    parent = _make_candidate(inplane_periodic=(False, False))
    with pytest.raises(ManipulationCapabilityError):
        RightGrainTranslation().execute(_context(parent, dy=100.0, dz=0.0))


def test_dx_moving_atoms_outside_supported_interval_is_rejected():
    parent = _make_candidate()
    with pytest.raises(ManipulationCapabilityError, match="half-open x interval"):
        RightGrainTranslation().execute(_context(parent, dy=0.0, dz=0.0, dx=100.0))


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf],
    ids=["nan", "infinite"],
)
def test_non_finite_displacement_raises_configuration_error(value):
    parent = _make_candidate()
    with pytest.raises(ManipulationConfigurationError, match="finite real"):
        RightGrainTranslation().execute(_context(parent, dy=value, dz=0.0))


@pytest.mark.parametrize(
    "value",
    [True, "1.0"],
    ids=["boolean", "string"],
)
def test_non_real_displacement_raises_type_error(value):
    parent = _make_candidate()
    with pytest.raises(TypeError, match="finite real"):
        RightGrainTranslation().execute(_context(parent, dy=value, dz=0.0))


def test_periodic_translation_wraps_in_plane():
    parent = _make_candidate()
    result = RightGrainTranslation().execute(_context(parent, dy=9.0, dz=0.0))
    # Row 2's y is 4.0; 4.0 + 9.0 wraps around the y box [0, 10) to 3.0.
    wrapped = result.children[0].atoms["y"][2]
    assert wrapped == pytest.approx((4.0 + 9.0) % 10.0)
