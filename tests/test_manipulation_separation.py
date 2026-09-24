# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.manipulation.separation: InterfaceSeparation."""

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
from GBOpt.interface.model import InterfaceCandidate
from GBOpt.manipulation.separation import InterfaceSeparation
from GBOpt.manipulation.types import (
    ManipulationCapabilityError,
    ManipulationConfigurationError,
    ManipulationContext,
)

_TOLERANCE = 1.0e-8


def _atoms() -> np.ndarray:
    return np.asarray(
        [
            ("U", 2.0, 1.0, 1.0),
            ("O", 8.0, 2.0, 3.0),
            ("O", 12.0, 4.0, 5.0),
            ("U", 18.0, 6.0, 7.0),
        ],
        dtype=Atom.atom_dtype,
    )


def _labels() -> np.ndarray:
    return np.asarray(
        [LEFT_GRAIN_LABEL, LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
        dtype=np.int8,
    )


def _periodic_candidate(**overrides) -> InterfaceCandidate:
    kwargs = {
        "atoms": _atoms(),
        "box_dims": np.asarray([[0.0, 20.0], [0.0, 10.0], [0.0, 10.0]], dtype=float),
        "gb_plane_x": 10.0,
        "left_grain_x_bounds": (0.0, 10.0),
        "right_grain_x_bounds": (10.0, 20.0),
        "grain_labels": _labels(),
        "inplane_periodic": (True, True),
        "normal_topology": BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        "coordinate_tolerance": _TOLERANCE,
        "interface_separation": 0.0,
    }
    kwargs.update(overrides)
    return InterfaceCandidate(**kwargs)


def _slab_candidate(**overrides) -> InterfaceCandidate:
    kwargs = {
        "atoms": _atoms(),
        "box_dims": np.asarray([[0.0, 22.0], [0.0, 10.0], [0.0, 10.0]], dtype=float),
        "gb_plane_x": 10.0,
        "left_grain_x_bounds": (0.0, 10.0),
        "right_grain_x_bounds": (10.0, 20.0),
        "grain_labels": _labels(),
        "inplane_periodic": (True, True),
        "normal_topology": BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
        "coordinate_tolerance": _TOLERANCE,
        "interface_separation": 0.0,
    }
    kwargs.update(overrides)
    return InterfaceCandidate(**kwargs)


def _context(parent, candidate=None, **params) -> ManipulationContext:
    merged = {"candidate": candidate if candidate is not None else parent, **params}
    return ManipulationContext(
        parents=(parent,), rng=np.random.default_rng(0), params=merged
    )


def test_name_and_arity():
    operation = InterfaceSeparation()
    assert operation.name == "interface_separation"
    assert operation.arity == 1


def test_periodic_separation_expands_x_by_twice_separation():
    parent = _periodic_candidate()
    result = InterfaceSeparation().execute(_context(parent, interface_separation=0.6))
    child = result.children[0]
    assert child.box_dims[0, 1] == pytest.approx(parent.box_dims[0, 1] + 1.2)
    assert child.gb_plane_x == pytest.approx(parent.gb_plane_x + 0.3)
    np.testing.assert_allclose(child.left_grain_x_bounds, parent.left_grain_x_bounds)
    np.testing.assert_allclose(
        child.right_grain_x_bounds, np.asarray(parent.right_grain_x_bounds) + 0.6
    )
    assert child.interface_separation == pytest.approx(0.6)


def test_slab_separation_expands_x_by_separation_and_preserves_outer_vacuum():
    parent = _slab_candidate()
    result = InterfaceSeparation().execute(_context(parent, interface_separation=0.6))
    child = result.children[0]
    assert child.box_dims[0, 1] == pytest.approx(parent.box_dims[0, 1] + 0.6)
    left_vacuum_before = parent.left_grain_x_bounds[0] - parent.box_dims[0, 0]
    right_vacuum_before = parent.box_dims[0, 1] - parent.right_grain_x_bounds[1]
    left_vacuum_after = child.left_grain_x_bounds[0] - child.box_dims[0, 0]
    right_vacuum_after = child.box_dims[0, 1] - child.right_grain_x_bounds[1]
    assert left_vacuum_after == pytest.approx(left_vacuum_before)
    assert right_vacuum_after == pytest.approx(right_vacuum_before)


def test_records_parameters_and_lineage():
    parent = _periodic_candidate()
    result = InterfaceSeparation().execute(_context(parent, interface_separation=0.6))
    assert dict(result.parameters) == {"interface_separation": 0.6}
    assert dict(result.lineage) == {
        "operation": "interface_separation",
        "parent_count": 1,
    }


def test_rejects_non_candidate():
    parent = _periodic_candidate()
    context = ManipulationContext(
        parents=(parent,),
        rng=np.random.default_rng(0),
        params={"candidate": object(), "interface_separation": 0.6},
    )
    with pytest.raises(ManipulationConfigurationError, match="InterfaceCandidate"):
        InterfaceSeparation().execute(context)


def test_rejects_reapplication_to_already_separated_candidate():
    parent = _periodic_candidate()
    separated = InterfaceSeparation().execute(
        _context(parent, interface_separation=0.6)
    ).children[0]
    with pytest.raises(ManipulationCapabilityError, match="cannot be reapplied"):
        InterfaceSeparation().execute(
            _context(parent, candidate=separated, interface_separation=0.1)
        )


def test_rejects_mismatched_topology():
    parent = _periodic_candidate()
    other = _slab_candidate()
    with pytest.raises(ManipulationCapabilityError, match="topology does not match"):
        InterfaceSeparation().execute(
            _context(parent, candidate=other, interface_separation=0.1)
        )


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf],
    ids=["nan", "infinite"],
)
def test_non_finite_separation_raises_configuration_error(value):
    parent = _periodic_candidate()
    with pytest.raises(ManipulationConfigurationError, match="finite real"):
        InterfaceSeparation().execute(_context(parent, interface_separation=value))


@pytest.mark.parametrize(
    "value",
    [True, "0.5"],
    ids=["boolean", "string"],
)
def test_non_real_separation_raises_type_error(value):
    parent = _periodic_candidate()
    with pytest.raises(TypeError, match="finite real"):
        InterfaceSeparation().execute(_context(parent, interface_separation=value))


def test_negative_separation_is_rejected():
    parent = _periodic_candidate()
    with pytest.raises(ManipulationConfigurationError, match="nonnegative"):
        InterfaceSeparation().execute(_context(parent, interface_separation=-0.1))
