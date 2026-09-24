# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.manipulation.termination: GrainTerminationCycle."""

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
from GBOpt.interface.model import InterfaceCandidate
from GBOpt.manipulation.termination import GrainTerminationCycle
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


def _context(parent, **params) -> ManipulationContext:
    return ManipulationContext(
        parents=(parent,), rng=np.random.default_rng(0), params=params
    )


def test_name_and_arity():
    operation = GrainTerminationCycle()
    assert operation.name == "grain_termination_cycle"
    assert operation.arity == 1


def test_zero_shift_preserves_atoms():
    parent = _periodic_candidate()
    result = GrainTerminationCycle().execute(_context(parent))
    np.testing.assert_allclose(result.children[0].atoms["x"], parent.atoms["x"])


def test_left_phase_shift_cycles_left_grain_only():
    parent = _periodic_candidate()
    result = GrainTerminationCycle().execute(
        _context(parent, left_phase_shift=9.0)
    )
    child = result.children[0]
    # 2.0 + 9.0 = 11.0, wrapped into [0, 10) -> 1.0
    assert child.atoms["x"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(child.atoms["x"][2:], parent.atoms["x"][2:])


def test_slab_termination_cycle_preserves_box_and_vacuum():
    parent = _slab_candidate()
    result = GrainTerminationCycle().execute(
        _context(parent, right_phase_shift=5.0)
    )
    child = result.children[0]
    np.testing.assert_allclose(child.box_dims, parent.box_dims)
    np.testing.assert_allclose(child.left_grain_x_bounds, parent.left_grain_x_bounds)
    np.testing.assert_allclose(child.right_grain_x_bounds, parent.right_grain_x_bounds)


def test_records_parameters_and_lineage():
    parent = _periodic_candidate()
    result = GrainTerminationCycle().execute(
        _context(
            parent,
            left_phase_shift=1.0,
            right_phase_shift=2.0,
            right_dy=0.5,
            right_dz=0.25,
        )
    )
    assert dict(result.parameters) == {
        "left_phase_shift": 1.0,
        "right_phase_shift": 2.0,
        "right_dy": 0.5,
        "right_dz": 0.25,
    }
    assert dict(result.lineage) == {
        "operation": "grain_termination_cycle",
        "parent_count": 1,
    }


def test_unknown_topology_is_rejected():
    parent = _periodic_candidate(normal_topology=BoundaryNormalTopology.UNKNOWN)
    with pytest.raises(ManipulationCapabilityError, match="known boundary-normal"):
        GrainTerminationCycle().execute(_context(parent))


def test_periodic_topology_requires_zero_vacuum_bounds():
    # A periodic candidate whose right bound doesn't reach the box edge is invalid for
    # periodic cycling; construct via a slab-shaped box reused with periodic topology.
    parent = _periodic_candidate(
        box_dims=np.asarray([[0.0, 22.0], [0.0, 10.0], [0.0, 10.0]], dtype=float),
    )
    with pytest.raises(ManipulationCapabilityError, match="zero-vacuum"):
        GrainTerminationCycle().execute(_context(parent))


def test_slab_without_vacuum_interval_is_rejected():
    parent = _slab_candidate(
        box_dims=np.asarray([[0.0, 20.0], [0.0, 10.0], [0.0, 10.0]], dtype=float),
    )
    with pytest.raises(ManipulationCapabilityError, match="free-surface or vacuum"):
        GrainTerminationCycle().execute(_context(parent))


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf],
    ids=["nan", "infinite"],
)
def test_non_finite_phase_shift_raises_configuration_error(value):
    parent = _periodic_candidate()
    with pytest.raises(ManipulationConfigurationError, match="finite real"):
        GrainTerminationCycle().execute(_context(parent, left_phase_shift=value))


@pytest.mark.parametrize(
    "value",
    [True, "1.0"],
    ids=["boolean", "string"],
)
def test_non_real_phase_shift_raises_type_error(value):
    parent = _periodic_candidate()
    with pytest.raises(TypeError, match="finite real"):
        GrainTerminationCycle().execute(_context(parent, right_dy=value))
