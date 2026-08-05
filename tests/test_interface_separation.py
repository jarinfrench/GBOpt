from __future__ import annotations

import math
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import (
    GBManipulator,
    GBManipulatorValueError,
    InterfaceCandidate,
    Parent,
)
from GBOpt.UnitCell import UnitCell

LEFT_GRAIN_LABEL = 0
RIGHT_GRAIN_LABEL = 1


DTYPE = np.dtype(
    [
        ("name", "U2"),
        ("x", float),
        ("y", float),
        ("z", float),
        ("site_id", np.int64),
    ]
)


def _synthetic_manipulator(
    topology: BoundaryNormalTopology,
    *,
    box_x=(2.0, 12.0),
    plane=7.0,
    left_bounds=None,
    right_bounds=None,
    two_parents=False,
):
    xlo, xhi = box_x
    if left_bounds is None:
        left_bounds = (xlo, plane)
    if right_bounds is None:
        right_bounds = (plane, xhi)
    left = np.array(
        [
            ("U", left_bounds[0] + 0.25, 0.0, 1.0, 10),
            ("O", left_bounds[1] - 0.25, 2.0, 3.0, 11),
        ],
        dtype=DTYPE,
    )
    right = np.array(
        [
            ("O", right_bounds[0] + 0.25, 4.0, 5.0, 20),
            ("U", right_bounds[1] - 0.25, 6.0, 7.0, 21),
        ],
        dtype=DTYPE,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    parent = SimpleNamespace(
        left_grain=left,
        right_grain=right,
        whole_system=np.hstack((left, right)),
        box_dims=np.array([[xlo, xhi], [-1.0, 9.0], [0.0, 10.0]]),
        gb_plane_x=plane,
        inplane_periodic=(True, True),
        coordinate_tolerance=1.0e-10,
        left_grain_x_bounds=np.asarray(left_bounds, dtype=float),
        right_grain_x_bounds=np.asarray(right_bounds, dtype=float),
        normal_topology=topology,
        periodic_outer_x_interface=(
            topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
        ),
        grain_labels=labels,
    )
    manipulator = object.__new__(GBManipulator)
    manipulator._GBManipulator__one_parent = not two_parents
    manipulator._GBManipulator__parents = [
        parent, copy(parent) if two_parents else None]
    manipulator._GBManipulator__rng = np.random.default_rng(7)
    manipulator._GBManipulator__candidate_grain_labels = labels.copy()
    return manipulator, parent


def _slab_manipulator(*, left_bounds=(4.0, 8.0), right_bounds=(8.0, 14.0)):
    return _synthetic_manipulator(
        BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
        box_x=(2.0, 16.0),
        plane=8.0,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
    )


def test_parent_candidate_is_geometry_bearing_defensive_and_read_only():
    manipulator, parent = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    parent_before = parent.whole_system.copy()

    candidate = manipulator.make_parent_candidate()

    assert np.array_equal(candidate.atoms, parent_before)
    assert np.array_equal(candidate.grain_labels, np.array([0, 0, 1, 1]))
    assert candidate.normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    assert candidate.periodic_outer_x_interface
    atoms = candidate.atoms
    labels = candidate.grain_labels
    assert not atoms.flags.writeable
    assert not labels.flags.writeable
    with pytest.raises(ValueError):
        atoms[0]["x"] = 99.0
    with pytest.raises(ValueError):
        labels[0] = 1
    assert np.array_equal(parent.whole_system, parent_before)


def test_geometry_companions_match_existing_fixed_cell_apis():
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )

    translated_atoms = manipulator.translate_right_grain(0.5, 0.25)
    translated = manipulator.make_translation_candidate(0.5, 0.25)
    assert np.array_equal(translated.atoms, translated_atoms)

    cycled_atoms = manipulator.cycle_grain_terminations(
        left_phase_shift=0.5,
        right_phase_shift=0.75,
        right_dy=0.25,
    )
    cycled = manipulator.make_termination_candidate(
        left_phase_shift=0.5,
        right_phase_shift=0.75,
        right_dy=0.25,
    )
    assert np.array_equal(cycled.atoms, cycled_atoms)
    assert np.array_equal(cycled.box_dims, translated.box_dims)
    assert cycled.gb_plane_x == translated.gb_plane_x


def test_slab_termination_companion_is_grain_local_and_geometry_preserving():
    manipulator, parent = _slab_manipulator(
        left_bounds=(4.0, 8.0), right_bounds=(8.0, 14.0)
    )
    parent_before = parent.whole_system.copy()

    atoms = manipulator.cycle_slab_terminations(
        left_phase_shift=0.5,
        right_phase_shift=1.0,
    )
    candidate = manipulator.make_slab_termination_candidate(
        left_phase_shift=0.5,
        right_phase_shift=1.0,
    )

    assert np.array_equal(candidate.atoms, atoms)
    assert candidate.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    assert candidate.interface_separation == 0.0
    assert candidate.box_dims.tolist() == parent.box_dims.tolist()
    assert candidate.gb_plane_x == parent.gb_plane_x
    assert candidate.left_grain_x_bounds.tolist() == [4.0, 8.0]
    assert candidate.right_grain_x_bounds.tolist() == [8.0, 14.0]
    np.testing.assert_allclose(candidate.atoms["x"][:2], [4.75, 4.25])
    np.testing.assert_allclose(candidate.atoms["x"][2:], [9.25, 8.75])
    assert np.array_equal(candidate.atoms["y"], parent.whole_system["y"])
    assert np.array_equal(candidate.atoms["z"], parent.whole_system["z"])
    assert np.array_equal(candidate.atoms["site_id"], parent.whole_system["site_id"])
    assert np.array_equal(candidate.atoms["name"], parent.whole_system["name"])
    assert np.array_equal(candidate.grain_labels, parent.grain_labels)
    assert np.array_equal(parent.whole_system, parent_before)


def test_slab_termination_zero_phase_is_exact_identity():
    manipulator, parent = _slab_manipulator()

    candidate = manipulator.make_slab_termination_candidate()

    assert np.array_equal(candidate.atoms, parent.whole_system)
    assert np.array_equal(candidate.box_dims, parent.box_dims)
    assert np.array_equal(candidate.left_grain_x_bounds, parent.left_grain_x_bounds)
    assert np.array_equal(candidate.right_grain_x_bounds, parent.right_grain_x_bounds)


def test_slab_termination_wraps_each_physical_upper_face_to_its_lower_face():
    manipulator, _ = _slab_manipulator()

    candidate = manipulator.make_slab_termination_candidate(
        left_phase_shift=0.25,
        right_phase_shift=0.25,
    )

    assert candidate.atoms[1]["x"] == pytest.approx(4.0)
    assert candidate.atoms[3]["x"] == pytest.approx(8.0)
    assert np.all((candidate.atoms["x"][:2] >= 4.0) & (candidate.atoms["x"][:2] < 8.0))
    assert np.all((candidate.atoms["x"][2:] >= 8.0) & (candidate.atoms["x"][2:] < 14.0))


def test_slab_termination_combines_right_inplane_registry_shift():
    manipulator, parent = _slab_manipulator()

    candidate = manipulator.make_slab_termination_candidate(
        right_phase_shift=0.5,
        right_dy=7.0,
        right_dz=6.0,
    )

    assert np.array_equal(candidate.atoms[:2], parent.left_grain)
    np.testing.assert_allclose(candidate.atoms["x"][2:], [8.75, 8.25])
    np.testing.assert_allclose(candidate.atoms["y"][2:], [1.0, 3.0])
    np.testing.assert_allclose(candidate.atoms["z"][2:], [1.0, 3.0])


def test_slab_termination_preserves_nonperiodic_inplane_rejection():
    manipulator, parent = _slab_manipulator()
    parent.inplane_periodic = (False, True)

    valid = manipulator.make_slab_termination_candidate(right_dy=0.5)
    np.testing.assert_allclose(valid.atoms["y"][2:], parent.right_grain["y"] + 0.5)

    with pytest.raises(GBManipulatorValueError, match="nonperiodic half-open y"):
        manipulator.cycle_slab_terminations(right_dy=5.5)


def test_slab_termination_composes_with_interface_separation():
    manipulator, _ = _slab_manipulator(
        left_bounds=(3.5, 8.0), right_bounds=(8.0, 13.0)
    )
    terminated = manipulator.make_slab_termination_candidate(
        left_phase_shift=0.5,
        right_phase_shift=0.75,
    )
    old_left_vacuum = terminated.left_grain_x_bounds[0] - terminated.box_dims[0, 0]
    old_right_vacuum = terminated.box_dims[0, 1] - terminated.right_grain_x_bounds[1]

    separated = manipulator.apply_interface_separation(
        terminated,
        interface_separation=1.25,
    )

    assert separated.interface_separation == pytest.approx(1.25)
    assert separated.box_dims[0].tolist() == pytest.approx([2.0, 17.25])
    assert separated.gb_plane_x == pytest.approx(8.625)
    assert separated.left_grain_x_bounds.tolist() == pytest.approx([3.5, 8.0])
    assert separated.right_grain_x_bounds.tolist() == pytest.approx([9.25, 14.25])
    assert (
        separated.left_grain_x_bounds[0] - separated.box_dims[0, 0]
        == pytest.approx(old_left_vacuum)
    )
    assert (
        separated.box_dims[0, 1] - separated.right_grain_x_bounds[1]
        == pytest.approx(old_right_vacuum)
    )
    labels = terminated.grain_labels
    np.testing.assert_allclose(
        separated.atoms["x"][labels == LEFT_GRAIN_LABEL],
        terminated.atoms["x"][labels == LEFT_GRAIN_LABEL],
    )
    np.testing.assert_allclose(
        separated.atoms["x"][labels == RIGHT_GRAIN_LABEL],
        terminated.atoms["x"][labels == RIGHT_GRAIN_LABEL] + 1.25,
    )


@pytest.mark.parametrize(
    "topology",
    [
        BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        BoundaryNormalTopology.UNKNOWN,
    ],
)
def test_slab_termination_rejects_non_slab_topology(topology):
    manipulator, _ = _synthetic_manipulator(topology)

    with pytest.raises(GBManipulatorValueError, match="single-interface slab"):
        manipulator.cycle_slab_terminations()


def test_slab_termination_rejects_two_parent_manipulator():
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
        box_x=(2.0, 16.0),
        plane=8.0,
        left_bounds=(4.0, 8.0),
        right_bounds=(8.0, 14.0),
        two_parents=True,
    )

    with pytest.raises(GBManipulatorValueError, match="exactly one parent"):
        manipulator.cycle_slab_terminations()


@pytest.mark.parametrize(
    ("left_bounds", "right_bounds", "message"),
    [
        ((4.0, 7.5), (8.0, 14.0), "contiguous"),
        ((4.0, 8.0), (8.5, 14.0), "contiguous"),
        ((2.0, 8.0), (8.0, 16.0), "vacuum"),
    ],
)
def test_slab_termination_rejects_unsupported_physical_geometry(
    left_bounds,
    right_bounds,
    message,
):
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
        box_x=(2.0, 16.0),
        plane=8.0,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
    )

    with pytest.raises(GBManipulatorValueError, match=message):
        manipulator.cycle_slab_terminations()


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("left_phase_shift", np.nan),
        ("right_phase_shift", np.inf),
        ("right_dy", True),
        ("right_dz", "1.0"),
    ],
)
def test_slab_termination_rejects_nonfinite_or_nonreal_values(argument, value):
    manipulator, _ = _slab_manipulator()

    with pytest.raises(GBManipulatorValueError, match="finite real"):
        manipulator.cycle_slab_terminations(**{argument: value})


def test_slab_termination_gbmaker_integration_preserves_vacuum_geometry():
    theta = math.radians(36.869898)
    with pytest.warns(DeprecationWarning):
        gb = GBMaker(
            a0=3.0,
            structure="rocksalt",
            gb_thickness=5.0,
            misorientation=[theta, 0.0, 0.0, 0.0, -theta / 2.0],
            atom_types=("Na", "Cl"),
            repeat_factor=(2, 3),
            x_dim_min=10.0,
            vacuum=2.0,
            interaction_distance=4.0,
        )
    manipulator = GBManipulator(gb, seed=17)
    base = manipulator.make_parent_candidate()

    candidate = manipulator.make_slab_termination_candidate(
        left_phase_shift=0.25,
        right_phase_shift=0.5,
        right_dy=0.125,
    )

    assert candidate.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    assert np.array_equal(candidate.box_dims, base.box_dims)
    assert candidate.gb_plane_x == base.gb_plane_x
    assert np.array_equal(candidate.left_grain_x_bounds, base.left_grain_x_bounds)
    assert np.array_equal(candidate.right_grain_x_bounds, base.right_grain_x_bounds)
    assert len(candidate.atoms) == len(base.atoms)
    assert np.array_equal(
        np.unique(candidate.atoms["name"], return_counts=True),
        np.unique(base.atoms["name"], return_counts=True),
    )
    left_vacuum = candidate.left_grain_x_bounds[0] - candidate.box_dims[0, 0]
    right_vacuum = candidate.box_dims[0, 1] - candidate.right_grain_x_bounds[1]
    assert left_vacuum == pytest.approx(2.0)
    assert right_vacuum == pytest.approx(2.0)


def test_zero_separation_is_exact_identity_and_nonmutating():
    manipulator, parent = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    base = manipulator.make_translation_candidate(0.5, 0.25)
    parent_before = parent.whole_system.copy()

    separated = manipulator.apply_interface_separation(
        base, interface_separation=0.0
    )

    assert np.array_equal(separated.atoms, base.atoms)
    assert np.array_equal(separated.box_dims, base.box_dims)
    assert separated.gb_plane_x == base.gb_plane_x
    assert np.array_equal(separated.left_grain_x_bounds, base.left_grain_x_bounds)
    assert np.array_equal(separated.right_grain_x_bounds, base.right_grain_x_bounds)
    assert np.array_equal(separated.grain_labels, base.grain_labels)
    assert np.array_equal(parent.whole_system, parent_before)


def test_periodic_separation_expands_both_physical_interfaces():
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        box_x=(2.0, 12.0),
        plane=7.0,
    )
    base = manipulator.make_parent_candidate()
    s = 0.75

    separated = manipulator.apply_interface_separation(
        base, interface_separation=s
    )

    assert separated.box_dims[0].tolist() == pytest.approx([2.0, 13.5])
    assert separated.gb_plane_x == pytest.approx(7.375)
    assert separated.left_grain_x_bounds.tolist() == pytest.approx([2.0, 7.0])
    assert separated.right_grain_x_bounds.tolist() == pytest.approx([7.75, 12.75])
    labels = base.grain_labels
    np.testing.assert_allclose(
        separated.atoms["x"][labels == 0], base.atoms["x"][labels == 0]
    )
    np.testing.assert_allclose(
        separated.atoms["x"][labels == 1], base.atoms["x"][labels == 1] + s
    )
    assert separated.right_grain_x_bounds[0] - \
        separated.left_grain_x_bounds[1] == pytest.approx(s)
    assert separated.box_dims[0, 1] - \
        separated.right_grain_x_bounds[1] == pytest.approx(s)


def test_slab_separation_preserves_asymmetric_outer_vacuum_widths():
    manipulator, _ = _slab_manipulator(
        left_bounds=(3.5, 8.0), right_bounds=(8.0, 13.0)
    )
    base = manipulator.make_parent_candidate()
    old_left_vacuum = base.left_grain_x_bounds[0] - base.box_dims[0, 0]
    old_right_vacuum = base.box_dims[0, 1] - base.right_grain_x_bounds[1]
    s = 1.25

    separated = manipulator.apply_interface_separation(
        base, interface_separation=s
    )

    assert separated.box_dims[0].tolist() == pytest.approx([2.0, 17.25])
    assert separated.gb_plane_x == pytest.approx(8.625)
    assert separated.left_grain_x_bounds.tolist() == pytest.approx([3.5, 8.0])
    assert separated.right_grain_x_bounds.tolist() == pytest.approx([9.25, 14.25])
    new_left_vacuum = separated.left_grain_x_bounds[0] - separated.box_dims[0, 0]
    new_right_vacuum = separated.box_dims[0, 1] - separated.right_grain_x_bounds[1]
    assert new_left_vacuum == pytest.approx(old_left_vacuum)
    assert new_right_vacuum == pytest.approx(old_right_vacuum)


def test_separation_uses_labels_and_preserves_interleaved_row_order_and_fields():
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    parent = manipulator._GBManipulator__parents[0]
    order = np.array([0, 2, 1, 3])
    atoms = parent.whole_system[order]
    labels = parent.grain_labels[order]
    base = InterfaceCandidate(
        atoms=atoms,
        box_dims=parent.box_dims,
        gb_plane_x=parent.gb_plane_x,
        left_grain_x_bounds=parent.left_grain_x_bounds,
        right_grain_x_bounds=parent.right_grain_x_bounds,
        grain_labels=labels,
        inplane_periodic=parent.inplane_periodic,
        normal_topology=parent.normal_topology,
        coordinate_tolerance=parent.coordinate_tolerance,
    )

    separated = manipulator.apply_interface_separation(
        base, interface_separation=0.5
    )

    assert separated.atoms["site_id"].tolist() == atoms["site_id"].tolist()
    assert separated.atoms["name"].tolist() == atoms["name"].tolist()
    np.testing.assert_allclose(
        separated.atoms["x"], atoms["x"] + 0.5 * (labels == 1)
    )
    assert np.array_equal(separated.atoms["y"], atoms["y"])
    assert np.array_equal(separated.atoms["z"], atoms["z"])
    assert np.array_equal(separated.grain_labels, labels)


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf, True, 1 + 2j])
def test_separation_rejects_invalid_values(bad):
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    base = manipulator.make_parent_candidate()
    with pytest.raises(GBManipulatorValueError):
        manipulator.apply_interface_separation(base, interface_separation=bad)


def test_separation_rejects_unknown_topology_two_parents_and_reapplication():
    unknown, _ = _synthetic_manipulator(BoundaryNormalTopology.UNKNOWN)
    with pytest.raises(GBManipulatorValueError, match="known"):
        unknown.apply_interface_separation(
            unknown.make_parent_candidate(), interface_separation=0.5
        )

    two, _ = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL, two_parents=True
    )
    one, _ = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    with pytest.raises(GBManipulatorValueError, match="exactly one parent"):
        two.apply_interface_separation(
            one.make_parent_candidate(), interface_separation=0.5
        )

    base = one.make_parent_candidate()
    separated = one.apply_interface_separation(base, interface_separation=0.5)
    with pytest.raises(GBManipulatorValueError, match="reapplied"):
        one.apply_interface_separation(separated, interface_separation=0.5)


def test_slab_requires_a_vacuum_interval():
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    )
    with pytest.raises(GBManipulatorValueError, match="vacuum"):
        manipulator.apply_interface_separation(
            manipulator.make_parent_candidate(), interface_separation=0.5
        )


def test_parent_infers_known_topology_from_gbmaker_vacuum_with_tolerance():
    atoms = np.array(
        [("U", 1.0, 1.0, 1.0), ("O", 6.0, 1.0, 1.0)],
        dtype=Atom.atom_dtype,
    )

    def fake(vacuum):
        return SimpleNamespace(
            vacuum_thickness=vacuum,
            right_grain=atoms[1:],
            left_grain=atoms[:1],
            whole_system=atoms,
            y_dim=10.0,
            z_dim=10.0,
            gb_thickness=2.0,
            unit_cell=UnitCell(),
            radius=1.0,
            box_dims=np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]]),
            inplane_periodic=(True, True),
            epsilon=1.0e-8,
            gb_plane_x=5.0,
        )

    periodic = Parent.__new__(Parent)
    periodic._Parent__init_by_gbmaker(fake(1.0e-10))
    assert periodic.normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    assert np.array_equal(periodic.left_grain_x_bounds, [0.0, 5.0])

    slab = Parent.__new__(Parent)
    slab._Parent__init_by_gbmaker(fake(1.5))
    assert slab.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    assert np.array_equal(slab.left_grain_x_bounds, [1.5, 5.0])
    assert np.array_equal(slab.right_grain_x_bounds, [5.0, 8.5])
