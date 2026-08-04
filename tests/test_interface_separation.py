from __future__ import annotations

from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.FileGrainOwnership import (
    CandidateFileMapping,
    GrainOwnership,
    LEFT_GRAIN_LABEL,
    RIGHT_GRAIN_LABEL,
    reload_explicit_manipulator,
)
from GBOpt.GBMinimizer import GeneticAlgorithmMinimizer
from GBOpt.GBManipulator import (
    GBManipulator,
    GBManipulatorValueError,
    InterfaceCandidate,
    Parent,
)
from GBOpt.UnitCell import UnitCell


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
    manipulator._GBManipulator__parents = [parent, copy(parent) if two_parents else None]
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


def _write_data(path, candidate: InterfaceCandidate, *, order=None):
    atoms = candidate.atoms
    if order is None:
        order = np.arange(len(atoms))
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("Interface candidate\n\n")
        stream.write(f"{len(atoms)} atoms\n")
        stream.write("2 atom types\n")
        for axis, (lower, upper) in zip("xyz", candidate.box_dims):
            stream.write(f"{lower:.12f} {upper:.12f} {axis}lo {axis}hi\n")
        stream.write("\nAtoms\n\n")
        type_map = {"U": 1, "O": 2}
        for row in order:
            atom = atoms[row]
            stream.write(
                f"{row + 1} {type_map[str(atom['name'])]} "
                f"{atom['x']:.12f} {atom['y']:.12f} {atom['z']:.12f}\n"
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
    assert separated.right_grain_x_bounds[0] - separated.left_grain_x_bounds[1] == pytest.approx(s)
    assert separated.box_dims[0, 1] - separated.right_grain_x_bounds[1] == pytest.approx(s)


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


def test_candidate_mapping_and_ownership_preserve_separated_geometry():
    manipulator, _ = _slab_manipulator()
    separated = manipulator.apply_interface_separation(
        manipulator.make_parent_candidate(), interface_separation=0.5
    )

    direct_ownership = GrainOwnership.from_interface_candidate(separated)
    assert direct_ownership.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    assert np.array_equal(
        direct_ownership.left_grain_x_bounds, separated.left_grain_x_bounds
    )

    mapping = CandidateFileMapping.from_interface_candidate(separated)
    assert mapping.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    assert not mapping.periodic_outer_x_interface
    assert np.array_equal(mapping.box_dims, separated.box_dims)
    assert np.array_equal(mapping.left_grain_x_bounds, separated.left_grain_x_bounds)
    assert np.array_equal(mapping.right_grain_x_bounds, separated.right_grain_x_bounds)
    ownership = mapping.ownership_for_file_ids(mapping.atom_ids[::-1])
    assert ownership.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    assert np.array_equal(ownership.left_grain_x_bounds, separated.left_grain_x_bounds)
    assert np.array_equal(ownership.right_grain_x_bounds, separated.right_grain_x_bounds)


@pytest.mark.parametrize(
    "topology, factory",
    [
        (
            BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            lambda: _synthetic_manipulator(
                BoundaryNormalTopology.PERIODIC_BICRYSTAL
            )[0],
        ),
        (BoundaryNormalTopology.SINGLE_INTERFACE_SLAB, lambda: _slab_manipulator()[0]),
    ],
)
def test_separated_geometry_survives_write_and_explicit_reload(
    tmp_path, topology, factory
):
    manipulator = factory()
    separated = manipulator.apply_interface_separation(
        manipulator.make_parent_candidate(), interface_separation=0.5
    )
    mapping = CandidateFileMapping.from_interface_candidate(separated)
    output = tmp_path / f"{topology.value}.data"
    _write_data(output, separated, order=np.arange(len(separated.atoms))[::-1])
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fluorite", 5.454, ("U", "O"))

    reloaded = reload_explicit_manipulator(
        output,
        candidate_mapping=mapping,
        unit_cell=unit_cell,
        gb_thickness=4.0,
        type_dict={"U": 1, "O": 2},
    )
    parent = reloaded.parents[0]
    assert parent.normal_topology is topology
    assert parent.periodic_outer_x_interface == (
        topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    assert np.array_equal(parent.box_dims, separated.box_dims)
    assert parent.gb_plane_x == separated.gb_plane_x
    assert np.array_equal(parent.left_grain_x_bounds, separated.left_grain_x_bounds)
    assert np.array_equal(parent.right_grain_x_bounds, separated.right_grain_x_bounds)
    assert np.array_equal(parent.grain_labels, separated.grain_labels)

    minimizer = GeneticAlgorithmMinimizer.__new__(GeneticAlgorithmMinimizer)
    remapped = minimizer._candidate_file_mapping(
        reloaded, parent.whole_system
    )
    assert remapped.normal_topology is topology
    assert np.array_equal(remapped.box_dims, separated.box_dims)
    assert remapped.gb_plane_x == separated.gb_plane_x
    assert np.array_equal(
        remapped.left_grain_x_bounds, separated.left_grain_x_bounds
    )
    assert np.array_equal(
        remapped.right_grain_x_bounds, separated.right_grain_x_bounds
    )


def test_grain_ownership_accepts_plane_inside_empty_central_interval():
    ownership = GrainOwnership(
        atom_ids=np.arange(1, 5),
        labels=np.array([0, 0, 1, 1]),
        gb_plane_x=7.25,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(2.0, 7.0),
        right_grain_x_bounds=(7.5, 12.5),
        coordinate_tolerance=1.0e-10,
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
    )
    assert ownership.gb_plane_x == pytest.approx(7.25)
    assert ownership.left_grain_x_bounds.tolist() == pytest.approx([2.0, 7.0])
    assert ownership.right_grain_x_bounds.tolist() == pytest.approx([7.5, 12.5])


def test_legacy_false_boolean_is_unknown_not_slab():
    ownership = GrainOwnership(
        atom_ids=np.arange(1, 3),
        labels=np.array([0, 1]),
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-10,
        periodic_outer_x_interface=False,
    )
    assert ownership.normal_topology is BoundaryNormalTopology.UNKNOWN
    assert not ownership.periodic_outer_x_interface


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
