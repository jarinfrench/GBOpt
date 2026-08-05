from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import Mock

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundarySpec import CSLExactSpec
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.FileGrainOwnership import (
    LEFT_GRAIN_LABEL,
    RIGHT_GRAIN_LABEL,
    GrainOwnership,
)
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import (
    GBManipulator,
    GBManipulatorValueError,
    InterfaceCandidate,
    Parent,
    ParentValueError,
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


def _gbmaker_stub(
    topology: BoundaryNormalTopology,
    *,
    box_x: tuple[float, float] = (2.0, 12.0),
    plane: float = 7.0,
    left_bounds: tuple[float, float] | None = None,
    right_bounds: tuple[float, float] | None = None,
    inplane_periodic: tuple[bool, bool] = (True, True),
) -> GBMaker:
    """Return a specced GBMaker stub consumed through the public constructors."""
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
    vacuum = float(left_bounds[0] - xlo)

    system = Mock(spec=GBMaker)
    system.vacuum_thickness = vacuum
    system.normal_topology = topology
    system.right_grain = right
    system.left_grain = left
    system.whole_system = np.hstack((left, right))
    system.y_dim = 10.0
    system.z_dim = 10.0
    system.gb_thickness = 2.0
    system.unit_cell = UnitCell()
    system.radius = 1.0
    system.box_dims = np.array([[xlo, xhi], [-1.0, 9.0], [0.0, 10.0]])
    system.inplane_periodic = inplane_periodic
    system.epsilon = 1.0e-10
    system.gb_plane_x = plane
    return system


def _file_backed_synthetic_manipulator(
    topology: BoundaryNormalTopology,
    *,
    box_x: tuple[float, float],
    plane: float,
    left_bounds: tuple[float, float],
    right_bounds: tuple[float, float],
    inplane_periodic: tuple[bool, bool],
) -> tuple[GBManipulator, Parent]:
    """Construct asymmetric or unknown synthetic geometry through public file I/O."""
    xlo, xhi = box_x
    atoms = np.array(
        [
            ("U", left_bounds[0] + 0.25, 0.0, 1.0),
            ("O", left_bounds[1] - 0.25, 2.0, 3.0),
            ("O", right_bounds[0] + 0.25, 4.0, 5.0),
            ("U", right_bounds[1] - 0.25, 6.0, 7.0),
        ],
        dtype=Atom.atom_dtype,
    )
    labels = np.array(
        [LEFT_GRAIN_LABEL, LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
        dtype=np.int8,
    )
    ownership = GrainOwnership(
        atom_ids=np.arange(1, 5, dtype=np.int64),
        labels=labels,
        gb_plane_x=plane,
        inplane_periodic=inplane_periodic,
        left_grain_x_bounds=left_bounds,
        right_grain_x_bounds=right_bounds,
        coordinate_tolerance=1.0e-10,
        normal_topology=topology,
    )
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fluorite", 5.454, ("U", "O"))

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            suffix=".data",
            delete=False,
            encoding="utf-8",
            newline="\n",
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write("Synthetic interface candidate\n\n")
            stream.write("4 atoms\n")
            stream.write("2 atom types\n")
            stream.write(f"{xlo:.12f} {xhi:.12f} xlo xhi\n")
            stream.write("-1.000000000000 9.000000000000 ylo yhi\n")
            stream.write("0.000000000000 10.000000000000 zlo zhi\n")
            stream.write("\nAtom Type Labels\n\n")
            stream.write("1 U\n")
            stream.write("2 O\n")
            stream.write("\nAtoms\n\n")
            for atom_id, atom in enumerate(atoms, start=1):
                stream.write(
                    f"{atom_id} {atom['name']} {atom['x']:.12f} "
                    f"{atom['y']:.12f} {atom['z']:.12f}\n"
                )

        manipulator = GBManipulator(
            str(temporary_path),
            unit_cell=unit_cell,
            gb_thickness=2.0,
            type_dict={"U": 1, "O": 2},
            grain_ownership=ownership,
            seed=7,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return manipulator, manipulator.parents[0]


def _synthetic_manipulator(
    topology: BoundaryNormalTopology,
    *,
    box_x: tuple[float, float] = (2.0, 12.0),
    plane: float = 7.0,
    left_bounds: tuple[float, float] | None = None,
    right_bounds: tuple[float, float] | None = None,
    inplane_periodic: tuple[bool, bool] = (True, True),
    two_parents: bool = False,
) -> tuple[GBManipulator, Parent]:
    xlo, xhi = box_x
    if left_bounds is None:
        left_bounds = (xlo, plane)
    if right_bounds is None:
        right_bounds = (plane, xhi)

    left_vacuum = float(left_bounds[0] - xlo)
    right_vacuum = float(xhi - right_bounds[1])
    inferred_topology = (
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
        if np.isclose(left_vacuum, 0.0, atol=1.0e-12, rtol=0.0)
        and np.isclose(right_vacuum, 0.0, atol=1.0e-12, rtol=0.0)
        else BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    )
    if (
        topology is not inferred_topology
        or not np.isclose(left_vacuum, right_vacuum, atol=1.0e-12, rtol=0.0)
        or not np.isclose(left_bounds[1], plane, atol=1.0e-12, rtol=0.0)
        or not np.isclose(right_bounds[0], plane, atol=1.0e-12, rtol=0.0)
    ):
        if two_parents:
            raise AssertionError(
                "The file-backed synthetic helper supports one parent only."
            )
        return _file_backed_synthetic_manipulator(
            topology,
            box_x=box_x,
            plane=plane,
            left_bounds=left_bounds,
            right_bounds=right_bounds,
            inplane_periodic=inplane_periodic,
        )

    system = _gbmaker_stub(
        topology,
        box_x=box_x,
        plane=plane,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
        inplane_periodic=inplane_periodic,
    )
    second = (
        _gbmaker_stub(
            topology,
            box_x=box_x,
            plane=plane,
            left_bounds=left_bounds,
            right_bounds=right_bounds,
            inplane_periodic=inplane_periodic,
        )
        if two_parents
        else None
    )
    manipulator = GBManipulator(system, second, seed=7)
    return manipulator, manipulator.parents[0]


def _slab_manipulator(
    *,
    left_bounds: tuple[float, float] = (4.0, 8.0),
    right_bounds: tuple[float, float] = (8.0, 14.0),
    inplane_periodic: tuple[bool, bool] = (True, True),
) -> tuple[GBManipulator, Parent]:
    return _synthetic_manipulator(
        BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
        box_x=(2.0, 16.0),
        plane=8.0,
        left_bounds=left_bounds,
        right_bounds=right_bounds,
        inplane_periodic=inplane_periodic,
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


@pytest.mark.parametrize(
    "labels",
    [
        np.array([False, False, True, True]),
        np.array([0.0, 0.0, 1.0, 1.0]),
    ],
)
def test_interface_candidate_rejects_coercive_grain_labels(labels):
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    base = manipulator.make_parent_candidate()

    with pytest.raises(GBManipulatorValueError, match="integer left/right"):
        InterfaceCandidate(
            atoms=base.atoms,
            box_dims=base.box_dims,
            gb_plane_x=base.gb_plane_x,
            left_grain_x_bounds=base.left_grain_x_bounds,
            right_grain_x_bounds=base.right_grain_x_bounds,
            grain_labels=labels,
            inplane_periodic=base.inplane_periodic,
            normal_topology=base.normal_topology,
            coordinate_tolerance=base.coordinate_tolerance,
        )


@pytest.mark.parametrize(
    "periodic",
    [
        pytest.param((1, 0), id="integer-flags"),
        pytest.param(("False", "True"), id="string-flags"),
    ],
)
def test_interface_candidate_rejects_coercive_periodicity(periodic):
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    base = manipulator.make_parent_candidate()

    with pytest.raises(GBManipulatorValueError, match="periodic"):
        InterfaceCandidate(
            atoms=base.atoms,
            box_dims=base.box_dims,
            gb_plane_x=base.gb_plane_x,
            left_grain_x_bounds=base.left_grain_x_bounds,
            right_grain_x_bounds=base.right_grain_x_bounds,
            grain_labels=base.grain_labels,
            inplane_periodic=periodic,
            normal_topology=base.normal_topology,
            coordinate_tolerance=base.coordinate_tolerance,
        )


def test_interface_candidate_normalizes_boolean_array_periodicity():
    manipulator, _ = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    base = manipulator.make_parent_candidate()
    periodic = np.asarray([True, False], dtype=np.bool_)

    candidate = InterfaceCandidate(
        atoms=base.atoms,
        box_dims=base.box_dims,
        gb_plane_x=base.gb_plane_x,
        left_grain_x_bounds=base.left_grain_x_bounds,
        right_grain_x_bounds=base.right_grain_x_bounds,
        grain_labels=base.grain_labels,
        inplane_periodic=periodic,
        normal_topology=base.normal_topology,
        coordinate_tolerance=base.coordinate_tolerance,
    )

    assert candidate.inplane_periodic == (True, False)
    periodic[0] = False
    assert candidate.inplane_periodic == (True, False)


def test_fixed_cell_operations_return_complete_candidates():
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
    assert np.array_equal(cycled.grain_labels, np.array([0, 0, 1, 1]))


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
    assert np.array_equal(
        candidate.grain_labels,
        np.array(
            [LEFT_GRAIN_LABEL, LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
            dtype=np.int8,
        ),
    )
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
    manipulator, parent = _slab_manipulator(inplane_periodic=(False, True))

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
    boundary = CSLExactSpec(
        axis=[0, 0, 1],
        plane=[1, 0, 0],
        quat=[3, 0, 0, 1],
    )
    gb = GBMaker.from_boundary_spec(
        3.0,
        "rocksalt",
        ("Na", "Cl"),
        boundary,
        mode="exact",
        gb_thickness=5.0,
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
    manipulator, parent = _synthetic_manipulator(
        BoundaryNormalTopology.PERIODIC_BICRYSTAL
    )
    order = np.array([0, 2, 1, 3])
    atoms = parent.whole_system[order]
    labels = manipulator.make_parent_candidate().grain_labels[order]
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


def _parent_gbmaker_stub(
    *,
    vacuum: float,
    topology: BoundaryNormalTopology,
) -> GBMaker:
    atoms = np.array(
        [("U", 1.0, 1.0, 1.0), ("O", 6.0, 1.0, 1.0)],
        dtype=Atom.atom_dtype,
    )
    system = Mock(spec=GBMaker)
    system.vacuum_thickness = vacuum
    system.normal_topology = topology
    system.right_grain = atoms[1:]
    system.left_grain = atoms[:1]
    system.whole_system = atoms
    system.y_dim = 10.0
    system.z_dim = 10.0
    system.gb_thickness = 2.0
    system.unit_cell = UnitCell()
    system.radius = 1.0
    system.box_dims = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    system.inplane_periodic = (True, True)
    system.epsilon = 1.0e-8
    system.gb_plane_x = 5.0
    return system


def test_parent_consumes_explicit_gbmaker_topology():
    periodic = Parent(
        _parent_gbmaker_stub(
            vacuum=0.0,
            topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        )
    )
    assert periodic.normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL
    assert np.array_equal(periodic.left_grain_x_bounds, [0.0, 5.0])

    slab = Parent(
        _parent_gbmaker_stub(
            vacuum=1.5,
            topology=BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
        )
    )
    assert slab.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
    assert np.array_equal(slab.left_grain_x_bounds, [1.5, 5.0])
    assert np.array_equal(slab.right_grain_x_bounds, [5.0, 8.5])


def test_parent_rejects_gbmaker_topology_vacuum_conflict():
    system = _parent_gbmaker_stub(
        vacuum=1.0,
        topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
    )

    with pytest.raises(ParentValueError, match="zero vacuum"):
        Parent(system)
