# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import copy
import filecmp
import importlib
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundarySpec import CSLExactSpec, FiveDOFSpec, PQSpec
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import (
    GBManipulator,
    GBManipulatorTypeError,
    GBManipulatorValueError,
    Parent,
    ParentCorruptedFileError,
    ParentFileMissingDataError,
    ParentFileNotFoundError,
    ParentsProxyIndexError,
    ParentsProxyTypeError,
    ParentsProxyValueError,
    ParentValueError,
    _calculate_local_order,
    _create_periodic_image_neighborhoods,
    _create_periodic_neighbor_list,
    _delaunay_insertion_sites,
    _floor_stoichiometric_count,
    _get_stoichiometric_change,
    _grid_insertion_sites,
    _minimum_periodic_distances,
    _ParentsProxy,
    _tetrahedron_circumcenters,
    _tile_periodic_positions_once,
)
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL, GrainOwnership
from GBOpt.UnitCell import UnitCell

pytestmark = pytest.mark.filterwarnings(
    "ignore:File-backed Parent initialization without explicit grain ownership is "
    "deprecated.*:DeprecationWarning"
)

_TEST_DIR = Path(__file__).resolve().parent
_INPUT_DIR = _TEST_DIR / "inputs"
_ASYMMETRIC_PQ_SPEC = PQSpec(
    P=[[-1, -1, 6], [1, -1, 0], [3, 3, 1]],
    Q=[[1, 1, 12], [1, -1, 0], [6, 6, -1]],
    basis_mode="supplied",
)


def structured_array_equal(array1, array2):
    if array1.dtype != array2.dtype:
        return False

    for field in array1.dtype.names:
        if np.issubdtype(array1[field].dtype, np.number):
            if not np.allclose(array1[field], array2[field]):
                return False
        else:
            if not np.array_equal(array1[field], array2[field]):
                return False

    return True


_SIGMA5_TILT_EXACT_SPEC = CSLExactSpec(
    axis=(0, 0, 1),
    plane=(3, 1, 0),
    quat=(3, 0, 0, 1),
    sigma=5,
)
_SIGMA5_TWIST_EXACT_SPEC = CSLExactSpec(
    axis=(1, 0, 0),
    plane=(1, 0, 0),
    quat=(3, 1, 0, 0),
    sigma=5,
)


def _make_exact_gb(
    a0,
    structure,
    atom_types,
    *,
    boundary=_SIGMA5_TILT_EXACT_SPEC,
    gb_thickness=10.0,
    repeat_factor=2,
    x_dim_min=10.0,
    vacuum=10.0,
    interaction_distance=1.0,
):
    """Return a compact exact boundary for manipulator integration tests."""
    return GBMaker.from_boundary_spec(
        a0,
        structure,
        atom_types,
        boundary,
        mode="exact",
        gb_thickness=gb_thickness,
        repeat_factor=repeat_factor,
        x_dim_min=x_dim_min,
        vacuum=vacuum,
        interaction_distance=interaction_distance,
    )


def _make_approximate_gb(
    a0,
    structure,
    atom_types,
    misorientation,
    *,
    gb_thickness,
    **kwargs,
):
    """Return an approximate boundary through the supported spec API."""
    return GBMaker.from_boundary_spec(
        a0,
        structure,
        atom_types,
        FiveDOFSpec(misorientation),
        mode="approximate",
        gb_thickness=gb_thickness,
        **kwargs,
    )


def _make_asymmetric_exact_gb():
    """Return a compact exact boundary whose physical interface is off-center."""
    return GBMaker.from_boundary_spec(
        1.0,
        "sc",
        "Cu",
        _ASYMMETRIC_PQ_SPEC,
        mode="exact",
        gb_thickness=1.0,
        repeat_factor=(2, 2),
        x_dim_min=1.0,
        vacuum=0.0,
        interaction_distance=0.1,
        mismatch_tol=0.005,
        mismatch_max_cells=50,
        strain_grain="both",
    )


def _local_orders_for_structure(structure, atoms):
    unit_cell = UnitCell()
    unit_cell.init_by_structure(structure, 1.0, atoms)
    basis = unit_cell.asarray()
    basis_positions = np.column_stack((basis["x"], basis["y"], basis["z"]))
    basis_types = unit_cell.names(asint=True)
    rmax = 1.05

    translation_indices = np.array(
        np.meshgrid(
            np.arange(-1, 2),
            np.arange(-1, 2),
            np.arange(-1, 2),
            indexing="ij",
        )
    ).reshape(3, -1).T
    translations = translation_indices @ unit_cell.conventional
    supercell_positions = (
        translations[:, np.newaxis, :] + basis_positions[np.newaxis, :, :]
    ).reshape(-1, 3)
    supercell_types = np.tile(basis_types, len(translations))

    central_position = basis_positions[0]
    distances = np.linalg.norm(supercell_positions - central_position, axis=1)
    neighborhood = (distances > 1e-12) & (distances < rmax)

    atom = np.array([basis_types[0], *central_position], dtype=np.float64)
    ideal_neighbors = np.column_stack(
        (supercell_types[neighborhood], supercell_positions[neighborhood])
    )
    distorted_neighbors = ideal_neighbors.copy()
    distorted_neighbors[:, 1:] = central_position + (
        ideal_neighbors[:, 1:] - central_position
    ) * np.array([1.0, 1.15, 0.85])

    kwargs = {
        "unit_cell_types": basis_types,
        "unit_cell_a0": unit_cell.a0,
        "N": len(basis),
        "Delta": 0.05,
        "Rmax": rmax,
    }
    ideal_order = _calculate_local_order(atom, ideal_neighbors, **kwargs)
    distorted_order = _calculate_local_order(atom, distorted_neighbors, **kwargs)
    return ideal_order, distorted_order, len(ideal_neighbors)


cross_structure = pytest.mark.slow(
    reason="Supplementary local-order regression across additional crystal templates"
)


@pytest.mark.parametrize(
    ("structure", "atoms"),
    (
        pytest.param("fcc", "Cu", id="fcc"),
        pytest.param("fluorite", ("U", "O"), id="fluorite"),
        pytest.param("bcc", "Fe", marks=cross_structure, id="bcc"),
        pytest.param("sc", "Po", marks=cross_structure, id="sc"),
        pytest.param("diamond", "C", marks=cross_structure, id="diamond"),
        pytest.param(
            "rocksalt",
            ("Na", "Cl"),
            marks=cross_structure,
            id="rocksalt",
        ),
        pytest.param(
            "zincblende",
            ("Zn", "S"),
            marks=cross_structure,
            id="zincblende",
        ),
    ),
)
def test_local_order_is_higher_for_ideal_crystal_neighborhoods(structure, atoms):
    ideal_order, distorted_order, neighbor_count = _local_orders_for_structure(
        structure, atoms
    )

    assert neighbor_count > 0
    assert np.isfinite(ideal_order)
    assert np.isfinite(distorted_order)
    assert ideal_order >= 0.0
    assert distorted_order >= 0.0
    assert ideal_order > distorted_order + 1e-8


def _synthetic_manipulator(
    unit_cell,
    atoms,
    seed=100,
    *,
    box_dims=None,
    inplane_periodic=(False, False),
    coordinate_tolerance=1e-8,
):
    # Bypass full GBMaker construction so stoichiometric mutator tests can
    # isolate selection logic with a tiny deterministic parent.
    if box_dims is None:
        box_dims = np.asarray(
            [
                [-20.0, 20.0],
                [-20.0, 20.0],
                [-20.0, 20.0],
            ],
            dtype=float,
        )

    parent = SimpleNamespace(
        unit_cell=unit_cell,
        whole_system=atoms,
        gb_atoms=atoms,
        gb_indices=np.arange(len(atoms), dtype=np.intp),
        box_dims=np.asarray(box_dims, dtype=float),
        inplane_periodic=tuple(inplane_periodic),
        coordinate_tolerance=float(coordinate_tolerance),
    )

    manipulator = object.__new__(GBManipulator)
    manipulator._GBManipulator__one_parent = True
    manipulator._GBManipulator__parents = [parent, None]
    manipulator.rng = np.random.default_rng(seed)

    return manipulator


def test_insert_atoms_with_stoichiometry_uses_selected_neighbor_site_ids(monkeypatch):
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fluorite", 5.454, ("U", "O"))
    atoms = np.array(
        [
            ("U", 0.0, 0.0, 0.0),
            ("O", 0.0, 2.0, 0.0),
            ("O", 0.0, 0.0, 2.0),
            ("U", 0.0, 0.0, 1.0),
            ("U", 0.0, 1.0, 1.0),
            ("U", 0.0, 3.0, 1.0),
            ("O", 0.0, 1.0, 0.5),
            ("O", 0.0, 1.0, 1.5),
            ("O", 0.0, 2.0, 0.5),
            ("O", 0.0, 2.0, 1.5),
            ("O", 0.0, 3.0, 0.5),
            ("O", 0.0, 3.0, 1.5),
        ],
        dtype=Atom.atom_dtype,
    )
    manipulator = _synthetic_manipulator(
        unit_cell,
        atoms,
        box_dims=np.array([[-1.0, 4.0], [0.0, 4.0], [-1.0, 4.0]], dtype=float),
        inplane_periodic=(True, False)
    )

    class FixedChoiceRng:
        def __init__(self):
            self.results = iter((np.array([3]), np.array([5, 7])))

        def choice(self, choices, size, replace=False, p=None):
            del p
            choices = np.asarray(choices)
            result = next(self.results)
            assert len(result) == size
            assert replace is False
            assert np.all(np.isin(result, choices))
            return result

    recorded_sites = {}

    def sparse_site_neighbors(_cutoff, positions, center_indices, _box_dims, periodic):
        recorded_sites["positions"] = np.asarray(positions, dtype=float)
        np.testing.assert_array_equal(center_indices, np.array([3], dtype=np.intp))
        assert periodic == (False, True, False)
        return ([np.array([5, 7], dtype=np.intp)], [np.array([1.0, 1.0])])

    manipulator.rng = FixedChoiceRng()
    possible_sites = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5],
            [0.0, 1.5, 0.5],
            [0.0, 2.0, 0.5],
            [0.0, 2.5, 0.5],
            [0.0, 3.0, 0.5],
            [0.0, 3.5, 0.5],
        ],
        dtype=float,
    )
    probabilities = np.full(len(possible_sites), 1.0 / len(possible_sites))
    module = importlib.import_module("GBOpt.GBManipulator")

    monkeypatch.setattr(
        module,
        "_grid_insertion_sites",
        lambda *_args, **_kwargs: (possible_sites, probabilities),
    )
    monkeypatch.setattr(module, "_create_periodic_neighbor_list", sparse_site_neighbors)

    _, new_atoms = manipulator.insert_atoms(
        num_to_insert=3,
        method="grid",
        keep_ratio=True,
        return_positions=True,
    )

    sites = recorded_sites["positions"]
    inserted_by_position = {
        tuple(float(value) for value in (atom["x"], atom["y"], atom["z"])): str(
            atom["name"]
        )
        for atom in new_atoms
    }
    assert inserted_by_position == {
        tuple(sites[3]): "U",
        tuple(sites[5]): "O",
        tuple(sites[7]): "O",
    }


@pytest.mark.integration(
    reason="Exercises GBMaker, soft-mode displacement, and LAMMPS write/read together"
)
def test_displace_along_soft_modes_preserves_multitype_numeric_roundtrip(tmp_path):
    seed = 100
    gb = _make_exact_gb(
        3.0,
        "rocksalt",
        ("Na", "Cl"),
        gb_thickness=5.0,
        repeat_factor=(2, 3),
        x_dim_min=10.0,
        vacuum=2.0,
        interaction_distance=4.0,
    )
    manipulator = GBManipulator(gb, seed=seed)

    child = manipulator.displace_along_soft_modes(mesh_size=1, num_q=1)[0]

    np.testing.assert_array_equal(child["name"], gb.whole_system["name"])

    output_path = tmp_path / "displaced.data"
    gb.write_lammps(
        str(output_path),
        child,
        gb.box_dims,
        type_as_int=True,
    )
    loaded = GBManipulator(
        str(output_path),
        unit_cell=gb.unit_cell,
        gb_thickness=gb.gb_thickness,
        seed=seed,
    )

    np.testing.assert_array_equal(
        loaded.parents[0].whole_system["name"],
        child["name"],
    )


class TestGBManipulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 100
        cls.tilt = _make_exact_gb(
            1.0,
            "fcc",
            "Cu",
            gb_thickness=10.0,
            interaction_distance=1.0,
            repeat_factor=2,
        )
        cls.twist = _make_exact_gb(
            1.0,
            "fcc",
            "Cu",
            boundary=_SIGMA5_TWIST_EXACT_SPEC,
            gb_thickness=10.0,
            interaction_distance=1.0,
            repeat_factor=2,
        )

    def setUp(self):
        self.a0 = 1.0
        self.structure = 'fcc'
        self.gb_thickness = 10.0
        self.atom_types = 'Cu'
        self.misorientation = [math.radians(36.869898), 0, 0, 0, 0]
        self.file1 = str(_INPUT_DIR / "basic_dump_test1.txt")
        self.file2 = str(_INPUT_DIR / "basic_dump_test2.txt")
        self.seed = 100
        self.manipulator_tilt = GBManipulator(self.tilt, seed=self.seed)
        self.manipulator_twist = GBManipulator(self.twist, seed=self.seed)

    def test_init_with_one_gbmaker_parent(self):
        self.assertIsNotNone(self.manipulator_tilt.parents[0])
        self.assertIsNone(self.manipulator_tilt.parents[1])

    def test_init_with_two_gbmaker_parents(self):
        manipulator = GBManipulator(self.tilt, self.tilt)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNotNone(manipulator.parents[1])

    def test_init_with_one_snapshot(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0, self.atom_types)
        gb_thickness = 10
        manipulator = GBManipulator(
            self.file1, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNone(manipulator.parents[1])

    def test_init_with_two_snapshots(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0, self.atom_types)
        gb_thickness = 10
        manipulator = GBManipulator(
            self.file1, self.file2, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNotNone(manipulator.parents[1])

    def test_init_with_mixed_input(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.structure, self.a0, self.atom_types)
        gb_thickness = 10
        manipulator = GBManipulator(
            self.tilt, self.file1, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator.parents[0])
        self.assertIsNotNone(manipulator.parents[1])

        manipulator2 = GBManipulator(
            self.file1, self.tilt, unit_cell=unit_cell, gb_thickness=gb_thickness)
        self.assertIsNotNone(manipulator2.parents[0])
        self.assertIsNotNone(manipulator2.parents[1])

        # test with GBMaker instance and file, without unit cell or gb_thickness
        manipulator3 = GBManipulator(self.tilt, self.file1)
        self.assertEqual(
            manipulator3.parents[0].unit_cell, manipulator3.parents[1].unit_cell)
        self.assertEqual(
            manipulator3.parents[0].gb_thickness, manipulator3.parents[1].gb_thickness)

    def test_grain_translation(self):
        new_system = self.manipulator_tilt.translate_right_grain(1.0, 0.5)
        self.assertTrue(np.allclose(self.tilt.whole_system['x'], new_system['x']))
        self.assertTrue(not np.allclose(self.tilt.whole_system['y'], new_system['y']))
        self.assertTrue(not np.allclose(self.tilt.whole_system['z'], new_system['z']))
        self.assertTrue(all(self.tilt.whole_system['name'] == new_system['name']))

    def test_grain_translation_warning(self):
        manipulator = GBManipulator(self.tilt, self.tilt)
        with self.assertWarnsRegex(
            UserWarning,
            "grain translation only occurring based on parent 1",
        ):
            _ = manipulator.translate_right_grain(1.0, 1.0)

    def test_slice_and_merge(self):
        manipulator = GBManipulator(self.tilt, self.tilt, seed=self.seed)
        new_system = manipulator.slice_and_merge()
        self.assertFalse(all(self.tilt.whole_system == new_system))

    def test_slice_and_merge_error(self):
        with self.assertRaises(GBManipulatorValueError):
            _ = self.manipulator_tilt.slice_and_merge()

    def test_remove_atoms(self):
        new_system = self.manipulator_tilt.remove_atoms(gb_fraction=0.10)
        self.assertGreater(len(self.tilt.whole_system), len(new_system))

    def test_remove_atoms_fraction_error(self):
        with self.assertRaises(GBManipulatorValueError):
            _ = self.manipulator_tilt.remove_atoms(gb_fraction=0.50)

    def test_remove_atoms_2_parent_warning(self):
        manipulator = GBManipulator(self.tilt, self.tilt, seed=self.seed)
        with self.assertWarnsRegex(
            UserWarning,
            "Atom removal only occurring based on parent 1",
        ):
            _ = manipulator.remove_atoms(gb_fraction=0.10)
            _ = manipulator.remove_atoms(gb_fraction=0.10)

    def test_remove_atoms_calculated_fraction_warning(self):
        with self.assertWarnsRegex(UserWarning, "zero atoms"):
            _ = self.manipulator_tilt.remove_atoms(gb_fraction=1e-7)

    def test_remove_atoms_with_specific_number(self):
        new_system = self.manipulator_tilt.remove_atoms(num_to_remove=1)
        self.assertEqual(len(self.tilt.whole_system)-1, len(new_system))

    def test_remove_atoms_with_stoichiometry_removes_one_fluorite_formula_unit(self):
        unit_cell = UnitCell()
        unit_cell.init_by_structure("fluorite", 1.0, ("U", "O"))
        atoms = np.array(
            [
                ("U", 0.0, 0.0, 0.0),
                ("O", 0.1, 0.0, 0.0),
                ("O", 0.0, 0.1, 0.0),
                ("U", 4.0, 0.0, 0.0),
                ("O", 4.1, 0.0, 0.0),
                ("O", 4.0, 0.1, 0.0),
                ("U", 8.0, 0.0, 0.0),
                ("O", 8.1, 0.0, 0.0),
                ("O", 8.0, 0.1, 0.0),
                ("U", 12.0, 0.0, 0.0),
                ("O", 12.1, 0.0, 0.0),
                ("O", 12.0, 0.1, 0.0),
            ],
            dtype=Atom.atom_dtype,
        )
        original_atoms = atoms.copy()

        manipulator = _synthetic_manipulator(unit_cell, atoms, seed=self.seed)

        with patch("GBOpt.GBManipulator._calculate_local_order", return_value=1.0):
            new_system, removed = manipulator.remove_atoms(
                num_to_remove=3,
                keep_ratio=True,
                return_positions=True,
            )

        np.testing.assert_array_equal(atoms, original_atoms)
        self.assertEqual(len(new_system), len(atoms) - 3)

        names, counts = np.unique(removed["name"], return_counts=True)
        self.assertEqual(dict(zip(names, counts)), {"O": 2, "U": 1})

    def test_insert_atoms(self):
        new_system_delaunay = self.manipulator_tilt.insert_atoms(
            fill_fraction=0.05, method='delaunay')
        self.assertGreater(len(new_system_delaunay), len(self.tilt.whole_system))
        new_system_grid = self.manipulator_tilt.insert_atoms(
            fill_fraction=0.10, method='grid')
        self.assertGreater(len(new_system_grid), len(self.tilt.whole_system))

    def test_insert_atoms_fraction_error(self):
        with self.assertRaises(GBManipulatorValueError):
            _ = self.manipulator_tilt.insert_atoms(
                fill_fraction=0.50, method='delaunay')

    def test_insert_atoms_2_parent_warning(self):
        manipulator = GBManipulator(self.tilt, self.tilt, seed=self.seed)
        with self.assertWarnsRegex(
            UserWarning,
            "Atom insertion only occurring based on parent 1",
        ):
            _ = manipulator.insert_atoms(fill_fraction=0.05, method='delaunay')

    def test_insert_atoms_calculated_fraction_warning(self):
        with self.assertWarnsRegex(UserWarning, "zero atoms"):
            _ = self.manipulator_tilt.insert_atoms(
                fill_fraction=1e-7, method='delaunay')

        with self.assertWarnsRegex(UserWarning, "zero atoms"):
            _ = self.manipulator_tilt.insert_atoms(
                fill_fraction=1e-7, method='grid')

    def test_insert_atoms_invalid_method(self):
        with self.assertRaises(GBManipulatorValueError):
            _ = self.manipulator_tilt.insert_atoms(fill_fraction=0.10, method='invalid')

    def test_insert_atoms_with_specific_number(self):
        new_system_delaunay = self.manipulator_tilt.insert_atoms(
            method='delaunay', num_to_insert=1)
        self.assertEqual(len(self.tilt.whole_system) + 1, len(new_system_delaunay))
        new_system_grid = self.manipulator_tilt.insert_atoms(
            method='grid', num_to_insert=1)
        self.assertEqual(len(self.tilt.whole_system) + 1, len(new_system_grid))

    def test_type_preservation_with_numeric_roundtrip(self):
        gb = _make_exact_gb(
            3.0,
            "rocksalt",
            ("Na", "Cl"),
            gb_thickness=5.0,
            repeat_factor=(2, 3),
            x_dim_min=10.0,
            vacuum=2.0,
            interaction_distance=4.0,
        )
        expected_types = {"Na", "Cl"}
        base_names = gb.whole_system["name"]
        self.assertEqual(set(base_names), expected_types)

        def roundtrip_names(atoms):
            with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                gb.write_lammps(
                    temp_file.name,
                    atoms,
                    gb.box_dims,
                    type_as_int=True,
                )
                loaded = GBManipulator(
                    temp_file.name,
                    unit_cell=gb.unit_cell,
                    gb_thickness=gb.gb_thickness,
                    seed=self.seed,
                )
                return loaded.parents[0].whole_system["name"]

        manipulator = GBManipulator(gb, seed=self.seed)

        translated = manipulator.translate_right_grain(0.1, 0.2)
        self.assertTrue(np.array_equal(translated["name"], base_names))
        self.assertEqual(set(roundtrip_names(translated)), expected_types)

        with patch("GBOpt.GBManipulator._calculate_local_order", return_value=1.0):
            removed = manipulator.remove_atoms(num_to_remove=2, keep_ratio=True)
        self.assertEqual(set(roundtrip_names(removed)), expected_types)

        inserted = manipulator.insert_atoms(
            num_to_insert=2,
            method="grid",
            keep_ratio=True,
        )
        self.assertTrue(
            np.array_equal(inserted["name"][:len(base_names)], base_names)
        )
        self.assertEqual(set(roundtrip_names(inserted)), expected_types)

        manipulator_two = GBManipulator(gb, gb, seed=self.seed)
        sliced = manipulator_two.slice_and_merge()
        self.assertEqual(set(roundtrip_names(sliced)), expected_types)

    @pytest.mark.slow
    def test_displace_along_soft_modes_base(self):
        # test the base case
        child = self.manipulator_tilt.displace_along_soft_modes()
        self.assertEqual(len(child), 1)
        self.assertFalse(structured_array_equal(
            child[0], self.manipulator_tilt.parents[0].whole_system))

    @pytest.mark.slow
    def test_displace_along_soft_modes_with_displacement_threshold(self):
        # test the case with a displacement threshold specified
        child = self.manipulator_tilt.displace_along_soft_modes(1.0)
        self.assertEqual(len(child), 1)
        self.assertFalse(structured_array_equal(
            child[0], self.manipulator_tilt.parents[0].whole_system))

    @pytest.mark.slow
    def test_displace_along_soft_modes_diff_mesh(self):
        # test differing mesh size
        child = self.manipulator_tilt.displace_along_soft_modes(mesh_size=2)
        self.assertEqual(len(child), 1)
        self.assertFalse(structured_array_equal(
            child[0], self.manipulator_tilt.parents[0].whole_system))

    @pytest.mark.slow
    def test_displace_along_soft_modes_num_q_vecs(self):
        # test number of q vectors
        child = self.manipulator_tilt.displace_along_soft_modes(num_q=20)
        self.assertEqual(len(child), 1)
        self.assertFalse(structured_array_equal(
            child[0], self.manipulator_tilt.parents[0].whole_system))

    @pytest.mark.slow
    def test_displace_along_soft_modes_num_child_structures(self):
        # test number of child structures
        children = self.manipulator_tilt.displace_along_soft_modes(num_children=2)
        self.assertEqual(len(children), 2)
        self.assertFalse(structured_array_equal(
            children[0], self.manipulator_tilt.parents[0].whole_system))
        self.assertFalse(structured_array_equal(
            children[1], self.manipulator_tilt.parents[0].whole_system))
        self.assertFalse(structured_array_equal(children[0], children[1]))

    @pytest.mark.slow
    def test_displace_along_soft_modes_simple_case(self):
        # While we end up using the indicated file for the actual atomic configuration,
        # that configuration was developed using this set of parameters
        GB = _make_approximate_gb(
            3.54,
            "fcc",
            "Cu",
            np.zeros(5),
            gb_thickness=5.0,
            repeat_factor=6,
            x_dim_min=10,
            vacuum=10,
            interaction_distance=5,
        )
        manipulator = GBManipulator(
            './tests/inputs/Cu_single_crystal_with_displaced_atom.txt', unit_cell=GB.unit_cell, gb_thickness=5)
        child1 = manipulator.displace_along_soft_modes()[0]
        child2 = manipulator.displace_along_soft_modes(subtract_displacement=True)[0]
        self.assertFalse(structured_array_equal(child1, child2))

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            GB.write_lammps(temp_file.name, child1, GB.box_dims)
            self.assertTrue(
                filecmp.cmp(
                    temp_file.name,
                    './tests/gold/soft_phonon_mode_displacement_added.txt',
                    shallow=False
                )
            )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            GB.write_lammps(temp_file.name, child2, GB.box_dims)
            self.assertTrue(
                filecmp.cmp(
                    temp_file.name,
                    './tests/gold/soft_phonon_mode_displacement_subtracted.txt',
                    shallow=False
                )
            )

    def test_apply_group_symmetry(self):
        manipulator = GBManipulator(self.tilt, seed=self.seed)
        with self.assertRaises(NotImplementedError):
            _ = manipulator.apply_group_symmetry("group")

    def test_parents_getter(self):
        manipulator = GBManipulator(self.tilt)
        parents = manipulator.parents
        self.assertTrue(isinstance(parents, _ParentsProxy))
        self.assertTrue(isinstance(parents[0], Parent))
        self.assertIsNone(parents[1])

    def test_parents_setter(self):
        manipulator = GBManipulator(self.tilt)
        self.assertIsNone(manipulator.parents[1])
        manipulator.parents[1] = Parent(self.tilt)
        self.assertIsNotNone(manipulator.parents[1])
        manipulator.parents = [Parent(self.file1, unit_cell=self.tilt.unit_cell),
                               Parent(self.file2, unit_cell=self.tilt.unit_cell)]
        self.assertFalse(None in manipulator.parents)

        with self.assertRaises(GBManipulatorValueError):
            manipulator.parents = Parent(self.file1, unit_cell=self.tilt.unit_cell)

        with self.assertRaises(GBManipulatorValueError):
            manipulator.parents = [Parent(self.file1, unit_cell=self.tilt.unit_cell), 1]

    def test_write_lammps_after_manipulate(self):
        manipulator1 = GBManipulator(self.tilt)
        manipulator2 = GBManipulator(self.tilt, self.tilt)
        p1 = manipulator1.translate_right_grain(1, 1)
        p2 = manipulator2.slice_and_merge()
        p3 = manipulator1.insert_atoms(fill_fraction=0.05, method='delaunay')
        # p4 = manipulator1.remove_atoms(0.2)
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            self.tilt.write_lammps(temp_file.name, p1, self.tilt.box_dims)
            self.tilt.write_lammps(temp_file.name, p2, self.tilt.box_dims)
            self.tilt.write_lammps(temp_file.name, p3, self.tilt.box_dims)
            # self.tilt.write_lammps(temp_file.name, p4, self.tilt.box_dims)


class TestParent(unittest.TestCase):
    def setUp(self):
        self.unit_cell = UnitCell()
        self.unit_cell.init_by_structure('fcc', 1.0, 'Cu')
        self.GB = _make_exact_gb(
            1.0,
            "fcc",
            "Cu",
            gb_thickness=10.0,
            repeat_factor=2,
            interaction_distance=1.0,
        )
        self.parent = Parent(self.GB)
        self.file = 'tests/inputs/basic_dump_test1.txt'

    def test_parent_init(self):
        parent1 = Parent(self.GB)
        self.assertGreater(len(parent1.left_grain), 0)
        self.assertGreater(len(parent1.right_grain), 0)
        self.assertEqual(len(parent1.left_grain) +
                         len(parent1.right_grain), len(parent1.whole_system))

        parent2 = Parent(
            self.file, unit_cell=self.unit_cell, gb_thickness=20)
        self.assertGreater(len(parent2.left_grain), 0)
        self.assertGreater(len(parent2.right_grain), 0)
        self.assertEqual(len(parent2.left_grain) +
                         len(parent2.right_grain), len(parent2.whole_system))
        self.assertEqual(parent2.gb_thickness, 20)
        self.assertEqual(parent2.whole_system[0]['name'], 'H')

    def test_parent_getters(self):
        parent = Parent(self.GB)
        self.assertEqual(repr(parent.unit_cell), repr(self.GB.unit_cell))
        self.assertEqual(parent.gb_thickness, self.GB.gb_thickness)
        self.assertTrue(np.allclose(parent.box_dims, self.GB.box_dims))
        # Parent calculates x_dim differently than GB
        self.assertNotEqual(parent.x_dim, self.GB.x_dim)
        self.assertEqual(parent.y_dim, self.GB.y_dim)
        self.assertEqual(parent.z_dim, self.GB.z_dim)

        x_gb = parent.gb_plane_x
        left_cut = x_gb - parent.gb_thickness / 2.0
        right_cut = x_gb + parent.gb_thickness / 2.0
        left_gb = parent.left_grain[parent.left_grain["x"] > left_cut]
        right_gb = parent.right_grain[parent.right_grain["x"] < right_cut]
        np.testing.assert_array_equal(parent.gb_atoms, np.hstack((left_gb, right_gb)))
        np.testing.assert_array_equal(parent.left_grain, self.GB.left_grain)
        np.testing.assert_array_equal(parent.right_grain, self.GB.right_grain)
        np.testing.assert_array_equal(parent.whole_system, self.GB.whole_system)
        self.assertEqual(parent.unit_cell, self.GB.unit_cell)

    def test_parent_snapshot_init_errors(self):
        with self.assertRaises(ParentValueError):
            _ = Parent(self.file)
        with self.assertRaises(ParentFileNotFoundError):
            _ = Parent("tests/inputs/file_not_found.txt", unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent("tests/inputs/file_without_box_bounds.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent("tests/inputs/file_with_invalid_box_bounds.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent("tests/inputs/file_with_invalid_box_bounds2.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent("tests/inputs/file_without_atoms.txt",
                       unit_cell=self.unit_cell)
        with self.assertRaises(ParentFileMissingDataError):
            _ = Parent("tests/inputs/file_missing_required_info.txt",
                       unit_cell=self.unit_cell)

    def test_read_lammps_with_typelabel(self):
        parent = Parent("tests/inputs/lammps_dump_with_typelabel_test.txt",
                        unit_cell=self.unit_cell, gb_thickness=20)
        self.assertEqual(parent.whole_system[0]['name'], 'Cu')

    def test_read_lammps_input(self):
        uc = UnitCell()
        uc.init_by_structure('fcc', 3.54, 'Cu')
        parent1 = Parent("tests/inputs/lammps_input_with_labels.txt",
                         unit_cell=uc, gb_thickness=10)
        parent2 = Parent("tests/inputs/lammps_input_without_labels.txt",
                         unit_cell=uc, gb_thickness=10)
        self.assertEqual(len(parent1.whole_system), 792)
        self.assertTrue(np.isclose(parent1.whole_system[0]['x'], 1.77))
        self.assertTrue(np.isclose(parent1.whole_system[0]['y'], 1.77))
        self.assertTrue(np.isclose(parent1.whole_system[0]['z'], 3.54))
        self.assertTrue(np.allclose(parent1.box_dims, np.array(
            [[-10, 30], [0, 21.24], [0, 21.24]])))
        self.assertTrue(all(parent1.whole_system['name'] == 'Cu'))

        self.assertEqual(len(parent2.whole_system), 792)
        self.assertTrue(np.isclose(parent2.whole_system[0]['x'], 1.77))
        self.assertTrue(np.isclose(parent2.whole_system[0]['y'], 1.77))
        self.assertTrue(np.isclose(parent2.whole_system[0]['z'], 3.54))
        self.assertTrue(np.allclose(parent2.box_dims, np.array(
            [[-10, 30], [0, 21.24], [0, 21.24]])))
        self.assertTrue(all(parent2.whole_system['name'] == 'H'))

    def test_read_lammps_input_multiple_atom_types(self):
        uc = UnitCell()
        uc.init_by_structure('fcc', 3.54, 'Cu')
        parent = Parent(
            "tests/inputs/lammps_input_multiple_atom_types.txt",
            unit_cell=uc, gb_thickness=10)
        self.assertTrue(
            all(np.unique(parent.whole_system['name']) == np.array(['Cu', 'Fe', 'Ni'])))

    def test_read_lammps_input_errors(self):
        uc = UnitCell()
        uc.init_by_structure('fcc', 354, 'Cu')
        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent(
                "tests/inputs/lammps_input_multiple_atom_types_missing_labels.txt",
                unit_cell=uc)

        with self.assertRaises(ParentCorruptedFileError):
            _ = Parent(
                "tests/inputs/lammps_input_multiple_atom_types_wrong_num_types.txt",
                unit_cell=uc)

    def test_unknown_file_type(self):
        with self.assertRaises(ParentValueError):
            _ = Parent('tests/inputs/unknown_file_type.txt',
                       unit_cell=self.unit_cell, gb_thickness=20)

    def test_file_too_short(self):
        with self.assertRaises(ParentValueError):
            _ = Parent("tests/inputs/file_too_short.txt",
                       unit_cell=self.unit_cell, gb_thickness=20)


class TestParentGBRegion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gbm = _make_asymmetric_exact_gb()
        cls.parent = Parent(cls.gbm)
        cls.d_hkl = max(
            cls.gbm.spacing["x"]["left"],
            cls.gbm.spacing["x"]["right"],
        )

    def test_fixture_interface_plane_is_not_box_midpoint(self):
        midpoint = float(np.mean(self.gbm.box_dims[0]))
        self.assertGreater(
            abs(self.gbm.gb_plane_x - midpoint),
            100.0 * self.gbm.epsilon,
        )

    def test_gb_indices_lie_within_symmetric_window(self):
        parent = self.parent
        x_gb = parent.gb_plane_x
        half = parent.gb_thickness / 2.0
        xs = parent.whole_system["x"][parent.gb_indices]
        self.assertGreater(len(xs), 0)
        self.assertTrue(np.all(xs > x_gb - half))
        self.assertTrue(np.all(xs < x_gb + half))

    def test_gb_indices_include_terminal_interface_layers(self):
        xs = self.parent.whole_system["x"][self.parent.gb_indices]
        left_terminal_x = float(np.max(self.parent.left_grain["x"]))
        right_terminal_x = float(np.min(self.parent.right_grain["x"]))

        self.assertTrue(
            np.any(np.isclose(xs, left_terminal_x)),
            msg="GB region must retain the terminal left-grain interface layer",
        )
        self.assertTrue(
            np.any(np.isclose(xs, right_terminal_x)),
            msg="GB region must retain the terminal right-grain interface layer",
        )

    def test_gb_plane_x_from_gbmaker_matches_source(self):
        self.assertAlmostEqual(self.parent.gb_plane_x, self.gbm.gb_plane_x, places=10)

    def test_legacy_file_path_gb_plane_x_inference_warns_and_remains_near_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asymmetric_boundary.dat"
            self.gbm.write_lammps(str(path), type_as_int=True)
            unit_cell = UnitCell()
            unit_cell.init_by_structure("sc", 1.0, "Cu")
            with self.assertWarnsRegex(
                DeprecationWarning,
                "without explicit grain ownership is deprecated",
            ):
                parent_file = Parent(
                    str(path),
                    unit_cell=unit_cell,
                    gb_thickness=self.gbm.gb_thickness,
                )

        self.assertAlmostEqual(
            parent_file.gb_plane_x,
            self.gbm.gb_plane_x,
            delta=self.d_hkl,
            msg="Legacy inferred gb_plane_x should be within one d_hkl of source",
        )

    def test_gb_atoms_lie_within_interface_window(self):
        parent = self.parent
        x_gb = parent.gb_plane_x
        half = parent.gb_thickness / 2.0
        gb_xs = parent.gb_atoms["x"]
        self.assertTrue(np.all(gb_xs >= x_gb - half))
        self.assertTrue(np.all(gb_xs <= x_gb + half))


class TestParentProxy(unittest.TestCase):
    def setUp(self):
        self.unit_cell = UnitCell()
        self.unit_cell.init_by_structure('fcc', 1.0, 'Cu')
        self.manipulator = GBManipulator(
            'tests/inputs/basic_dump_test1.txt', unit_cell=self.unit_cell)
        self.parents_proxy = _ParentsProxy(self.manipulator)

    def test_getitem(self):
        parent = self.parents_proxy[0]
        self.assertIsInstance(parent, Parent)

    def test_setitem(self):
        new_parent = Parent(
            _make_exact_gb(
                3.61,
                "fcc",
                "Cu",
                gb_thickness=10.0,
                repeat_factor=2,
                interaction_distance=1.0,
            )
        )
        self.parents_proxy[0] = new_parent
        self.assertIs(self.parents_proxy[0], new_parent)

    def test_len(self):
        self.assertEqual(len(self.parents_proxy), 2)

    def test_setitem_errors(self):
        self.parents_proxy[0] = None
        new_parent = Parent(
            _make_exact_gb(
                1.0,
                "fcc",
                "Cu",
                gb_thickness=10.0,
                repeat_factor=2,
                interaction_distance=1.0,
            )
        )
        with self.assertRaises(ParentsProxyValueError):
            self.parents_proxy[1] = new_parent
        self.parents_proxy[0] = new_parent
        with self.assertRaises(ParentsProxyIndexError):
            self.parents_proxy[2] = new_parent
        with self.assertRaises(ParentsProxyTypeError):
            self.parents_proxy[0] = 1


def _write_owned_test_data(path, atoms, box_dims):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("Owned candidate\n\n")
        stream.write(f"{len(atoms)} atoms\n")
        stream.write(f"{len(np.unique(atoms['name']))} atom types\n")
        stream.writelines(
            f"{lower:.12f} {upper:.12f} {axis}lo {axis}hi\n"
            for axis, (lower, upper) in zip("xyz", box_dims, strict=True)
        )
        stream.write("\nAtoms\n\n")
        stream.writelines(
            f"{atom_id} {atom['name']} {atom['x']:.12f} "
            f"{atom['y']:.12f} {atom['z']:.12f}\n"
            for atom_id, atom in enumerate(atoms, start=1)
        )


def _owned_test_manipulator(
    tmp_path,
    *,
    suffix="a",
    coordinates=None,
    box_dims=None,
    gb_plane_x=5.0,
    inplane_periodic=(True, True),
    left_grain_x_bounds=None,
    right_grain_x_bounds=None,
    normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
):
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fcc", 3.52, "Ni")

    if coordinates is None:
        coordinates = [
            ("Ni", 6.5, 1.0, 1.0),
            ("Ni", 3.0, 2.0, 2.0),
            ("Ni", 4.0, 3.0, 3.0),
            ("Ni", 4.5, 4.0, 4.0),
            ("Ni", 5.5, 5.0, 5.0),
            ("Ni", 6.0, 6.0, 6.0),
            ("Ni", 7.0, 7.0, 7.0),
            ("Ni", 8.0, 8.0, 8.0),
        ]

    atoms = np.asarray(coordinates, dtype=Atom.atom_dtype)
    box = (
        np.asarray(
            [
                [0.0, 10.0],
                [0.0, 10.0],
                [0.0, 10.0],
            ],
            dtype=float,
        )
        if box_dims is None
        else np.asarray(box_dims, dtype=float)
    )
    path = tmp_path / f"owned_{suffix}.data"
    _write_owned_test_data(path, atoms, box)

    labels = np.asarray(
        [
            LEFT_GRAIN_LABEL,
            LEFT_GRAIN_LABEL,
            LEFT_GRAIN_LABEL,
            LEFT_GRAIN_LABEL,
            RIGHT_GRAIN_LABEL,
            RIGHT_GRAIN_LABEL,
            RIGHT_GRAIN_LABEL,
            RIGHT_GRAIN_LABEL,
        ],
        dtype=np.int8,
    )

    if left_grain_x_bounds is None:
        left_grain_x_bounds = (float(box[0, 0]), gb_plane_x)
    if right_grain_x_bounds is None:
        right_grain_x_bounds = (gb_plane_x, float(box[0, 1]))

    ownership = GrainOwnership(
        atom_ids=np.arange(1, len(atoms) + 1),
        labels=labels,
        gb_plane_x=gb_plane_x,
        inplane_periodic=inplane_periodic,
        left_grain_x_bounds=left_grain_x_bounds,
        right_grain_x_bounds=right_grain_x_bounds,
        coordinate_tolerance=1.0e-8,
        normal_topology=normal_topology,
    )

    manipulator = GBManipulator(
        str(path),
        unit_cell=unit_cell,
        gb_thickness=10.0,
        grain_ownership=ownership,
        seed=2,
    )
    return manipulator, atoms, labels, box


def test_explicit_ownership_file_parent_does_not_reclassify_crossing_atom(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        manipulator, _atoms, labels, _box = _owned_test_manipulator(tmp_path)

    assert not any(issubclass(item.category, DeprecationWarning) for item in caught)
    parent = manipulator.parents[0]

    assert np.array_equal(parent.grain_labels, labels)
    assert parent.left_grain[0]["x"] == pytest.approx(6.5)
    assert parent.gb_plane_x == pytest.approx(5.0)
    assert len(parent.left_grain) == 4
    assert len(parent.right_grain) == 4


def test_explicit_ownership_copy_and_translation_preserve_labels(tmp_path):
    manipulator, _atoms, labels, _box = _owned_test_manipulator(tmp_path)

    translated = manipulator.translate_right_grain(0.25, 0.5)
    expected = np.hstack(
        (
            labels[labels == LEFT_GRAIN_LABEL],
            labels[labels == RIGHT_GRAIN_LABEL],
        )
    )

    assert np.array_equal(manipulator.candidate_grain_labels, expected)
    assert np.any(translated["x"][expected == LEFT_GRAIN_LABEL] > 5.0)

    original_x = float(manipulator.parents[0].whole_system[0]["x"])

    for copied in (copy.copy(manipulator), copy.deepcopy(manipulator)):
        copied_labels = copied.candidate_grain_labels

        with pytest.raises(ValueError):
            copied_labels[0] = RIGHT_GRAIN_LABEL

        assert np.array_equal(copied.candidate_grain_labels, expected)

        copied.parents[0].whole_system[0]["x"] = original_x + 1.0

        assert manipulator.parents[0].whole_system[0]["x"] == pytest.approx(original_x)


def test_owned_slice_and_merge_normalizes_affine_equivalent_variable_cells(tmp_path):
    parent1_manipulator, atoms, labels, box1 = _owned_test_manipulator(
        tmp_path,
        suffix="variable_a",
        gb_plane_x=4.0,
    )
    box2 = np.asarray(
        [[-2.0, 18.0], [-1.0, 11.0], [5.0, 20.0]],
        dtype=float,
    )
    atoms2 = np.array(atoms, copy=True)
    for axis_name, axis_index in zip("xyz", range(3), strict=True):
        source_lo, source_hi = box1[axis_index]
        target_lo, target_hi = box2[axis_index]
        reduced = (atoms2[axis_name] - source_lo) / (source_hi - source_lo)
        atoms2[axis_name] = target_lo + reduced * (target_hi - target_lo)

    parent2_manipulator, _atoms2, labels2, _ = _owned_test_manipulator(
        tmp_path,
        suffix="variable_b",
        coordinates=atoms2,
        box_dims=box2,
        gb_plane_x=6.0,
        left_grain_x_bounds=(-2.0, 6.0),
        right_grain_x_bounds=(6.0, 18.0),
    )

    class FixedRandom:
        @staticmethod
        def random():
            return 0.5

    crossover = GBManipulator._from_parents(
        parent1_manipulator.parents[0],
        parent2_manipulator.parents[0],
        rng=FixedRandom(),
    )
    child = crossover.slice_and_merge()

    # The second parent is mapped into the first parent's current relaxed box before
    # slicing. With random() == 0.5, the slice is exactly the actual GB plane x=4.0,
    # not the box midpoint x=5.0.
    mask1 = atoms["x"] < 4.0
    mask2 = atoms["x"] >= 4.0
    expected = np.hstack((atoms[mask1], atoms[mask2]))
    expected_labels = np.hstack((labels[mask1], labels2[mask2]))

    np.testing.assert_array_equal(child["name"], expected["name"])
    for axis_name in "xyz":
        np.testing.assert_allclose(child[axis_name], expected[axis_name])
    np.testing.assert_array_equal(
        crossover.candidate_grain_labels,
        expected_labels,
    )


def test_explicit_ownership_removal_deletes_same_label_row(tmp_path):
    manipulator, atoms, labels, _box = _owned_test_manipulator(tmp_path)

    class FixedChoice:
        def __init__(self):
            self.selected = None

        def choice(self, choices, size=None, replace=False, p=None):
            values = np.asarray(choices)
            self.selected = int(values[-1])

            if size is None:
                return self.selected

            return np.asarray([self.selected])

    fixed_choice = FixedChoice()
    manipulator.rng = fixed_choice

    reduced, removed = manipulator.remove_atoms(
        num_to_remove=1,
        keep_ratio=False,
        return_positions=True,
    )

    assert fixed_choice.selected is not None
    assert len(reduced) == len(atoms) - 1
    assert len(removed) == 1
    assert np.array_equal(
        manipulator.candidate_grain_labels,
        np.delete(labels, fixed_choice.selected),
    )


def test_explicit_ownership_insertion_assigns_new_label_once(monkeypatch, tmp_path):
    manipulator, atoms, labels, _box = _owned_test_manipulator(tmp_path)

    possible_sites = np.array([[3.0, 1.0, 1.0]], dtype=float)
    probabilities = np.array([1.0])

    module = importlib.import_module("GBOpt.GBManipulator")
    monkeypatch.setattr(
        module,
        "_grid_insertion_sites",
        lambda *_args, **_kwargs: (possible_sites, probabilities),
    )

    inserted, new_atoms = manipulator.insert_atoms(
        num_to_insert=1,
        method="grid",
        keep_ratio=False,
        return_positions=True,
    )

    assert len(inserted) == len(atoms) + 1
    assert len(new_atoms) == 1
    assert new_atoms[0]["x"] == pytest.approx(3.0)

    assert np.array_equal(manipulator.candidate_grain_labels[:-1], labels)
    assert manipulator.candidate_grain_labels[-1] == LEFT_GRAIN_LABEL

    # Ownership is assigned when the atom is inserted. Moving the returned candidate
    # afterward must not cause geometric reclassification.
    inserted[-1]["x"] = 9.0

    assert manipulator.candidate_grain_labels[-1] == LEFT_GRAIN_LABEL


def test_explicit_ownership_slice_and_merge_uses_atom_masks_for_labels(
    tmp_path,
):
    manip1, atoms1, labels1, _box = _owned_test_manipulator(
        tmp_path,
        suffix="one",
    )

    coordinates2 = [
        (
            str(row["name"]),
            float(row["x"]) + 0.25,
            float(row["y"]),
            float(row["z"]),
        )
        for row in atoms1
    ]

    manip2, atoms2, labels2, _ = _owned_test_manipulator(
        tmp_path,
        suffix="two",
        coordinates=coordinates2,
    )

    class MidpointRng:
        def random(self):
            return 0.5

    child_manipulator = GBManipulator._from_parents(
        manip1.parents[0],
        manip2.parents[0],
        rng=MidpointRng(),
    )

    child = child_manipulator.slice_and_merge()

    mask1 = atoms1["x"] < 5.0
    mask2 = atoms2["x"] >= 5.0
    expected_labels = np.hstack((labels1[mask1], labels2[mask2]))

    assert np.array_equal(
        child_manipulator.candidate_grain_labels,
        expected_labels,
    )
    assert len(child) == len(expected_labels)


def test_periodic_wave_crossover_preserves_uo2_formula_and_tilt(tmp_path):
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fluorite", 5.454, ("U", "O"))
    box = np.asarray([[0.0, 10.0], [0.0, 12.0], [0.0, 8.0]])
    labels = np.asarray(
        [LEFT_GRAIN_LABEL] * 3 + [RIGHT_GRAIN_LABEL] * 3,
        dtype=np.int8,
    )
    first_atoms = np.asarray(
        [
            ("U", 3.0, 1.0, 1.0),
            ("O", 3.5, 3.0, 2.0),
            ("O", 4.0, 5.0, 3.0),
            ("U", 5.5, 7.0, 4.0),
            ("O", 6.0, 9.0, 5.0),
            ("O", 6.5, 11.0, 6.0),
        ],
        dtype=Atom.atom_dtype,
    )
    second_atoms = np.asarray(
        [
            ("U", 3.0, 1.5, 1.5),
            ("U", 3.5, 3.5, 2.5),
            ("O", 4.0, 5.5, 3.5),
            ("O", 5.5, 7.5, 4.5),
            ("O", 6.0, 9.5, 5.5),
            ("O", 6.5, 11.5, 6.5),
        ],
        dtype=Atom.atom_dtype,
    )

    def owned_parent(atoms, suffix):
        path = tmp_path / f"uo2_{suffix}.data"
        _write_owned_test_data(path, atoms, box)
        ownership = GrainOwnership(
            atom_ids=np.arange(1, len(atoms) + 1),
            labels=labels,
            gb_plane_x=5.0,
            inplane_periodic=(True, True),
            left_grain_x_bounds=(0.0, 5.0),
            right_grain_x_bounds=(5.0, 10.0),
            coordinate_tolerance=1.0e-8,
            normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        )
        return GBManipulator(
            str(path),
            unit_cell=unit_cell,
            gb_thickness=10.0,
            grain_ownership=ownership,
        ).parents[0]

    crossover = GBManipulator._from_parents(
        owned_parent(first_atoms, "first"),
        owned_parent(second_atoms, "second"),
        rng=np.random.default_rng(29),
    )
    child = crossover.slice_and_merge(
        surface_mode="periodic_wave",
        max_tilt_degrees=5.0,
    )

    names = np.asarray(child["name"]).astype(str)
    assert np.count_nonzero(names == "O") == 2 * np.count_nonzero(names == "U")
    assert len(crossover.candidate_grain_labels) == len(child)

    provenance = dict(crossover.last_crossover_provenance)
    assert provenance["surface_mode"] == "periodic_wave"
    slope_y = 2.0 * np.pi * provenance["amplitude_y"] / 12.0
    slope_z = 2.0 * np.pi * provenance["amplitude_z"] / 8.0
    assert np.hypot(slope_y, slope_z) <= np.tan(np.deg2rad(5.0))

    # Each sinusoidal component has an integral period across its box direction, so
    # the crossover surface meets itself at both periodic face pairs.
    phase_y = provenance["phase_y"]
    phase_z = provenance["phase_z"]
    assert np.sin(phase_y) == pytest.approx(np.sin(2.0 * np.pi + phase_y))
    assert np.sin(phase_z) == pytest.approx(np.sin(2.0 * np.pi + phase_z))


@pytest.mark.parametrize(
    ("surface_mode", "max_tilt", "match"),
    [
        pytest.param("tilted_plane", 5.0, "surface_mode", id="mode"),
        pytest.param("periodic_wave", -1.0, "max_tilt", id="negative-tilt"),
        pytest.param("periodic_wave", 90.0, "max_tilt", id="right-angle-tilt"),
    ],
)
def test_slice_and_merge_rejects_invalid_surface_policy(
    tmp_path,
    surface_mode,
    max_tilt,
    match,
):
    first, _atoms, _labels, _box = _owned_test_manipulator(tmp_path, suffix="p1")
    second, _atoms, _labels, _box = _owned_test_manipulator(tmp_path, suffix="p2")
    crossover = GBManipulator._from_parents(
        first.parents[0],
        second.parents[0],
        rng=np.random.default_rng(3),
    )

    with pytest.raises(GBManipulatorValueError, match=match):
        crossover.slice_and_merge(
            surface_mode=surface_mode,
            max_tilt_degrees=max_tilt,
        )


@pytest.mark.parametrize(
    ("second_parent_kwargs", "match"),
    [
        pytest.param(
            {
                "gb_plane_x": 5.25,
            },
            "affine-equivalent physical grain geometry",
            id="gb-plane",
        ),
        pytest.param(
            {
                "inplane_periodic": (True, False),
            },
            "matching boundary topology",
            id="inplane-periodicity",
        ),
    ],
)
def test_explicit_ownership_slice_and_merge_rejects_incompatible_parents(
    tmp_path,
    second_parent_kwargs,
    match,
):
    manip1, _atoms1, _labels1, _box = _owned_test_manipulator(
        tmp_path,
        suffix="compatible",
    )

    manip2, _atoms2, _labels2, _ = _owned_test_manipulator(
        tmp_path,
        suffix="incompatible",
        **second_parent_kwargs,
    )

    child_manipulator = GBManipulator._from_parents(
        manip1.parents[0],
        manip2.parents[0],
    )

    with pytest.raises(GBManipulatorValueError, match=match):
        child_manipulator.slice_and_merge()


def test_grain_ownership_rejected_for_two_parent_constructor(tmp_path):
    manipulator, _atoms, _labels, _box = _owned_test_manipulator(tmp_path)

    ownership = manipulator.parents[0].grain_ownership

    with pytest.raises(GBManipulatorValueError, match="single file-backed parent"):
        GBManipulator(
            str(tmp_path / "owned_a.data"),
            str(tmp_path / "owned_a.data"),
            unit_cell=manipulator.parents[0].unit_cell,
            grain_ownership=ownership,
        )


def test_explicit_ownership_inplane_translation_allows_crossed_right_atom(
    tmp_path,
):
    coordinates = [
        ("Ni", 3.0, 1.0, 1.0),
        ("Ni", 3.5, 2.0, 2.0),
        ("Ni", 4.0, 3.0, 3.0),
        ("Ni", 4.5, 4.0, 4.0),
        ("Ni", 4.75, 5.0, 5.0),
        ("Ni", 6.0, 6.0, 6.0),
        ("Ni", 7.0, 7.0, 7.0),
        ("Ni", 8.0, 8.0, 8.0),
    ]

    manipulator, _atoms, labels, _box = _owned_test_manipulator(
        tmp_path,
        suffix="right_crossing",
        coordinates=coordinates,
    )

    translated = manipulator.translate_right_grain(0.25, 0.5)
    candidate_labels = manipulator.candidate_grain_labels

    assert np.array_equal(
        candidate_labels,
        np.hstack(
            (
                labels[labels == LEFT_GRAIN_LABEL],
                labels[labels == RIGHT_GRAIN_LABEL],
            )
        ),
    )

    assert translated["x"][candidate_labels == RIGHT_GRAIN_LABEL][0] == pytest.approx(
        4.75
    )


def test_explicit_ownership_x_translation_rejects_crossed_right_atom(
    tmp_path,
):
    coordinates = [
        ("Ni", 3.0, 1.0, 1.0),
        ("Ni", 3.5, 2.0, 2.0),
        ("Ni", 4.0, 3.0, 3.0),
        ("Ni", 4.5, 4.0, 4.0),
        ("Ni", 4.75, 5.0, 5.0),
        ("Ni", 6.0, 6.0, 6.0),
        ("Ni", 7.0, 7.0, 7.0),
        ("Ni", 8.0, 8.0, 8.0),
    ]

    manipulator, _atoms, _labels, _box = _owned_test_manipulator(
        tmp_path,
        suffix="right_crossing_dx",
        coordinates=coordinates,
    )

    with pytest.raises(
        GBManipulatorValueError,
        match="outside the supported half-open x interval",
    ):
        manipulator.translate_right_grain(0.25, 0.5, dx=0.1)


def test_grain_ownership_requires_ownership_value_object(tmp_path):
    manipulator, _atoms, _labels, _box = _owned_test_manipulator(tmp_path)

    path = str(tmp_path / "owned_a.data")

    with pytest.raises(GBManipulatorTypeError, match="GrainOwnership"):
        GBManipulator(
            path,
            unit_cell=manipulator.parents[0].unit_cell,
            grain_ownership=object(),
        )


def test_explicit_ownership_parent_proxy_replacement_resets_candidate_labels(tmp_path):
    manipulator, _atoms, labels, _box = _owned_test_manipulator(
        tmp_path,
        suffix="proxy_source",
    )

    replacement, _atoms2, replacement_labels, _ = _owned_test_manipulator(
        tmp_path,
        suffix="proxy_replacement",
    )

    class FixedChoice:
        def choice(self, choices, size=None, replace=False, p=None):
            values = np.asarray(choices)
            selected = values[-1]

            if size is None:
                return selected

            return np.asarray([selected])

    manipulator.rng = FixedChoice()
    manipulator.remove_atoms(
        num_to_remove=1,
        keep_ratio=False,
    )

    assert len(manipulator.candidate_grain_labels) == len(labels) - 1

    manipulator.parents[0] = replacement.parents[0]
    assert np.array_equal(
        manipulator.candidate_grain_labels,
        replacement_labels,
    )


def test_periodic_image_neighborhood_retains_translated_self_images():
    atoms = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
    box_dims = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=float)
    neighbors = (
        _create_periodic_image_neighborhoods(
            1.01,
            atoms,
            np.array([0], dtype=np.intp),
            box_dims,
            (True, True, True),
        )[0]
    )

    assert len(neighbors) == 6

    distances = np.linalg.norm(neighbors[:, 1:] - atoms[0, 1:], axis=1)
    np.testing.assert_allclose(distances, np.ones(6))


def test_periodic_physical_neighbors_deduplicate_images_at_minimum_distance():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]], dtype=float)
    box_dims = np.array([[0.0, 10.0], [0.0, 1.0], [0.0, 10.0]], dtype=float)
    (neighbor_indices, neighbor_distances) = _create_periodic_neighbor_list(
        1.6,
        positions,
        np.array([0], dtype=np.intp),
        box_dims,
        (False, True, False),
    )

    np.testing.assert_array_equal(neighbor_indices[0], np.array([1], dtype=np.intp))
    np.testing.assert_allclose(neighbor_distances[0], np.array([0.4]))


def test_periodic_physical_neighbors_do_not_wrap_nonperiodic_x():
    positions = np.array([[0.1, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=float)
    box_dims = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=float)
    (neighbor_indices, neighbor_distances) = _create_periodic_neighbor_list(
        0.3,
        positions,
        np.array([0], dtype=np.intp),
        box_dims,
        (False, False, False),
    )

    assert neighbor_indices[0].size == 0
    assert neighbor_distances[0].size == 0


def test_remove_atoms_local_order_uses_atoms_beyond_partner_cutoff(monkeypatch):
    unit_cell = UnitCell()
    unit_cell.init_by_structure("rocksalt", 4.0, ("Na", "Cl"))

    atoms = np.array(
        [
            ("Na", 0.0, 0.0, 0.0),
            ("Cl", 0.0, 1.0, 0.0),
            ("Cl", 0.0, 5.0, 0.0),
            ("Na", 0.0, 8.0, 0.0),
            ("Na", 19.0, 19.0, 19.0),
            ("Cl", 18.0, 19.0, 19.0),
            ("Na", -19.0, -19.0, -19.0),
            ("Cl", -18.0, -19.0, -19.0),
        ],
        dtype=Atom.atom_dtype,
    )

    manipulator = _synthetic_manipulator(unit_cell, atoms)

    recorded = {}

    def record_local_order(atom, neighs, *_args):
        if np.allclose(atom[1:], np.array([0.0, 0.0, 0.0])):
            recorded["neighbors"] = np.array(neighs, copy=True)
        return 1.0

    class FirstChoiceRng:
        @staticmethod
        def choice(choices, size, replace=False, p=None):
            del replace, p
            return np.asarray(choices)[:size]

    module = importlib.import_module("GBOpt.GBManipulator")
    monkeypatch.setattr(module, "_calculate_local_order", record_local_order)
    manipulator.rng = FirstChoiceRng()
    reduced = manipulator.remove_atoms(num_to_remove=2, keep_ratio=True)
    local_neighbors = recorded["neighbors"]

    # This Cl is well beyond the short stoichiometric-partner cutoff, but inside Rmax
    # and therefore must participate in the local-order fingerprint.
    assert np.any(
        np.all(
            np.isclose(local_neighbors[:, 1:], np.array([0.0, 5.0, 0.0])),
            axis=1,
        )
    )

    # It must not, however, be selected as the short-range Cl partner.
    assert np.any((reduced["name"] == "Cl") & np.isclose(reduced["y"], 5.0))


def test_remove_atoms_skips_central_atom_without_stoichiometric_partner(
    monkeypatch,
):
    unit_cell = UnitCell()
    unit_cell.init_by_structure("rocksalt", 4.0, ("Na", "Cl"))

    atoms = np.array(
        [
            ("Na", 0.0, 0.0, 0.0),
            ("Cl", 0.0, 0.5, 0.0),
            ("Na", 0.0, 1.0, 0.0),
            ("Cl", 0.0, 1.5, 0.0),
            ("Na", 0.0, 2.0, 0.0),
            ("Cl", 0.0, 2.5, 0.0),
            ("Na", 0.0, 3.0, 0.0),
            ("Cl", 0.0, 3.5, 0.0),
        ],
        dtype=Atom.atom_dtype,
    )
    manipulator = _synthetic_manipulator(unit_cell, atoms)

    def removal_neighbors(_cutoff, _positions, center_indices, _box, periodic):
        np.testing.assert_array_equal(
            center_indices,
            np.array([0, 2, 4, 6], dtype=np.intp),
        )
        assert periodic == (False, False, False)
        return (
            [
                np.empty(0, dtype=np.intp),
                np.array([3], dtype=np.intp),
                np.array([5], dtype=np.intp),
                np.array([7], dtype=np.intp),
            ],
            [
                np.empty(0, dtype=float),
                np.array([0.5]),
                np.array([0.5]),
                np.array([0.5]),
            ],
        )

    class FirstChoiceRng:
        @staticmethod
        def choice(choices, size, replace=False, p=None):
            del replace, p
            return np.asarray(choices)[:size]

    module = importlib.import_module("GBOpt.GBManipulator")
    monkeypatch.setattr(module, "_calculate_local_order", lambda *_args: 1.0)
    monkeypatch.setattr(
        module,
        "_create_periodic_neighbor_list",
        removal_neighbors,
    )
    manipulator.rng = FirstChoiceRng()

    _, removed = manipulator.remove_atoms(
        num_to_remove=2,
        keep_ratio=True,
        return_positions=True,
    )

    assert len(removed) == 2
    assert set(removed["name"]) == {"Na", "Cl"}
    assert np.any((removed["name"] == "Na") & np.isclose(removed["y"], 1.0))
    assert np.any((removed["name"] == "Cl") & np.isclose(removed["y"], 1.5))


def test_remove_atoms_partner_selection_uses_periodic_minimum_image(monkeypatch):
    unit_cell = UnitCell()
    unit_cell.init_by_structure("rocksalt", 4.0, ("Na", "Cl"))

    atoms = np.array(
        [
            ("Na", 0.0, 0.1, 0.0),
            ("Cl", 0.0, 9.9, 0.0),
            ("Na", 0.0, 5.0, 0.0),
            ("Cl", 0.0, 5.5, 0.0),
            ("Na", 0.9, 3.0, 0.9),
            ("Cl", 0.9, 3.5, 0.9),
            ("Na", 0.9, 6.0, 0.9),
            ("Cl", 0.9, 6.5, 0.9),
        ],
        dtype=Atom.atom_dtype,
    )

    box_dims = np.array([[-1.0, 1.0], [0.0, 10.0], [-1.0, 1.0]], dtype=float)

    manipulator = (
        _synthetic_manipulator(
            unit_cell,
            atoms,
            box_dims=box_dims,
            inplane_periodic=(True, False),
        )
    )

    class FirstChoiceRng:
        @staticmethod
        def choice(choices, size, replace=False, p=None):
            del replace, p
            return np.asarray(choices)[:size]

    module = importlib.import_module("GBOpt.GBManipulator")
    monkeypatch.setattr(module, "_calculate_local_order", lambda *_args: 1.0)
    manipulator.rng = FirstChoiceRng()

    _, removed = (
        manipulator.remove_atoms(
            num_to_remove=2,
            keep_ratio=True,
            return_positions=True,
        )
    )

    assert set(removed["name"]) == {"Na", "Cl"}

    # The selected Na is at y=0.1. Its Cl partner at y=9.9 is only 0.2 Angstrom away
    # through the periodic y boundary.
    assert np.any((removed["name"] == "Cl") & np.isclose(removed["y"], 9.9))


def test_tetrahedron_circumcenter_is_equidistant_from_vertices():
    tetrahedra = np.array(
        [
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ]
    )

    centers = _tetrahedron_circumcenters(tetrahedra)

    np.testing.assert_allclose(centers, [[1.0, 1.0, 1.0]])

    distances = np.linalg.norm(tetrahedra[0] - centers[0], axis=1)
    np.testing.assert_allclose(distances, np.full(4, np.sqrt(3.0)))


def test_periodic_tiling_uses_box_width_not_atom_extent():
    positions = np.array([[0.0, 2.0, 0.0], [0.0, 8.0, 0.0]])
    box_dims = np.array([[-1.0, 1.0], [0.0, 10.0], [-1.0, 1.0]])

    tiled = _tile_periodic_positions_once(positions, box_dims, (False, True, False))

    np.testing.assert_allclose(
        np.unique(tiled[:, 1]),
        [-8.0, -2.0, 2.0, 8.0, 12.0, 18.0],
    )


def test_minimum_periodic_distance_crosses_box_boundary():
    queries = np.array([[0.0, 0.1, 0.0]])
    references = np.array([[0.0, 9.9, 0.0]])
    box_dims = np.array([[-1.0, 1.0], [0.0, 10.0], [-1.0, 1.0]])

    distances = _minimum_periodic_distances(
        queries,
        references,
        box_dims,
        (False, True, False),
        1e-8,
    )

    np.testing.assert_allclose(distances, [0.2])


def test_delaunay_insertion_site_is_tetrahedron_circumcenter():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ]
    )
    box_dims = np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])

    sites, probabilities = _delaunay_insertion_sites(
        positions,
        positions,
        atom_radius=1.0,
        box_dims=box_dims,
        periodic=(False, False, False),
        coordinate_tolerance=1e-8,
    )

    assert len(sites) == 1
    np.testing.assert_allclose(sites[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(probabilities, [1.0])


def test_grid_insertion_rejects_site_too_close_across_periodic_boundary():
    gb_positions = np.array([[0.0, 9.9, 0.0], [1.0, 5.0, 1.0]])
    box_dims = np.array([[0.0, 1.0], [0.0, 10.0], [0.0, 1.0]])

    sites, _ = _grid_insertion_sites(
        gb_positions,
        gb_positions,
        atom_radius=0.25,
        box_dims=box_dims,
        periodic=(False, True, False),
        coordinate_tolerance=1e-8,
    )

    assert not np.any(np.all(np.isclose(sites, [0.0, 0.0, 0.0]), axis=1))


def test_stoichiometric_change_uses_total_atom_count():
    assert _get_stoichiometric_change(6, {1: 1, 2: 2}) == {1: 2, 2: 4}


def test_stoichiometric_change_rejects_incompatible_atom_count():
    with pytest.raises(
        GBManipulatorValueError,
        match="must be a multiple of 3",
    ):
        _get_stoichiometric_change(5, {1: 1, 2: 2})


def test_floor_stoichiometric_count_rounds_down():
    assert _floor_stoichiometric_count(8, {1: 1, 2: 2}) == 6
    assert _floor_stoichiometric_count(2, {1: 1, 2: 2}) == 0


def test_insert_atoms_with_stoichiometry_inserts_requested_fluorite_atom_count(
    monkeypatch
):
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fluorite", 5.454, ("U", "O"))

    atoms = np.array(
        [
            ("U", 0.0, 0.0, 0.0),
            ("U", 0.0, 1.0, 0.0),
            ("U", 0.0, 2.0, 0.0),
            ("U", 0.0, 3.0, 0.0),
            ("O", 0.0, 0.0, 1.0),
            ("O", 0.0, 1.0, 1.0),
            ("O", 0.0, 2.0, 1.0),
            ("O", 0.0, 3.0, 1.0),
            ("O", 0.0, 0.0, 2.0),
            ("O", 0.0, 1.0, 2.0),
            ("O", 0.0, 2.0, 2.0),
            ("O", 0.0, 3.0, 2.0),
        ],
        dtype=Atom.atom_dtype,
    )

    manipulator = _synthetic_manipulator(unit_cell, atoms)

    possible_sites = np.array(
        [
            [0.0, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5],
        ]
    )
    probabilities = np.full(3, 1.0 / 3.0)

    module = importlib.import_module("GBOpt.GBManipulator")

    monkeypatch.setattr(
        module,
        "_grid_insertion_sites",
        lambda *_args, **_kwargs: (possible_sites, probabilities),
    )

    monkeypatch.setattr(
        module,
        "_create_periodic_neighbor_list",
        lambda *_args, **_kwargs: (
            [np.array([1, 2], dtype=np.intp)],
            [np.array([0.5, 1.0])],
        ),
    )

    class FirstChoiceRng:
        @staticmethod
        def choice(choices, size, replace=False, p=None):
            del replace, p
            return np.asarray(choices)[:size]

    manipulator.rng = FirstChoiceRng()

    _, inserted = manipulator.insert_atoms(
        num_to_insert=3,
        method="grid",
        keep_ratio=True,
        return_positions=True,
    )

    assert len(inserted) == 3
    assert np.count_nonzero(inserted["name"] == "U") == 1
    assert np.count_nonzero(inserted["name"] == "O") == 2


def test_insert_atoms_fill_fraction_uses_valid_insertion_site_count(monkeypatch):
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fluorite", 5.454, ("U", "O"))

    atoms = np.array(
        [
            ("U", 0.0, 0.0, 0.0),
            ("U", 0.0, 1.0, 0.0),
            ("U", 0.0, 2.0, 0.0),
            ("U", 0.0, 3.0, 0.0),
            ("U", 0.0, 4.0, 0.0),
            ("U", 0.0, 5.0, 0.0),
            ("U", 0.0, 6.0, 0.0),
            ("U", 0.0, 7.0, 0.0),
            ("O", 0.0, 0.0, 1.0),
            ("O", 0.0, 1.0, 1.0),
            ("O", 0.0, 2.0, 1.0),
            ("O", 0.0, 3.0, 1.0),
            ("O", 0.0, 4.0, 1.0),
            ("O", 0.0, 5.0, 1.0),
            ("O", 0.0, 6.0, 1.0),
            ("O", 0.0, 7.0, 1.0),
            ("O", 0.0, 0.0, 2.0),
            ("O", 0.0, 1.0, 2.0),
            ("O", 0.0, 2.0, 2.0),
            ("O", 0.0, 3.0, 2.0),
            ("O", 0.0, 4.0, 2.0),
            ("O", 0.0, 5.0, 2.0),
            ("O", 0.0, 6.0, 2.0),
            ("O", 0.0, 7.0, 2.0),
        ],
        dtype=Atom.atom_dtype,
    )

    manipulator = _synthetic_manipulator(unit_cell, atoms)

    possible_sites = np.column_stack(
        (
            np.zeros(24),
            np.arange(24, dtype=float),
            np.zeros(24),
        )
    )
    probabilities = np.full(24, 1.0 / 24.0)

    module = importlib.import_module("GBOpt.GBManipulator")

    monkeypatch.setattr(
        module,
        "_grid_insertion_sites",
        lambda *_args, **_kwargs: (possible_sites, probabilities),
    )

    # Three U centers, each with sufficient O partner sites.
    monkeypatch.setattr(
        module,
        "_create_periodic_neighbor_list",
        lambda *_args, **_kwargs: (
            [
                np.array([3, 4], dtype=np.intp),
                np.array([5, 6], dtype=np.intp),
            ],
            [
                np.array([1.0, 1.0]),
                np.array([1.0, 1.0]),
            ],
        ),
    )

    class FirstChoiceRng:
        @staticmethod
        def choice(choices, size, replace=False, p=None):
            del replace, p
            return np.asarray(choices)[:size]

    manipulator.rng = FirstChoiceRng()

    _, inserted = manipulator.insert_atoms(
        fill_fraction=0.25,
        method="grid",
        keep_ratio=True,
        return_positions=True,
    )

    assert len(inserted) == 6
    assert np.count_nonzero(inserted["name"] == "U") == 2
    assert np.count_nonzero(inserted["name"] == "O") == 4


def test_insert_atoms_fill_fraction_is_capped_at_gb_limit(monkeypatch):
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fcc", 3.52, "Ni")

    atoms = np.array(
        [
            ("Ni", 0.0, 0.0, 0.0),
            ("Ni", 1.0, 0.0, 0.0),
            ("Ni", 2.0, 0.0, 0.0),
            ("Ni", 3.0, 0.0, 0.0),
            ("Ni", 4.0, 0.0, 0.0),
            ("Ni", 5.0, 0.0, 0.0),
            ("Ni", 6.0, 0.0, 0.0),
            ("Ni", 7.0, 0.0, 0.0),
        ],
        dtype=Atom.atom_dtype,
    )

    manipulator = _synthetic_manipulator(unit_cell, atoms, seed=17)

    possible_sites = np.array(
        [
            [0.5, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [2.5, 1.0, 1.0],
            [3.5, 1.0, 1.0],
            [4.5, 1.0, 1.0],
            [5.5, 1.0, 1.0],
            [6.5, 1.0, 1.0],
            [7.5, 1.0, 1.0],
            [0.5, 2.0, 1.0],
            [1.5, 2.0, 1.0],
            [2.5, 2.0, 1.0],
            [3.5, 2.0, 1.0],
        ],
        dtype=float,
    )
    probabilities = np.full(
        len(possible_sites),
        1.0 / len(possible_sites),
    )

    module = importlib.import_module("GBOpt.GBManipulator")
    monkeypatch.setattr(
        module,
        "_delaunay_insertion_sites",
        lambda *_args, **_kwargs: (possible_sites, probabilities),
    )

    with pytest.warns(
        UserWarning,
        match="25% GB-region limit permits at most 2 atoms",
    ):
        new_system, inserted_atoms = manipulator.insert_atoms(
            fill_fraction=0.25,
            method="delaunay",
            return_positions=True,
        )

    assert len(possible_sites) == 12
    assert int(0.25 * len(possible_sites)) == 3
    assert len(inserted_atoms) == 2
    assert len(new_system) == len(atoms) + 2


if __name__ == '__main__':
    unittest.main()
