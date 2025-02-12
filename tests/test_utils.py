import math
import tempfile
import unittest

import numpy as np

from GBOpt.Atom import Atom
from GBOpt.GBMaker import GBMaker
from GBOpt.Utils import (
    _flatten,
    _validate_array,
    approximate_rotation_matrix_as_int,
    get_points_inside_box,
    write_lammps,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Common test setup
        self.a0 = 3.61
        self.structure = 'fcc'
        self.atom_types = 'Cu'
        self.gb_thickness = 10.0
        theta = math.radians(36.869898)
        self.misorientation = np.array(
            [theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.repeat_factor = 6
        self.x_dim = 60.0
        self.vacuum = 10.0
        self.gb_id = 0
        self.interaction_distance = 10
        self.gbm = GBMaker(self.a0, self.structure, self.gb_thickness,
                           self.misorientation, self.atom_types,
                           repeat_factor=self.repeat_factor, x_dim=self.x_dim,
                           vacuum=self.vacuum,
                           interaction_distance=self.interaction_distance,
                           gb_id=self.gb_id)

    def test_write_lammps_with_gbmaker(self):
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            write_lammps(temp_file.name, self.gbm)
            with open(temp_file.name, 'r') as f:
                content = f.readlines()
            self.assertGreater(len(content), 0)
            self.assertIn("atoms", content[2].lower())
            self.assertIn("atom types", content[3].lower())

    def test_write_lammps_with_atoms_and_box_size(self):
        atoms = self.gbm.whole_system
        box_sizes = self.gbm.box_dims
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            write_lammps(temp_file.name, atoms=atoms, box_sizes=box_sizes)
            with open(temp_file.name, 'r') as f:
                content = f.readlines()
            self.assertGreater(len(content), 0)
            self.assertIn("atoms", content[2].lower())
            self.assertIn("atom types", content[3].lower())

    def test_lammps_file_formatting(self):
        atoms = np.array([('Cu', 0.0, 0.0, 0.0), ('H', 1.0, 1.0, 1.0)],
                         dtype=Atom.atom_dtype)
        box_sizes = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            write_lammps(temp_file.name, atoms=atoms, box_sizes=box_sizes)
            with open(temp_file.name, 'r') as f:
                content = f.readlines()
            self.assertEqual(content[2].strip(), '2 atoms')
            self.assertEqual(content[3].strip(), '2 atom types')

        with tempfile.NamedTemporaryFile(delete=True) as temp_file:
            write_lammps(temp_file.name, atoms=atoms,
                         box_sizes=box_sizes, type_as_int=False)
            with open(temp_file.name, 'r') as f:
                content = f.readlines()

            self.assertEqual(content[8].strip(), 'Atom Type Labels')
            self.assertEqual(content[10].strip(), '1 Cu')
            self.assertEqual(content[11].strip(), '2 H')

    def test_approximate_rotation_matrix_as_int(self):
        rotation_matrix = np.array([[0.70710678, 0.5, 0.5],
                                    [0.70710678, -0.5, -0.5],
                                    [0.0, 0.70710678, -0.70710678]])

        approx_matrix = approximate_rotation_matrix_as_int(rotation_matrix)

        expected_matrix = np.array([[141421, 100000, 100000],
                                    [141421, -100000, -100000],
                                    [0, 1, -1]])

        np.testing.assert_array_equal(approx_matrix, expected_matrix)

        approx_matrix = approximate_rotation_matrix_as_int(rotation_matrix, precision=1)
        expected_matrix = np.array([[7, 5, 5], [7, -5, -5], [0, 1, -1]])
        np.testing.assert_array_equal(approx_matrix, expected_matrix)

        approx_matrix = approximate_rotation_matrix_as_int(
            rotation_matrix, precision=10)
        expected_matrix = np.array(
            [
                [35355339, 25000000, 25000000],
                [35355339, -25000000, -25000000],
                [0, 1, -1]
            ]
        )

        np.testing.assert_array_equal(approx_matrix, expected_matrix)

        with self.assertWarns(UserWarning):
            _ = approximate_rotation_matrix_as_int(rotation_matrix, precision=0)

    def test_flatten(self):
        array = [1, 2, 3, 4, 5]
        result = _flatten(array)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertTrue(np.allclose(result, np.array(array)))
        self.assertTrue(result.shape == (5,))

        array2 = np.array([[1, 2], [3, 4], [5, 6]])
        result = _flatten(array2)
        self.assertTrue(result.shape == (6,))

    def test_validate_array(self):
        array1 = [1, 2, 3, 4, 5]
        array2 = np.array([[1, 2], [3, 4], [5, 6]])

        with self.assertRaises(TypeError):
            _validate_array(array1)

        with self.assertRaises(ValueError):
            _validate_array(array2, expected_shape=(6,))

        with self.assertRaises(ValueError):
            _validate_array(array2, expected_dtype=Atom.atom_dtype)

        with self.assertRaises(ValueError):
            _validate_array(array2, expected_shape=(3, 2),
                            expected_dtype=Atom.atom_dtype)

    def test_get_points_inside_box(self):
        atoms = np.array([('H', 0.0, 0.0, 0.0), ('H', 1.0, 1.0, 1.0)],
                         dtype=Atom.atom_dtype)
        box_dim1 = [0, 0, 0, 0.5, 0.5, 0.5]
        box_dim2 = [[0, 0, 0], [0.5, 0.5, 0.5]]
        box_dim3 = np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
        box_dim3_v2 = [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]
        box_dim4 = [[0, 0.5], [0, 0.5], [0.5]]
        box_dim5 = [0, 0, 0, -1, 0.5, 0.5]
        box_dim6 = [0, 0, 0, 0.5, 0.5, -2]

        result1 = get_points_inside_box(atoms, box_dim1)
        result2 = get_points_inside_box(atoms, box_dim2)
        result3 = get_points_inside_box(atoms, box_dim3)
        result3_v2 = get_points_inside_box(atoms, box_dim3_v2)

        np.testing.assert_array_equal(result1, atoms[0])
        np.testing.assert_array_equal(result2, atoms[0])
        np.testing.assert_array_equal(result3, np.array([], dtype=Atom.atom_dtype))
        np.testing.assert_array_equal(result3_v2, atoms[1])

        with self.assertRaises(ValueError):
            _ = get_points_inside_box(atoms, box_dim4)

        with self.assertRaises(ValueError):
            _ = get_points_inside_box(atoms, box_dim5)

        with self.assertRaises(ValueError):
            _ = get_points_inside_box(atoms, box_dim6)
