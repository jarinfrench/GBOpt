# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import math
import unittest
from collections import Counter
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.UnitCell import (
    RationalBasis,
    UnitCell,
    UnitCellTypeError,
    UnitCellValueError,
)


class TestUnitCell(unittest.TestCase):

    def test_fcc_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fcc', a0=1.0, atoms='Cu')
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(2) * 0.25)
        self.assertEqual(len(cell.unit_cell), 4)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0]
                    ]
                )
            )
        )
        names = cell.names()
        self.assertEqual(len(set(names)), 1)
        self.assertEqual(len(positions), len(names))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )
        self.assertTrue(list(cell.ideal_bond_lengths.keys()) == [(1, 1)])
        self.assertTrue(math.isclose(cell.ideal_bond_lengths[(1, 1)], 1 / np.sqrt(2)))
        self.assertEqual(cell.ratio, {1: 1})

    def test_bcc_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='bcc', a0=1.0, atoms='Fe')
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(3) * 0.25)
        self.assertEqual(len(cell.unit_cell), 2)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5]
                    ]
                )
            )
        )
        names = cell.names()
        self.assertEqual(len(set(names)), 1)
        self.assertEqual(len(positions), len(names))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0]
                    ]
                )
            )
        )
        self.assertTrue(list(cell.ideal_bond_lengths.keys()) == [(1, 1)])
        self.assertTrue(math.isclose(cell.ideal_bond_lengths[(1, 1)], np.sqrt(3) / 2))
        self.assertEqual(cell.ratio, {1: 1})

    def test_sc_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='sc', a0=1.0, atoms='H')
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, 0.5)
        self.assertEqual(len(cell.unit_cell), 1)
        positions = cell.positions()
        self.assertTrue(np.allclose(positions, np.array([[0.0, 0.0, 0.0]])))
        names = cell.names()
        self.assertEqual(len(set(names)), 1)
        self.assertEqual(len(positions), len(names))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ]
                )
            )
        )
        self.assertTrue(list(cell.ideal_bond_lengths.keys()) == [(1, 1)])
        self.assertTrue(math.isclose(cell.ideal_bond_lengths[(1, 1)], 1))
        self.assertEqual(cell.ratio, {1: 1})

    def test_diamond_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='diamond', a0=1.0, atoms='C')
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(3) * 0.125)
        self.assertEqual(len(cell.unit_cell), 8)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.25, 0.25, 0.25],
                        [0.75, 0.75, 0.25],
                        [0.75, 0.25, 0.75],
                        [0.25, 0.75, 0.75]
                    ]
                )
            )
        )
        names = cell.names()
        self.assertEqual(len(set(names)), 1)
        self.assertEqual(len(positions), len(names))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )
        self.assertTrue(list(cell.ideal_bond_lengths.keys()) == [(1, 1)])
        self.assertTrue(math.isclose(cell.ideal_bond_lengths[(1, 1)], np.sqrt(3) / 4))
        self.assertEqual(cell.ratio, {1: 1})

    def test_fluorite_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fluorite', a0=1.0, atoms=['Ca', 'F'])
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(3) * 0.125)
        self.assertEqual(len(cell.unit_cell), 12)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.25, 0.25, 0.25],
                        [0.25, 0.25, 0.75],
                        [0.25, 0.75, 0.25],
                        [0.25, 0.75, 0.75],
                        [0.75, 0.25, 0.25],
                        [0.75, 0.25, 0.75],
                        [0.75, 0.75, 0.25],
                        [0.75, 0.75, 0.75],
                    ]
                )
            )
        )
        names = cell.names()
        self.assertEqual(len(set(names)), 2)
        for i in range(4):
            self.assertEqual(names[i], 'Ca')
        for i in range(4, 12):
            self.assertEqual(names[i], 'F')
        self.assertEqual(len(positions), len(names))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )
        ideal_fluorite_bonds = {
            (1, 1): 1 / np.sqrt(2),
            (1, 2): np.sqrt(3) / 4,
            (2, 2): 0.5
        }
        self.assertTrue(cell.ideal_bond_lengths.keys() == ideal_fluorite_bonds.keys())
        self.assertTrue(all([math.isclose(cell.ideal_bond_lengths[i],
                        ideal_fluorite_bonds[i]) for i in ideal_fluorite_bonds.keys()]))
        self.assertEqual(cell.ratio, {1: 1, 2: 2})

    def test_rocksalt_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='rocksalt', a0=1.0, atoms=['Na', 'Cl'])
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, 0.25)
        self.assertEqual(len(cell.unit_cell), 8)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0, 0, 0],
                        [0, 0.5, 0.5],
                        [0.5, 0, 0.5],
                        [0.5, 0.5, 0],
                        [0, 0, 0.5],
                        [0, 0.5, 0],
                        [0.5, 0, 0],
                        [0.5, 0.5, 0.5],
                    ]
                )
            )
        )
        names = cell.names()
        self.assertEqual(len(set(names)), 2)
        for i in range(4):
            self.assertEqual(names[i], 'Na')
        for i in range(4, 8):
            self.assertEqual(names[i], 'Cl')
        self.assertEqual(len(positions), len(names))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )
        ideal_rocksalt_bonds = {
            (1, 1): 1 / np.sqrt(2),
            (1, 2): 0.5,
            (2, 2): 1 / np.sqrt(2)
        }
        self.assertTrue(cell.ideal_bond_lengths.keys() == ideal_rocksalt_bonds.keys())
        self.assertTrue(all([math.isclose(cell.ideal_bond_lengths[i],
                        ideal_rocksalt_bonds[i]) for i in ideal_rocksalt_bonds.keys()]))
        self.assertEqual(cell.ratio, {1: 1, 2: 1})

    def test_zincblende_initialization(self):
        cell = UnitCell()
        cell.init_by_structure(structure='zincblende', a0=1.0, atoms=['Zn', 'S'])
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, math.sqrt(3) * 0.125)
        self.assertEqual(len(cell.unit_cell), 8)
        positions = cell.positions()
        self.assertTrue(
            np.allclose(
                positions,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                        [0.25, 0.25, 0.25],
                        [0.75, 0.75, 0.25],
                        [0.75, 0.25, 0.75],
                        [0.25, 0.75, 0.75],
                    ]
                )
            )
        )
        names = cell.names()
        self.assertEqual(len(set(names)), 2)
        for i in range(4):
            self.assertEqual(names[i], 'Zn')
        for i in range(4, 8):
            self.assertEqual(names[i], 'S')
        self.assertEqual(len(positions), len(names))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, -1.0]
                    ]
                )
            )
        )
        ideal_zincblende_bonds = {
            (1, 1): 1 / np.sqrt(2),
            (1, 2): np.sqrt(3) / 4,
            (2, 2): 1 / np.sqrt(2)
        }
        self.assertTrue(cell.ideal_bond_lengths.keys() == ideal_zincblende_bonds.keys())
        self.assertTrue(all([math.isclose(cell.ideal_bond_lengths[i],
                        ideal_zincblende_bonds[i]) for i in ideal_zincblende_bonds.keys()]))
        self.assertEqual(cell.ratio, {1: 1, 2: 1})

    def test_custom_initialization(self):
        cell = UnitCell()
        custom_unit_cell = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        custom_conventional = np.array([[1.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        custom_reciprocal = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        custom_ideal_bonds = {
            (1, 1): 1,
            (1, 2): 1,
            (2, 2): 1
        }
        custom_ratio = {1: 1, 2: 1}
        custom_type_map = {'C': 1, 'H': 2}
        with self.assertRaises(UnitCellValueError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types='H',
                                a0=1.0,
                                conventional=custom_conventional,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds
                                )
        with self.assertRaises(UnitCellValueError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=['H', 'C'],
                                a0=1.0,
                                conventional=custom_conventional,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds,
                                ratio={1: 0, 2: 2}
                                )
        with self.assertRaises(UnitCellValueError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=['H', 'C'],
                                a0=1.0,
                                conventional=custom_conventional,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds,
                                ratio={0: 1, 1: 1}
                                )
        with self.assertRaises(UnitCellTypeError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=['H', 'C'],
                                a0=1.0,
                                conventional=custom_conventional,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds,
                                ratio="Error"
                                )
        with self.assertRaises(UnitCellTypeError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types=['H', 'C'],
                                a0=1.0,
                                conventional=custom_conventional,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds,
                                ratio="Error"
                                )
        cell.init_by_custom(unit_cell=custom_unit_cell,
                            unit_cell_types=['H', 'C'],
                            a0=1.0,
                            conventional=custom_conventional,
                            reciprocal=custom_reciprocal,
                            ideal_bond_lengths=custom_ideal_bonds,
                            ratio=custom_ratio
                            )
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(len(cell.unit_cell), 2)
        names = cell.names()
        self.assertEqual(names[0], 'H')
        self.assertEqual(names[1], 'C')
        positions = cell.positions()
        self.assertTrue(np.allclose(positions, custom_unit_cell))
        names = cell.names()
        self.assertEqual(len(set(names)), 2)
        self.assertEqual(len(positions), len(names))
        reciprocal = cell.reciprocal
        self.assertTrue(
            np.allclose(
                reciprocal,
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]
                    ]
                )
            )
        )
        self.assertTrue(cell.ideal_bond_lengths.keys() == custom_ideal_bonds.keys())
        self.assertTrue(all([math.isclose(cell.ideal_bond_lengths[i],
                        custom_ideal_bonds[i]) for i in custom_ideal_bonds.keys()]))
        self.assertEqual(cell.ratio, custom_ratio)
        self.assertEqual(cell.type_map, {'H': 1, 'C': 2})
        cell.type_map = custom_type_map
        self.assertEqual(cell.type_map, custom_type_map)

    def test_not_implemented_structure(self):
        cell = UnitCell()
        with self.assertRaises(NotImplementedError):
            cell.init_by_structure(structure='notimplemented', a0=1.0, atoms='H')

    def test_inits_with_multiple_atoms_one_atom_type(self):
        cell = UnitCell()
        with self.assertRaises(UnitCellValueError):
            cell.init_by_structure('fluorite', 1.0, 'Ca')
        with self.assertRaises(UnitCellValueError):
            cell.init_by_structure('rocksalt', 1.0, 'Na')
        with self.assertRaises(UnitCellValueError):
            cell.init_by_structure('zincblende', 1.0, 'Zn')

    def test_reciprocal_shape(self):
        cell = UnitCell()
        custom_unit_cell = np.array([[0.1, 0.2, 0.3]])
        custom_conventional = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        custom_reciprocal = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        custom_ideal_bonds = {
            (1, 1): 1,
            (1, 2): 1,
            (2, 2): 1
        }
        with self.assertRaises(UnitCellValueError):
            cell.init_by_custom(unit_cell=custom_unit_cell,
                                unit_cell_types='H',
                                a0=1.0,
                                conventional=custom_conventional,
                                reciprocal=custom_reciprocal,
                                ideal_bond_lengths=custom_ideal_bonds
                                )

    def test_lattice_parameter(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fcc', a0=3.54, atoms='Cu')
        cu_unit_cell = [
            Atom('Cu', 0.0, 0.0, 0.0),
            Atom('Cu', 0.0, 1.77, 1.77),
            Atom('Cu', 1.77, 0.0, 1.77),
            Atom('Cu', 1.77, 1.77, 0.0)
        ]
        cu_positions = [i['position'] for i in cu_unit_cell]
        cell_positions = [i['position'] for i in cell.unit_cell]
        self.assertEqual(cell_positions, cu_positions)

    def test_repr(self):
        cell = UnitCell()
        self.assertEqual(
            repr(cell),
            "UnitCell with 0 atoms\nLattice parameter (a0): 1.000 Å\nRadius: 0.000 Å\n"
            "Atoms: []\nReciprocal lattice:\n[[0. 0. 0.]\n [0. 0. 0.]\n [0. 0. 0.]]")
        cell.init_by_structure('sc', 1.0, 'H')
        self.assertEqual(
            repr(cell),
            "UnitCell with 1 atom\nLattice parameter (a0): 1.000 Å\nRadius: 0.500 Å\n"
            "Atoms: ['H': 0.000, 0.000, 0.000]\nReciprocal lattice:\n[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]"
        )
        custom_ideal_bonds = {
            (1, 1): 1,
            (1, 2): 1,
            (2, 2): 1
        }
        cell.init_by_custom(unit_cell=[[0, 0, 0]],
                            unit_cell_types='H',
                            a0=1.0,
                            conventional=[[1.0, 0.0, 0.0], [
                                0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            reciprocal=[[1.0, 0.0, 0.0], [
                                0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            ideal_bond_lengths=custom_ideal_bonds)
        self.assertEqual(
            repr(cell),
            "UnitCell with 1 atom\nLattice parameter (a0): 1.000 Å\nRadius: 0.500 Å\n"
            "Atoms: ['H': 0.000, 0.000, 0.000]\nReciprocal lattice:\n[[1. 0. 0.]\n [0. 1. 0.]\n [0. 0. 1.]]"
        )

    def test_asarray(self):
        cell = UnitCell()
        cell.init_by_structure('fcc', 1.0, 'Cu')
        expected = np.array([('Cu', 0.0, 0.0, 0.0),
                             ('Cu', 0.0, 0.5, 0.5),
                             ('Cu', 0.5, 0.0, 0.5),
                             ('Cu', 0.5, 0.5, 0.0)], dtype=Atom.atom_dtype)
        result = cell.asarray()
        self.assertTrue(all(result == expected))

        np.testing.assert_array_equal(result['name'], expected['name'])
        np.testing.assert_array_equal(result['x'], expected['x'])
        np.testing.assert_array_equal(result['y'], expected['y'])
        np.testing.assert_array_equal(result['z'], expected['z'])

    def test_init(self):
        cell = UnitCell()
        self.assertEqual(cell.unit_cell, [])
        self.assertTrue(np.allclose(cell.primitive, np.zeros((3, 3))))
        self.assertEqual(cell.a0, 1.0)
        self.assertEqual(cell.radius, 0.0)
        self.assertTrue(np.allclose(cell.reciprocal, np.zeros((3, 3))))
        self.assertEqual(cell.type_map, {})

    def test_names_as_ints(self):
        cell = UnitCell()
        cell.init_by_structure(structure='fcc', a0=1.0, atoms='Cu')
        self.assertTrue(np.allclose(cell.names(asint=True),
                        np.array([1, 1, 1, 1], dtype=int)))

        cell.init_by_structure(
            "fluorite", 5.52, ["Ca", "F"], type_map={"Ca": 2, "F": 1})
        self.assertTrue(all(cell.names(asint=True) == [
                        2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]))

    def test_setter(self):
        cell = UnitCell()
        cell.init_by_structure('fcc', 1.0, 'Ni')
        with self.assertRaises(UnitCellValueError):
            cell.a0 = -1.0
        cell.a0 = 2.0
        self.assertTrue(np.allclose(cell.primitive, np.array(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]])))
        self.assertEqual(cell.radius, math.sqrt(2) * 0.25 * 2)
        self.assertTrue(np.allclose(cell.reciprocal, np.array(
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]])))

    def test_fcc_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("fcc", 1.0, "Cu")
        self.assertAlmostEqual(cell.nn_distance(1), 1/np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2), 1)
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(6)/2)
        self.assertAlmostEqual(cell.nn_distance(4), np.sqrt(2))

        cell.a0 = 3.54
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(6) / 2)
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0 * np.sqrt(2))

    def test_bcc_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("bcc", 1.0, "Fe")
        self.assertAlmostEqual(cell.nn_distance(1), np.sqrt(3) / 2)
        self.assertAlmostEqual(cell.nn_distance(2), 1)
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(4), np.sqrt(11) / 2)

        cell.a0 = 2.86
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 * np.sqrt(3) / 2)
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0 * np.sqrt(11) / 2)

    def test_sc_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("sc", 1.0, "Po")
        self.assertAlmostEqual(cell.nn_distance(1), 1)
        self.assertAlmostEqual(cell.nn_distance(2), np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(3))
        self.assertAlmostEqual(cell.nn_distance(4), 2)

        cell.a0 = 3.345
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0 * np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(3))
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0 * 2)

    def test_diamond_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("diamond", 1.0, "C")
        self.assertAlmostEqual(cell.nn_distance(1), np.sqrt(3) / 4)
        self.assertAlmostEqual(cell.nn_distance(2), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(11) / 4)
        self.assertAlmostEqual(cell.nn_distance(4), 1)

        cell.a0 = 3.567
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 * np.sqrt(3) / 4)
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(11) / 4)
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0)

    def test_fluorite_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("fluorite", 1.0, ["U", "O"])
        self.assertAlmostEqual(cell.nn_distance(1), np.sqrt(3)/4)
        self.assertAlmostEqual(cell.nn_distance(2), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(11)/4)
        self.assertAlmostEqual(cell.nn_distance(4), 1)

        self.assertAlmostEqual(cell.nn_distance(1, 1), 1/np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 1), 1)
        self.assertAlmostEqual(cell.nn_distance(3, 1), np.sqrt(6)/2)
        self.assertAlmostEqual(cell.nn_distance(4, 1), np.sqrt(2))

        self.assertAlmostEqual(cell.nn_distance(1, 2), 0.5)
        self.assertAlmostEqual(cell.nn_distance(2, 2), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3, 2), np.sqrt(3) / 2)
        self.assertAlmostEqual(cell.nn_distance(4, 2), 1)

        cell.a0 = 5.454
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 * np.sqrt(3)/4)
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0 * 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(11)/4)
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0)

        self.assertAlmostEqual(cell.nn_distance(1, 1), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 1), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3, 1), cell.a0 * np.sqrt(6) / 2)
        self.assertAlmostEqual(cell.nn_distance(4, 1), cell.a0 * np.sqrt(2))

        self.assertAlmostEqual(cell.nn_distance(1, 2), cell.a0 * 0.5)
        self.assertAlmostEqual(cell.nn_distance(2, 2), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3, 2), cell.a0 * np.sqrt(3) / 2)
        self.assertAlmostEqual(cell.nn_distance(4, 2), cell.a0)

    def test_rocksalt_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("rocksalt", 1.0, ["Na", "Cl"])
        self.assertAlmostEqual(cell.nn_distance(1), 0.5)
        self.assertAlmostEqual(cell.nn_distance(2), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(3) / 2)
        self.assertAlmostEqual(cell.nn_distance(4), 1)

        self.assertAlmostEqual(cell.nn_distance(1, 1), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 1), 1)
        self.assertAlmostEqual(cell.nn_distance(3, 1), np.sqrt(6)/2)
        self.assertAlmostEqual(cell.nn_distance(4, 1), np.sqrt(2))

        self.assertAlmostEqual(cell.nn_distance(1, 2), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 2), 1)
        self.assertAlmostEqual(cell.nn_distance(3, 2), np.sqrt(6)/2)
        self.assertAlmostEqual(cell.nn_distance(4, 2), np.sqrt(2))

        cell.a0 = 5.454
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 * 0.5)
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(3) / 2)
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0)

        self.assertAlmostEqual(cell.nn_distance(1, 1), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 1), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3, 1), cell.a0 * np.sqrt(6) / 2)
        self.assertAlmostEqual(cell.nn_distance(4, 1), cell.a0 * np.sqrt(2))

        self.assertAlmostEqual(cell.nn_distance(1, 2), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 2), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3, 2), cell.a0 * np.sqrt(6) / 2)
        self.assertAlmostEqual(cell.nn_distance(4, 2), cell.a0 * np.sqrt(2))

    def test_zincblende_nn_calculations(self):
        cell = UnitCell()
        cell.init_by_structure("zincblende", 1.0, ["Zn", "S"])
        self.assertAlmostEqual(cell.nn_distance(1), np.sqrt(3) / 4)
        self.assertAlmostEqual(cell.nn_distance(2), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(11) / 4)
        self.assertAlmostEqual(cell.nn_distance(4), 1)

        self.assertAlmostEqual(cell.nn_distance(1, 1), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 1), 1)
        self.assertAlmostEqual(cell.nn_distance(3, 1), np.sqrt(6)/2)
        self.assertAlmostEqual(cell.nn_distance(4, 1), np.sqrt(2))

        self.assertAlmostEqual(cell.nn_distance(1, 2), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 2), 1)
        self.assertAlmostEqual(cell.nn_distance(3, 2), np.sqrt(6)/2)
        self.assertAlmostEqual(cell.nn_distance(4, 2), np.sqrt(2))

        cell.a0 = 5.454
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 * np.sqrt(3) / 4)
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(11) / 4)
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0)

        self.assertAlmostEqual(cell.nn_distance(1, 1), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 1), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3, 1), cell.a0 * np.sqrt(6) / 2)
        self.assertAlmostEqual(cell.nn_distance(4, 1), cell.a0 * np.sqrt(2))

        self.assertAlmostEqual(cell.nn_distance(1, 2), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 2), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3, 2), cell.a0 * np.sqrt(6) / 2)
        self.assertAlmostEqual(cell.nn_distance(4, 2), cell.a0 * np.sqrt(2))

    def notest_custom_nn_calculations(self):
        cell = UnitCell()
        custom_unit_cell = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        custom_conventional = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        custom_reciprocal = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        custom_ideal_bonds = {
            (1, 1): 1,
            (1, 2): 1,
            (2, 2): 1
        }
        custom_ratio = {1: 1, 2: 1}
        cell.init_by_custom(unit_cell=custom_unit_cell,
                            unit_cell_types=['H', 'C'],
                            a0=1.0,
                            conventional=custom_conventional,
                            reciprocal=custom_reciprocal,
                            ideal_bond_lengths=custom_ideal_bonds,
                            ratio=custom_ratio
                            )
        self.assertAlmostEqual(cell.nn_distance(1), np.sqrt(3)/4)
        self.assertAlmostEqual(cell.nn_distance(2), 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), np.sqrt(11)/4)
        self.assertAlmostEqual(cell.nn_distance(4), 1)

        self.assertAlmostEqual(cell.nn_distance(1, 1), 1/np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2, 1), 1)
        self.assertAlmostEqual(cell.nn_distance(3, 1), np.sqrt(6)/2)
        self.assertAlmostEqual(cell.nn_distance(4, 1), np.sqrt(2))

        self.assertAlmostEqual(cell.nn_distance(1, 2), np.sqrt(3)/4)
        self.assertAlmostEqual(cell.nn_distance(2, 2), np.sqrt(11)/4)
        self.assertAlmostEqual(cell.nn_distance(3, 2), np.sqrt(19)/4)
        self.assertAlmostEqual(cell.nn_distance(4, 2), np.sqrt(6)/2)

        cell.a0 = 3.3
        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 * np.sqrt(3)/4)
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0 * 1 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(11)/4)
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0)

        self.assertAlmostEqual(cell.nn_distance(1), cell.a0 / np.sqrt(2))
        self.assertAlmostEqual(cell.nn_distance(2), cell.a0)
        self.assertAlmostEqual(cell.nn_distance(3), cell.a0 * np.sqrt(6) / 2)
        self.assertAlmostEqual(cell.nn_distance(4), cell.a0 * np.sqrt(2))

        self.assertAlmostEqual(cell.nn_distance(1, 2), cell.a0 * np.sqrt(3)/4)
        self.assertAlmostEqual(cell.nn_distance(2, 2), cell.a0 * np.sqrt(11)/4)
        self.assertAlmostEqual(cell.nn_distance(3, 2), cell.a0 * np.sqrt(19)/4)
        self.assertAlmostEqual(cell.nn_distance(4, 2), cell.a0 * np.sqrt(6)/2)

    def test_type_map_property(self):
        cell = UnitCell()
        cell.init_by_structure("fcc", 1.0, "Cu")

        self.assertEqual(cell.type_map, {"Cu": 1})

        cell.init_by_structure("fluorite", 2.0, ["U", "O"])
        self.assertEqual(cell.type_map, {"U": 1, "O": 2})
        cell.type_map = {"O": 1, "U": 2}
        self.assertEqual(cell.type_map, {"U": 2, "O": 1})
        cell.type_map = {1: "U", 2: "O"}
        self.assertEqual(cell.type_map, {"U": 1, "O": 2})

        with self.assertRaises(UnitCellValueError):
            cell.type_map = {2: "U", 3: "O"}

        with self.assertRaises(UnitCellValueError):
            cell.type_map = {"O": 2, "U": 3}

        with self.assertRaises(UnitCellTypeError):
            cell.type_map = "Error"

    def test_types_method(self):
        cell = UnitCell()
        cell.init_by_structure("fluorite", 5.52, ["Ca", "F"])
        self.assertTrue(all(cell.types() == [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]))
        cell.type_map = {"F": 1, "Ca": 2}
        self.assertTrue(all(cell.types() == [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]))


BUILTIN_RATIONAL_BASIS_CASES = [
    pytest.param(
        "sc",
        "H",
        1,
        ("H",),
        [[0, 0, 0]],
        id="sc",
    ),
    pytest.param(
        "bcc",
        "Fe",
        2,
        ("Fe", "Fe"),
        [[0, 0, 0], [1, 1, 1]],
        id="bcc",
    ),
    pytest.param(
        "fcc",
        "Cu",
        2,
        ("Cu",) * 4,
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        id="fcc",
    ),
    pytest.param(
        "diamond",
        "C",
        4,
        ("C",) * 8,
        [
            [0, 0, 0],
            [0, 2, 2],
            [2, 0, 2],
            [2, 2, 0],
            [1, 1, 1],
            [3, 3, 1],
            [3, 1, 3],
            [1, 3, 3],
        ],
        id="diamond",
    ),
    pytest.param(
        "fluorite",
        ["U", "O"],
        4,
        ("U",) * 4 + ("O",) * 8,
        [
            [0, 0, 0],
            [0, 2, 2],
            [2, 0, 2],
            [2, 2, 0],
            [1, 1, 1],
            [1, 1, 3],
            [1, 3, 1],
            [1, 3, 3],
            [3, 1, 1],
            [3, 1, 3],
            [3, 3, 1],
            [3, 3, 3],
        ],
        id="fluorite",
    ),
    pytest.param(
        "rocksalt",
        ["Na", "Cl"],
        2,
        ("Na",) * 4 + ("Cl",) * 4,
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        id="rocksalt",
    ),
    pytest.param(
        "zincblende",
        ["Zn", "S"],
        4,
        ("Zn",) * 4 + ("S",) * 4,
        [
            [0, 0, 0],
            [0, 2, 2],
            [2, 0, 2],
            [2, 2, 0],
            [1, 1, 1],
            [3, 3, 1],
            [3, 1, 3],
            [1, 3, 3],
        ],
        id="zincblende",
    ),
]


INVALID_RATIONAL_BASES = [
    pytest.param(
        UnitCellTypeError,
        {"names": ("A",), "numerators": [[0, 0, 0]], "denominator": 1.0},
        id="noninteger-denominator",
    ),
    pytest.param(
        UnitCellValueError,
        {"names": ("A",), "numerators": [[0, 0, 0]], "denominator": 0},
        id="nonpositive-denominator",
    ),
    pytest.param(
        UnitCellValueError,
        {"names": ("A",), "numerators": [0, 0, 0], "denominator": 2},
        id="one-dimensional-numerators",
    ),
    pytest.param(
        UnitCellValueError,
        {"names": (), "numerators": np.empty((0, 3)), "denominator": 2},
        id="empty-basis",
    ),
    pytest.param(
        UnitCellValueError,
        {
            "names": ("A", "B"),
            "numerators": [[0, 0, 0]],
            "denominator": 2,
        },
        id="name-coordinate-count-mismatch",
    ),
    pytest.param(
        UnitCellTypeError,
        {"names": ("A",), "numerators": [[0.0, 0, 0]], "denominator": 2},
        id="noninteger-numerator",
    ),
    pytest.param(
        UnitCellValueError,
        {"names": ("A",), "numerators": [[2, 0, 0]], "denominator": 2},
        id="numerator-outside-half-open-cell",
    ),
    pytest.param(
        UnitCellValueError,
        {
            "names": ("A", "B"),
            "numerators": [[0, 0, 0], [0, 0, 0]],
            "denominator": 2,
        },
        id="duplicate-coordinate",
    ),
]


class TestRationalBasis:

    @pytest.mark.parametrize(
        ("structure", "atoms", "denominator", "expected_names", "numerators"),
        BUILTIN_RATIONAL_BASIS_CASES,
    )
    def test_builtin_structures_expose_exact_rational_bases(
            self, structure, atoms, denominator, expected_names, numerators):
        cell = UnitCell()
        cell.init_by_structure(structure=structure, a0=5.454, atoms=atoms)

        basis = cell.rational_basis

        assert basis is not None
        assert basis.denominator == denominator
        assert basis.names == expected_names
        assert basis.numerators.dtype == object
        np.testing.assert_array_equal(
            basis.numerators,
            np.array(numerators, dtype=object),
        )

        expected_cartesian = np.asarray(numerators, dtype=float) * cell.a0 / denominator
        atom_array = cell.asarray()
        actual_cartesian = np.column_stack(
            (atom_array["x"], atom_array["y"], atom_array["z"])
        )
        np.testing.assert_allclose(actual_cartesian, expected_cartesian)
        assert tuple(atom_array["name"]) == expected_names

    def test_fluorite_rational_basis_has_expected_species_counts(self):
        cell = UnitCell()
        cell.init_by_structure("fluorite", 5.454, ["U", "O"])

        assert Counter(cell.rational_basis.names) == {"U": 4, "O": 8}

    def test_rational_basis_copies_inputs_and_is_immutable(self):
        names = ["A", "B"]
        numerators = np.array([[0, 0, 0], [1, 1, 1]], dtype=object)
        basis = RationalBasis(names=names, numerators=numerators, denominator=2)

        names[0] = "changed"
        numerators[0, 0] = 1

        assert basis.names == ("A", "B")
        np.testing.assert_array_equal(
            basis.numerators,
            np.array([[0, 0, 0], [1, 1, 1]], dtype=object),
        )
        assert not basis.numerators.flags.writeable
        with pytest.raises(ValueError):
            basis.numerators[0, 0] = 1
        with pytest.raises(ValueError):
            basis.numerators.setflags(write=True)
        with pytest.raises(FrozenInstanceError):
            basis.denominator = 4

    @pytest.mark.parametrize(("error", "kwargs"), INVALID_RATIONAL_BASES)
    def test_rational_basis_rejects_invalid_data(self, error, kwargs):
        with pytest.raises(error):
            RationalBasis(**kwargs)

    def test_uninitialized_cell_has_no_exact_rational_basis(self):
        assert UnitCell().rational_basis is None

    def test_custom_cell_has_no_exact_rational_basis(self):
        cell = UnitCell()
        cell.init_by_custom(
            unit_cell=np.array([[0.0, 0.0, 0.0]]),
            unit_cell_types=["H"],
            a0=1.0,
            conventional=np.eye(3),
            reciprocal=np.eye(3),
            ideal_bond_lengths={(1, 1): 1.0},
        )

        assert cell.rational_basis is None


if __name__ == '__main__':
    unittest.main()
