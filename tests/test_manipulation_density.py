# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Regression tests for two bugs found/fixed while extracting
``GBOpt.manipulation.density`` from ``GBManipulator.insert_atoms``/``remove_atoms``
(see ROADMAP_BRANCH_BACKLOG.md's R18 entry):

1. The multitype ``keep_ratio=False`` random type-count split used the global,
   unseeded NumPy RNG (``np.random.choice``) instead of the manipulator's own seeded
   generator. Fixed by routing it through ``_random_type_counts(..., rng)``.
2. ``select_insertion_sites``'s multitype ``keep_ratio=False`` branch passed the
   full-length ``probabilities`` array to ``rng.choice`` alongside a shrinking subset
   of ``available_indices``, which raises once a prior type has consumed some sites.
   Fixed by slicing and renormalizing ``probabilities`` to ``available_indices``.
"""

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.interface.model import InterfaceCandidate
from GBOpt.manipulation.density import (
    AtomRemoval,
    _random_type_counts,
    select_insertion_sites,
    select_removal_indices,
)
from GBOpt.manipulation.types import ManipulationContext
from GBOpt.UnitCell import UnitCell


def test_random_type_counts_does_not_use_global_rng(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("global np.random.choice must not be called")

    monkeypatch.setattr(np.random, "choice", _fail)
    rng = np.random.default_rng(0)
    counts = _random_type_counts(10, 3, rng)
    assert counts.sum() == 10
    assert len(counts) == 3
    assert (counts > 0).all()


def test_random_type_counts_is_deterministic_for_a_given_rng_seed():
    first = _random_type_counts(20, 4, np.random.default_rng(42))
    second = _random_type_counts(20, 4, np.random.default_rng(42))
    assert np.array_equal(first, second)


def test_select_removal_indices_multitype_keep_ratio_false_does_not_use_global_rng(
    monkeypatch,
):
    def _fail(*args, **kwargs):
        raise AssertionError("global np.random.choice must not be called")

    monkeypatch.setattr(np.random, "choice", _fail)

    atoms = np.zeros((12, 4))
    atoms[:6, 0] = 1
    atoms[6:, 0] = 2
    atoms[:, 1] = np.arange(12)
    positions = atoms[:, 1:]
    gb_atom_indices = np.arange(12)

    indices = select_removal_indices(
        atoms=atoms,
        positions=positions,
        gb_atom_indices=gb_atom_indices,
        type_map={1: "A", 2: "B"},
        ratio={1: 1, 2: 1},
        unit_cell=None,
        num_to_remove=4,
        keep_ratio=False,
        rng=np.random.default_rng(1),
    )
    assert len(indices) == 4
    assert len(set(indices)) == 4


def test_select_insertion_sites_multitype_keep_ratio_false_handles_shrinking_sites():
    """Regression for the probability/available-sites length mismatch.

    With 3 types and keep_ratio=False, every type after the first draws from a
    strictly smaller ``available_indices`` than ``len(probabilities)`` -- the shape
    that used to raise ``ValueError: a and p must have same size``.
    """
    num_sites = 9
    possible_sites = np.arange(num_sites)[:, None].astype(float)
    probabilities = np.full(num_sites, 1.0 / num_sites)

    atoms_to_add = select_insertion_sites(
        possible_sites=possible_sites,
        probabilities=probabilities,
        type_map={1: "A", 2: "B", 3: "C"},
        ratio={1: 1, 2: 1, 3: 1},
        unit_cell=None,
        num_to_insert=6,
        keep_ratio=False,
        rng=np.random.default_rng(7),
    )

    all_indices = [idx for indices in atoms_to_add.values() for idx in indices]
    assert len(all_indices) == 6
    assert len(set(all_indices)) == 6


def test_select_insertion_sites_multitype_keep_ratio_false_does_not_use_global_rng(
    monkeypatch,
):
    def _fail(*args, **kwargs):
        raise AssertionError("global np.random.choice must not be called")

    monkeypatch.setattr(np.random, "choice", _fail)

    num_sites = 9
    possible_sites = np.arange(num_sites)[:, None].astype(float)
    probabilities = np.full(num_sites, 1.0 / num_sites)

    select_insertion_sites(
        possible_sites=possible_sites,
        probabilities=probabilities,
        type_map={1: "A", 2: "B", 3: "C"},
        ratio={1: 1, 2: 1, 3: 1},
        unit_cell=None,
        num_to_insert=6,
        keep_ratio=False,
        rng=np.random.default_rng(7),
    )


def test_atom_removal_execute_converts_atom_names_using_unit_cell_type_map():
    """Regression for a bug found while adding GBManipulator.make_removal_candidate.

    ``AtomRemoval.execute`` used to invert ``unit_cell.type_map`` (already
    ``dict[str, int]``, name -> integer type) before looking up each atom's numeric
    type by name, producing an int-keyed dict and raising ``KeyError`` on the string
    lookup for any real unit cell. No existing test called ``AtomRemoval.execute``
    directly, so this went uncaught until a real ``GBMaker``-sourced unit cell (type
    names like ``"Cu"``) was exercised through ``make_removal_candidate``.
    """
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fcc", 1.0, "Cu")

    atoms = np.asarray(
        [
            ("Cu", 0.0, 0.0, 0.0),
            ("Cu", 1.0, 0.0, 0.0),
            ("Cu", 2.0, 0.0, 0.0),
            ("Cu", 3.0, 0.0, 0.0),
        ],
        dtype=Atom.atom_dtype,
    )
    box_dims = np.asarray([[0.0, 4.0], [0.0, 4.0], [0.0, 4.0]], dtype=float)
    candidate = InterfaceCandidate(
        atoms=atoms,
        box_dims=box_dims,
        gb_plane_x=2.0,
        left_grain_x_bounds=(0.0, 2.0),
        right_grain_x_bounds=(2.0, 4.0),
        grain_labels=np.asarray([0, 0, 1, 1], dtype=np.int8),
        inplane_periodic=(True, True),
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        coordinate_tolerance=1.0e-8,
        interface_separation=0.0,
    )
    context = ManipulationContext(
        parents=(candidate,),
        rng=np.random.default_rng(0),
        params={
            "unit_cell": unit_cell,
            "gb_thickness": 4.0,
            "num_to_remove": 1,
            "keep_ratio": True,
        },
    )

    result = AtomRemoval().execute(context)

    assert len(result.children[0].atoms) == 3
