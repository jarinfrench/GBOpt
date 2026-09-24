# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.CandidateLoader import CandidateLoader
from GBOpt.FileGrainOwnership import GrainOwnershipError
from GBOpt.GBManipulator import GBManipulator
from GBOpt.UnitCell import UnitCell


def _owned_candidate() -> tuple[np.ndarray, np.ndarray, np.ndarray, UnitCell]:
    atoms = np.asarray(
        [
            ("Ni", 6.5, 1.0, 1.0),
            ("Ni", 3.0, 2.0, 2.0),
            ("Ni", 4.0, 3.0, 3.0),
            ("Ni", 4.5, 4.0, 4.0),
            ("Ni", 5.5, 5.0, 5.0),
            ("Ni", 6.0, 6.0, 6.0),
            ("Ni", 7.0, 7.0, 7.0),
            ("Ni", 8.0, 8.0, 8.0),
        ],
        dtype=Atom.atom_dtype,
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    box_dims = np.asarray(
        [[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]],
        dtype=float,
    )
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fcc", 3.52, "Ni")
    return atoms, labels, box_dims, unit_cell


def test_write_candidate_mapping_ids_equal_write_result_ids(tmp_path: Path) -> None:
    atoms, labels, box_dims, _unit_cell = _owned_candidate()
    path = tmp_path / "candidate.data"

    result, mapping = CandidateLoader().write_candidate(
        path,
        atoms,
        labels,
        box_dims=box_dims,
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(0.0, 5.0),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-8,
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        type_map={"Ni": 1},
    )

    assert result.target == path
    assert path.is_file()
    assert np.array_equal(mapping.atom_ids, result.atom_ids)
    assert np.array_equal(mapping.atom_ids, np.arange(1, len(atoms) + 1))
    assert np.array_equal(mapping.labels, labels)


def test_write_candidate_rejects_invalid_geometry(tmp_path: Path) -> None:
    atoms, labels, _box_dims, _unit_cell = _owned_candidate()
    bad_box_dims = np.asarray([[10.0, 0.0], [0.0, 10.0], [0.0, 10.0]], dtype=float)

    with pytest.raises(GrainOwnershipError):
        CandidateLoader().write_candidate(
            tmp_path / "candidate.data",
            atoms,
            labels,
            box_dims=bad_box_dims,
            gb_plane_x=5.0,
            inplane_periodic=(True, True),
            right_grain_x_bounds=(5.0, 10.0),
            coordinate_tolerance=1.0e-8,
        )


def test_write_then_reload_round_trip(tmp_path: Path) -> None:
    atoms, labels, box_dims, unit_cell = _owned_candidate()
    loader = CandidateLoader()
    path = tmp_path / "candidate.data"

    _result, mapping = loader.write_candidate(
        path,
        atoms,
        labels,
        box_dims=box_dims,
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(0.0, 5.0),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-8,
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        type_map={"Ni": 1},
    )

    manipulator = loader.reload(
        path,
        candidate_mapping=mapping,
        unit_cell=unit_cell,
        gb_thickness=10.0,
        type_dict={"Ni": 1},
    )

    assert isinstance(manipulator, GBManipulator)
    assert np.array_equal(manipulator.parents[0].grain_labels, labels)
    assert np.array_equal(manipulator.parents[0].initial_atom_ids, mapping.atom_ids)


def test_reload_rejects_atom_count_change(tmp_path: Path) -> None:
    atoms, labels, box_dims, unit_cell = _owned_candidate()
    loader = CandidateLoader()
    path = tmp_path / "candidate.data"
    _result, mapping = loader.write_candidate(
        path,
        atoms,
        labels,
        box_dims=box_dims,
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(0.0, 5.0),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-8,
        type_map={"Ni": 1},
    )

    truncated = tmp_path / "truncated.data"
    _write_result, _truncated_mapping = loader.write_candidate(
        truncated,
        atoms[:-1],
        labels[:-1],
        box_dims=box_dims,
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(0.0, 5.0),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-8,
        type_map={"Ni": 1},
    )

    with pytest.raises(GrainOwnershipError, match="atom count does not match"):
        loader.reload(
            truncated,
            candidate_mapping=mapping,
            unit_cell=unit_cell,
            gb_thickness=10.0,
            type_dict={"Ni": 1},
        )


def test_reload_allow_variable_cell_requires_boolean(tmp_path: Path) -> None:
    atoms, labels, box_dims, unit_cell = _owned_candidate()
    loader = CandidateLoader()
    path = tmp_path / "candidate.data"
    _result, mapping = loader.write_candidate(
        path,
        atoms,
        labels,
        box_dims=box_dims,
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(0.0, 5.0),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1.0e-8,
        type_map={"Ni": 1},
    )

    with pytest.raises(TypeError, match="allow_variable_cell must be a Boolean"):
        loader.reload(
            path,
            candidate_mapping=mapping,
            unit_cell=unit_cell,
            gb_thickness=10.0,
            type_dict={"Ni": 1},
            allow_variable_cell="yes",
        )


def test_file_grain_ownership_does_not_import_gbmanipulator() -> None:
    """FileGrainOwnership no longer needs a local import of GBManipulator (#75)."""
    script = dedent(
        """
        import types
        import importlib.util
        import sys

        spec = importlib.util.find_spec("GBOpt")
        stub = types.ModuleType("GBOpt")
        stub.__path__ = spec.submodule_search_locations
        sys.modules["GBOpt"] = stub

        import GBOpt.FileGrainOwnership
        assert "GBOpt.GBManipulator" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_gbmanipulator_does_not_import_candidate_loader() -> None:
    """GBManipulator does not import CandidateLoader, confirming no reintroduced cycle."""
    script = dedent(
        """
        import types
        import importlib.util
        import sys

        spec = importlib.util.find_spec("GBOpt")
        stub = types.ModuleType("GBOpt")
        stub.__path__ = spec.submodule_search_locations
        sys.modules["GBOpt"] = stub

        import GBOpt.GBManipulator
        assert "GBOpt.CandidateLoader" not in sys.modules
        assert "GBOpt.FileGrainOwnership" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
