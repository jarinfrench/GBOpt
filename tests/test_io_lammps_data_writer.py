# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundarySpec import FiveDOFSpec
from GBOpt.GBMaker import GBMaker
from GBOpt.io.lammps.data_writer import LammpsDataWriter
from GBOpt.io.lammps.types import LammpsWriteError
from GBOpt.io.types import StructureData, StructureValueError, WriteResult


def _make_triclinic_gb() -> GBMaker:
    """Same fixture as ``tests.test_gbmaker.TestGBMakerTriclinic.setUp``."""
    theta = math.radians(36.869898)
    misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
    return GBMaker.from_boundary_spec(
        3.61,
        "fcc",
        "Cu",
        FiveDOFSpec(misorientation),
        mode="approximate",
        gb_thickness=10.0,
        repeat_factor=6,
        x_dim_min=60.0,
        vacuum=10.0,
        interaction_distance=10,
    )


def _orthogonal_structure() -> tuple[np.ndarray, StructureData]:
    atoms = np.array(
        [("Cu", 0.0, 0.0, 0.0), ("H", 1.0, 1.0, 1.0)], dtype=Atom.atom_dtype
    )
    box_sizes = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    cell = np.diag(box_sizes[:, 1] - box_sizes[:, 0])
    origin = box_sizes[:, 0].copy()
    return atoms, StructureData(atoms, cell, origin)


# --------------------------------------------------------------------------------------
# LammpsDataWriter formatting, direct
# --------------------------------------------------------------------------------------


def test_orthogonal_five_column_form(tmp_path: Path) -> None:
    _, structure = _orthogonal_structure()
    path = tmp_path / "out.data"

    result = LammpsDataWriter().write(path, structure)

    content = path.read_text().splitlines()
    assert content[2] == "2 atoms"
    assert content[3] == "2 atom types"
    assert "Atom Type Labels" in content
    assert content[content.index("Atom Type Labels") + 2] == "1 Cu"
    assert content[content.index("Atom Type Labels") + 3] == "2 H"
    assert isinstance(result, WriteResult)
    assert result.format == "lammps_data"
    np.testing.assert_array_equal(result.atom_ids, [1, 2])
    assert result.losses == ()
    assert result.digest is None


def test_type_as_int_uses_type_map_when_it_covers_all_species(tmp_path: Path) -> None:
    _, structure = _orthogonal_structure()
    path = tmp_path / "out.data"

    LammpsDataWriter().write(
        path, structure, type_as_int=True, type_map={"H": 5, "Cu": 9}
    )

    lines = path.read_text().splitlines()
    atoms_start = lines.index("Atoms") + 2
    assert lines[atoms_start] == "1 9 0.000000 0.000000 0.000000"
    assert lines[atoms_start + 1] == "2 5 1.000000 1.000000 1.000000"


def test_type_as_int_falls_back_to_enumeration_when_type_map_incomplete(
    tmp_path: Path,
) -> None:
    _, structure = _orthogonal_structure()
    path = tmp_path / "out.data"

    LammpsDataWriter().write(path, structure, type_as_int=True, type_map={"Cu": 9})

    lines = path.read_text().splitlines()
    atoms_start = lines.index("Atoms") + 2
    # 'H' is not in type_map, so the whole mapping falls back to first-seen order.
    assert lines[atoms_start] == "1 1 0.000000 0.000000 0.000000"
    assert lines[atoms_start + 1] == "2 2 1.000000 1.000000 1.000000"


def test_six_column_charge_form(tmp_path: Path) -> None:
    atoms = np.array(
        [("U", 0.0, 0.0, 0.0), ("O", 0.25, 0.25, 0.25), ("O", 0.25, 0.25, 0.75)],
        dtype=Atom.atom_dtype,
    )
    box_sizes = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    cell = np.diag(box_sizes[:, 1] - box_sizes[:, 0])
    origin = box_sizes[:, 0].copy()
    structure = StructureData(atoms, cell, origin)
    charges = {"U": 2.4, "O": -1.2}
    path = tmp_path / "out.data"

    LammpsDataWriter().write(path, structure, charges=charges)

    lines = path.read_text().splitlines()
    atoms_start = lines.index("Atoms") + 2
    assert lines[atoms_start] == "1 U 2.400000 0.000000 0.000000 0.000000"
    assert lines[atoms_start + 1] == "2 O -1.200000 0.250000 0.250000 0.250000"
    assert lines[atoms_start + 2] == "3 O -1.200000 0.250000 0.250000 0.750000"


def test_six_column_charge_form_type_as_int_mutates_charges_with_int_keys(
    tmp_path: Path,
) -> None:
    atoms = np.array(
        [("U", 0.0, 0.0, 0.0), ("O", 0.25, 0.25, 0.25)], dtype=Atom.atom_dtype
    )
    box_sizes = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    cell = np.diag(box_sizes[:, 1] - box_sizes[:, 0])
    origin = box_sizes[:, 0].copy()
    structure = StructureData(atoms, cell, origin)
    charges = {"U": 2.4, "O": -1.2}
    path = tmp_path / "out.data"

    LammpsDataWriter().write(path, structure, charges=charges, type_as_int=True)

    lines = path.read_text().splitlines()
    atoms_start = lines.index("Atoms") + 2
    # np.unique sorts names alphabetically ('O' before 'U'), so 'O' -> 1, 'U' -> 2.
    assert lines[atoms_start] == "1 2 2.400000 0.000000 0.000000 0.000000"
    assert lines[atoms_start + 1] == "2 1 -1.200000 0.250000 0.250000 0.250000"
    # Established side effect: name-keyed charges are mutated in place to also carry
    # the resolved numeric-type-ID keys.
    assert charges[2] == pytest.approx(2.4)
    assert charges[1] == pytest.approx(-1.2)


def test_triclinic_tilt_line(tmp_path: Path) -> None:
    atoms, _ = _orthogonal_structure()
    box_sizes = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    cell = np.diag(box_sizes[:, 1] - box_sizes[:, 0])
    cell[1, 0] = 1.5
    cell[2, 0] = -0.5
    cell[2, 1] = 0.25
    structure = StructureData(atoms, cell, box_sizes[:, 0])
    path = tmp_path / "out.data"

    LammpsDataWriter().write(path, structure, triclinic=True)

    lines = path.read_text().splitlines()
    tilt_line = next(line for line in lines if line.endswith("xy xz yz"))
    assert tilt_line == "1.500000 -0.500000 0.250000 xy xz yz"


def test_non_triclinic_omits_tilt_line(tmp_path: Path) -> None:
    _, structure = _orthogonal_structure()
    path = tmp_path / "out.data"

    LammpsDataWriter().write(path, structure, triclinic=False)

    content = path.read_text()
    assert "xy xz yz" not in content


def test_rejects_non_structure_data(tmp_path: Path) -> None:
    with pytest.raises(LammpsWriteError):
        LammpsDataWriter().write(tmp_path / "out.data", object())


def test_rejects_non_int_or_str_charge_keys(tmp_path: Path) -> None:
    _, structure = _orthogonal_structure()
    with pytest.raises(LammpsWriteError):
        LammpsDataWriter().write(
            tmp_path / "out.data", structure, charges={1.5: 2.0}
        )


def test_rejects_non_numeric_charge_values(tmp_path: Path) -> None:
    _, structure = _orthogonal_structure()
    with pytest.raises(LammpsWriteError):
        LammpsDataWriter().write(
            tmp_path / "out.data", structure, charges={"Cu": "not-a-number"}
        )


# --------------------------------------------------------------------------------------
# WriteResult metadata
# --------------------------------------------------------------------------------------


def test_write_result_declares_dropped_external_ids_and_periodicity(
    tmp_path: Path,
) -> None:
    atoms = np.array([("Cu", 0.0, 0.0, 0.0)], dtype=Atom.atom_dtype)
    cell = np.diag([10.0, 10.0, 10.0])
    origin = np.zeros(3)
    structure = StructureData(
        atoms,
        cell,
        origin,
        periodicity=(True, True, True),
        external_ids=np.array([42], dtype=np.int64),
        frame_index=3,
    )
    path = tmp_path / "out.data"

    result = LammpsDataWriter().write(path, structure)

    assert any("external_ids" in loss for loss in result.losses)
    assert any("periodicity" in loss for loss in result.losses)
    assert any("frame_index" in loss for loss in result.losses)
    # Sequential candidate-local IDs are assigned regardless of the source external_ids.
    np.testing.assert_array_equal(result.atom_ids, [1])


def test_write_result_digest_only_computed_when_requested(tmp_path: Path) -> None:
    _, structure = _orthogonal_structure()
    path = tmp_path / "out.data"

    without_digest = LammpsDataWriter().write(path, structure)
    assert without_digest.digest is None

    with_digest = LammpsDataWriter().write(path, structure, compute_digest=True)
    assert with_digest.digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_result_rejects_malformed_fields() -> None:
    with pytest.raises(StructureValueError):
        WriteResult("out.data", "", np.array([1], dtype=np.int64))
    with pytest.raises(StructureValueError):
        WriteResult("out.data", "lammps_data", np.array([[1]], dtype=np.int64))


# --------------------------------------------------------------------------------------
# Equivalence against GBMaker.write_lammps()'s current behavior
# --------------------------------------------------------------------------------------


def test_matches_gbmaker_write_lammps_orthogonal(tmp_path: Path) -> None:
    gb = _make_triclinic_gb()
    atoms = gb.whole_system
    box_sizes = gb.box_dims

    via_gbmaker = tmp_path / "via_gbmaker.data"
    gb.write_lammps(str(via_gbmaker), atoms, box_sizes)

    cell = np.diag(box_sizes[:, 1] - box_sizes[:, 0])
    structure = StructureData(atoms, cell, box_sizes[:, 0])
    via_writer = tmp_path / "via_writer.data"
    LammpsDataWriter().write(
        via_writer, structure, type_map=gb.unit_cell.type_map
    )

    assert via_gbmaker.read_text() == via_writer.read_text()


def test_matches_gbmaker_write_lammps_type_as_int_and_charges(tmp_path: Path) -> None:
    gb = _make_triclinic_gb()
    atoms = gb.whole_system
    box_sizes = gb.box_dims
    charges_for_gbmaker = {"Cu": 1.5}
    charges_for_writer = {"Cu": 1.5}

    via_gbmaker = tmp_path / "via_gbmaker.data"
    gb.write_lammps(
        str(via_gbmaker), atoms, box_sizes, type_as_int=True,
        charges=charges_for_gbmaker,
    )

    cell = np.diag(box_sizes[:, 1] - box_sizes[:, 0])
    structure = StructureData(atoms, cell, box_sizes[:, 0])
    via_writer = tmp_path / "via_writer.data"
    LammpsDataWriter().write(
        via_writer,
        structure,
        type_as_int=True,
        charges=charges_for_writer,
        type_map=gb.unit_cell.type_map,
    )

    assert via_gbmaker.read_text() == via_writer.read_text()


def test_matches_gbmaker_write_lammps_triclinic(tmp_path: Path) -> None:
    gb = _make_triclinic_gb()

    via_gbmaker = tmp_path / "via_gbmaker.data"
    gb.write_lammps(str(via_gbmaker), triclinic=True)

    # Reproduce GBMaker.write_lammps()'s triclinic pre-rotation independently, using
    # only its public surface plus the same private tilt-computation method its own
    # test suite (tests/test_gbmaker.py) already exercises via name-mangled access.
    atoms = gb.whole_system.copy()
    box_sizes = gb.box_dims
    xy, xz, yz, theta = gb._GBMaker__get_triclinic_params()
    cell = np.diag(box_sizes[:, 1] - box_sizes[:, 0])
    cell[1, 0] = xy
    cell[2, 0] = xz
    cell[2, 1] = yz
    ct, st = math.cos(theta), math.sin(theta)
    Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
    rotated = (Rx @ positions.T).T
    atoms["x"], atoms["y"], atoms["z"] = rotated.T
    structure = StructureData(atoms, cell, box_sizes[:, 0])

    via_writer = tmp_path / "via_writer.data"
    LammpsDataWriter().write(
        via_writer, structure, type_map=gb.unit_cell.type_map, triclinic=True
    )

    assert via_gbmaker.read_text() == via_writer.read_text()


# --------------------------------------------------------------------------------------
# Import-boundary: the writer must not import the GBMaker construction facade
# --------------------------------------------------------------------------------------


def test_data_writer_module_does_not_import_gbmaker() -> None:
    import subprocess
    import sys

    # GBOpt/__init__.py itself unconditionally imports GBOpt.GBMaker (it re-exports
    # GBMaker as part of the package's public surface), so a plain
    # "import GBOpt.io.lammps.data_writer; assert 'GBOpt.GBMaker' not in sys.modules"
    # would fail regardless of what data_writer.py itself imports -- the parent
    # package's __init__.py always runs first and pulls GBMaker in on its own. To
    # isolate what data_writer's *own* import graph pulls in, stub out "GBOpt" with an
    # empty module (its on-disk __path__ preserved via importlib.util.find_spec, which
    # locates the package without executing it) before importing the submodule, so
    # GBOpt/__init__.py's body never runs in this subprocess.
    script = (
        "import importlib.util, sys, types\n"
        "spec = importlib.util.find_spec('GBOpt')\n"
        "stub = types.ModuleType('GBOpt')\n"
        "stub.__path__ = spec.submodule_search_locations\n"
        "sys.modules['GBOpt'] = stub\n"
        "import GBOpt.io.lammps.data_writer\n"
        "assert 'GBOpt.GBMaker' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
