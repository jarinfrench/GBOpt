# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Old/new LAMMPS parser equivalence tests (R11, issue #72).

Exercises the pre-R11-shaped compatibility wrappers (``LammpsAtomData``-returning,
now living in ``GBOpt.io.lammps.compat`` and re-exported unchanged from
``GBOpt.FileGrainOwnership``) side by side with the new ``StructureData``-returning
readers, across the acceptance criteria's required cases: malformed files,
duplicate/missing IDs, column permutations, type maps, non-finite values, and
multi-frame dumps. The two call shapes must always agree: same exception (or none),
same atom IDs, species, coordinates, and box geometry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from GBOpt.io.lammps.compat import (
    read_lammps_data_file,
    read_lammps_dump_file,
)
from GBOpt.io.lammps.data import LammpsDataReader
from GBOpt.io.lammps.dump import LammpsDumpReader
from GBOpt.io.lammps.types import LammpsAtomData, LammpsDataError
from GBOpt.io.types import StructureData


def _assert_equivalent(structure: StructureData, legacy: LammpsAtomData) -> None:
    assert np.array_equal(structure.external_ids, legacy.atom_ids)
    np.testing.assert_array_equal(structure.atoms, legacy.atoms)
    widths = np.diagonal(structure.cell)
    expected_box = np.stack([structure.origin, structure.origin + widths], axis=1)
    np.testing.assert_allclose(expected_box, legacy.box_dims)
    assert structure.periodicity == legacy.boundary_periodic
    assert structure.frame_index == legacy.selected_frame


def _write_data_file(
    path: Path, rows: list[str], *, n_types: int = 2, n_atoms: int | None = None
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("Synthetic\n\n")
        stream.write(f"{n_atoms if n_atoms is not None else len(rows)} atoms\n")
        stream.write(f"{n_types} atom types\n")
        stream.write("0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n")
        stream.write("\nAtoms\n\n")
        for row in rows:
            stream.write(f"{row}\n")


def _write_dump_file(
    path: Path, rows: list[str], *, n_atoms: int | None = None, second_frame: bool = False
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n")
        stream.write(f"{n_atoms if n_atoms is not None else len(rows)}\n")
        stream.write("ITEM: BOX BOUNDS pp pp pp\n")
        stream.write("0.0 10.0\n0.0 10.0\n0.0 10.0\n")
        stream.write("ITEM: ATOMS id typelabel x y z\n")
        for row in rows:
            stream.write(f"{row}\n")
        if second_frame:
            stream.write("ITEM: TIMESTEP\n1\nITEM: NUMBER OF ATOMS\n0\n")
            stream.write("ITEM: BOX BOUNDS pp pp pp\n0.0 1.0\n0.0 1.0\n0.0 1.0\n")
            stream.write("ITEM: ATOMS id typelabel x y z\n")


@pytest.mark.parametrize(
    "rows,type_dict",
    [
        (["1 1 1.0 1.0 1.0", "2 2 2.0 2.0 2.0"], {"U": 1, "O": 2}),
        (["2 U 1.0 1.0 1.0", "1 O 2.0 2.0 2.0"], None),
        (["1 O 1.0 1.0 1.0", "2 1 2.0 2.0 2.0"], {"U": 1, "O": 2}),
    ],
)
def test_data_reader_and_compat_wrapper_agree_on_valid_files(
    tmp_path: Path, rows: list[str], type_dict: dict[str, int] | None
) -> None:
    path = tmp_path / "data.lmp"
    _write_data_file(path, rows)

    structure = LammpsDataReader().read(path, type_dict=type_dict)
    legacy = read_lammps_data_file(path, type_dict=type_dict)

    _assert_equivalent(structure, legacy)


@pytest.mark.parametrize(
    "rows,type_dict",
    [
        pytest.param(["1 1 1.0 1.0 1.0", "1 2 2.0 2.0 2.0"], {"U": 1, "O": 2}, id="duplicate-ids"),
        pytest.param(["1 1 nan 1.0 1.0"], {"U": 1, "O": 2}, id="non-finite"),
        pytest.param(["1 1 1.0 1.0"], {"U": 1, "O": 2}, id="short-row"),
        pytest.param(["1 999 1.0 1.0 1.0"], None, id="unknown-type"),
    ],
)
def test_data_reader_and_compat_wrapper_agree_on_malformed_files(
    tmp_path: Path, rows: list[str], type_dict: dict[str, int]
) -> None:
    path = tmp_path / "bad.lmp"
    _write_data_file(path, rows, n_types=2)

    with pytest.raises(LammpsDataError):
        LammpsDataReader().read(path, type_dict=type_dict)
    with pytest.raises(LammpsDataError):
        read_lammps_data_file(path, type_dict=type_dict)


def test_data_reader_and_compat_wrapper_agree_on_declared_row_count_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing_row.lmp"
    _write_data_file(path, ["1 1 1.0 1.0 1.0"], n_types=2, n_atoms=2)

    with pytest.raises(LammpsDataError):
        LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})
    with pytest.raises(LammpsDataError):
        read_lammps_data_file(path, type_dict={"U": 1, "O": 2})


def test_data_reader_and_compat_wrapper_agree_across_type_map_directions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "typed.lmp"
    _write_data_file(path, ["1 2 1.0 1.0 1.0"])

    for type_dict in ({"U": 1, "O": 2}, {1: "U", 2: "O"}):
        structure = LammpsDataReader().read(path, type_dict=type_dict)
        legacy = read_lammps_data_file(path, type_dict=type_dict)
        _assert_equivalent(structure, legacy)
        assert structure.atoms["name"][0] == "O"


def test_dump_reader_and_compat_wrapper_agree_on_first_frame(tmp_path: Path) -> None:
    path = tmp_path / "dump.lmp"
    _write_dump_file(path, ["2 O 3.0 1.0 7.0", "1 U 7.0 2.0 8.0"], second_frame=True)

    structure = LammpsDumpReader().read(path)
    legacy = read_lammps_dump_file(path)

    _assert_equivalent(structure, legacy)
    assert structure.frame_index == 0


@pytest.mark.parametrize(
    "rows,n_atoms",
    [
        pytest.param(["1 U 1.0 1.0 1.0", "1 O 2.0 2.0 2.0"], None, id="duplicate-ids"),
        pytest.param(["1 U 1.0 1.0 1.0"], 2, id="missing-row"),
        pytest.param(["1 U nan 1.0 1.0"], None, id="non-finite"),
    ],
)
def test_dump_reader_and_compat_wrapper_agree_on_malformed_frames(
    tmp_path: Path, rows: list[str], n_atoms: int | None
) -> None:
    path = tmp_path / "bad_dump.lmp"
    _write_dump_file(path, rows, n_atoms=n_atoms)

    with pytest.raises(LammpsDataError):
        LammpsDumpReader().read(path)
    with pytest.raises(LammpsDataError):
        read_lammps_dump_file(path)
