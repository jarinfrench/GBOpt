# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

from pathlib import Path

import pytest

from GBOpt.io.lammps.dispatch import read_structure_file


def _write_data_file(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("Data\n\n1 atoms\n1 atom types\n")
        stream.write("0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n")
        stream.write("\nAtoms\n\n1 1 1.0 1.0 1.0\n")


def _write_dump_file(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n")
        stream.write("ITEM: BOX BOUNDS pp pp pp\n")
        stream.write("0.0 10.0\n0.0 10.0\n0.0 10.0\n")
        stream.write("ITEM: ATOMS id typelabel x y z\n1 U 1.0 1.0 1.0\n")


def test_dispatches_data_file_to_data_reader(tmp_path: Path) -> None:
    path = tmp_path / "structure.data"
    _write_data_file(path)

    structure = read_structure_file(path, type_dict={"U": 1})

    assert structure.frame_index is None
    assert structure.periodicity is None


def test_dispatches_dump_file_to_dump_reader(tmp_path: Path) -> None:
    path = tmp_path / "structure.dump"
    _write_dump_file(path)

    structure = read_structure_file(path)

    assert structure.frame_index == 0
    assert structure.periodicity == (True, True, True)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_structure_file(tmp_path / "missing.lmp")
