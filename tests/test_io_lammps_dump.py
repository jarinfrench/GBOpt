# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from GBOpt.io.lammps.dump import LammpsDumpReader
from GBOpt.io.lammps.types import LammpsDataError


def _write_dump_frame(
    stream,
    *,
    timestep: int,
    n_atoms: int,
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    bounds_flags: str,
    attributes: str,
    rows: list[str],
) -> None:
    stream.write("ITEM: TIMESTEP\n")
    stream.write(f"{timestep}\n")
    stream.write("ITEM: NUMBER OF ATOMS\n")
    stream.write(f"{n_atoms}\n")
    stream.write(f"ITEM: BOX BOUNDS {bounds_flags}\n")
    for lower, upper in box:
        stream.write(f"{lower} {upper}\n")
    stream.write(f"ITEM: ATOMS {attributes}\n")
    for row in rows:
        stream.write(f"{row}\n")


_DEFAULT_BOX = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))


def test_reads_first_frame_with_typelabel(tmp_path: Path) -> None:
    path = tmp_path / "dump.lmp"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        _write_dump_frame(
            stream,
            timestep=0,
            n_atoms=2,
            box=_DEFAULT_BOX,
            bounds_flags="pp pp pp",
            attributes="id typelabel x y z",
            rows=["2 O 3.0 1.0 7.0", "1 U 7.0 2.0 8.0"],
        )

    structure = LammpsDumpReader().read(path)

    assert np.array_equal(structure.external_ids, np.array([2, 1]))
    assert structure.atoms[0]["name"] == "O"
    assert structure.periodicity == (True, True, True)
    assert structure.frame_index == 0
    assert structure.charges is None
    assert np.array_equal(np.diagonal(structure.cell), [10.0, 10.0, 10.0])


def test_ignores_later_frames_even_when_first_frame_valid(tmp_path: Path) -> None:
    path = tmp_path / "multi.lmp"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        _write_dump_frame(
            stream,
            timestep=0,
            n_atoms=1,
            box=_DEFAULT_BOX,
            bounds_flags="pp pp pp",
            attributes="id typelabel x y z",
            rows=["1 U 1.0 1.0 1.0"],
        )
        # Second frame is malformed; must not affect first-frame parsing.
        stream.write("ITEM: TIMESTEP\n1\nITEM: garbage\n")

    structure = LammpsDumpReader().read(path)
    assert structure.atoms.size == 1


def test_malformed_first_frame_fails_even_if_later_frame_valid(tmp_path: Path) -> None:
    path = tmp_path / "bad_first.lmp"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\nnot-a-number\n")
        _write_dump_frame(
            stream,
            timestep=1,
            n_atoms=1,
            box=_DEFAULT_BOX,
            bounds_flags="pp pp pp",
            attributes="id typelabel x y z",
            rows=["1 U 1.0 1.0 1.0"],
        )

    with pytest.raises(LammpsDataError):
        LammpsDumpReader().read(path)


def test_rejects_unexpected_content_between_frames(tmp_path: Path) -> None:
    path = tmp_path / "extra_row.lmp"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        _write_dump_frame(
            stream,
            timestep=0,
            n_atoms=1,
            box=_DEFAULT_BOX,
            bounds_flags="pp pp pp",
            attributes="id typelabel x y z",
            rows=["1 U 1.0 1.0 1.0", "2 U 2.0 2.0 2.0"],
        )

    with pytest.raises(LammpsDataError):
        LammpsDumpReader().read(path)


def test_rejects_duplicate_ids_in_first_frame(tmp_path: Path) -> None:
    path = tmp_path / "dup.lmp"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        _write_dump_frame(
            stream,
            timestep=0,
            n_atoms=2,
            box=_DEFAULT_BOX,
            bounds_flags="pp pp pp",
            attributes="id typelabel x y z",
            rows=["1 U 1.0 1.0 1.0", "1 O 2.0 2.0 2.0"],
        )

    with pytest.raises(LammpsDataError, match="unique"):
        LammpsDumpReader().read(path)


def test_requires_unambiguous_boundary_flags(tmp_path: Path) -> None:
    path = tmp_path / "no_flags.lmp"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        _write_dump_frame(
            stream,
            timestep=0,
            n_atoms=1,
            box=_DEFAULT_BOX,
            bounds_flags="",
            attributes="id typelabel x y z",
            rows=["1 U 1.0 1.0 1.0"],
        )

    structure = LammpsDumpReader().read(path)
    assert structure.periodicity is None


def test_column_permutation_numeric_type_vs_typelabel(tmp_path: Path) -> None:
    numeric_path = tmp_path / "numeric.lmp"
    label_path = tmp_path / "label.lmp"
    with numeric_path.open("w", encoding="utf-8", newline="\n") as stream:
        _write_dump_frame(
            stream, timestep=0, n_atoms=1, box=_DEFAULT_BOX, bounds_flags="pp pp pp",
            attributes="id type x y z", rows=["1 1 1.0 1.0 1.0"],
        )
    with label_path.open("w", encoding="utf-8", newline="\n") as stream:
        _write_dump_frame(
            stream, timestep=0, n_atoms=1, box=_DEFAULT_BOX, bounds_flags="pp pp pp",
            attributes="id typelabel x y z", rows=["1 U 1.0 1.0 1.0"],
        )

    numeric = LammpsDumpReader().read(numeric_path, type_dict={1: "U"})
    label = LammpsDumpReader().read(label_path)

    assert numeric.atoms["name"][0] == label.atoms["name"][0] == "U"


def test_rejects_non_finite_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.lmp"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        _write_dump_frame(
            stream, timestep=0, n_atoms=1, box=_DEFAULT_BOX, bounds_flags="pp pp pp",
            attributes="id typelabel x y z", rows=["1 U nan 1.0 1.0"],
        )

    with pytest.raises(LammpsDataError):
        LammpsDumpReader().read(path)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LammpsDumpReader().read(tmp_path / "missing.lmp")
