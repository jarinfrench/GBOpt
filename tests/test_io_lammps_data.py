# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from GBOpt.io.lammps.data import LammpsDataReader
from GBOpt.io.lammps.types import LammpsDataError


def _write_data_file(
    path: Path,
    rows: list[str],
    *,
    n_atoms: int | None = None,
    n_types: int = 2,
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.0, 10.0), (0.0, 10.0), (0.0, 10.0),
    ),
    extra_after_atoms: list[str] | None = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("Synthetic structure\n\n")
        stream.write(f"{n_atoms if n_atoms is not None else len(rows)} atoms\n")
        stream.write(f"{n_types} atom types\n")
        for axis, (lower, upper) in zip("xyz", box, strict=True):
            stream.write(f"{lower} {upper} {axis}lo {axis}hi\n")
        stream.write("\nAtoms\n\n")
        for row in rows:
            stream.write(f"{row}\n")
        for line in extra_after_atoms or []:
            stream.write(f"{line}\n")


def test_reads_five_column_form_and_preserves_file_row_order(tmp_path: Path) -> None:
    path = tmp_path / "data.lmp"
    _write_data_file(path, ["2 2 3.0 1.0 7.0", "1 1 7.0 2.0 8.0"])

    structure = LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})

    assert np.array_equal(structure.external_ids, np.array([2, 1]))
    assert structure.atoms[0]["name"] == "O"
    assert structure.atoms[0]["x"] == pytest.approx(3.0)
    assert structure.atoms[1]["name"] == "U"
    assert structure.atoms[1]["z"] == pytest.approx(8.0)
    assert structure.charges is None
    assert structure.periodicity is None
    assert structure.frame_index is None
    assert np.array_equal(np.diagonal(structure.cell), [10.0, 10.0, 10.0])
    assert np.array_equal(structure.origin, [0.0, 0.0, 0.0])


def test_reads_six_column_form_and_captures_charges(tmp_path: Path) -> None:
    path = tmp_path / "charged.lmp"
    _write_data_file(path, ["1 1 -1.5 2.0 3.0 4.0", "2 2 0.75 5.0 6.0 7.0"])

    structure = LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})

    assert structure.charges is not None
    assert np.array_equal(structure.charges, [-1.5, 0.75])


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LammpsDataReader().read(tmp_path / "missing.lmp")


def test_rejects_triclinic_box(tmp_path: Path) -> None:
    path = tmp_path / "triclinic.lmp"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("Triclinic\n\n1 atoms\n1 atom types\n")
        stream.write("0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n")
        stream.write("1.0 2.0 3.0 xy xz yz\n\nAtoms\n\n1 1 1.0 1.0 1.0\n")

    with pytest.raises(LammpsDataError):
        LammpsDataReader().read(path, type_dict={"Ni": 1})


def test_rejects_duplicate_atom_ids(tmp_path: Path) -> None:
    path = tmp_path / "dup.lmp"
    _write_data_file(path, ["1 1 1.0 1.0 1.0", "1 2 2.0 2.0 2.0"])

    with pytest.raises(LammpsDataError, match="unique"):
        LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})


def test_rejects_row_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "missing_row.lmp"
    _write_data_file(path, ["1 1 1.0 1.0 1.0"], n_atoms=2)

    with pytest.raises(LammpsDataError, match="expected 2 atom rows"):
        LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})


def test_column_permutation_type_label_vs_numeric_type(tmp_path: Path) -> None:
    numeric_path = tmp_path / "numeric.lmp"
    label_path = tmp_path / "label.lmp"
    _write_data_file(numeric_path, ["1 1 1.0 1.0 1.0"])
    _write_data_file(label_path, ["1 U 1.0 1.0 1.0"])

    numeric = LammpsDataReader().read(numeric_path, type_dict={"U": 1, "O": 2})
    label = LammpsDataReader().read(label_path, type_dict={"U": 1, "O": 2})

    assert numeric.atoms["name"][0] == label.atoms["name"][0] == "U"


def test_type_dict_id_to_name_and_name_to_id_forms_agree(tmp_path: Path) -> None:
    path = tmp_path / "typed.lmp"
    _write_data_file(path, ["1 2 1.0 1.0 1.0"])

    by_id = LammpsDataReader().read(path, type_dict={1: "U", 2: "O"})
    by_name = LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})

    assert by_id.atoms["name"][0] == by_name.atoms["name"][0] == "O"


def test_rejects_non_finite_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.lmp"
    _write_data_file(path, ["1 1 nan 1.0 1.0"])

    with pytest.raises(LammpsDataError):
        LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})


def test_rejects_non_finite_charge(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite_charge.lmp"
    _write_data_file(path, ["1 1 nan 1.0 1.0 1.0"])

    with pytest.raises(LammpsDataError, match="charge must be finite"):
        LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})


def test_rejects_unknown_type_id_without_type_dict(tmp_path: Path) -> None:
    path = tmp_path / "unknown_type.lmp"
    _write_data_file(path, ["1 9999 1.0 1.0 1.0"], n_types=9999)

    with pytest.raises(LammpsDataError):
        LammpsDataReader().read(path)


def test_rejects_extra_content_after_atoms_section(tmp_path: Path) -> None:
    path = tmp_path / "extra.lmp"
    _write_data_file(path, ["1 1 1.0 1.0 1.0"], extra_after_atoms=["2 2 2.0 2.0 2.0"])

    with pytest.raises(LammpsDataError, match="unexpected extra content"):
        LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})


def test_accepts_standard_section_after_atoms(tmp_path: Path) -> None:
    path = tmp_path / "with_masses.lmp"
    _write_data_file(path, ["1 1 1.0 1.0 1.0"], extra_after_atoms=["Masses", "", "1 1.0"])

    structure = LammpsDataReader().read(path, type_dict={"U": 1, "O": 2})
    assert structure.atoms.size == 1
