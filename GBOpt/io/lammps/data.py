# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""``LammpsDataReader``: parses the orthogonal LAMMPS data format into ``StructureData``.

This module is a leaf with respect to its ``io.lammps`` siblings: it depends only on
``tokens`` and ``types`` (plus the generic ``GBOpt.io.types``), and does not import
``dump`` or ``dispatch``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from GBOpt.Atom import Atom
from GBOpt.io.lammps.tokens import (
    _INTEGER_TOKEN,
    _normalize_type_mapping,
    _resolve_type_id_species,
    _strict_id_token,
    _strict_species_name,
    _strict_type_id,
)
from GBOpt.io.lammps.types import LammpsDataError
from GBOpt.io.types import StructureData

_SECTION_HEADERS = {
    "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
    "Masses", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
    "Dihedral Coeffs", "Improper Coeffs",
}


class LammpsDataReader:
    """Reads the orthogonal LAMMPS data format emitted by ``GBMaker``.

    Both the five-column atomic form and the six-column charge form are supported. File
    row order is preserved so callers can align ownership by atom ID.
    """

    def read(
        self, path: str | Path, *, type_dict: Mapping[object, object] | None = None
    ) -> StructureData:
        """Read one LAMMPS data file into a neutral structure snapshot.

        :param path: Path to the LAMMPS data file.
        :param type_dict: Keyword argument, optional, defaults to ``None``. Mapping in
            ``species -> type ID`` or ``type ID -> species`` form.
        :return: The parsed neutral structure. ``charges`` is populated only when every
            atom row in the file supplied a charge column; otherwise it is ``None``.
        :raises FileNotFoundError: If ``path`` does not identify an existing file.
        :raises LammpsDataError: If the file or type mapping is malformed, ambiguous,
            unsupported, or inconsistent with its declared counts and bounds.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(str(file_path))
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 6:
            raise LammpsDataError("LAMMPS data file is too short")

        n_atoms: int | None = None
        n_types: int | None = None
        box: dict[str, tuple[float, float]] = {}
        id_to_name = _normalize_type_mapping(type_dict)
        atoms_line: int | None = None
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            parts = stripped.split()
            if len(parts) == 2 and parts[1] == "atoms":
                try:
                    n_atoms = int(parts[0])
                except ValueError as exc:
                    raise LammpsDataError("invalid atom count") from exc
            elif len(parts) == 3 and parts[1:] == ["atom", "types"]:
                try:
                    n_types = int(parts[0])
                except ValueError as exc:
                    raise LammpsDataError("invalid atom-type count") from exc
            elif len(parts) >= 4 and parts[-2:] in (
                ["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]
            ):
                axis = parts[-2][0]
                try:
                    bounds = (float(parts[0]), float(parts[1]))
                except ValueError as exc:
                    raise LammpsDataError(f"invalid {axis} box bounds") from exc
                box[axis] = bounds
            elif len(parts) >= 6 and parts[-3:] == ["xy", "xz", "yz"]:
                raise LammpsDataError(
                    "explicit ownership supports orthogonal LAMMPS data boxes only"
                )
            elif stripped == "Atom Type Labels":
                index += 1
                while index < len(lines) and not lines[index].strip():
                    index += 1
                while index < len(lines):
                    label_parts = lines[index].strip().split()
                    if not label_parts:
                        break
                    if (
                        len(label_parts) != 2
                        or not _INTEGER_TOKEN.fullmatch(label_parts[0])
                    ):
                        break
                    type_id = _strict_type_id(int(label_parts[0]))
                    if n_types is not None and type_id > n_types:
                        raise LammpsDataError(
                            f"atom type id {type_id} exceeds declared atom-type "
                            f"count {n_types}"
                        )

                    species = _strict_species_name(label_parts[1])
                    # ``Atom Type Labels`` is file-local serialization metadata and is
                    # therefore authoritative for numeric type tokens in this file. A
                    # caller-provided ``type_dict`` is only a fallback for files without
                    # labels. In particular, evaluator artifacts may assign different
                    # transient numeric type IDs while emitting named atom rows;
                    # persistent species identity is validated per atom ID later by
                    # ``CandidateFileMapping`` rather than by requiring one global
                    # numeric type assignment across evaluations.
                    id_to_name[type_id] = species
                    index += 1
                continue
            elif stripped.startswith("Atoms"):
                atoms_line = index
                break
            index += 1

        if n_atoms is None or n_atoms < 0:
            raise LammpsDataError("missing or invalid atom count")
        if n_types is None or n_types <= 0:
            raise LammpsDataError("missing or invalid atom-type count")
        if set(box) != {"x", "y", "z"}:
            raise LammpsDataError("missing orthogonal box bounds")
        box_dims = np.asarray([box[axis] for axis in "xyz"], dtype=float)
        if not np.all(np.isfinite(box_dims)) or np.any(box_dims[:, 0] >= box_dims[:, 1]):
            raise LammpsDataError("box bounds must be finite and strictly ordered")
        if atoms_line is None:
            raise LammpsDataError("missing Atoms section")

        inverse_default = {number: name for name, number in Atom._numbers.items()}
        rows: list[tuple[str, float, float, float]] = []
        ids: list[int] = []
        row_charges: list[float | None] = []
        index = atoms_line + 1
        while index < len(lines) and len(rows) < n_atoms:
            stripped = lines[index].split("#", 1)[0].strip()
            index += 1
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) not in (5, 6):
                raise LammpsDataError(
                    "Atoms rows must use 'id type x y z' or 'id type charge x y z'"
                )
            atom_id = _strict_id_token(parts[0])
            type_token = parts[1]
            coordinate_start = 2 if len(parts) == 5 else 3
            if len(parts) == 6:
                try:
                    charge = float(parts[2])
                except ValueError as exc:
                    raise LammpsDataError("atom charge must be numeric") from exc
                if not np.isfinite(charge):
                    raise LammpsDataError("atom charge must be finite")
            else:
                charge = None
            if _INTEGER_TOKEN.fullmatch(type_token):
                type_id = _strict_type_id(int(type_token))
                if type_id > n_types:
                    raise LammpsDataError(
                        f"atom type id {type_id} exceeds declared atom-type count "
                        f"{n_types}"
                    )

                name = _resolve_type_id_species(type_id, id_to_name, inverse_default)
            else:
                name = _strict_species_name(type_token)
            try:
                coordinates = tuple(
                    float(value)
                    for value in parts[coordinate_start:coordinate_start + 3]
                )
            except ValueError as exc:
                raise LammpsDataError("atom coordinates must be numeric") from exc
            if len(coordinates) != 3 or not np.all(np.isfinite(coordinates)):
                raise LammpsDataError(
                    "atom coordinates must contain three finite values"
                )
            ids.append(atom_id)
            rows.append((name, *coordinates))
            row_charges.append(charge)

        if len(rows) != n_atoms:
            raise LammpsDataError(f"expected {n_atoms} atom rows, found {len(rows)}")
        # The handoff format emitted by GBMaker ends after Atoms.  Reject additional
        # atom-like records instead of silently trusting a header count that
        # understates the serialized structure.  Standard later LAMMPS sections
        # remain acceptable.
        for remaining in lines[index:]:
            stripped = remaining.split("#", 1)[0].strip()
            if not stripped:
                continue
            if stripped in _SECTION_HEADERS or any(
                stripped.startswith(f"{header} ") for header in _SECTION_HEADERS
            ):
                break
            raise LammpsDataError("unexpected extra content in the Atoms section")

        atom_ids = np.asarray(ids, dtype=np.int64)
        if np.unique(atom_ids).size != atom_ids.size:
            raise LammpsDataError("atom IDs must be unique")
        atoms = np.asarray(rows, dtype=Atom.atom_dtype)
        if len(np.unique(atoms["name"])) > n_types:
            raise LammpsDataError(
                "atom rows contain more species than declared atom types"
            )

        cell = np.diag(box_dims[:, 1] - box_dims[:, 0])
        origin = box_dims[:, 0].copy()
        charges = (
            np.asarray(row_charges, dtype=float)
            if row_charges and all(value is not None for value in row_charges)
            else None
        )
        return StructureData(
            atoms,
            cell,
            origin,
            external_ids=atom_ids,
            charges=charges,
        )
