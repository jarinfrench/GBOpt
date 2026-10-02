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


def _parse_atom_type_labels(
    lines: list[str], index: int, n_types: int | None, id_to_name: dict[int, str]
) -> int:
    """Parse an ``Atom Type Labels`` block starting at ``index``.

    :param lines: Full file content, split into lines.
    :param index: Index of the first line after the ``Atom Type Labels`` header.
    :param n_types: Declared atom-type count, if already parsed.
    :param id_to_name: Type-ID-to-species mapping; mutated in place with each entry.
    :return: The index of the first line after the parsed block.
    :raises LammpsDataError: If a labeled type ID exceeds ``n_types``, or a species
        name is invalid.
    """
    while index < len(lines) and not lines[index].strip():
        index += 1
    while index < len(lines):
        label_parts = lines[index].strip().split()
        if not label_parts:
            break
        if len(label_parts) != 2 or not _INTEGER_TOKEN.fullmatch(label_parts[0]):
            break
        type_id = _strict_type_id(int(label_parts[0]))
        if n_types is not None and type_id > n_types:
            raise LammpsDataError(
                f"atom type id {type_id} exceeds declared atom-type count {n_types}"
            )

        species = _strict_species_name(label_parts[1])
        # ``Atom Type Labels`` is file-local serialization metadata and is therefore
        # authoritative for numeric type tokens in this file. A caller-provided
        # ``type_dict`` is only a fallback for files without labels. In particular,
        # evaluator artifacts may assign different transient numeric type IDs while
        # emitting named atom rows; persistent species identity is validated per atom
        # ID later by ``CandidateFileMapping`` rather than by requiring one global
        # numeric type assignment across evaluations.
        id_to_name[type_id] = species
        index += 1
    return index


def _parse_count_token(token: str, label: str) -> int:
    """Parse a leading integer count token, raising with a descriptive message.

    :param token: The token expected to hold an integer count.
    :param label: Human-readable name of the count, used in the error message.
    :return: The parsed integer.
    :raises LammpsDataError: If ``token`` is not a valid integer.
    """
    try:
        return int(token)
    except ValueError as exc:
        raise LammpsDataError(f"invalid {label}") from exc


def _parse_box_bound(parts: list[str]) -> tuple[str, tuple[float, float]]:
    """Parse one ``<lo> <hi> <axis>lo <axis>hi`` box-bound header line.

    :param parts: The line's whitespace-split tokens.
    :return: ``(axis, (lower, upper))``.
    :raises LammpsDataError: If the bound values are not valid floats.
    """
    axis = parts[-2][0]
    try:
        bounds = (float(parts[0]), float(parts[1]))
    except ValueError as exc:
        raise LammpsDataError(f"invalid {axis} box bounds") from exc
    return axis, bounds


def _parse_header_lines(
    lines: list[str], id_to_name: dict[int, str]
) -> tuple[int | None, int | None, dict[str, tuple[float, float]], int | None]:
    """Scan header lines for the atom/type counts, box bounds, and type labels.

    :param lines: Full file content, split into lines.
    :param id_to_name: Type-ID-to-species mapping; mutated in place with any
        ``Atom Type Labels`` entries found.
    :return: ``(n_atoms, n_types, box, atoms_line)`` as parsed so far; ``atoms_line``
        is the index of the line beginning with ``Atoms``, or ``None`` if the header
        ends without one.
    :raises LammpsDataError: If a count or box bound is malformed, or the box is
        triclinic.
    """
    n_atoms: int | None = None
    n_types: int | None = None
    box: dict[str, tuple[float, float]] = {}
    atoms_line: int | None = None
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        parts = stripped.split()
        if len(parts) == 2 and parts[1] == "atoms":
            n_atoms = _parse_count_token(parts[0], "atom count")
        elif len(parts) == 3 and parts[1:] == ["atom", "types"]:
            n_types = _parse_count_token(parts[0], "atom-type count")
        elif len(parts) >= 4 and parts[-2:] in (
            ["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]
        ):
            axis, bounds = _parse_box_bound(parts)
            box[axis] = bounds
        elif len(parts) >= 6 and parts[-3:] == ["xy", "xz", "yz"]:
            raise LammpsDataError(
                "explicit ownership supports orthogonal LAMMPS data boxes only"
            )
        elif stripped == "Atom Type Labels":
            index = _parse_atom_type_labels(lines, index + 1, n_types, id_to_name)
            continue
        elif stripped.startswith("Atoms"):
            atoms_line = index
            break
        index += 1
    return n_atoms, n_types, box, atoms_line


def _validate_header(
    n_atoms: int | None,
    n_types: int | None,
    box: dict[str, tuple[float, float]],
    atoms_line: int | None,
) -> np.ndarray:
    """Validate parsed header fields and compute the per-axis box bounds.

    :param n_atoms: Parsed atom count, or ``None`` if not found.
    :param n_types: Parsed atom-type count, or ``None`` if not found.
    :param box: Per-axis ``(lower, upper)`` bounds parsed from the header.
    :param atoms_line: Index of the ``Atoms`` section header, or ``None`` if not
        found.
    :return: A ``(3, 2)`` array of box bounds, ordered ``x, y, z``.
    :raises LammpsDataError: If any header field is missing or invalid.
    """
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
    return box_dims


def _parse_atom_row(
    stripped: str,
    n_types: int,
    id_to_name: dict[int, str],
    inverse_default: dict[int, str],
) -> tuple[tuple[str, float, float, float], int, float | None]:
    """Parse one ``Atoms`` section row, with any trailing comment already removed.

    :param stripped: The row text, stripped and with its trailing ``#`` comment (if
        any) removed.
    :param n_types: Declared atom-type count, for numeric type-ID bounds checking.
    :param id_to_name: Type-ID-to-species mapping accumulated from the header.
    :param inverse_default: Fallback type-ID-to-species mapping from
        ``Atom._numbers``.
    :return: ``((species, x, y, z), atom_id, charge)``; ``charge`` is ``None`` for the
        five-column atomic form.
    :raises LammpsDataError: If the row's column count, type token, coordinates, or
        charge are malformed.
    """
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
                f"atom type id {type_id} exceeds declared atom-type count {n_types}"
            )

        name = _resolve_type_id_species(type_id, id_to_name, inverse_default)
    else:
        name = _strict_species_name(type_token)
    try:
        coordinates = tuple(
            float(value) for value in parts[coordinate_start:coordinate_start + 3]
        )
    except ValueError as exc:
        raise LammpsDataError("atom coordinates must be numeric") from exc
    if len(coordinates) != 3 or not np.all(np.isfinite(coordinates)):
        raise LammpsDataError("atom coordinates must contain three finite values")
    return (name, *coordinates), atom_id, charge


def _parse_atoms_section(
    lines: list[str],
    atoms_line: int,
    n_atoms: int,
    n_types: int,
    id_to_name: dict[int, str],
) -> tuple[list[tuple[str, float, float, float]], list[int], list[float | None], int]:
    """Parse up to ``n_atoms`` rows following the ``Atoms`` section header.

    :param lines: Full file content, split into lines.
    :param atoms_line: Index of the ``Atoms`` section header line.
    :param n_atoms: Declared atom count; parsing stops once this many rows are read.
    :param n_types: Declared atom-type count, for numeric type-ID bounds checking.
    :param id_to_name: Type-ID-to-species mapping accumulated from the header.
    :return: ``(rows, ids, row_charges, index)``, where ``index`` is the line index
        immediately after the parsed rows.
    :raises LammpsDataError: If any row is malformed.
    """
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
        row, atom_id, charge = _parse_atom_row(
            stripped, n_types, id_to_name, inverse_default
        )
        ids.append(atom_id)
        rows.append(row)
        row_charges.append(charge)
    return rows, ids, row_charges, index


def _check_trailing_content(lines: list[str], index: int) -> None:
    """Reject unexpected content following the parsed ``Atoms`` rows.

    :param lines: Full file content, split into lines.
    :param index: Line index immediately after the parsed ``Atoms`` rows.
    :raises LammpsDataError: If a non-blank, non-section-header line follows.
    """
    # The handoff format emitted by GBMaker ends after Atoms.  Reject additional
    # atom-like records instead of silently trusting a header count that
    # understates the serialized structure.  Standard later LAMMPS sections remain
    # acceptable.
    for remaining in lines[index:]:
        stripped = remaining.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped in _SECTION_HEADERS or any(
            stripped.startswith(f"{header} ") for header in _SECTION_HEADERS
        ):
            break
        raise LammpsDataError("unexpected extra content in the Atoms section")


def _build_structure(
    rows: list[tuple[str, float, float, float]],
    ids: list[int],
    row_charges: list[float | None],
    n_types: int,
    box_dims: np.ndarray,
) -> StructureData:
    """Assemble validated atom rows and box bounds into a ``StructureData``.

    :param rows: Parsed ``(species, x, y, z)`` rows in file order.
    :param ids: Parsed atom IDs, aligned with ``rows``.
    :param row_charges: Parsed per-row charges (or ``None``), aligned with ``rows``.
    :param n_types: Declared atom-type count, for species-count validation.
    :param box_dims: Per-axis ``(lower, upper)`` box bounds, ordered ``x, y, z``.
    :return: The assembled neutral structure.
    :raises LammpsDataError: If atom IDs are not unique or species outnumber
        ``n_types``.
    """
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
    return StructureData(atoms, cell, origin, external_ids=atom_ids, charges=charges)


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

        id_to_name = _normalize_type_mapping(type_dict)
        n_atoms, n_types, box, atoms_line = _parse_header_lines(lines, id_to_name)
        box_dims = _validate_header(n_atoms, n_types, box, atoms_line)
        assert n_atoms is not None and n_types is not None and atoms_line is not None

        rows, ids, row_charges, index = _parse_atoms_section(
            lines, atoms_line, n_atoms, n_types, id_to_name
        )
        if len(rows) != n_atoms:
            raise LammpsDataError(f"expected {n_atoms} atom rows, found {len(rows)}")
        _check_trailing_content(lines, index)

        return _build_structure(rows, ids, row_charges, n_types, box_dims)
