# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""``LammpsDumpReader``: parses the first frame of a LAMMPS dump into ``StructureData``.

This module is a leaf with respect to its ``io.lammps`` siblings: it depends only on
``tokens`` and ``types`` (plus the generic ``GBOpt.io.types``), and does not import
``data`` or ``dispatch``.
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


def _dump_boundary_flags(tokens: list[str]) -> tuple[bool, bool, bool] | None:
    """Decode orthogonal LAMMPS dump boundary flags.

    :param tokens: Tokens following ``ITEM: BOX BOUNDS``.
    :return: Per-axis periodicity when three valid flags are present; otherwise
        ``None``.
    """
    if len(tokens) < 3:
        return None
    flags = tokens[-3:]
    if not all(len(flag) == 2 and set(flag).issubset(set("pfsm")) for flag in flags):
        return None
    return tuple(flag == "pp" for flag in flags)


def _parse_timestep_and_count(lines: list[str]) -> tuple[int, int]:
    """Parse the ``ITEM: TIMESTEP`` and ``ITEM: NUMBER OF ATOMS`` records.

    :param lines: Full file content, split into lines.
    :return: ``(n_atoms, index)``, where ``index`` is the line index of the
        ``ITEM: BOX BOUNDS`` record.
    :raises LammpsDataError: If the timestep or atom-count records are missing or
        malformed.
    """
    if not lines or lines[0].strip() != "ITEM: TIMESTEP":
        raise LammpsDataError("LAMMPS dump must begin with ITEM: TIMESTEP")
    try:
        int(lines[1].strip())
    except (IndexError, ValueError) as exc:
        raise LammpsDataError("selected dump frame has an invalid timestep") from exc
    index = 2
    if index >= len(lines) or lines[index].strip() != "ITEM: NUMBER OF ATOMS":
        raise LammpsDataError("selected dump frame is missing NUMBER OF ATOMS")
    try:
        n_atoms = int(lines[index + 1].strip())
    except (IndexError, ValueError) as exc:
        raise LammpsDataError("selected dump frame has an invalid atom count") from exc
    if n_atoms < 0:
        raise LammpsDataError("selected dump frame atom count must be nonnegative")
    return n_atoms, index + 2


def _parse_box_bounds(
    lines: list[str], index: int
) -> tuple[np.ndarray, tuple[bool, bool, bool] | None, int]:
    """Parse the ``ITEM: BOX BOUNDS`` record and its three bound rows.

    :param lines: Full file content, split into lines.
    :param index: Line index of the ``ITEM: BOX BOUNDS`` record.
    :return: ``(box_dims, boundary_periodic, index)``, where ``box_dims`` is a
        ``(3, 2)`` array of bounds ordered ``x, y, z``, and ``index`` is the line
        index of the ``ITEM: ATOMS`` record.
    :raises LammpsDataError: If the record is missing, non-orthogonal, or its bounds
        are malformed.
    """
    if index >= len(lines) or not lines[index].startswith("ITEM: BOX BOUNDS"):
        raise LammpsDataError("selected dump frame is missing BOX BOUNDS")
    boundary_periodic = _dump_boundary_flags(lines[index].split()[3:])
    bounds_rows: list[tuple[float, float]] = []
    for offset in range(1, 4):
        try:
            parts = lines[index + offset].split()
            if len(parts) != 2:
                raise ValueError
            lower, upper = float(parts[0]), float(parts[1])
        except (IndexError, ValueError) as exc:
            raise LammpsDataError(
                "explicit ownership supports orthogonal two-column dump bounds only"
            ) from exc
        bounds_rows.append((lower, upper))
    box_dims = np.asarray(bounds_rows, dtype=float)
    if not np.all(np.isfinite(box_dims)) or np.any(box_dims[:, 0] >= box_dims[:, 1]):
        raise LammpsDataError("selected dump frame box bounds are invalid")
    return box_dims, boundary_periodic, index + 4


def _parse_atoms_header(
    lines: list[str], index: int
) -> tuple[list[str], str, dict[str, int], int]:
    """Parse the ``ITEM: ATOMS`` record's declared per-row attributes.

    :param lines: Full file content, split into lines.
    :param index: Line index of the ``ITEM: ATOMS`` record.
    :return: ``(attributes, species_attr, attr_index, index)``, where
        ``species_attr`` is ``"typelabel"`` or ``"type"``, ``attr_index`` maps
        ``id``/``species_attr``/``x``/``y``/``z`` to their column positions, and
        ``index`` is the line index of the first atom row.
    :raises LammpsDataError: If the record is missing a required attribute.
    """
    if index >= len(lines) or not lines[index].startswith("ITEM: ATOMS"):
        raise LammpsDataError("selected dump frame is missing ATOMS")
    attributes = lines[index].split()[2:]
    for required in ("id", "x", "y", "z"):
        if required not in attributes:
            raise LammpsDataError(
                f"selected dump frame is missing atom attribute {required!r}"
            )
    if "typelabel" in attributes:
        species_attr = "typelabel"
    elif "type" in attributes:
        species_attr = "type"
    else:
        raise LammpsDataError("selected dump frame requires type or typelabel")
    attr_index = {
        name: attributes.index(name) for name in ("id", species_attr, "x", "y", "z")
    }
    return attributes, species_attr, attr_index, index + 1


def _parse_dump_atom_row(
    parts: list[str],
    attributes: list[str],
    attr_index: dict[str, int],
    species_attr: str,
    id_to_name: dict[int, str],
    inverse_default: dict[int, str],
) -> tuple[tuple[str, float, float, float], int]:
    """Parse one already-tokenized dump atom row.

    :param parts: The row's whitespace-split tokens.
    :param attributes: Declared per-row attribute names, for the row-length check.
    :param attr_index: Mapping of ``id``/``species_attr``/``x``/``y``/``z`` to column
        positions.
    :param species_attr: ``"typelabel"`` or ``"type"``.
    :param id_to_name: Type-ID-to-species mapping, used when ``species_attr`` is
        ``"type"``.
    :param inverse_default: Fallback type-ID-to-species mapping from
        ``Atom._numbers``.
    :return: ``((species, x, y, z), atom_id)``.
    :raises LammpsDataError: If the row is short, or its type or coordinates are
        malformed.
    """
    if len(parts) < len(attributes):
        raise LammpsDataError("selected dump frame contains a short atom row")
    atom_id = _strict_id_token(parts[attr_index["id"]])
    species_token = parts[attr_index[species_attr]]

    if species_attr == "typelabel":
        species = _strict_species_name(species_token)
    else:
        if not _INTEGER_TOKEN.fullmatch(species_token):
            raise LammpsDataError("dump type values must be integral")

        type_id = _strict_type_id(int(species_token))
        species = _resolve_type_id_species(type_id, id_to_name, inverse_default)
    try:
        xyz = tuple(float(parts[attr_index[axis]]) for axis in ("x", "y", "z"))
    except ValueError as exc:
        raise LammpsDataError("dump coordinates must be numeric") from exc
    if not np.all(np.isfinite(xyz)):
        raise LammpsDataError("dump coordinates must be finite")
    return (species, *xyz), atom_id


def _parse_atom_rows(
    lines: list[str],
    index: int,
    n_atoms: int,
    attributes: list[str],
    attr_index: dict[str, int],
    species_attr: str,
    id_to_name: dict[int, str],
) -> tuple[list[tuple[str, float, float, float]], list[int], int]:
    """Parse exactly ``n_atoms`` dump atom rows starting at ``index``.

    :param lines: Full file content, split into lines.
    :param index: Line index of the first atom row.
    :param n_atoms: Declared frame atom count.
    :param attributes: Declared per-row attribute names.
    :param attr_index: Mapping of ``id``/``species_attr``/``x``/``y``/``z`` to column
        positions.
    :param species_attr: ``"typelabel"`` or ``"type"``.
    :param id_to_name: Type-ID-to-species mapping, used when ``species_attr`` is
        ``"type"``.
    :return: ``(rows, ids, next_index)``, where ``next_index`` is the line index
        immediately after the parsed rows.
    :raises LammpsDataError: If fewer than ``n_atoms`` rows are available, or any row
        is malformed.
    """
    inverse_default = {number: name for name, number in Atom._numbers.items()}
    ids: list[int] = []
    rows: list[tuple[str, float, float, float]] = []
    for row_index in range(n_atoms):
        if index + row_index >= len(lines) or lines[index + row_index].startswith(
            "ITEM:"
        ):
            raise LammpsDataError(
                f"selected dump frame expected {n_atoms} atom rows, found {row_index}"
            )
        parts = lines[index + row_index].split()
        row, atom_id = _parse_dump_atom_row(
            parts, attributes, attr_index, species_attr, id_to_name, inverse_default
        )
        ids.append(atom_id)
        rows.append(row)
    return rows, ids, index + n_atoms


def _check_trailing_frame(lines: list[str], next_index: int) -> None:
    """Reject unexpected content between the parsed frame and the next one.

    :param lines: Full file content, split into lines.
    :param next_index: Line index immediately after the parsed atom rows.
    :raises LammpsDataError: If a non-blank line other than the next frame's
        ``ITEM: TIMESTEP`` follows.
    """
    while next_index < len(lines) and not lines[next_index].strip():
        next_index += 1
    if next_index < len(lines) and lines[next_index].strip() != "ITEM: TIMESTEP":
        raise LammpsDataError(
            "selected dump frame contains unexpected content after its atom rows"
        )


class LammpsDumpReader:
    """Reads exactly the first frame of an orthogonal LAMMPS dump.

    Multi-frame dumps are never concatenated. Validation applies only to frame zero; a
    malformed first frame is an error even if a later frame is valid.
    """

    def read(
        self, path: str | Path, *, type_dict: Mapping[object, object] | None = None
    ) -> StructureData:
        """Read frame zero of one LAMMPS dump into a neutral structure snapshot.

        :param path: Path to the LAMMPS dump file.
        :param type_dict: Keyword argument, optional, defaults to ``None``. Mapping in
            ``species -> type ID`` or ``type ID -> species`` form.
        :return: The parsed neutral structure for dump frame zero. ``charges`` is
            always ``None``; LAMMPS dump atom rows carry no charge column in this
            reader.
        :raises FileNotFoundError: If ``path`` does not identify an existing file.
        :raises LammpsDataError: If the type mapping, frame structure, bounds,
            topology, atom attributes, IDs, species, or coordinates are malformed or
            ambiguous.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(str(file_path))
        lines = file_path.read_text(encoding="utf-8").splitlines()

        n_atoms, index = _parse_timestep_and_count(lines)
        box_dims, boundary_periodic, index = _parse_box_bounds(lines, index)
        attributes, species_attr, attr_index, index = _parse_atoms_header(
            lines, index
        )

        id_to_name = _normalize_type_mapping(type_dict)
        rows, ids, next_index = _parse_atom_rows(
            lines, index, n_atoms, attributes, attr_index, species_attr, id_to_name
        )
        _check_trailing_frame(lines, next_index)

        atom_ids = np.asarray(ids, dtype=np.int64)
        if np.unique(atom_ids).size != atom_ids.size:
            raise LammpsDataError("atom IDs must be unique")

        cell = np.diag(box_dims[:, 1] - box_dims[:, 0])
        origin = box_dims[:, 0].copy()
        return StructureData(
            np.asarray(rows, dtype=Atom.atom_dtype),
            cell,
            origin,
            periodicity=boundary_periodic,
            external_ids=atom_ids,
            frame_index=0,
        )
