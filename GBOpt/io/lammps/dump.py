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
        if not lines or lines[0].strip() != "ITEM: TIMESTEP":
            raise LammpsDataError("LAMMPS dump must begin with ITEM: TIMESTEP")
        try:
            int(lines[1].strip())
        except (IndexError, ValueError) as exc:
            raise LammpsDataError(
                "selected dump frame has an invalid timestep"
            ) from exc
        index = 2
        if index >= len(lines) or lines[index].strip() != "ITEM: NUMBER OF ATOMS":
            raise LammpsDataError("selected dump frame is missing NUMBER OF ATOMS")
        try:
            n_atoms = int(lines[index + 1].strip())
        except (IndexError, ValueError) as exc:
            raise LammpsDataError(
                "selected dump frame has an invalid atom count"
            ) from exc
        if n_atoms < 0:
            raise LammpsDataError("selected dump frame atom count must be nonnegative")
        index += 2
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
                    "explicit ownership supports orthogonal two-column dump bounds "
                    "only"
                ) from exc
            bounds_rows.append((lower, upper))
        box_dims = np.asarray(bounds_rows, dtype=float)
        if not np.all(np.isfinite(box_dims)) or np.any(box_dims[:, 0] >= box_dims[:, 1]):
            raise LammpsDataError("selected dump frame box bounds are invalid")
        index += 4
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
        id_to_name = _normalize_type_mapping(type_dict)
        inverse_default = {number: name for name, number in Atom._numbers.items()}
        ids: list[int] = []
        rows: list[tuple[str, float, float, float]] = []
        index += 1
        for row_index in range(n_atoms):
            if index + row_index >= len(lines) or lines[index + row_index].startswith(
                "ITEM:"
            ):
                raise LammpsDataError(
                    f"selected dump frame expected {n_atoms} atom rows, found "
                    f"{row_index}"
                )
            parts = lines[index + row_index].split()
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
            ids.append(atom_id)
            rows.append((species, *xyz))
        next_index = index + n_atoms
        while next_index < len(lines) and not lines[next_index].strip():
            next_index += 1
        if (
            next_index < len(lines)
            and lines[next_index].strip() != "ITEM: TIMESTEP"
        ):
            raise LammpsDataError(
                "selected dump frame contains unexpected content after its atom rows"
            )
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
