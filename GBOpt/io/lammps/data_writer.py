# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""``LammpsDataWriter``: formats a ``StructureData`` as the LAMMPS data format.

This module is a leaf with respect to its ``io.lammps`` siblings: it depends only on
``types`` (plus the generic ``GBOpt.io.types``), and does not import ``data``, ``dump``,
``dispatch``, or ``compat`` -- writing and reading are independent concerns, mirroring
``data.py``/``dump.py``'s existing sibling relationship. It also does not import
``GBOpt.GBMaker``: it is a pure formatter over the neutral ``StructureData``/write-option
inputs it is given, with no knowledge of grain construction, orientation planning, or
periodicity policy. Any orientation-dependent computation (e.g. the restricted-triclinic
tilt rotation) is the caller's responsibility, applied to the atom positions and cell
before constructing the ``StructureData`` passed in here.
"""

from __future__ import annotations

import hashlib
from collections.abc import MutableMapping
from numbers import Number
from pathlib import Path
from typing import TextIO

import numpy as np

from GBOpt.io.lammps.types import LammpsWriteError
from GBOpt.io.types import StructureData, StructureValueError, WriteResult


def _resolve_name_to_int(
    atom_names: np.ndarray, type_map: MutableMapping[str, int] | None
) -> dict[str, int]:
    """Resolve species names to numeric type IDs, mirroring legacy ``write_lammps``.

    :param atom_names: Sorted-unique species names present in the atoms to write.
    :param type_map: Caller-supplied species-to-numeric-ID mapping, or ``None``.
    :return: A mapping used verbatim (filtered to ``atom_names``, in ``type_map``'s
        order) when every name in ``atom_names`` is a key of ``type_map``; otherwise
        numeric IDs assigned in ``atom_names``'s (first-seen/alphabetical) order.
    """
    normalized_type_map = dict(type_map) if type_map else {}
    if set(atom_names).issubset(normalized_type_map.keys()):
        return {
            name: normalized_type_map[name]
            for name in normalized_type_map
            if name in atom_names
        }
    return {name: i + 1 for i, name in enumerate(atom_names)}


def _validate_and_resolve_charges(
    charges: MutableMapping[object, float] | None,
    atom_names: np.ndarray,
    name_to_int: dict[str, int],
    *,
    type_as_int: bool,
) -> None:
    """Validate ``charges`` and, when applicable, mutate in numeric-type-ID keys.

    :param charges: Per-species (or numeric-type-ID) charge mapping, or ``None``.
    :param atom_names: Sorted-unique species names present in the atoms to write.
    :param name_to_int: Resolved species-to-numeric-ID mapping.
    :param type_as_int: Whether atom types will be written as numeric IDs.
    :raises LammpsWriteError: If ``charges`` has non-``int``/``str`` keys or non-numeric
        values.
    """
    if charges is None:
        return
    if not all(isinstance(key, (int, str)) for key in charges):
        raise LammpsWriteError(
            "'charges' keys are required to be integers or strings."
        )
    if not all(isinstance(value, Number) for value in charges.values()):
        raise LammpsWriteError("'charges' values are required to be numeric.")
    if type_as_int and all(isinstance(key, str) for key in charges):
        for name in atom_names:
            charges[name_to_int[name]] = charges[name]


def _resolve_box_bounds(structure: StructureData) -> np.ndarray:
    """Return per-axis lower/upper box bounds from ``structure``'s cell and origin.

    :param structure: Neutral structure whose ``cell`` diagonal gives box widths.
    :return: A ``(3, 2)`` array of ``[lower, upper]`` bounds per axis.
    :raises LammpsWriteError: If the resulting bounds are malformed.
    """
    try:
        widths = np.diagonal(structure.cell)
        return np.stack([structure.origin, structure.origin + widths], axis=1)
    except StructureValueError as exc:
        raise LammpsWriteError(str(exc)) from exc


def _write_header(
    fdata: TextIO,
    atoms: np.ndarray,
    atom_names: np.ndarray,
    box_sizes: np.ndarray,
    cell: np.ndarray,
    name_to_int: dict[str, int],
    *,
    precision: int,
    type_as_int: bool,
    triclinic: bool,
) -> None:
    """Write the comment, counts, box-bounds, tilt, and type-label header sections."""
    joined_names = "".join(atom_names)
    fdata.write(f"Crystalline {joined_names} atoms\n\n")

    fdata.write(f"{len(atoms)} atoms\n")
    fdata.write(f"{len(set(atoms['name']))} atom types\n")
    fdata.write(
        f"{box_sizes[0][0]:.{precision}f} {box_sizes[0][1]:.{precision}f} xlo xhi\n"
    )
    fdata.write(
        f"{box_sizes[1][0]:.{precision}f} {box_sizes[1][1]:.{precision}f} ylo yhi\n"
    )
    fdata.write(
        f"{box_sizes[2][0]:.{precision}f} {box_sizes[2][1]:.{precision}f} zlo zhi\n"
    )
    if triclinic:
        xy, xz, yz = float(cell[1, 0]), float(cell[2, 0]), float(cell[2, 1])
        fdata.write(
            f"{xy:.{precision}f} {xz:.{precision}f} {yz:.{precision}f} xy xz yz\n"
        )

    if not type_as_int:
        fdata.write("\nAtom Type Labels\n\n")
        fdata.writelines(f"{value} {name}\n" for name, value in name_to_int.items())


def _format_atom_line(
    index: int,
    label: object,
    pos: list[float],
    charge: float | None,
    *,
    precision: int,
) -> str:
    """Format one ``Atoms`` section row, five- or six-column depending on ``charge``."""
    if charge is not None:
        return (
            f"{index} {label} {charge:.{precision}f} "
            f"{pos[0]:.{precision}f} {pos[1]:.{precision}f} "
            f"{pos[2]:.{precision}f}\n"
        )
    return (
        f"{index} {label} {pos[0]:.{precision}f} "
        f"{pos[1]:.{precision}f} {pos[2]:.{precision}f}\n"
    )


def _write_atoms_section(
    fdata: TextIO,
    atoms: np.ndarray,
    name_to_int: dict[str, int],
    charges: MutableMapping[object, float] | None,
    *,
    precision: int,
    type_as_int: bool,
) -> None:
    """Write the ``Atoms`` section, one row per atom in file-row order."""
    fdata.write("\nAtoms\n\n")
    for i, (name, *pos) in enumerate(atoms):
        if charges is not None:
            charge = charges[name_to_int[name]] if type_as_int else charges[name]
        else:
            charge = None
        label = name_to_int[name] if type_as_int else name
        fdata.write(_format_atom_line(i + 1, label, pos, charge, precision=precision))


def _declared_losses(structure: StructureData) -> tuple[str, ...]:
    """Describe source ``StructureData`` fields this format cannot represent."""
    losses: list[str] = []
    if structure.external_ids is not None:
        losses.append(
            "source StructureData.external_ids were not preserved; sequential "
            "candidate-local atom IDs (1..N) were assigned on write"
        )
    if structure.periodicity is not None:
        losses.append(
            "source StructureData.periodicity is not encoded in the LAMMPS data format"
        )
    if structure.charges is not None:
        losses.append(
            "source StructureData.charges was not written; the 'charges' write "
            "option, not StructureData.charges, is the sole per-atom charge source"
        )
    if structure.frame_index is not None:
        losses.append(
            "source StructureData.frame_index is not encoded in the LAMMPS data format"
        )
    return tuple(losses)


class LammpsDataWriter:
    """Writes the orthogonal or restricted-triclinic LAMMPS data format.

    Reproduces ``GBMaker.write_lammps()``'s established formatting exactly: atom
    ordering, sequential atom-ID assignment, type labels, charges, coordinate precision,
    box headers, and tilt factors. Both the five-column atomic form and the six-column
    charge form are supported.
    """

    def write(
        self,
        path: str | Path,
        structure: StructureData,
        *,
        type_as_int: bool = False,
        precision: int = 6,
        charges: MutableMapping[object, float] | None = None,
        type_map: MutableMapping[str, int] | None = None,
        triclinic: bool = False,
        compute_digest: bool = False,
    ) -> WriteResult:
        """Write one neutral structure snapshot to a LAMMPS data file.

        :param path: Destination path for the LAMMPS data file.
        :param structure: Neutral structure to serialize. ``cell`` is read as LAMMPS
            box-vector rows: the diagonal gives the box widths and, when ``triclinic``
            is set, ``cell[1, 0]``/``cell[2, 0]``/``cell[2, 1]`` give the ``xy``/``xz``/
            ``yz`` tilt factors. Atom positions are written as given -- any
            restricted-triclinic rotation must already be applied by the caller.
        :param type_as_int: Keyword argument, optional, defaults to ``False``. Whether
            to write atom types as a numeric ID (using ``type_map``, falling back to
            first-seen order) rather than a species name.
        :param precision: Keyword argument, optional, defaults to ``6``. Decimal
            precision for written float values.
        :param charges: Keyword argument, optional, defaults to ``None``. Per-species
            (or, when ``type_as_int`` and all keys are strings, additionally per-numeric-
            type-ID after this call) charge values. When given, every atom row is
            written in six-column charge form; ``None`` writes the five-column atomic
            form. Mutated in place to add numeric-type-ID keys when applicable, mirroring
            established behavior.
        :param type_map: Keyword argument, optional, defaults to ``None``. Species-to-
            numeric-type-ID mapping. Used verbatim only when every species present in
            ``structure.atoms`` is a key of it; otherwise numeric IDs are assigned in
            first-seen order.
        :param triclinic: Keyword argument, optional, defaults to ``False``. Whether to
            emit the ``xy xz yz`` tilt-factor header line.
        :param compute_digest: Keyword argument, optional, defaults to ``False``.
            Whether to compute a content digest of the written file for
            ``WriteResult.digest``.
        :return: Outcome metadata, including the row-to-external-ID mapping this write
            assigned.
        :raises LammpsWriteError: If ``structure`` is not a ``StructureData``, or
            ``charges`` is malformed.
        """
        if not isinstance(structure, StructureData):
            raise LammpsWriteError("structure must be a GBOpt.io.StructureData")

        atoms = structure.atoms
        atom_names = np.unique(atoms["name"])
        name_to_int = _resolve_name_to_int(atom_names, type_map)
        _validate_and_resolve_charges(
            charges, atom_names, name_to_int, type_as_int=type_as_int
        )
        box_sizes = _resolve_box_bounds(structure)

        file_path = Path(path)
        with file_path.open("w", encoding="utf-8") as fdata:
            _write_header(
                fdata,
                atoms,
                atom_names,
                box_sizes,
                structure.cell,
                name_to_int,
                precision=precision,
                type_as_int=type_as_int,
                triclinic=triclinic,
            )
            _write_atoms_section(
                fdata,
                atoms,
                name_to_int,
                charges,
                precision=precision,
                type_as_int=type_as_int,
            )

        digest = (
            hashlib.sha256(file_path.read_bytes()).hexdigest()
            if compute_digest
            else None
        )

        return WriteResult(
            file_path,
            "lammps_data",
            np.arange(1, len(atoms) + 1, dtype=np.int64),
            digest=digest,
            losses=_declared_losses(structure),
        )


__all__ = ["LammpsDataWriter"]
