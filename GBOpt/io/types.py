# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared data types and exceptions for the ``GBOpt.io`` structure I/O layer.

Contains the I/O exception hierarchy, the canonical neutral ``StructureData``
representation, and the ``StructureReader``/``StructureWriter`` protocols that
format-specific reader/writer implementations satisfy. ``StructureData`` is deliberately
independent of grain ownership, optimization policy, and calculator execution: it
carries only what a structure file can unambiguously express (atoms, cell, origin,
periodicity, optional external atom IDs, optional charges, and minimal frame metadata),
none of which requires knowing how a caller will use the structure. No file parsing,
format dispatch, or ownership/reload logic belongs here; this module is a pure
data-definition layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from GBOpt.Atom import Atom


class StructureIOError(Exception):
    """Base for structure I/O contract errors in the ``GBOpt.io`` package."""


class StructureFormatError(StructureIOError, ValueError):
    """Raised when structure file content is malformed, ambiguous, or unsupported."""


class StructureValueError(StructureIOError, ValueError):
    """Raised when a ``StructureData`` field value is invalid."""


def _readonly_copy(values: object, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Return an independent read-only NumPy array.

    :param values: Source array-like whose values will be copied.
    :param dtype: Keyword argument, optional, defaults to ``None``. Requested result
        dtype.
    :return: A copied array with mutation disabled.
    """
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, init=False)
class StructureData:
    """Canonical neutral atomic-structure representation.

    :param atoms: One-dimensional structured atom array (``Atom.atom_dtype``: species
        name plus Cartesian x, y, z coordinates).
    :param cell: 3 by 3 array of lattice-vector rows. Readers that only support
        orthogonal boxes populate a diagonal matrix; off-diagonal entries are reserved
        for future non-orthogonal support.
    :param origin: Length-3 array giving the lattice's lower-corner origin point.
    :param periodicity: Per-axis periodicity flags ``(x, y, z)``, or ``None`` when the
        source format does not encode periodicity (e.g. a LAMMPS data file).
    :param external_ids: Length-``len(atoms)`` array of serialization-local atom IDs in
        atom-row order, or ``None`` when the source format has none. These IDs are
        local to one serialized file and are not a persistent atom identity across
        reads, writes, or reloads.
    :param charges: Length-``len(atoms)`` array of per-atom charges, or ``None`` when
        the source format does not encode charges.
    :param frame_index: Zero-based index of the frame this structure was read from,
        for multi-frame formats, or ``None`` for single-frame formats.
    """

    atoms: np.ndarray
    cell: np.ndarray
    origin: np.ndarray
    periodicity: tuple[bool, bool, bool] | None
    external_ids: np.ndarray | None
    charges: np.ndarray | None
    frame_index: int | None

    def __init__(
        self,
        atoms: np.ndarray,
        cell: np.ndarray,
        origin: np.ndarray,
        *,
        periodicity: tuple[bool, bool, bool] | None = None,
        external_ids: np.ndarray | None = None,
        charges: np.ndarray | None = None,
        frame_index: int | None = None,
    ) -> None:
        """Construct one immutable neutral structure snapshot.

        :param atoms: One-dimensional structured atom array (``Atom.atom_dtype``).
        :param cell: 3 by 3 array-like of lattice-vector rows.
        :param origin: Length-3 array-like lattice origin point.
        :param periodicity: Keyword argument, optional, defaults to ``None``. Per-axis
            periodicity flags ``(x, y, z)``.
        :param external_ids: Keyword argument, optional, defaults to ``None``.
            Serialization-local atom IDs parallel to ``atoms``.
        :param charges: Keyword argument, optional, defaults to ``None``. Per-atom
            charges parallel to ``atoms``.
        :param frame_index: Keyword argument, optional, defaults to ``None``.
            Zero-based source-frame index.
        :raises StructureValueError: If any field is malformed or inconsistent with
            ``atoms``.
        """
        atoms_array = np.asarray(atoms)
        if atoms_array.ndim != 1 or atoms_array.dtype != Atom.atom_dtype:
            raise StructureValueError(
                "atoms must be a one-dimensional Atom.atom_dtype structured array"
            )

        try:
            cell_array = _readonly_copy(cell, dtype=float)
        except (TypeError, ValueError) as exc:
            raise StructureValueError("cell must be a real 3 by 3 array-like") from exc
        if cell_array.shape != (3, 3) or not np.all(np.isfinite(cell_array)):
            raise StructureValueError("cell must be a finite 3 by 3 array")

        try:
            origin_array = _readonly_copy(origin, dtype=float)
        except (TypeError, ValueError) as exc:
            raise StructureValueError("origin must be a real length-3 array-like") from exc
        if origin_array.shape != (3,) or not np.all(np.isfinite(origin_array)):
            raise StructureValueError("origin must be a finite length-3 array")

        if periodicity is not None:
            try:
                x_p, y_p, z_p = periodicity
            except (TypeError, ValueError) as exc:
                raise StructureValueError(
                    "periodicity must be a three-value sequence of bools"
                ) from exc
            flags = (x_p, y_p, z_p)
            if not all(isinstance(flag, (bool, np.bool_)) for flag in flags):
                raise StructureValueError("periodicity flags must be bools")
            periodicity = (bool(x_p), bool(y_p), bool(z_p))

        n_atoms = atoms_array.size
        normalized_ids: np.ndarray | None = None
        if external_ids is not None:
            try:
                normalized_ids = _readonly_copy(external_ids, dtype=np.int64)
            except (TypeError, ValueError) as exc:
                raise StructureValueError(
                    "external_ids must be an integer array-like"
                ) from exc
            if normalized_ids.shape != (n_atoms,):
                raise StructureValueError(
                    "external_ids must be parallel to atoms"
                )

        normalized_charges: np.ndarray | None = None
        if charges is not None:
            try:
                normalized_charges = _readonly_copy(charges, dtype=float)
            except (TypeError, ValueError) as exc:
                raise StructureValueError("charges must be a real array-like") from exc
            if normalized_charges.shape != (n_atoms,):
                raise StructureValueError("charges must be parallel to atoms")
            if not np.all(np.isfinite(normalized_charges)):
                raise StructureValueError("charges must be finite")

        if frame_index is not None:
            if isinstance(frame_index, (bool, np.bool_)) or not isinstance(
                frame_index, (int, np.integer)
            ):
                raise StructureValueError("frame_index must be a non-Boolean integer")
            if frame_index < 0:
                raise StructureValueError("frame_index must be non-negative")
            frame_index = int(frame_index)

        object.__setattr__(self, "atoms", _readonly_copy(atoms_array, dtype=Atom.atom_dtype))
        object.__setattr__(self, "cell", cell_array)
        object.__setattr__(self, "origin", origin_array)
        object.__setattr__(self, "periodicity", periodicity)
        object.__setattr__(self, "external_ids", normalized_ids)
        object.__setattr__(self, "charges", normalized_charges)
        object.__setattr__(self, "frame_index", frame_index)


@dataclass(frozen=True, slots=True, init=False)
class WriteResult:
    """Outcome metadata from one ``StructureWriter`` write.

    Carries the candidate-local row-to-external-ID mapping a write assigned, so a later
    ownership reload (R14) can align a re-read file back to the rows of the
    ``StructureData`` that produced it, plus an explicit account of what the write did
    not preserve. Deliberately format-neutral, like ``StructureData`` itself: nothing
    here is LAMMPS-specific, even though ``GBOpt.io.lammps`` is this type's only
    producer today.

    :param target: Destination path the structure was written to.
    :param format: Format identifier the writer produced (e.g. ``"lammps_data"``).
    :param atom_ids: Length-``len(structure.atoms)`` array of external atom IDs assigned
        in atom-row order. Index ``i`` is the row-to-external-ID mapping for row ``i``;
        these IDs are local to this one write and carry no persistent atom identity
        across writes, reads, or reloads.
    :param digest: Content signature of the written file, or ``None`` when not
        requested.
    :param losses: Human-readable descriptions of source ``StructureData`` information
        this write did not preserve (e.g. periodicity flags, pre-existing external IDs).
        Empty when nothing was dropped.
    """

    target: Path
    format: str
    atom_ids: np.ndarray
    digest: str | None
    losses: tuple[str, ...]

    def __init__(
        self,
        target: str | Path,
        format: str,
        atom_ids: np.ndarray,
        *,
        digest: str | None = None,
        losses: tuple[str, ...] = (),
    ) -> None:
        """Construct one immutable write outcome.

        :param target: Destination path the structure was written to.
        :param format: Format identifier the writer produced.
        :param atom_ids: Array-like of external atom IDs assigned, in atom-row order.
        :param digest: Keyword argument, optional, defaults to ``None``. Content
            signature of the written file.
        :param losses: Keyword argument, optional, defaults to ``()``. Descriptions of
            information the write did not preserve.
        :raises StructureValueError: If any field is malformed.
        """
        target_path = Path(target)

        if not isinstance(format, str) or not format:
            raise StructureValueError("format must be a non-empty string")

        try:
            ids_array = _readonly_copy(atom_ids, dtype=np.int64)
        except (TypeError, ValueError) as exc:
            raise StructureValueError(
                "atom_ids must be an integer array-like"
            ) from exc
        if ids_array.ndim != 1:
            raise StructureValueError("atom_ids must be one-dimensional")

        if digest is not None and not isinstance(digest, str):
            raise StructureValueError("digest must be a string or None")

        normalized_losses = tuple(losses)
        if not all(isinstance(loss, str) for loss in normalized_losses):
            raise StructureValueError("losses must be a sequence of strings")

        object.__setattr__(self, "target", target_path)
        object.__setattr__(self, "format", format)
        object.__setattr__(self, "atom_ids", ids_array)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "losses", normalized_losses)


@runtime_checkable
class StructureReader(Protocol):
    """Protocol satisfied by a format-specific structure reader."""

    def read(self, path: object, **kwargs: object) -> StructureData:
        """Read one structure snapshot from ``path``.

        :param path: Path to the structure file.
        :param kwargs: Format-specific reader options.
        :return: The parsed neutral structure.
        """
        ...


@runtime_checkable
class StructureWriter(Protocol):
    """Protocol satisfied by a format-specific structure writer."""

    def write(self, path: object, structure: StructureData, **kwargs: object) -> None:
        """Write one structure snapshot to ``path``.

        :param path: Destination path for the structure file.
        :param structure: Neutral structure to serialize.
        :param kwargs: Format-specific writer options.
        """
        ...


__all__ = [
    "StructureIOError",
    "StructureFormatError",
    "StructureValueError",
    "StructureData",
    "WriteResult",
    "StructureReader",
    "StructureWriter",
]
