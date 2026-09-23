# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared data types and exceptions for the ``GBOpt.io.lammps`` package.

Contains ``LammpsDataError`` (the established public exception raised by the LAMMPS
readers), ``LammpsWriteError`` (raised by ``LammpsDataWriter`` for invalid write
options), and ``LammpsAtomData`` (the legacy parsed-snapshot value type returned by the
compatibility wrapper functions moved from ``GBOpt.FileGrainOwnership``). None of these
types encode parsing or formatting logic; both exceptions extend the generic
``GBOpt.io`` exception hierarchy so callers that already catch
``GBOpt.io.StructureFormatError``/``StructureIOError`` or ``ValueError`` continue to
work unchanged, and ``LammpsAtomData`` is a thin, LAMMPS-specific view kept only for
compatibility with existing callers of the pre-R11 reader functions -- new code should
consume ``GBOpt.io.StructureData`` via ``LammpsDataReader``/``LammpsDumpReader`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from GBOpt.Atom import Atom
from GBOpt.io.types import StructureFormatError, StructureIOError, _readonly_copy


class LammpsDataError(StructureFormatError):
    """Raised when a LAMMPS data or dump file cannot be read unambiguously."""


class LammpsWriteError(StructureIOError, ValueError):
    """Raised when ``LammpsDataWriter`` is given invalid structure data or options."""


@dataclass(frozen=True, slots=True, init=False)
class LammpsAtomData:
    """Parsed atom IDs, species/coordinates, box bounds, and optional BC flags."""

    _atom_ids: np.ndarray
    _atoms: np.ndarray
    _box_dims: np.ndarray
    boundary_periodic: tuple[bool, bool, bool] | None
    selected_frame: int | None

    def __init__(
        self,
        atom_ids: np.ndarray,
        atoms: np.ndarray,
        box_dims: np.ndarray,
        *,
        boundary_periodic: tuple[bool, bool, bool] | None = None,
        selected_frame: int | None = None,
    ) -> None:
        """Construct one immutable parsed LAMMPS atom snapshot.

        :param atom_ids: Serialization-local atom IDs in file-row order.
        :param atoms: Structured atom rows containing species and Cartesian coordinates.
        :param box_dims: Orthogonal lower and upper bounds for x, y, and z.
        :param boundary_periodic: Keyword argument, optional, defaults to ``None``.
            Per-axis periodicity decoded from a dump header.
        :param selected_frame: Keyword argument, optional, defaults to ``None`` for data
            files. Selected zero-based dump frame index.
        """
        object.__setattr__(self, "_atom_ids", _readonly_copy(atom_ids, dtype=np.int64))
        object.__setattr__(self, "_atoms", _readonly_copy(atoms, dtype=Atom.atom_dtype))
        object.__setattr__(self, "_box_dims", _readonly_copy(box_dims, dtype=float))
        object.__setattr__(self, "boundary_periodic", boundary_periodic)
        object.__setattr__(self, "selected_frame", selected_frame)

    @property
    def atom_ids(self) -> np.ndarray:
        """Parsed atom IDs in file-row order."""
        return _readonly_copy(self._atom_ids)

    @property
    def atoms(self) -> np.ndarray:
        """Parsed structured atom rows."""
        return _readonly_copy(self._atoms)

    @property
    def box_dims(self) -> np.ndarray:
        """Parsed orthogonal box bounds."""
        return _readonly_copy(self._box_dims)


__all__ = ["LammpsAtomData", "LammpsDataError", "LammpsWriteError"]
