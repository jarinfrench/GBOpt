# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Structure I/O package: canonical structure representation and format readers/writers.

The package-level surface exposes ``StructureData``, the canonical neutral atomic
structure representation (atoms, cell, origin, periodicity, optional external atom IDs
and charges, minimal frame metadata); ``WriteResult``, the format-neutral outcome
metadata (row-to-external-ID mapping, optional digest, declared losses) a
``StructureWriter`` returns; the ``StructureIOError`` exception hierarchy; and the
``StructureReader``/``StructureWriter`` protocols format-specific implementations
satisfy. ``types`` is this package's leaf module and holds no parsing, formatting, or
dispatch logic.

Format-specific readers and writers live in their own subpackages (currently
``GBOpt.io.lammps``), which build on these neutral types without this package depending
back on any of them. No grain-ownership, optimization-policy, or calculator-execution
logic belongs here or in any format subpackage under it -- that remains in
``GBOpt.GrainOwnership`` and ``GBOpt.FileGrainOwnership``, which now build on top of
``GBOpt.io.lammps`` instead of implementing LAMMPS parsing themselves.
"""

from .types import (
    StructureData,
    StructureFormatError,
    StructureIOError,
    StructureReader,
    StructureValueError,
    StructureWriter,
    WriteResult,
)

__all__ = [
    # Exceptions
    "StructureIOError",
    "StructureFormatError",
    "StructureValueError",
    # Canonical structure representation
    "StructureData",
    "WriteResult",
    # Reader/writer protocols
    "StructureReader",
    "StructureWriter",
]
