# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""LAMMPS structure readers and writer, extracted from ``GBOpt.FileGrainOwnership``
(R11, #72) and ``GBOpt.GBMaker`` (R12, #73).

The package-level surface exposes ``LammpsDataReader``/``LammpsDumpReader`` (returning
the neutral ``GBOpt.io.StructureData``) and ``read_structure_file`` (format dispatch
between them); ``LammpsDataWriter`` (formatting a ``StructureData`` to the LAMMPS data
format and returning a ``GBOpt.io.WriteResult``); ``LammpsDataError``/
``LammpsWriteError``; and the legacy ``LammpsAtomData``-returning
``read_lammps_data_file``/``read_lammps_dump_file``/``read_lammps_structure_file``
wrappers kept for compatibility with pre-R11 callers. ``GBOpt.FileGrainOwnership`` now
imports all five of the latter from here rather than defining them itself, and
``GBOpt.GBMaker.write_lammps()`` delegates to ``LammpsDataWriter`` rather than
formatting LAMMPS output itself, per the compatibility-facade convention established by
``GBOpt.GBMaker``/``GBOpt.gbmaker``.

``tokens`` (atom-ID/species/type-ID token validation and type-map normalization) and
``types`` (``LammpsDataError``, ``LammpsWriteError``, ``LammpsAtomData``) are this
package's leaves; ``data``, ``dump``, and ``data_writer`` are siblings that each depend
only on ``tokens``/``types`` (``data_writer`` does not even need ``tokens``) and do not
import one another; ``dispatch`` composes ``data``/``dump`` (mirroring
``gbmaker.assembly``'s relationship to ``exact_grain``/``approximate_grain``); ``compat``
builds the legacy ``LammpsAtomData`` wrappers on top of ``dispatch``/``data``/``dump``.
``data_writer`` does not import ``GBOpt.GBMaker`` -- it is a pure formatter over
``StructureData`` and plain write options, with no grain-construction or orientation-
planning knowledge. No grain-ownership inference, candidate/file mapping, or
evaluator-reload logic belongs in this package -- that remains in
``GBOpt.FileGrainOwnership``, which is this package's only intended ownership-side
consumer.
"""

from .compat import (
    read_lammps_data_file,
    read_lammps_dump_file,
    read_lammps_structure_file,
)
from .data import LammpsDataReader
from .data_writer import LammpsDataWriter
from .dispatch import read_structure_file
from .dump import LammpsDumpReader
from .types import LammpsAtomData, LammpsDataError, LammpsWriteError

__all__ = [
    # Exceptions
    "LammpsDataError",
    "LammpsWriteError",
    # Readers (return GBOpt.io.StructureData)
    "LammpsDataReader",
    "LammpsDumpReader",
    "read_structure_file",
    # Writer (returns GBOpt.io.WriteResult)
    "LammpsDataWriter",
    # Legacy compatibility wrappers (return LammpsAtomData)
    "LammpsAtomData",
    "read_lammps_data_file",
    "read_lammps_dump_file",
    "read_lammps_structure_file",
]
