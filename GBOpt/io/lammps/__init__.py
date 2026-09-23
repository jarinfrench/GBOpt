# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""LAMMPS structure readers, extracted from ``GBOpt.FileGrainOwnership`` (R11, #72).

The package-level surface exposes ``LammpsDataReader``/``LammpsDumpReader`` (returning
the neutral ``GBOpt.io.StructureData``) and ``read_structure_file`` (format dispatch
between them), plus ``LammpsDataError`` and the legacy ``LammpsAtomData``-returning
``read_lammps_data_file``/``read_lammps_dump_file``/``read_lammps_structure_file``
wrappers kept for compatibility with pre-R11 callers.
``GBOpt.FileGrainOwnership`` now imports all five of the latter from here rather than
defining them itself, per the compatibility-facade convention established by
``GBOpt.GBMaker``/``GBOpt.gbmaker``.

``tokens`` (atom-ID/species/type-ID token validation and type-map normalization) and
``types`` (``LammpsDataError``, ``LammpsAtomData``) are this package's leaves; ``data``
and ``dump`` are siblings that each depend only on ``tokens``/``types`` and do not
import one another; ``dispatch`` composes them (mirroring ``gbmaker.assembly``'s
relationship to ``exact_grain``/``approximate_grain``); ``compat`` builds the legacy
``LammpsAtomData`` wrappers on top of ``dispatch``/``data``/``dump``. No grain-ownership
inference, candidate/file mapping, or evaluator-reload logic belongs in this package --
that remains in ``GBOpt.FileGrainOwnership``, which is this package's only intended
ownership-side consumer.
"""

from .compat import (
    read_lammps_data_file,
    read_lammps_dump_file,
    read_lammps_structure_file,
)
from .data import LammpsDataReader
from .dispatch import read_structure_file
from .dump import LammpsDumpReader
from .types import LammpsAtomData, LammpsDataError

__all__ = [
    # Exception
    "LammpsDataError",
    # Readers (return GBOpt.io.StructureData)
    "LammpsDataReader",
    "LammpsDumpReader",
    "read_structure_file",
    # Legacy compatibility wrappers (return LammpsAtomData)
    "LammpsAtomData",
    "read_lammps_data_file",
    "read_lammps_dump_file",
    "read_lammps_structure_file",
]
