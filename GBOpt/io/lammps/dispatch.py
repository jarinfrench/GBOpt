# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Format dispatch across the LAMMPS structure readers.

Composes ``LammpsDataReader`` and ``LammpsDumpReader`` -- the same relationship
``GBOpt.gbmaker.assembly`` has to ``exact_grain``/``approximate_grain`` -- to pick the
right reader from a file's first line, without either reader depending on the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from GBOpt.io.lammps.data import LammpsDataReader
from GBOpt.io.lammps.dump import LammpsDumpReader
from GBOpt.io.types import StructureData


def read_structure_file(
    path: str | Path, *, type_dict: Mapping[object, object] | None = None
) -> StructureData:
    """Read a supported LAMMPS data file or first dump frame.

    :param path: Path to a LAMMPS data or dump file.
    :param type_dict: Keyword argument, optional, defaults to ``None``. Mapping in
        ``species -> type ID`` or ``type ID -> species`` form.
    :return: The parsed neutral structure.
    :raises FileNotFoundError: If ``path`` does not identify an existing file.
    :raises LammpsDataError: If the selected reader rejects malformed, unsupported, or
        ambiguous file content or type metadata.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    with file_path.open(encoding="utf-8") as stream:
        first = stream.readline().strip()
    if first == "ITEM: TIMESTEP":
        return LammpsDumpReader().read(file_path, type_dict=type_dict)
    return LammpsDataReader().read(file_path, type_dict=type_dict)
