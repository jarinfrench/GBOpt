# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Legacy ``LammpsAtomData``-returning compatibility wrappers.

Reproduces the exact pre-R11 ``read_lammps_data_file``/``read_lammps_dump_file``/
``read_lammps_structure_file`` signatures and return shape (``LammpsAtomData``, an
orthogonal ``box_dims`` pair rather than the neutral ``StructureData`` cell/origin
pair) on top of the new readers, so ``GBOpt.FileGrainOwnership`` -- and any external
caller that imported these functions directly from it -- keeps working unchanged.
New code should call ``LammpsDataReader``/``LammpsDumpReader``/``read_structure_file``
directly instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from GBOpt.io.lammps.data import LammpsDataReader
from GBOpt.io.lammps.dispatch import read_structure_file
from GBOpt.io.lammps.dump import LammpsDumpReader
from GBOpt.io.lammps.types import LammpsAtomData
from GBOpt.io.types import StructureData


def _to_lammps_atom_data(structure: StructureData) -> LammpsAtomData:
    """Convert a neutral structure snapshot to the legacy ``LammpsAtomData`` shape.

    :param structure: Neutral structure produced by a LAMMPS reader. ``cell`` is
        assumed diagonal, as every ``GBOpt.io.lammps`` reader produces, so it losslessly
        maps back to an orthogonal ``(3, 2)`` bounds pair.
    :return: The equivalent legacy parsed-snapshot value.
    """
    assert structure.external_ids is not None
    widths = np.diagonal(structure.cell)
    lower = structure.origin
    box_dims = np.stack([lower, lower + widths], axis=1)
    return LammpsAtomData(
        structure.external_ids,
        structure.atoms,
        box_dims,
        boundary_periodic=structure.periodicity,
        selected_frame=structure.frame_index,
    )


def read_lammps_data_file(
    path: str | Path, *, type_dict: Mapping[object, object] | None = None
) -> LammpsAtomData:
    """Read the orthogonal LAMMPS data format emitted by ``GBMaker``.

    :param path: Path to the LAMMPS data file.
    :param type_dict: Keyword argument, optional, defaults to ``None``. Mapping in
        ``species -> type ID`` or ``type ID -> species`` form.
    :return: Immutable parsed atom IDs, atom rows, and box bounds.
    :raises FileNotFoundError: If ``path`` does not identify an existing file.
    :raises LammpsDataError: If the file or type mapping is malformed, ambiguous,
        unsupported, or inconsistent with its declared counts and bounds.
    """
    return _to_lammps_atom_data(LammpsDataReader().read(path, type_dict=type_dict))


def read_lammps_dump_file(
    path: str | Path, *, type_dict: Mapping[object, object] | None = None
) -> LammpsAtomData:
    """Read exactly the first frame of an orthogonal LAMMPS dump.

    :param path: Path to the LAMMPS dump file.
    :param type_dict: Keyword argument, optional, defaults to ``None``. Mapping in
        ``species -> type ID`` or ``type ID -> species`` form.
    :return: Immutable parsed data for dump frame zero.
    :raises FileNotFoundError: If ``path`` does not identify an existing file.
    :raises LammpsDataError: If the type mapping, frame structure, bounds, topology,
        atom attributes, IDs, species, or coordinates are malformed or ambiguous.
    """
    return _to_lammps_atom_data(LammpsDumpReader().read(path, type_dict=type_dict))


def read_lammps_structure_file(
    path: str | Path, *, type_dict: Mapping[object, object] | None = None
) -> LammpsAtomData:
    """Read a supported LAMMPS data file or first dump frame.

    :param path: Path to a LAMMPS data or dump file.
    :param type_dict: Keyword argument, optional, defaults to ``None``. Mapping in
        ``species -> type ID`` or ``type ID -> species`` form.
    :return: Immutable parsed atom and box data.
    :raises FileNotFoundError: If ``path`` does not identify an existing file.
    :raises LammpsDataError: If the selected reader rejects malformed, unsupported, or
        ambiguous file content or type metadata.
    """
    return _to_lammps_atom_data(read_structure_file(path, type_dict=type_dict))
