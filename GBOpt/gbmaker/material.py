# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Resolve crystal unit-cell material state for GBMaker construction.

Builds the immutable ``MaterialState`` contract, including the constructed
``UnitCell``, from crystal-identity inputs. Unit-cell construction errors (species
count, unrecognized element symbols, unsupported structures) propagate unchanged so
callers see the same ``UnitCell``/``Atom`` exception types raised by
``UnitCell.init_by_structure`` today; they are not translated into
``GBMakerConstructionError``. No orientation, dimension, or grain-generation logic
belongs here.
"""

from __future__ import annotations

from GBOpt.UnitCell import UnitCell

from .types import MaterialState


def resolve_material_state(
    a0: float,
    structure: str,
    atom_types: str | tuple[str, ...],
) -> MaterialState:
    """Build a resolved MaterialState with its constructed unit cell attached.

    :param a0: Crystal lattice parameter (Angstroms).
    :param structure: Crystal structure name.
    :param atom_types: Atom type string or tuple of atom type strings.
    :return: Validated MaterialState carrying the constructed UnitCell.
    :raises GBMakerConstructionValueError: If a0, structure, or atom_types fail
        MaterialState validation.
    :raises UnitCellError: If unit-cell construction fails for the validated material
        identity (unsupported structure, wrong atom-type count, invalid type).
    :raises AtomValueError: If an atom type is not a recognized element symbol.
    """
    material = MaterialState(a0=a0, structure=structure, atom_types=atom_types)
    unit_cell = UnitCell()
    unit_cell.init_by_structure(material.structure, material.a0, material.atom_types)
    return MaterialState(
        a0=material.a0,
        structure=material.structure,
        atom_types=material.atom_types,
        unit_cell=unit_cell,
    )
