# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Central ownership-aware candidate write/reload service.

``CandidateLoader`` combines persistent grain identity (:class:`GBOpt.GrainOwnership.
GrainOwnership`, via :class:`GBOpt.FileGrainOwnership.CandidateFileMapping`) with the
candidate-local row-to-ID mapping a real :class:`GBOpt.io.lammps.LammpsDataWriter` write
assigns (:class:`GBOpt.io.WriteResult`), reads an evaluator-returned structure through the
R11 readers, validates the round trip, and reconstructs a validated
:class:`GBOpt.GBManipulator.GBManipulator`.

This is the one module in the ownership stack that imports ``GBManipulator`` at module
scope. Nothing reachable from ``GBManipulator``'s own import graph imports this module, so
doing so here does not reintroduce the cycle that previously required
``GBOpt.FileGrainOwnership.reload_explicit_manipulator`` to import ``GBManipulator``
locally inside its own function body. ``FileGrainOwnership.reload_explicit_manipulator``
is now a thin compatibility wrapper over :meth:`CandidateLoader.reload`, kept for existing
callers per issue #75's "keep existing entry points as wrappers during migration".
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np

from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.FileGrainOwnership import (
    CandidateFileMapping,
    GrainOwnershipError,
    read_lammps_structure_file,
)
from GBOpt.GBManipulator import GBManipulator
from GBOpt.io import StructureData, StructureValueError, WriteResult
from GBOpt.io.lammps import LammpsDataWriter, LammpsWriteError


class CandidateLoader:
    """Authoritative write and reload service for one explicit-ownership candidate."""

    def write_candidate(
        self,
        path: str | Path,
        atoms: np.ndarray,
        labels: np.ndarray,
        *,
        box_dims: np.ndarray,
        gb_plane_x: float,
        inplane_periodic: tuple[bool, bool],
        right_grain_x_bounds: np.ndarray | tuple[float, float],
        coordinate_tolerance: float,
        left_grain_x_bounds: np.ndarray | tuple[float, float] | None = None,
        periodic_outer_x_interface: bool | None = None,
        normal_topology: BoundaryNormalTopology | str | None = None,
        type_map: MutableMapping[str, int] | None = None,
        precision: int = 15,
    ) -> tuple[WriteResult, CandidateFileMapping]:
        """Write one candidate and build its mapping from the write's own outcome.

        The returned mapping's candidate-local atom IDs are taken from the write's real
        ``WriteResult.atom_ids`` rather than being independently assumed, so the mapping
        used for a later reload is always traceable to an actual serialization.

        :param path: Destination path for the candidate LAMMPS data file.
        :param atoms: Candidate atom rows in row order.
        :param labels: Persistent grain labels aligned with ``atoms``.
        :param box_dims: Keyword argument, required. Candidate orthogonal box bounds.
        :param gb_plane_x: Keyword argument, required. Nominal central interface plane
            in angstroms.
        :param inplane_periodic: Keyword argument, required. Explicit y/z periodicity
            flags.
        :param right_grain_x_bounds: Keyword argument, required. Physical right-grain x
            bounds.
        :param coordinate_tolerance: Keyword argument, required. Positive geometry
            tolerance in angstroms.
        :param left_grain_x_bounds: Keyword argument, optional, defaults to ``None``.
            Physical left-grain x bounds.
        :param periodic_outer_x_interface: Keyword argument, optional, defaults to
            ``None``. Legacy topology compatibility flag.
        :param normal_topology: Keyword argument, optional, defaults to ``None``.
            Explicit boundary-normal topology.
        :param type_map: Keyword argument, optional, defaults to ``None``. Species-to-
            numeric-type-ID mapping passed through to the writer.
        :param precision: Keyword argument, optional, defaults to ``15``. Decimal
            precision for written float values.
        :return: The write's outcome metadata and the resulting candidate mapping.
        :raises GrainOwnershipError: If the candidate geometry, species, or ownership are
            invalid.
        """
        bounds = np.asarray(box_dims, dtype=float)
        cell = np.diag(bounds[:, 1] - bounds[:, 0])
        origin = bounds[:, 0].copy()
        try:
            structure = StructureData(atoms, cell, origin)
        except StructureValueError as exc:
            raise GrainOwnershipError(str(exc)) from exc

        try:
            result = LammpsDataWriter().write(
                path,
                structure,
                precision=precision,
                type_map=type_map,
            )
        except LammpsWriteError as exc:
            raise GrainOwnershipError(str(exc)) from exc

        mapping = CandidateFileMapping(
            atom_ids=result.atom_ids,
            labels=labels,
            species=np.asarray(structure.atoms["name"], dtype="U8"),
            box_dims=bounds,
            gb_plane_x=gb_plane_x,
            inplane_periodic=inplane_periodic,
            right_grain_x_bounds=right_grain_x_bounds,
            coordinate_tolerance=coordinate_tolerance,
            periodic_outer_x_interface=periodic_outer_x_interface,
            left_grain_x_bounds=left_grain_x_bounds,
            normal_topology=normal_topology,
        )
        return result, mapping

    def reload(
        self,
        returned_structure: str | Path,
        *,
        candidate_mapping: CandidateFileMapping,
        unit_cell: Any,
        gb_thickness: float,
        type_dict: Mapping[object, object] | None = None,
        allow_variable_cell: bool = False,
    ) -> GBManipulator:
        """Validate and reconstruct an evaluator-returned owned candidate.

        This is the authoritative reload path for explicit-ownership GA execution.

        :param returned_structure: Path to the evaluator-returned LAMMPS structure.
        :param candidate_mapping: Keyword argument, required. Expected candidate IDs,
            species, ownership, geometry, and topology.
        :param unit_cell: Keyword argument, required. Unit cell used to initialize the
            reloaded manipulator parent.
        :param gb_thickness: Keyword argument, required. Grain-boundary region thickness
            in angstroms.
        :param type_dict: Keyword argument, optional, defaults to ``None``. Mapping in
            ``species -> type ID`` or ``type ID -> species`` form.
        :param allow_variable_cell: Keyword argument, optional, defaults to ``False``.
            Allow evaluator-returned orthogonal box bounds to differ from the submitted
            candidate. Persistent labels remain aligned by transient atom ID while the GB
            plane and physical grain x bounds are affinely remapped into the returned box.
            The evaluator is expected to dilate structure geometry affinely with the cell.
        :return: Manipulator reconstructed with explicit ownership aligned by atom ID.
        :raises TypeError: If ``allow_variable_cell`` is not Boolean.
        :raises FileNotFoundError: If the returned structure file does not exist.
        :raises LammpsDataError: If the returned LAMMPS file or type mapping is
            malformed, unsupported, or ambiguous.
        :raises GrainOwnershipError: If evaluator output changes candidate atom count,
            IDs, species, disallowed box geometry, topology, or ownership alignment.
        """
        if not isinstance(allow_variable_cell, (bool, np.bool_)):
            raise TypeError("allow_variable_cell must be a Boolean")
        variable_cell = bool(allow_variable_cell)

        snapshot = read_lammps_structure_file(returned_structure, type_dict=type_dict)
        file_ids = snapshot.atom_ids
        if snapshot.atoms.size != candidate_mapping.expected_count:
            raise GrainOwnershipError(
                "evaluator output atom count does not match the candidate"
            )
        expected_ids = candidate_mapping.atom_ids
        if not np.array_equal(np.sort(file_ids), expected_ids):
            raise GrainOwnershipError(
                "evaluator output atom IDs do not match the candidate"
            )
        order = np.argsort(file_ids, kind="stable")
        expected_species = candidate_mapping.species
        actual_species = np.asarray(snapshot.atoms["name"], dtype="U8")[order]
        if not np.array_equal(actual_species, expected_species):
            raise GrainOwnershipError(
                "evaluator output changed species/type for one or more atom IDs"
            )
        tolerance = candidate_mapping.coordinate_tolerance
        box_changed = not np.allclose(
            snapshot.box_dims, candidate_mapping.box_dims, atol=tolerance, rtol=0.0
        )
        if box_changed and not variable_cell:
            raise GrainOwnershipError(
                "evaluator output changed box bounds; variable-cell relaxation is "
                "not enabled"
            )
        if snapshot.selected_frame is not None and snapshot.boundary_periodic is None:
            raise GrainOwnershipError(
                "evaluator dump does not encode unambiguous boundary topology"
            )
        if snapshot.boundary_periodic is not None:
            expected_periodic = (
                candidate_mapping.periodic_outer_x_interface,
                *candidate_mapping.inplane_periodic,
            )
            if snapshot.boundary_periodic != expected_periodic:
                raise GrainOwnershipError("evaluator output changed boundary topology")
        ownership = candidate_mapping.ownership_for_file_ids(
            file_ids,
            box_dims=snapshot.box_dims if box_changed else None,
        )

        manipulator = GBManipulator(
            str(returned_structure),
            unit_cell=unit_cell,
            gb_thickness=gb_thickness,
            type_dict=type_dict,
            grain_ownership=ownership,
        )
        parent = manipulator.parents[0]
        if (
            parent.grain_labels is None
            or len(parent.grain_labels) != candidate_mapping.expected_count
        ):
            raise GrainOwnershipError("reloaded ownership length does not match atom count")
        return manipulator


__all__ = ["CandidateLoader"]
