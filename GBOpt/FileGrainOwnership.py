"""LAMMPS serialization and reload support for explicit grain ownership.

This module owns transient candidate/file mappings and validated reconstruction.
Persistent grain identity itself is defined in :mod:`GBOpt.GrainOwnership`. LAMMPS file
parsing itself (data and dump formats) moved to :mod:`GBOpt.io.lammps` in R11 (#72);
``LammpsDataError``, ``LammpsAtomData``, ``read_lammps_data_file``,
``read_lammps_dump_file``, and ``read_lammps_structure_file`` are re-exported here
unchanged as a compatibility facade over that package, per the convention established
by ``GBOpt.GBMaker``'s facade over ``GBOpt.gbmaker``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GrainOwnership import (
    GrainOwnership,
    GrainOwnershipError,
    _readonly_copy,
    _strict_finite_real,
)
from GBOpt.io.lammps import (
    LammpsAtomData,
    LammpsDataError,
    read_lammps_data_file,
    read_lammps_dump_file,
    read_lammps_structure_file,
)

if TYPE_CHECKING:
    from GBOpt.GBManipulator import GBManipulator

__all__ = [
    "LammpsDataError",
    "LammpsAtomData",
    "read_lammps_data_file",
    "read_lammps_dump_file",
    "read_lammps_structure_file",
    "CandidateFileMapping",
    "reload_explicit_manipulator",
]


def _strict_box_dims(value: object) -> np.ndarray:
    """Validate orthogonal candidate box bounds.

    :param value: Box bounds to validate.
    :return: A floating-point array with shape ``(3, 2)``.
    :raises GrainOwnershipError: If the bounds are malformed or unordered.
    """
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3, 2):
        raise GrainOwnershipError("candidate box_dims must have shape (3, 2)")

    bounds = np.empty((3, 2), dtype=float)
    for axis in range(3):
        bounds[axis, 0] = _strict_finite_real(
            f"candidate box_dims[{axis}, 0]",
            raw[axis, 0],
        )
        bounds[axis, 1] = _strict_finite_real(
            f"candidate box_dims[{axis}, 1]",
            raw[axis, 1],
        )

    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise GrainOwnershipError("candidate box bounds must be strictly ordered")
    return bounds


def _remap_x_geometry(
    value: float | np.ndarray,
    source_box: np.ndarray,
    target_box: np.ndarray,
) -> float | np.ndarray:
    """Affinely map x-directed geometry between orthogonal boxes.

    The mapped geometry preserves its reduced x coordinate. This is the geometry
    transformation applied by an orthogonal variable-cell relaxation when atoms are
    dilated with the box; persistent grain labels are intentionally not inferred from
    the transformed coordinates.

    :param value: Scalar x coordinate or one-dimensional x-coordinate array.
    :param source_box: Source orthogonal box bounds.
    :param target_box: Target orthogonal box bounds.
    :return: Affinely transformed scalar or array in the target x box.
    """
    source_lo, source_hi = source_box[0]
    target_lo, target_hi = target_box[0]
    reduced = (np.asarray(value, dtype=float) - source_lo) / (source_hi - source_lo)
    mapped = target_lo + reduced * (target_hi - target_lo)
    if np.ndim(value) == 0:
        return float(mapped)
    return np.asarray(mapped, dtype=float)


def _strict_candidate_species(value: object) -> str:
    """Validate one candidate species symbol.

    :param value: Species value to validate.
    :return: The validated element symbol.
    :raises GrainOwnershipError: If the value is unsupported.
    """
    if not isinstance(value, str) or value not in Atom._numbers:
        raise GrainOwnershipError(f"unsupported candidate species label: {value!r}")
    return value


@dataclass(frozen=True, slots=True, init=False)
class CandidateFileMapping:
    """One candidate's transient serialization map and expected geometry."""

    _atom_ids: np.ndarray
    _labels: np.ndarray
    _species: np.ndarray
    _box_dims: np.ndarray
    gb_plane_x: float
    inplane_periodic: tuple[bool, bool]
    _left_grain_x_bounds: np.ndarray
    _right_grain_x_bounds: np.ndarray
    coordinate_tolerance: float
    _normal_topology: BoundaryNormalTopology

    def __init__(
        self,
        *,
        atom_ids: np.ndarray,
        labels: np.ndarray,
        species: np.ndarray,
        box_dims: np.ndarray,
        gb_plane_x: float,
        inplane_periodic: tuple[bool, bool],
        right_grain_x_bounds: np.ndarray | tuple[float, float],
        coordinate_tolerance: float,
        periodic_outer_x_interface: bool | None = None,
        left_grain_x_bounds: np.ndarray | tuple[float, float] | None = None,
        normal_topology: BoundaryNormalTopology | str | None = None,
    ) -> None:
        """Construct one deterministic candidate/file round-trip mapping.

        Candidate IDs are serialization-local identifiers assigned in candidate row
        order and must therefore be exactly ``1..N``.

        :param atom_ids: Keyword argument, required. Candidate-local IDs in row order.
        :param labels: Keyword argument, required. Persistent grain labels aligned with
            candidate rows.
        :param species: Keyword argument, required. Expected element symbols aligned
            with candidate rows.
        :param box_dims: Keyword argument, required. Expected orthogonal box bounds.
        :param gb_plane_x: Keyword argument, required. Nominal central interface plane
            in angstroms.
        :param inplane_periodic: Keyword argument, required. Explicit y/z periodicity
            flags.
        :param right_grain_x_bounds: Keyword argument, required. Physical right-grain x
            bounds.
        :param coordinate_tolerance: Keyword argument, required. Positive geometry
            tolerance in angstroms.
        :param periodic_outer_x_interface: Keyword argument, optional, defaults to
            ``None``. Legacy topology compatibility flag.
        :param left_grain_x_bounds: Keyword argument, optional, defaults to ``None``.
            Physical left-grain x bounds.
        :param normal_topology: Keyword argument, optional, defaults to ``None``.
            Explicit boundary-normal topology.
        :raises GrainOwnershipError: If the mapping, species, geometry, IDs, or topology
            are invalid.
        """
        bounds = _strict_box_dims(box_dims)
        plane = _strict_finite_real("candidate gb_plane_x", gb_plane_x)
        if not bounds[0, 0] < plane < bounds[0, 1]:
            raise GrainOwnershipError("candidate gb_plane_x must lie inside the x box")

        if left_grain_x_bounds is None:
            left_grain_x_bounds = (float(bounds[0, 0]), plane)

        ownership = GrainOwnership(
            atom_ids=atom_ids,
            labels=labels,
            gb_plane_x=plane,
            inplane_periodic=inplane_periodic,
            left_grain_x_bounds=left_grain_x_bounds,
            right_grain_x_bounds=right_grain_x_bounds,
            coordinate_tolerance=coordinate_tolerance,
            periodic_outer_x_interface=periodic_outer_x_interface,
            normal_topology=normal_topology,
        )

        ownership_ids = ownership.atom_ids
        canonical_ids = np.arange(
            1,
            ownership_ids.size + 1,
            dtype=np.int64,
        )
        if not np.array_equal(ownership_ids, canonical_ids):
            raise GrainOwnershipError(
                "candidate atom_ids must be exactly 1..N in candidate row order"
            )

        raw_species = np.asarray(species, dtype=object)
        if raw_species.ndim != 1 or raw_species.size != ownership_ids.size:
            raise GrainOwnershipError(
                "candidate species length must equal candidate atom ID count"
            )

        normalized_species = np.asarray(
            [
                _strict_candidate_species(value)
                for value in raw_species.tolist()
            ],
            dtype=Atom.atom_dtype["name"],
        )

        tolerance = ownership.coordinate_tolerance
        left_bounds = ownership.left_grain_x_bounds
        if left_bounds is None:
            raise GrainOwnershipError(
                "candidate mapping requires explicit left-grain x bounds"
            )
        right_bounds = ownership.right_grain_x_bounds

        if (
            left_bounds[0] < bounds[0, 0] - tolerance
            or right_bounds[1] > bounds[0, 1] + tolerance
        ):
            raise GrainOwnershipError(
                "candidate physical grain x bounds must lie inside the candidate box"
            )

        if (
            left_bounds[1] > plane + tolerance
            or right_bounds[0] < plane - tolerance
        ):
            raise GrainOwnershipError(
                "candidate physical grain bounds must not cross gb_plane_x"
            )

        object.__setattr__(self, "_atom_ids", _readonly_copy(ownership_ids))
        object.__setattr__(self, "_labels", ownership.labels)
        object.__setattr__(self, "_species", _readonly_copy(normalized_species))
        object.__setattr__(self, "_box_dims", _readonly_copy(bounds, dtype=float))
        object.__setattr__(self, "gb_plane_x", ownership.gb_plane_x)
        object.__setattr__(self, "inplane_periodic", ownership.inplane_periodic)
        object.__setattr__(self, "_left_grain_x_bounds", left_bounds)
        object.__setattr__(self, "_right_grain_x_bounds", right_bounds)
        object.__setattr__(self, "coordinate_tolerance", tolerance)
        object.__setattr__(self, "_normal_topology", ownership.normal_topology)

    @classmethod
    def from_candidate(
        cls,
        atoms: np.ndarray,
        labels: np.ndarray,
        *,
        box_dims: np.ndarray,
        gb_plane_x: float,
        inplane_periodic: tuple[bool, bool],
        right_grain_x_bounds: np.ndarray | tuple[float, float],
        coordinate_tolerance: float,
        periodic_outer_x_interface: bool | None = None,
        left_grain_x_bounds: np.ndarray | tuple[float, float] | None = None,
        normal_topology: BoundaryNormalTopology | str | None = None,
    ) -> CandidateFileMapping:
        """Construct a mapping from candidate atom and ownership arrays.

        :param atoms: One-dimensional structured candidate atom array containing a
            ``name`` field.
        :param labels: Persistent grain labels aligned with candidate rows.
        :param box_dims: Keyword argument, required. Expected orthogonal box bounds.
        :param gb_plane_x: Keyword argument, required. Nominal central interface plane
            in angstroms.
        :param inplane_periodic: Keyword argument, required. Explicit y/z periodicity
            flags.
        :param right_grain_x_bounds: Keyword argument, required. Physical right-grain x
            bounds.
        :param coordinate_tolerance: Keyword argument, required. Positive geometry
            tolerance in angstroms.
        :param periodic_outer_x_interface: Keyword argument, optional, defaults to
            ``None``. Legacy topology compatibility flag.
        :param left_grain_x_bounds: Keyword argument, optional, defaults to ``None``.
            Physical left-grain x bounds.
        :param normal_topology: Keyword argument, optional, defaults to ``None``.
            Explicit boundary-normal topology.
        :return: Deterministic candidate-to-file mapping with consecutive atom IDs.
        :raises GrainOwnershipError: If candidate arrays or mapping metadata are
            malformed or inconsistent.
        """
        structured = np.asarray(atoms)
        if (
            structured.ndim != 1
            or structured.dtype.names is None
            or "name" not in structured.dtype.names
        ):
            raise GrainOwnershipError(
                "candidate atoms must be a one-dimensional structured atom array"
            )
        candidate_labels = np.asarray(labels)
        if candidate_labels.ndim != 1 or candidate_labels.size != structured.size:
            raise GrainOwnershipError(
                "ownership length must equal candidate atom count"
            )
        atom_ids = np.arange(1, structured.size + 1, dtype=np.int64)
        return cls(
            atom_ids=atom_ids,
            labels=candidate_labels,
            species=np.asarray(structured["name"], dtype="U8"),
            box_dims=box_dims,
            gb_plane_x=gb_plane_x,
            inplane_periodic=inplane_periodic,
            left_grain_x_bounds=left_grain_x_bounds,
            right_grain_x_bounds=right_grain_x_bounds,
            coordinate_tolerance=coordinate_tolerance,
            periodic_outer_x_interface=periodic_outer_x_interface,
            normal_topology=normal_topology,
        )

    @classmethod
    def from_interface_candidate(cls, candidate: Any) -> CandidateFileMapping:
        """Construct a mapping from an interface-candidate value object.

        :param candidate: Interface-candidate-like object providing atoms, ownership,
            geometry, periodicity, and topology.
        :return: Deterministic candidate-to-file mapping with consecutive atom IDs.
        :raises GrainOwnershipError: If candidate data violate a mapping invariant.
        """
        return cls.from_candidate(
            candidate.atoms,
            candidate.grain_labels,
            box_dims=candidate.box_dims,
            gb_plane_x=candidate.gb_plane_x,
            inplane_periodic=candidate.inplane_periodic,
            left_grain_x_bounds=candidate.left_grain_x_bounds,
            right_grain_x_bounds=candidate.right_grain_x_bounds,
            coordinate_tolerance=candidate.coordinate_tolerance,
            normal_topology=candidate.normal_topology,
        )

    @property
    def atom_ids(self) -> np.ndarray:
        """Candidate-local serialization IDs."""
        return _readonly_copy(self._atom_ids)

    @property
    def labels(self) -> np.ndarray:
        """Persistent candidate grain labels."""
        return _readonly_copy(self._labels)

    @property
    def species(self) -> np.ndarray:
        """Expected species for each candidate atom ID."""
        return _readonly_copy(self._species)

    @property
    def box_dims(self) -> np.ndarray:
        """Expected orthogonal candidate box bounds."""
        return _readonly_copy(self._box_dims)

    @property
    def left_grain_x_bounds(self) -> np.ndarray:
        """Expected physical left-grain x bounds."""
        return _readonly_copy(self._left_grain_x_bounds)

    @property
    def right_grain_x_bounds(self) -> np.ndarray:
        """Expected physical right-grain x bounds."""
        return _readonly_copy(self._right_grain_x_bounds)

    @property
    def normal_topology(self) -> BoundaryNormalTopology:
        """Expected boundary-normal topology."""
        return self._normal_topology

    @property
    def periodic_outer_x_interface(self) -> bool:
        """Expected outer-x periodic-interface state.

        :return: ``True`` when the expected topology has a periodic outer x interface;
            otherwise ``False``.
        """
        return self._normal_topology.periodic_outer_x_interface

    @property
    def expected_count(self) -> int:
        """Expected number of evaluator-output atoms."""
        return int(self._atom_ids.size)

    def ownership_for_file_ids(
        self,
        file_ids: np.ndarray,
        *,
        box_dims: np.ndarray | None = None,
    ) -> GrainOwnership:
        """Align mapped ownership to evaluator-output file-row IDs and geometry.

        :param file_ids: Evaluator-output atom IDs in file-row order.
        :param box_dims: Keyword argument, optional, defaults to ``None``. Returned
            orthogonal box bounds. When supplied, x-directed interface geometry is
            affinely transformed from the candidate box into these bounds.
        :return: Immutable ownership metadata aligned to ``file_ids``.
        :raises GrainOwnershipError: If the IDs are malformed or do not exactly match
            the candidate mapping, or if supplied box bounds are invalid.
        """
        plane = self.gb_plane_x
        left_bounds = self._left_grain_x_bounds
        right_bounds = self._right_grain_x_bounds
        if box_dims is not None:
            returned_box = _strict_box_dims(box_dims)
            plane = _remap_x_geometry(plane, self._box_dims, returned_box)
            left_bounds = _remap_x_geometry(
                left_bounds,
                self._box_dims,
                returned_box,
            )
            right_bounds = _remap_x_geometry(
                right_bounds,
                self._box_dims,
                returned_box,
            )
        base = GrainOwnership(
            atom_ids=self._atom_ids,
            labels=self._labels,
            gb_plane_x=plane,
            inplane_periodic=self.inplane_periodic,
            left_grain_x_bounds=left_bounds,
            right_grain_x_bounds=right_bounds,
            coordinate_tolerance=self.coordinate_tolerance,
            normal_topology=self._normal_topology,
        )
        return base.aligned_to(file_ids)


def reload_explicit_manipulator(
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
    :param gb_thickness: Keyword argument, required. Grain-boundary region thickness in
        angstroms.
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
    :raises LammpsDataError: If the returned LAMMPS file or type mapping is malformed,
        unsupported, or ambiguous.
    :raises GrainOwnershipError: If evaluator output changes candidate atom count, IDs,
        species, disallowed box geometry, topology, or ownership alignment.
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
    # Local import avoids a module cycle with GBManipulator -> FileGrainOwnership.
    from GBOpt.GBManipulator import GBManipulator

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
