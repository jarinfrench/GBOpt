# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Manipulate grain-boundary structures while preserving interface geometry.

This module owns in-memory structural transformations. External file ownership,
calculator evaluation, and optimizer policy do not belong here.
"""

import copy as copy_module
import multiprocessing as mp
import warnings
from numbers import Real
from os.path import isfile

import numpy as np

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import (
    BoundaryNormalTopology,
    normalize_boundary_normal_topology,
)
from GBOpt.GBMaker import GBMaker
from GBOpt.GrainOwnership import (
    LEFT_GRAIN_LABEL,
    RIGHT_GRAIN_LABEL,
    GrainOwnership,
    GrainOwnershipError,
)
from GBOpt.interface import InterfaceCandidate
from GBOpt.interface.types import (
    InterfaceCandidateTypeError,
    InterfaceCandidateValueError,
)
from GBOpt.io import StructureData
from GBOpt.io.lammps import LammpsDataError, read_structure_file
from GBOpt.manipulation import (
    GrainTerminationCycle,
    InterfaceSeparation,
    Manipulation,
    ManipulationArityError,
    ManipulationCapabilityError,
    ManipulationCompatibilityError,
    ManipulationConfigurationError,
    ManipulationContext,
    ManipulationRegistry,
    ManipulationResult,
    RightGrainTranslation,
    default_registry,
)
from GBOpt.manipulation.crossover import crossover_slice_and_merge
from GBOpt.manipulation.density import (
    _calculate_local_order as _calculate_local_order,  # re-exported for compatibility
)
from GBOpt.manipulation.density import (
    delaunay_insertion_sites,
    grid_insertion_sites,
    select_insertion_sites,
    select_removal_indices,
)
from GBOpt.manipulation.soft_mode import (
    _calculate_dynamical_matrix as _calculate_dynamical_matrix,  # re-exported for compatibility
)
from GBOpt.manipulation.soft_mode import soft_mode_displacement_atoms
from GBOpt.manipulation.translation import right_grain_translation_atoms
from GBOpt.UnitCell import UnitCell

# TODO: Generalize to interfaces, not just GBs


class GBManipulatorError(Exception):
    """Base class for exceptions in the GBManipulator class."""


class GBManipulatorValueError(GBManipulatorError, ValueError):
    """
    Exception raised in the GBManipulator class when an invalid value is assigned to a
    GBManipulator attribute.
    """


class GBManipulatorTypeError(GBManipulatorError, TypeError):
    """Exception raised in the GBManipulator class an invalid type is assigned to a
    GBManipulator attribute."""


class CompositionAwareCrossoverError(GBManipulatorValueError):
    """Raised when no exact formula-preserving crossover cut is available."""


def _validate_finite_real(name: str, value: object) -> float:
    """Return ``value`` as a finite float.

    :param name: Input name used in validation messages.
    :param value: Candidate finite real scalar.
    :return: Validated Python ``float``.
    :raises GBManipulatorValueError: If ``value`` is Boolean, non-real, or non-finite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise GBManipulatorTypeError(f"{name} must be a finite real value.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise GBManipulatorValueError(f"{name} must be a finite real value.")
    return normalized


def _strict_float_array(
    name: str,
    values: object,
    *,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Return a copied finite float array without coercing strings or Booleans.

    :param name: Input name used in validation messages.
    :param values: Candidate array-like object.
    :param shape: Keyword argument, required. Required array shape.
    :return: Copied float array with the requested shape.
    :raises GBManipulatorValueError: If shape or finiteness is invalid.
    :raises GBManipuylatorTypeError: If type is invalid.
    """
    raw = np.asarray(values, dtype=object)
    if raw.shape != shape:
        raise GBManipulatorValueError(
            f"{name} must have shape {shape}; got {raw.shape}.")
    normalized = np.empty(shape, dtype=float)
    for index in np.ndindex(shape):
        normalized[index] = _validate_finite_real(f"{name}{index}", raw[index])
    return normalized


def _normalize_inplane_periodic(value: object) -> tuple[bool, bool]:
    """Return strict y/z periodicity flags.

    :param value: Two Boolean periodicity flags.
    :return: Normalized ``(periodic_y, periodic_z)`` tuple.
    :raises GBManipulatorValueError: If the input is malformed.
    :raises GBManipulatorTypeError: If the input is coercive.
    """
    if not isinstance(value, (tuple, list)):
        raise GBManipulatorTypeError(
            "inplane_periodic must contain exactly two Boolean values"
        )
    if len(value) != 2:
        raise GBManipulatorValueError(
            "inplane_periodic must contain exactly two Boolean values"
        )
    normalized = []
    for axis_name, flag in zip(("y", "z"), value, strict=True):
        if not isinstance(flag, (bool, np.bool_)):
            raise GBManipulatorTypeError(
                f"{axis_name}-axis periodicity must be Boolean"
            )
        normalized.append(bool(flag))
    return normalized[0], normalized[1]


def _normalize_grain_labels(labels: object, *, expected_count: int) -> np.ndarray:
    """Return a read-only array of strict left/right labels.

    :param labels: Candidate-aligned grain labels.
    :param expected_count: Keyword argument, required. Required number of labels.
    :return: Read-only ``int8`` labels.
    :raises GBManipulatorValueError: If labels are malformed or omit a grain.
    :raises GBManipulatorTypeError: If labels are not integers.
    """
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.size != expected_count:
        raise GBManipulatorValueError(
            "grain_labels length must equal the candidate atom count"
        )
    if raw.dtype.kind not in ("i", "u"):
        raise GBManipulatorTypeError(
            "grain_labels must use an integer left/right label dtype"
        )
    if not np.all(np.isin(raw, (LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL))):
        raise GBManipulatorValueError(
            "grain_labels must contain only left and right labels"
        )
    result = np.array(raw, dtype=np.int8, copy=True)
    if result.size and (
        not np.any(result == LEFT_GRAIN_LABEL)
        or not np.any(result == RIGHT_GRAIN_LABEL)
    ):
        raise GBManipulatorValueError(
            "A nonempty interface candidate must contain both grains"
        )
    result.setflags(write=False)
    return result


def _readonly_copy(values: np.ndarray, *, dtype=None) -> np.ndarray:
    """Return a defensive read-only array copy.

    :param values: Array values to copy.
    :param dtype: Keyword argument, optional, defaults to ``None``. Optional output
        dtype.
    :return: Defensive read-only array copy.
    """
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _construct_interface_candidate(
    *,
    atoms: np.ndarray,
    box_dims: np.ndarray,
    gb_plane_x: float,
    left_grain_x_bounds: np.ndarray | tuple[float, float],
    right_grain_x_bounds: np.ndarray | tuple[float, float],
    grain_labels: np.ndarray,
    inplane_periodic: tuple[bool, bool],
    normal_topology: BoundaryNormalTopology | str,
    coordinate_tolerance: float,
    interface_separation: float = 0.0,
) -> InterfaceCandidate:
    """Construct an ``InterfaceCandidate``, translating its exceptions to this module's.

    ``InterfaceCandidate`` lives in the neutral ``GBOpt.interface`` package and raises
    its own independent exception hierarchy. Callers within this module rely on
    construction failures surfacing as ``GBManipulatorValueError``/
    ``GBManipulatorTypeError``, matching every other validation failure in this module.

    :param atoms: Keyword argument, required. Forwarded to ``InterfaceCandidate``.
    :param box_dims: Keyword argument, required. Forwarded to ``InterfaceCandidate``.
    :param gb_plane_x: Keyword argument, required. Forwarded to ``InterfaceCandidate``.
    :param left_grain_x_bounds: Keyword argument, required. Forwarded to
        ``InterfaceCandidate``.
    :param right_grain_x_bounds: Keyword argument, required. Forwarded to
        ``InterfaceCandidate``.
    :param grain_labels: Keyword argument, required. Forwarded to
        ``InterfaceCandidate``.
    :param inplane_periodic: Keyword argument, required. Forwarded to
        ``InterfaceCandidate``.
    :param normal_topology: Keyword argument, required. Forwarded to
        ``InterfaceCandidate``.
    :param coordinate_tolerance: Keyword argument, required. Forwarded to
        ``InterfaceCandidate``.
    :param interface_separation: Keyword argument, optional, defaults to ``0.0``.
        Forwarded to ``InterfaceCandidate``.
    :return: The constructed candidate.
    :raises GBManipulatorValueError: If candidate state is malformed or internally
        inconsistent.
    :raises GBManipulatorTypeError: If a candidate argument has an unsupported type.
    """
    try:
        return InterfaceCandidate(
            atoms=atoms,
            box_dims=box_dims,
            gb_plane_x=gb_plane_x,
            left_grain_x_bounds=left_grain_x_bounds,
            right_grain_x_bounds=right_grain_x_bounds,
            grain_labels=grain_labels,
            inplane_periodic=inplane_periodic,
            normal_topology=normal_topology,
            coordinate_tolerance=coordinate_tolerance,
            interface_separation=interface_separation,
        )
    except InterfaceCandidateTypeError as exc:
        raise GBManipulatorTypeError(str(exc)) from exc
    except InterfaceCandidateValueError as exc:
        raise GBManipulatorValueError(str(exc)) from exc


class ParentError(Exception):
    """Base class for exceptions in the Parent class."""


class ParentValueError(ParentError, ValueError):
    """
    Exception raised in the Parent class when an invalid value is assigned to a Parent
    attribute.
    """


class ParentFileNotFoundError(ParentError, FileNotFoundError):
    """
    Exception raised in the Parent class when the snapshot file is not found.
    """


class ParentCorruptedFileError(ParentError):
    """
    Exception raised in the Parent class when an error occurs while reading a snapshot.

    As of R13, this is the single exception Parent's file-backed construction raises
    for any structure-file reading failure -- malformed/unsupported file content,
    missing required data, or an invalid type mapping -- since these are all reported
    by ``GBOpt.io.lammps``'s readers as one flat ``LammpsDataError``, which no longer
    distinguishes them the way Parent's own removed file-syntax parsing once did.
    """


class ParentFileMissingDataError(ParentError):
    """
    Exception raised when data is missing from a snapshot that is otherwise formatted
    correctly.

    As of R13, Parent no longer raises this directly -- ``ParentCorruptedFileError``
    covers every structure-file reading failure (see its docstring). Retained only so
    existing ``except ParentFileMissingDataError`` callers keep importing successfully.
    """


class ParentsProxyError(Exception):
    """Base class for exceptions in the ParentsProxy class."""


class ParentsProxyValueError(ParentsProxyError, ValueError):
    """
    Exception raised in the ParentsProxy class when an invalid value is assigned to a
    ParentsProxy attribute.
    """


class ParentsProxyIndexError(ParentsProxyError, IndexError):
    """
    Exception raised in the ParentsProxy class when an invalid index is used. Valid
    indices are 0 and 1.
    """


class ParentsProxyTypeError(ParentsProxyError, TypeError):
    """
    Exception raised in the ParentsProxy class when an invalid type is assigned to a
    ParentsProxy attribute.
    """


def _box_dims_from_structure(structure: StructureData) -> np.ndarray:
    """Derive legacy orthogonal box bounds from a neutral structure's diagonal cell.

    Assumes ``structure.cell`` is diagonal, as every ``GBOpt.io.lammps`` reader
    produces; the same assumption and conversion ``GBOpt.io.lammps.compat`` uses to
    reconstruct ``LammpsAtomData.box_dims``.

    :param structure: Neutral structure snapshot with a diagonal cell.
    :return: A ``(3, 2)`` array of ``[lower, upper]`` bounds per axis.
    """
    widths = np.diagonal(structure.cell)
    lower = structure.origin
    return np.stack([lower, lower + widths], axis=1)


class Parent:
    """Legacy compatibility state used by :class:`GBManipulator` inputs.

    ``Parent`` retains material and derived GB-region context required by older
    manipulation methods.  It is not a separate parent-only structure domain model;
    composable manipulation state is represented by :class:`InterfaceCandidate`.

    :param system: A GBMaker instance or string containing the filename of a LAMMPS dump
        file.
    :param unit_cell: Required only if GB is specified using a LAMMPS dump file. Gives
        the nominal unit cell of the system.
    :param gb_thickness: Thickness of the GB region, optional, defaults to 10.
    :param type_dict: The map from integer to element string. The default mapping is
        1 -> 'H', 2 -> 'He', etc.
    :param grain_ownership: Keyword argument, optional, defaults to ``None``. Explicit
        persistent grain ownership for a single file-backed parent. File-backed
        construction without explicit ownership uses deprecated coordinate-based grain
        inference.
    """

    def __init__(
        self,
        system: GBMaker | str,
        *,
        unit_cell: UnitCell = None,
        gb_thickness: float = 10,
        type_dict: dict | None = None,
        grain_ownership: GrainOwnership | None = None,
    ) -> None:
        if grain_ownership is not None and not isinstance(
            grain_ownership, GrainOwnership
        ):
            raise ParentValueError("grain_ownership must be a GrainOwnership instance")
        if isinstance(system, GBMaker):
            if grain_ownership is not None:
                raise ParentValueError(
                    "grain_ownership is only valid for file-backed parents"
                )
            self.__init_by_gbmaker(system)
        else:
            if gb_thickness is None:  # defaults to 10 if passed in as None.
                gb_thickness = 10
            self.__init_by_file(
                system,
                unit_cell,
                gb_thickness,
                type_dict,
                grain_ownership,
            )
        self.__finish_init()

    @classmethod
    def from_structure(
        cls,
        structure: StructureData,
        *,
        unit_cell: UnitCell,
        gb_thickness: float = 10,
        grain_ownership: GrainOwnership | None = None,
    ) -> "Parent":
        """Construct a Parent directly from neutral structure data.

        This is the canonical construction seam: ``Parent(filename, ...)`` selects a
        reader for the file's format, produces a :class:`~GBOpt.io.StructureData`, and
        delegates to this method rather than parsing LAMMPS file syntax itself.

        :param structure: Neutral structure snapshot, typically produced by a
            ``GBOpt.io.lammps`` reader.
        :param unit_cell: Keyword argument, required. Nominal unit cell of the bulk
            structure.
        :param gb_thickness: Keyword argument, optional, defaults to ``10``. Thickness
            of the GB region, given in angstroms.
        :param grain_ownership: Keyword argument, optional, defaults to ``None``.
            Explicit persistent grain ownership. Without it, construction falls back to
            deprecated coordinate-based grain inference (see the class docstring).
        :return: A fully constructed Parent.
        :raises ParentValueError: If ``unit_cell`` is not given or ``grain_ownership``
            is not a ``GrainOwnership`` instance.
        :raises GrainOwnershipError: If ``grain_ownership`` is inconsistent with
            ``structure``.
        """
        if grain_ownership is not None and not isinstance(
            grain_ownership, GrainOwnership
        ):
            raise ParentValueError("grain_ownership must be a GrainOwnership instance")
        if not unit_cell:
            raise ParentValueError("Unit cell must be specified for files")
        if gb_thickness is None:  # defaults to 10 if passed in as None.
            gb_thickness = 10
        obj = cls.__new__(cls)
        obj.__init_from_structure_data(structure, unit_cell, gb_thickness, grain_ownership)
        obj.__finish_init()
        return obj

    def __finish_init(self) -> None:
        """Derive GB-region membership shared by every construction path.

        :raises AttributeError: If called before a construction path has populated this
            Parent's whole-system, grain, and geometry state.
        """
        x_gb = self.__gb_plane_x
        left_cut = x_gb - self.__gb_thickness / 2.0
        right_cut = x_gb + self.__gb_thickness / 2.0
        left_gb_mask = self.__left_grain["x"] > left_cut
        right_gb_mask = self.__right_grain["x"] < right_cut
        left_gb = self.__left_grain[left_gb_mask]
        right_gb = self.__right_grain[right_gb_mask]
        self.__gb_indices = np.where(
            (self.__whole_system["x"] > left_cut) & (
                self.__whole_system["x"] < right_cut)
        )[0]
        if self.__grain_ownership is None:
            self.__gb_atoms = np.hstack((left_gb, right_gb))
        else:
            # Persistent grain identity and geometric GB-region membership are
            # intentionally independent concepts.
            self.__gb_atoms = self.__whole_system[self.__gb_indices]
        self.__GBpos = self.__whole_system[
            np.where(
                np.logical_and(
                    self.__whole_system["x"] >= x_gb - self.__gb_thickness / 2,
                    self.__whole_system["x"] <= x_gb + self.__gb_thickness / 2
                )
            )
        ]

    def _to_interface_candidate(
        self,
        atoms: np.ndarray,
        grain_labels: np.ndarray,
        *,
        interface_separation: float = 0.0,
    ) -> InterfaceCandidate:
        """Convert candidate rows using this parent's interface geometry context.

        This method is the narrow compatibility bridge from legacy ``Parent`` state to
        the immutable candidate representation used by composable manipulation
        operations.  Parent/child status remains an optimizer role rather than a
        distinct candidate data type.

        :param atoms: Candidate atom rows.
        :param grain_labels: Candidate-aligned left/right labels.
        :param interface_separation: Keyword argument, optional, defaults to ``0.0``.
            Existing inserted interface separation in angstroms.
        :return: Immutable geometry-bearing interface candidate.
        :raises GBManipulatorValueError: If candidate rows or stored geometry are
            inconsistent.
        """
        return _construct_interface_candidate(
            atoms=atoms,
            box_dims=self.box_dims,
            gb_plane_x=self.gb_plane_x,
            left_grain_x_bounds=self.left_grain_x_bounds,
            right_grain_x_bounds=self.right_grain_x_bounds,
            grain_labels=grain_labels,
            inplane_periodic=self.inplane_periodic,
            normal_topology=self.normal_topology,
            coordinate_tolerance=self.coordinate_tolerance,
            interface_separation=interface_separation,
        )

    def __init_by_gbmaker(self, system: GBMaker) -> None:
        """
        Method for initializing the Parent using a GBMaker instance.

        :param system: The GBMaker instance.
        """
        self.__grain_ownership = None
        self.__initial_atom_ids = None
        self.__right_grain = system.right_grain
        self.__left_grain = system.left_grain
        self.__whole_system = system.whole_system
        self.__y_dim = system.y_dim
        self.__z_dim = system.z_dim
        self.__gb_thickness = system.gb_thickness
        self.__unit_cell = system.unit_cell
        self.__atom_radius = system.radius
        self.__box_dims = np.array(system.box_dims, dtype=float, copy=True)
        self.__inplane_periodic = tuple(system.inplane_periodic)
        self.__coordinate_tolerance = float(system.epsilon)
        try:
            self.__normal_topology = normalize_boundary_normal_topology(
                system.normal_topology
            )
        except (AttributeError, ValueError) as exc:
            raise ParentValueError(
                "GBMaker must provide valid boundary-normal topology metadata"
            ) from exc
        vacuum = float(system.vacuum_thickness)
        if self.__normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL:
            if not np.isclose(
                vacuum, 0.0, atol=self.__coordinate_tolerance, rtol=0.0
            ):
                raise ParentValueError(
                    "Periodic GBMaker topology requires zero vacuum thickness"
                )
            vacuum = 0.0
        elif self.__normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB:
            if vacuum <= self.__coordinate_tolerance:
                raise ParentValueError(
                    "Single-interface slab topology requires positive vacuum thickness"
                )
        else:
            raise ParentValueError(
                "GBMaker boundary-normal topology must be known explicitly"
            )
        self.__gb_plane_x = float(system.gb_plane_x)
        self.__left_grain_x_bounds = np.array(
            [self.__box_dims[0, 0] + vacuum, self.__gb_plane_x], dtype=float
        )
        self.__right_grain_x_bounds = np.array(
            [self.__gb_plane_x, self.__box_dims[0, 1] - vacuum], dtype=float
        )
        # We do not use GB.x_dim because this is limited to a single grain of the GB,
        # not the entire system.
        self.__x_dim = self.__box_dims[0][1] - self.__box_dims[0][0]

    def __init_by_file(
        self,
        system_file: str,
        unit_cell: UnitCell,
        gb_thickness: float,
        type_dict: dict | None,
        grain_ownership: GrainOwnership | None,
    ) -> None:
        """
        Method for initializing the Parent using a file.

        Selects a ``GBOpt.io.lammps`` reader by the file's own content (LAMMPS dump or
        LAMMPS data format) and delegates the resulting neutral structure to
        :meth:`from_structure`; this method itself does no LAMMPS file-syntax parsing.

        :param system_file: Filename of the atom structure file. Currently allowed
            formats: LAMMPS dump file, LAMMPS input (data) file.
        :param unit_cell: Nominal unit cell of the bulk structure.
        :param gb_thickness: Thickness of the GB region, given in angstroms.
        :param type_dict: Conversion from type number to type name, optional. Note that
            if this is not provided and the snapshot does not indicate the atom names,
            atom names are assumed started from "H".
        :raises ParentValueError: Exception raised if unit_cell is not passed in.
        :raises ParentFileNotFoundError: Exception raised if the specified file is not
            found.
        :raises ParentCorruptedFileError: Exception raised if the file's format is
            unrecognized, or its content, type mapping, geometry, or topology is
            malformed, ambiguous, or unsupported. See that exception's docstring for why
            this single exception now covers every reader failure.
        """
        if not unit_cell:
            raise ParentValueError("Unit cell must be specified for files")
        if not isfile(system_file):
            raise ParentFileNotFoundError(f"{system_file} does not exist.")
        try:
            structure = read_structure_file(system_file, type_dict=type_dict)
        except LammpsDataError as exc:
            raise ParentCorruptedFileError(
                f"Unable to read {system_file} as a supported LAMMPS structure file: "
                f"{exc}"
            ) from exc

        if grain_ownership is None:
            warnings.warn(
                "File-backed Parent initialization without explicit grain "
                "ownership is deprecated because gb_plane_x and grain "
                "membership must be inferred from coordinates. Supply "
                "grain_ownership with explicit interface metadata instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        self.__init_from_structure_data(structure, unit_cell, gb_thickness, grain_ownership)

    def __init_from_structure_data(
        self,
        structure: StructureData,
        unit_cell: UnitCell,
        gb_thickness: float,
        grain_ownership: GrainOwnership | None,
    ) -> None:
        """Populate this Parent's file-backed state from neutral structure data.

        With ``grain_ownership`` supplied, persistent grain identity is restored
        explicitly (see :meth:`__init_from_owned_structure`). Without it, grain
        membership and ``gb_plane_x`` are inferred from the box x-midpoint -- the
        deprecated legacy fallback documented on the class.

        :param structure: Neutral structure snapshot to construct from.
        :param unit_cell: Nominal unit cell of the bulk structure.
        :param gb_thickness: Thickness of the GB region, given in angstroms.
        :param grain_ownership: Explicit persistent grain ownership, or ``None`` for the
            legacy geometric fallback.
        :raises GrainOwnershipError: If ``grain_ownership`` is inconsistent with
            ``structure``.
        """
        self.__unit_cell = unit_cell
        self.__gb_thickness = gb_thickness
        self.__inplane_periodic = (True, True)
        self.__coordinate_tolerance = 1.0e-10
        self.__grain_ownership = None
        self.__initial_atom_ids = None
        self.__normal_topology = BoundaryNormalTopology.UNKNOWN

        if grain_ownership is not None:
            self.__init_from_owned_structure(structure, grain_ownership)
            return

        box_dims = _box_dims_from_structure(structure)
        self.__whole_system = np.array(structure.atoms, copy=True)
        self.__box_dims = box_dims
        self.__x_dim = float(box_dims[0, 1] - box_dims[0, 0])
        self.__y_dim = float(box_dims[1, 1] - box_dims[1, 0])
        self.__z_dim = float(box_dims[2, 1] - box_dims[2, 0])
        # TODO: Need a more robust calculation of where the GB is located.
        grain_cutoff = (box_dims[0, 1] - box_dims[0, 0]) / 2 + box_dims[0, 0]
        mask = self.__whole_system["x"] < grain_cutoff
        self.__left_grain = self.__whole_system[mask]
        self.__right_grain = self.__whole_system[~mask]
        self.__gb_plane_x = (
            max(self.__left_grain["x"]) + min(self.__right_grain["x"])) / 2
        self.__left_grain_x_bounds = np.array(
            [box_dims[0, 0], self.__gb_plane_x], dtype=float
        )
        self.__right_grain_x_bounds = np.array(
            [self.__gb_plane_x, box_dims[0, 1]], dtype=float
        )

    def __init_from_owned_structure(
        self,
        structure: StructureData,
        grain_ownership: GrainOwnership,
    ) -> None:
        """Restore a file-backed parent from explicit ownership and parsed rows.

        :param structure: Neutral structure snapshot in file-row order.
        :param grain_ownership: Explicit ownership keyed by serialization-local IDs.
        :raises GrainOwnershipError: If IDs, geometry, or topology are inconsistent.
        """
        file_ids = structure.external_ids
        # Every GBOpt.io.lammps reader always populates external_ids for LAMMPS data
        # and dump formats; GBOpt.io.lammps.compat._to_lammps_atom_data relies on the
        # same invariant.
        assert file_ids is not None
        aligned = grain_ownership.aligned_to(file_ids)
        order = np.argsort(file_ids, kind="stable")
        sorted_ids = file_ids[order]
        ownership = aligned.aligned_to(sorted_ids)
        atoms = structure.atoms[order]
        box_dims = _box_dims_from_structure(structure)
        tolerance = ownership.coordinate_tolerance
        left_bounds = ownership.left_grain_x_bounds
        right_bounds = ownership.right_grain_x_bounds
        if left_bounds is None:
            raise GrainOwnershipError(
                "explicit file loading requires left-grain x bounds"
            )
        if not box_dims[0, 0] < ownership.gb_plane_x < box_dims[0, 1]:
            raise GrainOwnershipError(
                "gb_plane_x must lie strictly inside the file x bounds"
            )
        if (
            left_bounds[0] < box_dims[0, 0] - tolerance
            or right_bounds[1] > box_dims[0, 1] + tolerance
        ):
            raise GrainOwnershipError(
                "physical grain x bounds must lie inside the file box"
            )
        if structure.periodicity is not None:
            expected = (
                ownership.periodic_outer_x_interface,
                *ownership.inplane_periodic,
            )
            if structure.periodicity != expected:
                raise GrainOwnershipError(
                    "file boundary topology does not match explicit ownership"
                )

        labels = ownership.labels
        self.__whole_system = np.array(atoms, copy=True)
        self.__left_grain = self.__whole_system[labels == LEFT_GRAIN_LABEL]
        self.__right_grain = self.__whole_system[labels == RIGHT_GRAIN_LABEL]
        if not len(self.__left_grain) or not len(self.__right_grain):
            raise GrainOwnershipError(
                "explicit ownership must contain both left and right grains"
            )
        self.__box_dims = np.array(box_dims, dtype=float, copy=True)
        self.__x_dim = float(self.__box_dims[0, 1] - self.__box_dims[0, 0])
        self.__y_dim = float(self.__box_dims[1, 1] - self.__box_dims[1, 0])
        self.__z_dim = float(self.__box_dims[2, 1] - self.__box_dims[2, 0])
        self.__gb_plane_x = ownership.gb_plane_x
        self.__inplane_periodic = ownership.inplane_periodic
        self.__coordinate_tolerance = tolerance
        self.__left_grain_x_bounds = np.array(left_bounds, dtype=float, copy=True)
        self.__right_grain_x_bounds = np.array(
            right_bounds,
            dtype=float,
            copy=True,
        )
        self.__normal_topology = ownership.normal_topology
        self.__initial_atom_ids = np.array(sorted_ids, dtype=np.int64, copy=True)
        self.__initial_atom_ids.setflags(write=False)
        self.__grain_ownership = GrainOwnership(
            atom_ids=self.__initial_atom_ids,
            labels=labels,
            gb_plane_x=self.__gb_plane_x,
            inplane_periodic=self.__inplane_periodic,
            left_grain_x_bounds=self.__left_grain_x_bounds,
            right_grain_x_bounds=self.__right_grain_x_bounds,
            coordinate_tolerance=self.__coordinate_tolerance,
            normal_topology=self.__normal_topology,
        )

    # Getters

    @property
    def left_grain(self) -> np.ndarray:
        return self.__left_grain

    @property
    def right_grain(self) -> np.ndarray:
        return self.__right_grain

    @property
    def whole_system(self) -> np.ndarray:
        return self.__whole_system

    @property
    def gb_atoms(self) -> np.ndarray:
        return self.__gb_atoms

    @property
    def unit_cell(self) -> UnitCell:
        return self.__unit_cell

    @property
    def gb_indices(self) -> np.ndarray:
        return self.__gb_indices

    @property
    def gb_thickness(self) -> float:
        return self.__gb_thickness

    @property
    def box_dims(self) -> np.ndarray:
        return self.__box_dims

    @property
    def x_dim(self) -> float:
        return self.__x_dim

    @property
    def y_dim(self) -> float:
        return self.__y_dim

    @property
    def z_dim(self) -> float:
        return self.__z_dim

    @property
    def gb_plane_x(self) -> float:
        """The physical central boundary plane or gap midpoint."""
        return self.__gb_plane_x

    @property
    def inplane_periodic(self) -> tuple[bool, bool]:
        """y/z periodicity flags."""
        return tuple(self.__inplane_periodic)

    @property
    def normal_topology(self) -> BoundaryNormalTopology:
        """Explicit boundary-normal topology."""
        return self.__normal_topology

    @property
    def periodic_outer_x_interface(self) -> bool:
        """Whether the outer x faces form a second interface."""
        return self.__normal_topology.periodic_outer_x_interface

    @property
    def coordinate_tolerance(self) -> float:
        """Coordinate tolerance in angstroms."""
        return self.__coordinate_tolerance

    @property
    def left_grain_x_bounds(self) -> np.ndarray:
        """Copy of the left physical grain interval."""
        return np.array(self.__left_grain_x_bounds, dtype=float, copy=True)

    @property
    def right_grain_x_bounds(self) -> np.ndarray:
        """Copy of the right physical grain interval."""
        return np.array(self.__right_grain_x_bounds, dtype=float, copy=True)

    @property
    def grain_ownership(self) -> GrainOwnership | None:
        """Defensive copy of explicit persistent ownership, when present."""
        if self.__grain_ownership is None:
            return None
        return copy_module.copy(self.__grain_ownership)

    @property
    def grain_labels(self) -> np.ndarray | None:
        """Persistent labels aligned with ``whole_system`` rows, when present."""
        if self.__grain_ownership is None:
            return None
        return self.__grain_ownership.labels

    @property
    def initial_atom_ids(self) -> np.ndarray | None:
        """Initial serialization IDs while they remain applicable."""
        if self.__initial_atom_ids is None:
            return None
        result = np.array(self.__initial_atom_ids, dtype=np.int64, copy=True)
        result.setflags(write=False)
        return result

    def __copy__(self):
        """Independent shallow copy preserving explicit ownership."""
        result = type(self).__new__(type(self))
        for name, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                value = value.copy()
            elif isinstance(value, GrainOwnership):
                value = copy_module.copy(value)
            setattr(result, name, value)
        return result

    def __deepcopy__(self, memo):
        """Independent deep copy preserving explicit ownership."""
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                copied = value.copy()
            else:
                copied = copy_module.deepcopy(value, memo)
            setattr(result, name, copied)
        return result


class _ParentsProxy:
    """
    Class for allowing for access to parents in the GBManipulator class by index.

    :param manipulator: The instance of GBManipulator that the ParentsProxy class acts
        for.
    """

    def __init__(self, manipulator) -> None:
        self.__manipulator = manipulator

    def __getitem__(self, index) -> Parent:
        return self.__manipulator._GBManipulator__parents[index]

    def __setitem__(self, index, value) -> None:
        """
        Method allowing for setting the parents of the GBManipulator class by index.

        :param index: The index to assign to. Valid values are 0 and 1, and the 0th
            index must be assigned to first.
        :param value: The Parent instance to assign to the GBManipulator parents
            attribute.
        :raises ParentsProxyIndexError: Exception raised when an index other than 0 or 1
            is passed in.
        :raises ParentsProxyTypeError: Exception raised when an incorrect type is passed
            in as to the parents attribute.
        :raises ParentsProxyValueError: Exception raised when attempting to assign to
            the second parent first. As most mutators act on the first parent, assigning
            to the first value is required.
        """
        if index not in (0, 1):  # Only valid values are 0 and 1: max of 2 parents.
            raise ParentsProxyIndexError("Index out of range. Index must be 0 or 1.")

        if not (value is None or isinstance(value, Parent)):
            raise ParentsProxyTypeError("Value must be None or a instance of Parent")

        # Since most of the manipulators act on the first parent, we make sure that
        # assignments are made first to index 0.
        if index == 1 and self.__manipulator._GBManipulator__parents[0] is None:
            raise ParentsProxyValueError("parents[0] must be assigned to first.")

        parents = self.__manipulator._GBManipulator__parents[:]
        parents[index] = value
        self.__manipulator._GBManipulator__parents = parents
        if parents[0] is not None:
            self.__manipulator._GBManipulator__candidate_grain_labels = (
                self.__manipulator._GBManipulator__initial_candidate_labels()
            )

    def __len__(self) -> int:
        """
        Method for returning the length of the parents list. This value should always be
        2, even if no parents are assigned.

        :return: 2, the length of the parents attribute.
        """
        return len(self.__manipulator._GBManipulator__parents)


class GBManipulator:
    """
    Class to manipulate atoms in the grain boundary region.

    :param system1: The GBMaker instance containing the generated GB or the filename
        containing the name of the LAMMPS dump file. First parent.
    :param system2: The GBMaker instance containing the generated GB or the filename
        containing the name of the LAMMPS dump file for the second parent, optional,
        defaults to None.
    :param unit_cell: The unit cell of the system. Required if GB1 or GB2 is a LAMMPS
        dump file.
    :param gb_thickness: Thickness of the GB region, optional, defaults to 10.
    :param seed: The seed for random number generation, optional, defaults to None
        (automatically seeded).
    :param type_dict: The mapping of integer to string types. If not specified, the
        default mapping is 1 -> 'H', 2 -> 'He', etc.
    :param grain_ownership: Keyword argument, optional, defaults to ``None``. Explicit
        persistent ownership for a single file-backed first parent.
    """

    def __init__(
        self,
        system1: GBMaker | str,
        system2: GBMaker | str = None,
        *,
        gb_thickness: float = None,
        unit_cell: UnitCell = None,
        seed: int = None,
        type_dict: dict | None = None,
        grain_ownership: GrainOwnership | None = None,
    ) -> None:
        if grain_ownership is not None and not isinstance(
            grain_ownership, GrainOwnership
        ):
            raise GBManipulatorTypeError(
                "grain_ownership must be a GrainOwnership instance"
            )
        # initialize the random number generator
        if not seed:
            self.__rng = np.random.default_rng()
        else:
            self.__rng = np.random.default_rng(seed=seed)

        self.__last_crossover_provenance: tuple[tuple[str, object], ...] | None = None

        self.__parents = [None, None]

        if not system2:
            # Some mutators require two parents, so we set __one_parent to True so we do
            # not attempt to perform those in the case that only one GB is passed in.
            self.__one_parent = True
            self.__set_parents(
                system1,
                unit_cell=unit_cell,
                gb_thickness=gb_thickness,
                type_dict=type_dict,
                grain_ownership=grain_ownership,
            )
        else:
            if grain_ownership is not None:
                raise GBManipulatorValueError(
                    "grain_ownership is only supported for a single file-backed parent"
                )
            self.__one_parent = False
            self.__set_parents(system1, system2, unit_cell=unit_cell,
                               gb_thickness=gb_thickness, type_dict=type_dict)
        self.__num_processes = mp.cpu_count() // 2 or 1
        self.__candidate_grain_labels = self.__initial_candidate_labels()

    @classmethod
    def _from_parents(
        cls,
        parent1: Parent,
        parent2: Parent | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> "GBManipulator":
        """Construct a manipulator from already validated parent state.

        :param parent1: First parent.
        :param parent2: Second parent, optional, defaults to ``None``.
        :param rng: Keyword argument, optional, defaults to ``None``. Random-number
            generator to attach to the manipulator.
        :return: Manipulator containing defensive parent copies.
        :raises GBManipulatorValueError: If either supplied parent is invalid.
        """
        if not isinstance(parent1, Parent) or (
            parent2 is not None and not isinstance(parent2, Parent)
        ):
            raise GBManipulatorValueError("_from_parents requires Parent instances")
        result = cls.__new__(cls)
        result.__rng = np.random.default_rng() if rng is None else rng
        result.__last_crossover_provenance = None
        result.__parents = [copy_module.copy(parent1), None]
        if parent2 is not None:
            result.__parents[1] = copy_module.copy(parent2)
        result.__one_parent = parent2 is None
        result.__num_processes = mp.cpu_count() // 2 or 1
        result.__candidate_grain_labels = result.__initial_candidate_labels()
        return result

    def __initial_candidate_labels(self) -> np.ndarray | None:
        """Return labels aligned with the first parent's current row order."""
        labels = self.__parents[0].grain_labels
        if labels is None:
            return None
        return _normalize_grain_labels(
            labels,
            expected_count=len(self.__parents[0].whole_system),
        )

    def __set_candidate_labels(self,
                               labels: np.ndarray | None,
                               expected_count: int,
                               ) -> None:
        """Store labels aligned with the most recently produced candidate."""
        if labels is None:
            self.__candidate_grain_labels = None
            return
        self.__candidate_grain_labels = _normalize_grain_labels(
            labels,
            expected_count=expected_count,
        )

    def __set_parents(
            self,
            system1: GBMaker | str,
            system2: GBMaker | str = None,
            *,
            unit_cell: UnitCell = None,
            gb_thickness: float = None,
            type_dict: dict | None = None,
            grain_ownership: GrainOwnership | None = None,
    ) -> None:
        """
        Method to assign the parent(s) that will create the child(ren).

        :param system1: The first parent.
        :param system2: The second parent, optional, defaults to None.
        :param unit_cell: Keyword argument. The nominal unit cell of the bulk structure,
            optional, defaults to None. Required only when system1 is of type str.
        :param gb_thickness: Keyword argument. The thickness of the GB region, optional,
            defaults to None. Note that if None is passed to the Parent class
            constructor, a value of 10 is assigned.
        :param type_dict: Keyword argument. Optional, defaults to an empty dict. The
            mapping from integer to elemental string. Default mapping is 1 -> 'H',
            2 -> 'He', etc.
        :param grain_ownership: Keyword argument, optional, defaults to ``None``.
            Explicit persistent ownership for a single file-backed first parent.
        """
        if type_dict is None:
            type_dict = unit_cell.type_map if unit_cell is not None else None
        self.__parents[0] = Parent(
            system1,
            unit_cell=unit_cell,
            gb_thickness=gb_thickness,
            type_dict=type_dict,
            grain_ownership=grain_ownership,
        )
        if system2 is not None:
            # If there are 2 parents, with the first one being of type GBMaker, and
            # unit_cell has not been passed in, we assume that the unit cell from the
            # GBMaker instance applies to the second system.
            if isinstance(system1, GBMaker) and isinstance(system2, str):
                if unit_cell is None:
                    unit_cell = system1.unit_cell
                if gb_thickness is None:
                    gb_thickness = system1.gb_thickness
            self.__parents[1] = Parent(
                system2, unit_cell=unit_cell, gb_thickness=gb_thickness, type_dict=type_dict)

    @property
    def candidate_grain_labels(self) -> np.ndarray | None:
        """Return labels aligned with the most recently produced candidate."""
        if self.__candidate_grain_labels is None:
            return None
        result = np.array(self.__candidate_grain_labels, dtype=np.int8, copy=True)
        result.setflags(write=False)
        return result

    @property
    def last_crossover_provenance(self) -> tuple[tuple[str, object], ...] | None:
        """Return immutable parameters for the most recent crossover.

        :return: Ordered crossover parameter pairs, or ``None`` before crossover.
        """
        return self.__last_crossover_provenance

    @property
    def rng(self):
        return self.__rng

    @rng.setter
    def rng(self, rng: np.random.default_rng) -> None:
        self.__rng = rng

    # TODO: Swap to use Atom class if it can be vectorized for each of these mutators.

    @staticmethod
    def __concatenated_labels(parent: Parent, count: int) -> np.ndarray:
        """Return left-then-right labels aligned with a parent-derived candidate.

        :param parent: Parent supplying left and right grain populations.
        :param count: Candidate row count.
        :return: Integer labels aligned with the concatenated grain rows.
        :raises GBManipulatorValueError: If the row count differs from the parent
            population.
        """
        left_count = len(parent.left_grain)
        right_count = len(parent.right_grain)
        if left_count + right_count != count:
            raise GBManipulatorValueError(
                "candidate row count does not match the parent grain populations"
            )
        return np.hstack(
            (
                np.full(left_count, LEFT_GRAIN_LABEL, dtype=np.int8),
                np.full(right_count, RIGHT_GRAIN_LABEL, dtype=np.int8),
            )
        )

    def __parent_candidate_geometry(self, index: int) -> InterfaceCandidate:
        """Convert ``self.__parents[index]`` into an immutable interface candidate.

        Reuses that parent's own geometry (physical box and grain bounds, coordinate
        tolerance, whatever its boundary-normal topology currently is) exactly as it is
        stored, so this conversion threads the same interface state through to a
        manipulation operation without recomputing or otherwise touching it. Does not
        require boundary-normal topology to be resolved -- callers that need that
        guarantee use ``__parent_candidate`` instead -- and does not update
        ``__candidate_grain_labels``; callers that mean to record the produced candidate
        as this manipulator's new "current" single-parent candidate do that themselves.

        :param index: Parent index to convert.
        :return: Geometry-bearing candidate for that parent.
        """
        parent = self.__parents[index]
        labels = parent.grain_labels
        if labels is None:
            labels = self.__concatenated_labels(parent, len(parent.whole_system))
        return parent._to_interface_candidate(parent.whole_system, labels)

    def __parent_candidate(self, index: int) -> InterfaceCandidate:
        """Convert ``self.__parents[index]`` into an immutable interface candidate.

        :param index: Parent index to convert.
        :return: Geometry-bearing candidate for that parent.
        :raises GBManipulatorValueError: If that parent's boundary-normal topology is
            unknown.
        """
        parent = self.__parents[index]
        if parent.normal_topology is BoundaryNormalTopology.UNKNOWN:
            raise GBManipulatorValueError(
                "a parent candidate requires known boundary-normal topology"
            )
        return self.__parent_candidate_geometry(index)

    @staticmethod
    def __translate_manipulation_error(
        func,
        *args,
        capability_exception: type[GBManipulatorValueError] = GBManipulatorValueError,
        **kwargs,
    ):
        """Call ``func(*args, **kwargs)``, translating manipulation-operation
        exceptions back to this class's own established public exception identities.

        A ``GBOpt.manipulation`` operation raises a plain ``TypeError`` for a
        malformed parameter type and ``ManipulationConfigurationError`` /
        ``ManipulationCompatibilityError`` (both ``ValueError``-rooted) for a
        parent-independent or parent-vs-parent validation failure, matching this
        class's own long-standing ``_validate_finite_real`` type/value split. This
        boundary is this class's counterpart to the ``__translate_construction_error``
        pattern documented for other extracted-logic wrappers in this codebase,
        adapted to the ``Manipulation`` protocol's own exception vocabulary instead of
        an ``InterfaceCandidate``-construction one.

        ``ManipulationCapabilityError`` (a failure specific to the actual parents/
        parameters given, not a structural mismatch) is translated to
        ``capability_exception`` instead of the generic ``GBManipulatorValueError``,
        so a caller needing a more specific established identity for that case (e.g.
        ``slice_and_merge``'s ``CompositionAwareCrossoverError``) can request it without
        a second, near-duplicate translation helper.

        :param func: Callable to invoke.
        :param args: Positional arguments forwarded to ``func``.
        :param capability_exception: Keyword argument, optional, defaults to
            ``GBManipulatorValueError``. Exception raised in place of
            ``ManipulationCapabilityError``.
        :param kwargs: Keyword arguments forwarded to ``func``.
        :return: ``func``'s return value.
        :raises GBManipulatorTypeError: If ``func`` raises ``TypeError``.
        :raises GBManipulatorValueError: If ``func`` raises
            ``ManipulationConfigurationError`` or ``ManipulationCompatibilityError``.
        """
        try:
            return func(*args, **kwargs)
        except TypeError as exc:
            raise GBManipulatorTypeError(str(exc)) from exc
        except ManipulationCapabilityError as exc:
            raise capability_exception(str(exc)) from exc
        except (ManipulationConfigurationError, ManipulationCompatibilityError) as exc:
            raise GBManipulatorValueError(str(exc)) from exc

    def make_parent_candidate(self) -> InterfaceCandidate:
        """Return the first parent as a complete immutable interface candidate.

        :return: Geometry-bearing parent candidate.
        :raises GBManipulatorValueError: If the manipulator has two parents or topology
            metadata is unavailable.
        """
        if not self.__one_parent:
            raise GBManipulatorValueError(
                "a parent candidate requires exactly one parent"
            )
        candidate = self.__parent_candidate(0)
        self.__set_candidate_labels(candidate.grain_labels, len(candidate.atoms))
        return candidate

    def __current_parent_candidates(self) -> tuple[InterfaceCandidate, ...]:
        """Return this manipulator's own current parent(s) as immutable candidates.

        :return: One candidate per active parent, in parent order.
        :raises GBManipulatorValueError: If a parent's boundary-normal topology is
            unknown.
        """
        count = 1 if self.__one_parent else 2
        return tuple(self.__parent_candidate(index) for index in range(count))

    def apply(
        self,
        manipulation: Manipulation,
        *,
        seed: int | None = None,
        **params: object,
    ) -> ManipulationResult:
        """Run ``manipulation`` against this manipulator's current parent(s).

        This is a generic facade seam: it converts this manipulator's own parent state
        into immutable candidates (preserving interface topology, ownership labels,
        physical grain bounds, and interface-separation state exactly as the rest of this
        class already constructs them via ``_to_interface_candidate``), checks the
        operation's declared arity against the number of parents actually available, and
        executes it. No built-in manipulation algorithm is routed through this seam; it
        exists so an operation defined outside ``GBOpt`` can run against real manipulator
        state.

        :param manipulation: Operation to execute; may be defined outside GBOpt.
        :param seed: Keyword argument, optional, defaults to ``None``. When given, used
            to construct a fresh, independent ``np.random.Generator`` for this call
            (``seed=0`` is a valid, deterministic seed, unlike this class's own
            constructor). When omitted, this manipulator's own random-number generator
            is used, and its state advances as a result of the call, matching every
            other randomized method on this class.
        :param params: Additional operation-specific keyword parameters, forwarded to
            the operation unmodified.
        :return: The operation's result.
        :raises ManipulationConfigurationError: If ``manipulation.arity`` is not a
            positive integer.
        :raises ManipulationArityError: If ``manipulation.arity`` does not match the
            number of parents this manipulator currently has.
        :raises GBManipulatorValueError: If a parent's boundary-normal topology is
            unknown.
        """
        parents = self.__current_parent_candidates()
        arity = manipulation.arity
        if not isinstance(arity, int) or isinstance(arity, bool) or arity < 1:
            raise ManipulationConfigurationError(
                f"{manipulation.name!r} arity must be a positive integer"
            )
        if arity != len(parents):
            raise ManipulationArityError(
                f"{manipulation.name!r} requires {arity} parent(s); this manipulator "
                f"currently provides {len(parents)}"
            )
        rng = self.__rng if seed is None else np.random.default_rng(seed)
        context = ManipulationContext(parents=parents, rng=rng, params=params)
        return manipulation.execute(context)

    def apply_named(
        self,
        name: str,
        *,
        seed: int | None = None,
        registry: ManipulationRegistry | None = None,
        **params: object,
    ) -> ManipulationResult:
        """Run the operation registered as ``name`` against this manipulator.

        :param name: Registered operation name.
        :param seed: Keyword argument, optional, defaults to ``None``. Forwarded to
            ``apply``.
        :param registry: Keyword argument, optional, defaults to ``None``. Registry to
            look ``name`` up in; defaults to ``GBOpt.manipulation.default_registry``.
        :param params: Additional operation-specific keyword parameters, forwarded to
            ``apply``.
        :return: The operation's result.
        :raises ManipulationLookupError: If ``name`` is not registered in the resolved
            registry.
        :raises ManipulationConfigurationError: If the registered operation's arity is
            not a positive integer.
        :raises ManipulationArityError: If the registered operation's arity does not
            match the number of parents this manipulator currently has.
        """
        active_registry = default_registry if registry is None else registry
        manipulation = active_registry.get(name)
        return self.apply(manipulation, seed=seed, **params)

    def translate_right_grain(
        self,
        dy: float,
        dz: float,
        *,
        dx: float = 0.0,
    ) -> np.ndarray:
        """Rigidly translate the right grain.

        Delegates to ``GBOpt.manipulation.translation.right_grain_translation_atoms``,
        the pure computational core also used by
        ``GBOpt.manipulation.RightGrainTranslation.execute``, translating its
        exceptions back to this class's own established public exception identities.
        Calls that function directly rather than routing through
        ``RightGrainTranslation.execute`` itself: explicit ownership is persistent
        state, and a relaxed right-grain atom may legitimately cross the nominal
        interface plane, so this method's long-standing contract of returning raw,
        un-revalidated atom rows cannot be satisfied by ``execute``'s
        ``InterfaceCandidate``-validated result (see
        ``right_grain_translation_atoms``'s own docstring).

        :param dy: Displacement in y in angstroms.
        :param dz: Displacement in z in angstroms.
        :param dx: Keyword argument, optional, defaults to ``0.0``. Displacement in x in
            angstroms.
        :return: Left-grain rows followed by translated right-grain rows.
        :raises GBManipulatorValueError: If a displacement is invalid or moves atoms
            outside a supported interval.
        """
        if not self.__one_parent:
            warnings.warn(
                "grain translation only occurring based on parent 1",
                UserWarning,
                stacklevel=2,
            )

        dx = _validate_finite_real("dx", dx)
        dy = _validate_finite_real("dy", dy)
        dz = _validate_finite_real("dz", dz)

        parent = self.__parents[0]
        atoms = self.__translate_manipulation_error(
            right_grain_translation_atoms,
            left_atoms=parent.left_grain,
            right_atoms=parent.right_grain,
            right_grain_x_bounds=parent.right_grain_x_bounds,
            box_dims=parent.box_dims,
            inplane_periodic=parent.inplane_periodic,
            coordinate_tolerance=parent.coordinate_tolerance,
            dx=dx,
            dy=dy,
            dz=dz,
        )
        labels = self.__concatenated_labels(parent, len(atoms))
        self.__set_candidate_labels(labels, len(atoms))
        return atoms

    def make_translation_candidate(
        self,
        dy: float,
        dz: float,
        *,
        dx: float = 0.0,
    ) -> InterfaceCandidate:
        """Return a geometry-bearing right-grain translation candidate.

        :param dy: Displacement in y in angstroms.
        :param dz: Displacement in z in angstroms.
        :param dx: Keyword argument, optional, defaults to ``0.0``. Displacement in x
            in angstroms.
        :return: Complete immutable translated candidate.
        :raises GBManipulatorValueError: If the manipulator does not have exactly one
            parent, topology is unknown, or a displacement is invalid.
        """
        if not self.__one_parent:
            raise GBManipulatorValueError(
                "a translation candidate requires exactly one parent"
            )

        parent = self.__parents[0]
        if parent.normal_topology is BoundaryNormalTopology.UNKNOWN:
            raise GBManipulatorValueError(
                "a translation candidate requires known boundary-normal topology"
            )

        context = ManipulationContext(
            parents=(self.__parent_candidate_geometry(0),),
            rng=self.__rng,
            params={"dx": dx, "dy": dy, "dz": dz},
        )
        result = self.__translate_manipulation_error(
            RightGrainTranslation().execute, context
        )
        child = result.children[0]
        self.__set_candidate_labels(child.grain_labels, len(child.atoms))
        return child

    def cycle_grain_terminations(
        self,
        *,
        left_phase_shift: float = 0.0,
        right_phase_shift: float = 0.0,
        right_dy: float = 0.0,
        right_dz: float = 0.0,
    ) -> np.ndarray:
        """Cycle grain-local terminations for a periodic bicrystal or slab.

        Each grain is cycled independently through its finite physical x interval. The
        left grain remains fixed in-plane, while the right grain may also be translated
        along the periodic in-plane directions.

        For a periodic bicrystal, the physical grain bounds must span the complete x box
        so that the central and outer periodic interfaces remain consistent.

        For a single-interface slab, the physical grain bounds must lie inside the x box
        with at least one free-surface or vacuum interval. Cycling a complete finite
        grain changes both its GB-facing and free-surface terminations; those
        terminations therefore remain coupled.

        :param left_phase_shift: Keyword argument, optional, defaults to ``0.0``.
            Left-grain x phase shift in angstroms.
        :param right_phase_shift: Keyword argument, optional, defaults to ``0.0``.
            Right-grain x phase shift in angstroms.
        :param right_dy: Keyword argument, optional, defaults to ``0.0``. Right-grain y
            translation in angstroms.
        :param right_dz: Keyword argument, optional, defaults to ``0.0``. Right-grain z
            translation in angstroms.
        :return: Left-grain rows followed by right-grain rows after termination cycling
            and optional right-grain in-plane translation.
        :raises GBManipulatorValueError: If the manipulator does not have exactly one
            parent, the boundary-normal topology is unknown, the physical grain or box
            geometry is invalid for that topology, parent atoms lie outside their
            physical grain bounds, or a displacement is invalid.
        """
        if not self.__one_parent:
            raise GBManipulatorValueError(
                "termination cycling requires exactly one parent"
            )

        context = ManipulationContext(
            parents=(self.__parent_candidate_geometry(0),),
            rng=self.__rng,
            params={
                "left_phase_shift": left_phase_shift,
                "right_phase_shift": right_phase_shift,
                "right_dy": right_dy,
                "right_dz": right_dz,
            },
        )
        result = self.__translate_manipulation_error(
            GrainTerminationCycle().execute, context
        )
        return np.array(result.children[0].atoms, copy=True)

    # TODO: Independent GB-only slab termination control that preserves each outer
    #   free-surface termination is deferred. The current operation intentionally cycles
    #   each complete finite grain and couples its GB-facing and surface phases.
    def make_termination_candidate(
        self,
        *,
        left_phase_shift: float = 0.0,
        right_phase_shift: float = 0.0,
        right_dy: float = 0.0,
        right_dz: float = 0.0,
    ) -> InterfaceCandidate:
        """Return a geometry-bearing termination and registry candidate.

        :param left_phase_shift: Keyword argument, optional, defaults to ``0.0``. Left
            x phase shift in angstroms.
        :param right_phase_shift: Keyword argument, optional, defaults to ``0.0``. Right
            x phase shift in angstroms.
        :param right_dy: Keyword argument, optional, defaults to ``0.0``. Right-grain y
            translation in angstroms.
        :param right_dz: Keyword argument, optional, defaults to ``0.0``. Right-grain z
            translation in angstroms.
        :return: Complete immutable candidate.
        :raises GBManipulatorValueError: If topology, geometry, or a displacement is
            invalid.
        """
        atoms = self.cycle_grain_terminations(
            left_phase_shift=left_phase_shift,
            right_phase_shift=right_phase_shift,
            right_dy=right_dy,
            right_dz=right_dz,
        )
        parent = self.__parents[0]
        labels = self.__concatenated_labels(parent, len(atoms))
        return parent._to_interface_candidate(atoms, labels)

    def apply_interface_separation(
        self,
        candidate: InterfaceCandidate,
        *,
        interface_separation: float,
    ) -> InterfaceCandidate:
        """Insert a topology-aware empty interval between the two grains.

        Periodic bicrystals expand the x box by twice the requested separation so both
        the central and outer periodic interfaces gain the same spacing. Slabs expand by
        the requested separation while preserving both outer vacuum widths.

        :param candidate: Geometry-bearing fixed-cell candidate from this manipulator.
        :param interface_separation: Keyword argument, required. Nonnegative central
            separation in angstroms.
        :return: Complete immutable separated candidate and updated geometry.
        :raises GBManipulatorValueError: If topology, candidate provenance, or
            separation is invalid.
        """
        if not self.__one_parent:
            raise GBManipulatorValueError(
                "interface separation requires exactly one parent"
            )

        context = ManipulationContext(
            parents=(self.__parent_candidate_geometry(0),),
            rng=self.__rng,
            params={"candidate": candidate, "interface_separation": interface_separation},
        )
        result = self.__translate_manipulation_error(
            InterfaceSeparation().execute, context
        )
        return result.children[0]

    def slice_and_merge(
        self,
        *,
        surface_mode: str = "normal_plane",
        max_tilt_degrees: float = 5.0,
    ) -> np.ndarray:
        """Construct an exact formula-preserving child from two parents.

        ``normal_plane`` uses a plane parallel to yz. ``periodic_wave`` uses a smooth
        sinusoidal surface that is continuous across the y/z periodic boundaries. Its
        combined maximum local tilt is bounded by ``max_tilt_degrees``.

        :param surface_mode: Keyword argument, optional, defaults to
            ``"normal_plane"``. Crossover surface mode.
        :param max_tilt_degrees: Keyword argument, optional, defaults to ``5.0``.
            Maximum combined local tilt for ``periodic_wave``, in degrees.
        :return: Formula-preserving child atom rows.
        :raises GBManipulatorValueError: If parent geometry or arguments are invalid.
        :raises CompositionAwareCrossoverError: If the parents are compositionally
            inadmissible or no positive-width admissible cut interval exists.
        """
        if self.__one_parent:
            raise GBManipulatorValueError(
                "Unable to slice and merge with only one parent.")
        parent1 = self.__parents[0]
        parent2 = self.__parents[1]

        new_positions, child_labels, provenance = self.__translate_manipulation_error(
            crossover_slice_and_merge,
            parent1,
            parent2,
            surface_mode=surface_mode,
            max_tilt_degrees=max_tilt_degrees,
            rng=self.__rng,
            capability_exception=CompositionAwareCrossoverError,
        )
        self.__set_candidate_labels(child_labels, len(new_positions))
        self.__last_crossover_provenance = tuple(provenance.items())

        return new_positions

    def remove_atoms(
        self,
        *,
        gb_fraction: float = None,
        num_to_remove: int = None,
        keep_ratio: bool = True,
        return_positions: bool = False,
    ) -> np.ndarray:
        """
        Removes *gb_fraction* of atoms or *num_to_remove* atom(s) in the GB region. Uses
        the local order parameter method of Lyakhov *et al.*, Computer Phys. Comm. 181
        (2010) 1623-1632.

        One of the following parameters must be specified.
        :param gb_fraction: Keyword argument. The fraction of atoms in the GB plane to
            remove. Must be less than 25% of the total number of atoms in the GB region.
        :param num_to_remove: Keyword argument. The specific number of atoms to remove.
            Maximum is 25% of the total number of atoms in the GB region.
        :param keep_ratio: Keyword argument. Whether or not to maintain stochiometric
            ratios. Default: True.
        :param return_positions: Keyword argument, optional, defaults to False. Flag to
            include the positions of the atoms removed into the array.
        :return: Atom positions after atom removal.
        """
        if not gb_fraction and not num_to_remove:
            raise GBManipulatorValueError(
                "gb_fraction or num_to_remove must be specified."
            )
        if not self.__one_parent:
            warnings.warn("Atom removal only occurring based on parent 1.")
        parent = self.__parents[0]
        atoms = Atom.as_array(parent.whole_system, type_map=parent.unit_cell.type_map)
        gb_atoms = Atom.as_array(parent.gb_atoms, type_map=parent.unit_cell.type_map)
        gb_atom_indices = parent.gb_indices
        type_map = parent.unit_cell.type_map
        positions = atoms[:, 1:]

        if gb_fraction is not None and (gb_fraction <= 0 or gb_fraction > 0.25):
            raise GBManipulatorValueError(
                f"Invalid value for gb_fraction ({gb_fraction=}). Must be "
                "0 < gb_fraction <= 0.25"
            )

        if num_to_remove is not None and (
            num_to_remove < 1 or num_to_remove > int(0.25 * len(gb_atoms))
        ):
            raise GBManipulatorValueError(
                "Invalid num_to_remove value. Must be >= 1, and must be less than or "
                "equal to 25% of the total number of atoms in the GB region."
            )
        if num_to_remove is None:
            num_to_remove = int(gb_fraction * len(gb_atoms))

        if num_to_remove == 0:
            warnings.warn(
                "Calculated fraction of atoms to remove is 0 "
                f"(int({gb_fraction}*{len(gb_atoms)}) = 0)"
            )
            self.__set_candidate_labels(
                getattr(parent, "grain_labels", None), len(parent.whole_system)
            )
            return atoms

        indices_to_remove = self.__translate_manipulation_error(
            select_removal_indices,
            atoms=atoms,
            positions=positions,
            gb_atom_indices=gb_atom_indices,
            type_map=type_map,
            ratio=parent.unit_cell.ratio,
            unit_cell=parent.unit_cell,
            num_to_remove=num_to_remove,
            keep_ratio=keep_ratio,
            rng=self.__rng,
        )
        pos = np.delete(parent.whole_system, indices_to_remove, axis=0)
        labels = getattr(parent, "grain_labels", None)
        retained_labels = (
            None
            if labels is None
            else np.delete(labels, indices_to_remove, axis=0)
        )
        self.__set_candidate_labels(retained_labels, len(pos))

        if return_positions:
            return (pos, parent.whole_system[indices_to_remove])
        else:
            return pos

    def insert_atoms(
        self,
        *,
        fill_fraction: float = None,
        num_to_insert: int = None,
        method: str = "delaunay",
        keep_ratio: bool = True,
        return_positions: bool = False,
    ) -> np.ndarray:
        """
        Inserts **fraction** atoms in the GB at empty lattice sites. "Empty" sites are
        determined through Delaunay triangulation (method="Delaunay") or through a grid
        with a resolution of 1 angstrom (method="grid").

        One of the following parameters must be specified.
        :param fill_fraction: Keyword argument. The fraction of empty lattice sites to
            fill. Must be less than or equal to 25% of the total number of atoms in the
            GB slab.
        :param num_to_insert: Keyword argument. The number of atoms to insert. Must be
            less than or equal to 25% of the total number of atoms in the GB slab.
        :param method: Keyword argument, optional, defaults to "delaunay". The method to
            use. Must be either "delaunay" or "grid."
        :param keep_ratio: Keyword argument, optional, defaults to True. Flag
            specifying whether or not to keep stoichiometric ratios in the system with
            the added atoms. If true, atoms are inserted at neighboring.
        :param return_positions: Keyword argument, optional, defaults to False. Flag to
            include the positions of the new atoms inserted into the array.
        :raises GBManipulatorValueError: Exception raised if an invalid method is
            specified.
        :return: Atom positions after atom insertion.
        """
        if not fill_fraction and not num_to_insert:
            raise GBManipulatorValueError(
                "fill_fraction or num_to_insert must be specified."
            )
        if not self.__one_parent:
            warnings.warn("Atom insertion only occurring based on parent 1.")
        parent = self.__parents[0]
        atoms = Atom.as_array(parent.whole_system, type_map=parent.unit_cell.type_map)
        gb_atoms = Atom.as_array(parent.gb_atoms, type_map=parent.unit_cell.type_map)
        type_map = parent.unit_cell.type_map
        type_map_inverse = {v: k for k, v in type_map.items()}

        if fill_fraction is not None and (fill_fraction <= 0 or fill_fraction > 0.25):
            raise GBManipulatorValueError(
                f"Invalid value for fill_fraction ({fill_fraction=}). Must be 0 < "
                "fill_fraction <= 0.25"
            )

        if num_to_insert is not None and (
            num_to_insert < 1 or num_to_insert > int(0.25 * len(gb_atoms))
        ):
            raise GBManipulatorValueError(
                "Invalid num_to_insert value. Must be >= 1, and must be less than or "
                "equal to 25% of the total number of atoms in the GB region.")

        if num_to_insert is None:
            num_to_insert = int(fill_fraction * len(gb_atoms))

        if num_to_insert == 0:
            warnings.warn(
                "Calculated fraction of atoms to insert is 0 "
                f"(int({fill_fraction}*{len(gb_atoms)}) = 0)"
            )
            self.__set_candidate_labels(
                getattr(parent, "grain_labels", None), len(parent.whole_system)
            )
            return atoms

        if method == "delaunay":
            possible_sites, probabilities = delaunay_insertion_sites(
                gb_atoms[:, 1:], parent.unit_cell.radius)
        elif method == "grid":
            possible_sites, probabilities = grid_insertion_sites(
                gb_atoms[:, 1:], parent.unit_cell.radius)
        else:
            raise GBManipulatorValueError(f"Unrecognized insert_atoms method: {method}")

        atoms_to_add = self.__translate_manipulation_error(
            select_insertion_sites,
            possible_sites=possible_sites,
            probabilities=probabilities,
            type_map=type_map,
            ratio=parent.unit_cell.ratio,
            unit_cell=parent.unit_cell,
            num_to_insert=num_to_insert,
            keep_ratio=keep_ratio,
            rng=self.__rng,
        )

        new_atoms = np.array(
            [
                (type_map_inverse[atom_type], *possible_sites[idx])
                for atom_type in atoms_to_add.keys()
                for idx in atoms_to_add[atom_type]
            ], dtype=Atom.atom_dtype
        )

        candidate = np.hstack((parent.whole_system, new_atoms))
        labels = getattr(parent, "grain_labels", None)
        if labels is None:
            candidate_labels = None
        else:
            left_bounds = parent.left_grain_x_bounds
            right_bounds = parent.right_grain_x_bounds
            tolerance = parent.coordinate_tolerance
            inserted_labels = np.empty(len(new_atoms), dtype=np.int8)
            for index, x_value in enumerate(new_atoms["x"]):
                x_coord = float(x_value)
                in_left = (
                    x_coord >= left_bounds[0] - tolerance
                    and x_coord < left_bounds[1]
                )
                in_right = (
                    x_coord >= right_bounds[0] - tolerance
                    and x_coord < right_bounds[1]
                )
                if in_left and not in_right:
                    inserted_labels[index] = LEFT_GRAIN_LABEL
                elif in_right and not in_left:
                    inserted_labels[index] = RIGHT_GRAIN_LABEL
                elif in_left and in_right:
                    inserted_labels[index] = (
                        LEFT_GRAIN_LABEL
                        if x_coord < parent.gb_plane_x
                        else RIGHT_GRAIN_LABEL
                    )
                else:
                    raise GBManipulatorValueError(
                        "inserted atom lies outside both explicit physical grain x "
                        "intervals"
                    )
            candidate_labels = np.hstack((labels, inserted_labels))
        self.__set_candidate_labels(candidate_labels, len(candidate))
        if return_positions:
            return (candidate, new_atoms)
        return candidate

    def displace_along_soft_modes(
        self,
        threshold: float = None,
        *,
        mesh_size: int = 4,
        num_q: int = 1,
        mode_index: int = 0,
        subtract_displacement: bool = False,
    ) -> np.ndarray:
        """
        Displace atoms along a single selected soft phonon mode.

        :param threshold: Maximum displacement of atoms allowed, optional, defaults to 1.5
            times the ideal bond length.
        :param mesh_size: Keyword argument. Specifies the size of the mesh for
            identifying unique q points. Optional. Defaults to 4.
        :param num_q: Keyword argument. Specifies the number of unique q points to use
            when calculating the dynamical matrix and determining the displacements.
            Optional. Defaults to 50.
        :param mode_index: Keyword argument. Selects which non-acoustic soft mode to
            displace along, ordered from softest (0) to next-softest (1), and so on.
            Optional. Defaults to 0.
        :param subtract_displacement: Keyword argument. Flag for subtracting, rather
            than adding the displacements from the eigenvectors to the original
            positions. Optional. Defaults to False (adds the displacements).
        :return: The grain boundary structure displaced along the selected mode.
        """
        if threshold is not None and threshold < 0:
            raise GBManipulatorValueError("d_max must be a positive float value.")
        if mesh_size < 1:
            raise GBManipulatorValueError("mesh_size must be >= 1.")
        if num_q < 1:
            raise GBManipulatorValueError("num_q must be >= 1.")
        if mode_index < 0:
            raise GBManipulatorValueError("mode_index must be >= 0.")
        parent = self.__parents[0]

        # NOTE: threshold is validated and defaulted here (matching every other
        # parameter's handling) but, as before R17/R18, never actually used below --
        # see REFACTOR_CLEANUP.md's open entry on this parameter.
        ideal_bonds = parent.unit_cell.ideal_bond_lengths
        if not threshold:
            threshold = 1.5 * max(ideal_bonds.values())

        return self.__translate_manipulation_error(
            soft_mode_displacement_atoms,
            structured_atoms=parent.whole_system,
            unit_cell=parent.unit_cell,
            gb_indices=parent.gb_indices,
            box_dims=parent.box_dims,
            gb_thickness=parent.gb_thickness,
            mesh_size=mesh_size,
            num_q=num_q,
            mode_index=mode_index,
            subtract_displacement=subtract_displacement,
        )

    def apply_group_symmetry(self, group: str) -> np.ndarray:
        """
        Apply the specified group symmetry to the GB region.

        :param group: One of the 230 crystallographic space groups.
        :raises NotImplementedError: Not currently implemented.
        :return: Atoms positions after applying group symmetry.
        """

        pos = self.__parents[0].whole_system
        raise NotImplementedError("This mutator has not been implemented yet.")
        return pos

    def __copy__(self):
        """Return an independent manipulator copy preserving ownership state."""
        result = type(self).__new__(type(self))
        result.__rng = copy_module.deepcopy(self.__rng)
        result.__parents = [
            copy_module.copy(parent) if parent is not None else None
            for parent in self.__parents
        ]
        result.__one_parent = self.__one_parent
        result.__num_processes = self.__num_processes
        result.__candidate_grain_labels = (
            None
            if self.__candidate_grain_labels is None
            else _readonly_copy(self.__candidate_grain_labels, dtype=np.int8)
        )
        return result

    def __deepcopy__(self, memo):
        """Return an independent deep copy preserving ownership state."""
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        result.__rng = copy_module.deepcopy(self.__rng, memo)
        result.__parents = [
            copy_module.deepcopy(parent, memo) if parent is not None else None
            for parent in self.__parents
        ]
        result.__one_parent = self.__one_parent
        result.__num_processes = self.__num_processes
        result.__candidate_grain_labels = (
            None
            if self.__candidate_grain_labels is None
            else _readonly_copy(self.__candidate_grain_labels, dtype=np.int8)
        )
        return result

    # Getter and setter methods for the parents
    @property
    def parents(self) -> list:
        return _ParentsProxy(self)

    @parents.setter
    def parents(self, value) -> None:
        if not isinstance(value, list) or len(value) != 2:
            raise GBManipulatorValueError(
                "The parents attribute must be a list with exactly 2 elements.")

        if any(not (v is None or isinstance(v, Parent)) for v in value):
            raise GBManipulatorValueError(
                "Both items in the parents list must be None or instances of Parent")

        self.__parents = value
        self.__candidate_grain_labels = self.__initial_candidate_labels()
