# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Manipulate grain-boundary structures while preserving interface geometry.

This module owns in-memory structural transformations. External file ownership,
calculator evaluation, and optimizer policy do not belong here.
"""

import copy as copy_module
import multiprocessing as mp
import warnings
from itertools import combinations_with_replacement
from numbers import Real
from os.path import isfile

import numpy as np
import scipy.sparse as sps
import spglib as spg
from numba import float64, jit, prange
from numba.typed import List
from scipy.spatial import ConvexHull, Delaunay, KDTree

from GBOpt._candidate_admissibility import (
    CandidateAdmissibilityError,
    composition_delta_is_formula_multiple,
    validate_formula_composition,
)
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
    ManipulationConfigurationError,
    ManipulationContext,
    ManipulationRegistry,
    ManipulationResult,
    RightGrainTranslation,
    default_registry,
)
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


def _affine_remap_axis_values(
    values: float | np.ndarray,
    source_bounds: np.ndarray,
    target_bounds: np.ndarray,
) -> float | np.ndarray:
    """Map Cartesian coordinates between one pair of affine-equivalent bounds.

    :param values: Scalar coordinate or coordinate array to transform.
    :param source_bounds: Two-element source interval.
    :param target_bounds: Two-element target interval.
    :return: Coordinate values with the same reduced positions in the target interval.
    """
    source_lo, source_hi = np.asarray(source_bounds, dtype=float)
    target_lo, target_hi = np.asarray(target_bounds, dtype=float)
    reduced = (np.asarray(values, dtype=float) - source_lo) / (source_hi - source_lo)
    mapped = target_lo + reduced * (target_hi - target_lo)
    if np.ndim(values) == 0:
        return float(mapped)
    return np.asarray(mapped, dtype=float)


def _affine_rescale_atoms(
    atoms: np.ndarray,
    source_box: np.ndarray,
    target_box: np.ndarray,
) -> np.ndarray:
    """Rescale atom coordinates between affine-equivalent orthogonal boxes.

    :param atoms: Structured atom rows to rescale.
    :param source_box: Source orthogonal box bounds.
    :param target_box: Target orthogonal box bounds.
    :return: Independent atom rows expressed in the target box.
    """
    rescaled = np.array(atoms, copy=True)
    for axis_name, axis_index in zip("xyz", range(3), strict=True):
        rescaled[axis_name] = _affine_remap_axis_values(
            rescaled[axis_name],
            source_box[axis_index],
            target_box[axis_index],
        )
    return rescaled


def _crossover_scalar_coordinates(
    atoms: np.ndarray,
    box_dims: np.ndarray,
    *,
    amplitude_y: float,
    amplitude_z: float,
    phase_y: float,
    phase_z: float,
) -> np.ndarray:
    """Project atoms onto one periodic-wave crossover coordinate.

    :param atoms: Structured atom rows in the crossover box.
    :param box_dims: Orthogonal crossover box bounds.
    :param amplitude_y: Keyword argument, required. y-periodic wave amplitude in
        angstroms.
    :param amplitude_z: Keyword argument, required. z-periodic wave amplitude in
        angstroms.
    :param phase_y: Keyword argument, required. y-periodic phase in radians.
    :param phase_z: Keyword argument, required. z-periodic phase in radians.
    :return: Scalar coordinates for comparison with the crossover offset.
    """
    box = np.asarray(box_dims, dtype=float)
    ylo, yhi = box[1]
    zlo, zhi = box[2]
    y_phase = 2.0 * np.pi * (atoms["y"] - ylo) / (yhi - ylo) + phase_y
    z_phase = 2.0 * np.pi * (atoms["z"] - zlo) / (zhi - zlo) + phase_z
    return (
        np.asarray(atoms["x"], dtype=float)
        - amplitude_y * np.sin(y_phase)
        - amplitude_z * np.sin(z_phase)
    )


def _admissible_crossover_intervals(
    first_atoms: np.ndarray,
    second_atoms: np.ndarray,
    first_coordinates: np.ndarray,
    second_coordinates: np.ndarray,
    *,
    lower: float,
    upper: float,
    species_ratio: tuple[tuple[str, int], ...],
) -> tuple[tuple[float, float], ...]:
    """Return positive-width cut intervals with a formula-vector count exchange.

    :param first_atoms: First-parent structured atom rows.
    :param second_atoms: Second-parent structured atom rows.
    :param first_coordinates: First-parent scalar crossover coordinates.
    :param second_coordinates: Second-parent scalar crossover coordinates.
    :param lower: Keyword argument, required. Inclusive offset lower bound.
    :param upper: Keyword argument, required. Exclusive offset upper bound.
    :param species_ratio: Keyword argument, required. Normalized formula vector.
    :return: Ordered, maximally merged admissible offset intervals.
    """
    species = tuple(name for name, _coefficient in species_ratio)
    first_names = np.asarray(first_atoms["name"]).astype(str)
    second_names = np.asarray(second_atoms["name"]).astype(str)
    first_counts = {
        name: int(np.count_nonzero((first_coordinates < lower) & (first_names == name)))
        for name in species
    }
    second_counts = {
        name: int(
            np.count_nonzero(
                (second_coordinates < lower) & (second_names == name)
            )
        )
        for name in species
    }
    events: list[tuple[float, int, str]] = []
    for parent_index, coordinates, names in (
        (0, first_coordinates, first_names),
        (1, second_coordinates, second_names),
    ):
        selected = np.flatnonzero((coordinates >= lower) & (coordinates < upper))
        events.extend(
            (float(coordinates[index]), parent_index, str(names[index]))
            for index in selected
        )
    events.sort(key=lambda event: event[0])

    intervals: list[tuple[float, float]] = []
    cursor = lower
    event_index = 0
    while event_index < len(events):
        coordinate = events[event_index][0]
        if coordinate > cursor and composition_delta_is_formula_multiple(
            first_counts,
            second_counts,
            species_ratio,
        ):
            if intervals and intervals[-1][1] == cursor:
                intervals[-1] = (intervals[-1][0], coordinate)
            else:
                intervals.append((cursor, coordinate))
        while event_index < len(events) and events[event_index][0] == coordinate:
            _value, parent_index, name = events[event_index]
            target = first_counts if parent_index == 0 else second_counts
            target[name] += 1
            event_index += 1
        cursor = coordinate
    if cursor < upper and composition_delta_is_formula_multiple(
        first_counts,
        second_counts,
        species_ratio,
    ):
        if intervals and intervals[-1][1] == cursor:
            intervals[-1] = (intervals[-1][0], upper)
        else:
            intervals.append((cursor, upper))
    return tuple(intervals)


def _sample_interval_by_width(
    intervals: tuple[tuple[float, float], ...],
    rng: np.random.Generator,
) -> float:
    """Sample uniformly over the union of positive-width intervals.

    :param intervals: Positive-width ordered intervals.
    :param rng: Optimizer-owned random-number generator.
    :return: Sample strictly inside one interval, selected in proportion to its width.
    """
    widths = np.asarray([upper - lower for lower, upper in intervals], dtype=float)
    total = float(np.sum(widths))
    target = min(float(rng.random()), np.nextafter(1.0, 0.0)) * total
    cumulative = 0.0
    selected_lower, selected_upper = intervals[-1]
    for interval, width in zip(intervals, widths, strict=True):
        if target < cumulative + width:
            selected_lower, selected_upper = interval
            break
        cumulative += float(width)
    fraction = min(float(rng.random()), np.nextafter(1.0, 0.0))
    cut = selected_lower + fraction * (selected_upper - selected_lower)
    if cut <= selected_lower:
        cut = float(np.nextafter(selected_lower, selected_upper))
    return cut


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


@jit(float64(float64, float64), nopython=True, cache=True)
def _gaussian(x: float, sigma: float = 0.02) -> float:
    """
    Calculates a Gaussian-smeared delta function at *x* given a standard deviation of
    *sigma*.

    :param x: Where to calculate the Gaussian-smeared delta function.
    :param sigma: Standard deviation of the Gaussian-smeared delta function, optional,
        defaults to 0.02.
    :return: Value of the Gaussian-smeared delta function at x.
    """
    prefactor = 1 / (sigma * np.sqrt(2 * np.pi))
    return prefactor * np.exp(-x * x / (2 * sigma * sigma))


@jit(nopython=True, cache=True)
def _calculate_fingerprint_vector(atom, neighs, NB, V, Btype, Delta, Rmax):
    """
    Calculates the fingerprint for *atom* as described in Lyakhov *et al.*,
    Computer Phys. Comm. 181 (2010) 1623-1632 (Eq. 4).

    :param np.ndarray atom: The atom we are calculating the fingerprint for.
    :param np.ndarray neighs: list of Atom containing the neighbors to **atom**.
    :param int NB: The number of atoms of type B neighbor to **atom**.
    :param float V: The volume of the unit cell in angstroms**3.
    :param int Btype: The type of neighbors we are interested in.
    :param float Delta: The discretization length for Rs in angstroms.
    :param float Rmax: The maximum distance from the *atom* to another atom to
        calculate the fingerprint.
    :return: The vector containing the fingerprint for *atom*.
    """
    Rs = np.arange(0, Rmax + Delta, Delta)

    fingerprint_vector = np.zeros_like(Rs)
    for idx, R in enumerate(Rs):
        local_sum = 0
        for neigh in neighs:
            if neigh[0] == Btype:
                diff = atom[1:] - neigh[1:]
                # Rij = np.linalg.norm(atom[1:] - neigh[1:])
                distance = np.sqrt(np.dot(diff, diff))
                delta = _gaussian(R - distance, 0.02)
                local_sum += delta / \
                    (4 * np.pi * distance * distance * (NB / V) * Delta)
        fingerprint_vector[idx] = local_sum - 1

    return fingerprint_vector


@jit(nopython=True, cache=True, parallel=True)
def _calculate_local_order(atom, neighs, unit_cell_types, unit_cell_a0, N, Delta, Rmax):
    """
    Calculates the local order parameter following Lyakhov *et al.*, Computer Phys.
    Comm. 181 (2010) 1623-1632 (Eq. 5).

    :param np.ndarray atom: Atom we are calculating the local order for.
    :param np.ndarray neighs: Neighbors of *atom*.
    :param np.ndarray unit_cell_types: The types of the atoms in the unit cell.
    :param float unit_cell_a0: The lattice parameter.
    :param int N: The number of atoms in the unit cell.
    :param float Delta: Bin size to calculate the fingerprint vector.
    :param float Rmax: Maximum distance from *atom* to consider as a neighbor to
        *atom* in angstroms.
    :return: The local order parameter for *atom* based on its neighbors.
    """
    local_sum = 0
    atom_types = np.unique(neighs[:, 0])
    V = unit_cell_a0 ** 3
    prefactor = Delta / (N * (V / N) ** (1 / 3))
    for Btype in atom_types:
        NB = np.sum(unit_cell_types == Btype)
        fingerprint = _calculate_fingerprint_vector(
            atom, neighs, NB, V, Btype, Delta, Rmax)
        local_sum += NB * prefactor * np.dot(fingerprint, fingerprint)
    return np.sqrt(local_sum)


def _create_neighbor_list(rcut: float, pos: np.ndarray) -> list:
    """
    Creates a neighbor list using a KDTree.

    :param rcut: Cutoff distance for considering an atom a neighbor to another.
    :param pos: The array of atom positions.
    :return: The neighbor list for the atoms in **pos**
    """
    kdtree = KDTree(pos)
    neighbor_list = kdtree.query_ball_tree(kdtree, r=rcut)
    # Remove an atom from using itself as a neighbor.
    for i, neighbor in enumerate(neighbor_list):
        neighbor.remove(i)
    return neighbor_list


# @jit(nopython=True, cache=True)
def _calculate_bond_hardness(parent, neighbor_list, ideal_bonds):
    atoms = parent.whole_system
    types = Atom.as_array(atoms)[:, 0]
    gb_indices = parent.gb_indices

    atom_info = {}
    for idx, atom in enumerate(atoms):
        a = Atom(*atom)  # convert this to an Atom
        if a.name not in atom_info:
            atom_info[a.name] = {
                "num": types[idx],
                "r_cov": a["r_cov"],
                "valence": a["valence"],
                "valence_electrons": a["valence_electrons"]
            }

    atom_type_to_name = {info["num"]: name for name, info in atom_info.items()}
    atom_name_to_type = {name: num for num, name in atom_type_to_name.items()}
    atom_types = list(atom_info.keys())

    n_of_bond_type = {
        (atom1, atom2): 0
        for atom1 in atom_types for atom2 in atom_types
    }

    for idx in gb_indices:
        for jdx in neighbor_list[idx]:
            if jdx < idx:
                continue
            n_of_bond_type[(atoms[idx]["name"], atoms[jdx]["name"])] += 1

    # We precompute half of Delta_k since it is used frequently.
    Delta_k = {}
    sorted_atom_type_to_name = sorted(atom_type_to_name)
    for type1, type2 in combinations_with_replacement(sorted_atom_type_to_name, 2):
        name1 = atom_type_to_name[type1]
        name2 = atom_type_to_name[type2]
        dk_tuple = (type1, type2)
        Delta_k[dk_tuple] = 0.5 * (ideal_bonds[(type1, type2)] -
                                   atom_info[name1]["r_cov"] - atom_info[name2]["r_cov"])
    bond_valence = np.sum(np.exp(-np.asarray(list(Delta_k.values())) / 0.37))

    y_dim = parent.box_dims[1, 1] - parent.box_dims[1, 0]
    z_dim = parent.box_dims[2, 1] - parent.box_dims[2, 0]
    V = parent.gb_thickness * y_dim * z_dim
    N = np.sum(list(n_of_bond_type.values()))
    Hij = np.zeros((len(atoms), len(atoms)))
    for i1 in gb_indices:
        atom1 = Atom(*atoms[i1])
        type1 = atom_name_to_type[atom1["name"]]
        i1_CN = atom1["valence"] / bond_valence
        for i2 in neighbor_list[i1]:
            atom2 = Atom(*atoms[i2])
            type2 = atom_name_to_type[atom2["name"]]
            dk_tuple = (type1, type2) if type1 <= type2 else (type2, type1)
            i1_electronegativity = 0.481 * \
                atom1["valence_electrons"] / \
                (atom1["r_cov"] + Delta_k[dk_tuple])
            i2_electronegativity = 0.481 * \
                atom2["valence_electrons"] / \
                (atom2["r_cov"] + Delta_k[dk_tuple])
            i2_CN = atom2["valence"] / bond_valence
            Xij = np.sqrt(i1_electronegativity / i1_CN * i2_electronegativity / i2_CN)
            fi = abs(i1_electronegativity - i2_electronegativity) / \
                (4*np.sqrt(i1_electronegativity * i2_electronegativity))
            Hij[i1, i2] = Xij / (V / N) * np.exp(-2.7 * fi)
            Hij[i2, i1] = Hij[i1, i2]

    return Hij


@jit(nopython=True, cache=True)
def _calculate_dynamical_matrix(hardness, positions, gb_atom_indices, neighbor_list, q_vec):
    num_gb_atoms = len(gb_atom_indices)
    Dij = np.zeros((3 * num_gb_atoms, 3 * num_gb_atoms), dtype=np.complex128)

    for d_i in prange(len(gb_atom_indices)):
        id1 = gb_atom_indices[d_i]
        for id2 in neighbor_list[id1]:
            if id2 not in gb_atom_indices:
                continue
            d_j = np.where(gb_atom_indices == id2)[0][0]
            rij = positions[id2] - positions[id1]
            exp_term = np.exp(1j * np.dot(q_vec, rij))
            for aa in range(3):
                for bb in range(3):
                    Dij[3 * d_i + aa, 3 * d_j + bb] = -hardness[id1, id2] * exp_term
                    if d_i == d_j:
                        Dij[3 * d_i + aa, 3 * d_j + bb] += hardness[id1, id2] * exp_term
    return Dij


def _get_stoichiometric_change(n_units: int, ratio: dict[int, int]) -> dict[int, int]:
    """_summary_

    Args:
        n_units: The number of atom units (defined as the sum of the values in the ratio
            dict) that will be modified. For example, given a ratio of
            {1: 1, 2: 2, 3: 3, 4: 1}, this number would indicate how many of type 1
            (and type 4) would be affected by the operation.
        ratio: The ratio of each atom type in the unit cell. Must be a dict where the
            keys and values are positive integers.

    Returns:
        dict[int, int]: The number of atoms of each type that will be changed.
    """

    return {atom_type: num * n_units for atom_type, num in ratio.items()}


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
    def __translate_manipulation_error(func, *args, **kwargs):
        """Call ``func(*args, **kwargs)``, translating manipulation-operation
        exceptions back to this class's own established public exception identities.

        A ``GBOpt.manipulation`` operation raises a plain ``TypeError`` for a
        malformed parameter type and ``ManipulationConfigurationError`` /
        ``ManipulationCapabilityError`` (both ``ValueError``-rooted) for every other
        validation failure, matching this class's own long-standing
        ``_validate_finite_real`` type/value split. This boundary is this class's
        counterpart to the ``__translate_construction_error`` pattern documented for
        other extracted-logic wrappers in this codebase, adapted to the
        ``Manipulation`` protocol's own exception vocabulary instead of an
        ``InterfaceCandidate``-construction one.

        :param func: Callable to invoke.
        :param args: Positional arguments forwarded to ``func``.
        :param kwargs: Keyword arguments forwarded to ``func``.
        :return: ``func``'s return value.
        :raises GBManipulatorTypeError: If ``func`` raises ``TypeError``.
        :raises GBManipulatorValueError: If ``func`` raises
            ``ManipulationConfigurationError`` or ``ManipulationCapabilityError``.
        """
        try:
            return func(*args, **kwargs)
        except TypeError as exc:
            raise GBManipulatorTypeError(str(exc)) from exc
        except (ManipulationConfigurationError, ManipulationCapabilityError) as exc:
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
        if surface_mode not in {"normal_plane", "periodic_wave"}:
            raise GBManipulatorValueError(
                "surface_mode must be 'normal_plane' or 'periodic_wave'"
            )
        tilt = _validate_finite_real("max_tilt_degrees", max_tilt_degrees)
        if tilt < 0.0 or tilt >= 90.0:
            raise GBManipulatorValueError(
                "max_tilt_degrees must satisfy 0 <= value < 90"
            )
        parent1 = self.__parents[0]
        parent2 = self.__parents[1]
        labels1 = parent1.grain_labels
        labels2 = parent2.grain_labels
        if (labels1 is None) != (labels2 is None):
            raise GBManipulatorValueError(
                "slice_and_merge requires both parents to use the same ownership mode"
            )
        if labels1 is not None:
            tolerance = max(parent1.coordinate_tolerance, parent2.coordinate_tolerance)
            if (
                parent1.inplane_periodic != parent2.inplane_periodic
                or parent1.normal_topology is not parent2.normal_topology
            ):
                raise GBManipulatorValueError(
                    "owned crossover requires matching boundary topology"
                )
            mapped_plane = _affine_remap_axis_values(
                parent2.gb_plane_x,
                parent2.box_dims[0],
                parent1.box_dims[0],
            )
            mapped_left_bounds = _affine_remap_axis_values(
                parent2.left_grain_x_bounds,
                parent2.box_dims[0],
                parent1.box_dims[0],
            )
            mapped_right_bounds = _affine_remap_axis_values(
                parent2.right_grain_x_bounds,
                parent2.box_dims[0],
                parent1.box_dims[0],
            )
            if (
                not np.isclose(
                    parent1.gb_plane_x,
                    mapped_plane,
                    atol=tolerance,
                    rtol=0.0,
                )
                or not np.allclose(
                    parent1.left_grain_x_bounds,
                    mapped_left_bounds,
                    atol=tolerance,
                    rtol=0.0,
                )
                or not np.allclose(
                    parent1.right_grain_x_bounds,
                    mapped_right_bounds,
                    atol=tolerance,
                    rtol=0.0,
                )
            ):
                raise GBManipulatorValueError(
                    "owned crossover requires affine-equivalent physical grain "
                    "geometry"
                )
        pos1 = parent1.whole_system
        pos2 = parent2.whole_system
        if labels1 is not None and not np.allclose(
            parent1.box_dims,
            parent2.box_dims,
            atol=tolerance,
            rtol=0.0,
        ):
            pos2 = _affine_rescale_atoms(
                pos2,
                parent2.box_dims,
                parent1.box_dims,
            )

        try:
            validate_formula_composition(pos1, parent1.unit_cell)
            validate_formula_composition(pos2, parent2.unit_cell)
        except CandidateAdmissibilityError as exc:
            raise CompositionAwareCrossoverError(str(exc)) from exc
        first_formula = parent1.unit_cell.formula_ratio
        second_formula = parent2.unit_cell.formula_ratio
        if first_formula != second_formula:
            raise CompositionAwareCrossoverError(
                "crossover parents use different normalized formula vectors"
            )

        amplitude_y = 0.0
        amplitude_z = 0.0
        phase_y = 0.0
        phase_z = 0.0
        if surface_mode == "periodic_wave" and tilt > 0.0:
            maximum_slope = np.tan(np.deg2rad(tilt))
            slope_radius = maximum_slope * np.sqrt(float(self.__rng.random()))
            slope_angle = 2.0 * np.pi * float(self.__rng.random())
            slope_y = slope_radius * np.cos(slope_angle)
            slope_z = slope_radius * np.sin(slope_angle)
            y_length = float(np.ptp(parent1.box_dims[1]))
            z_length = float(np.ptp(parent1.box_dims[2]))
            amplitude_y = slope_y * y_length / (2.0 * np.pi)
            amplitude_z = slope_z * z_length / (2.0 * np.pi)
            phase_y = 2.0 * np.pi * float(self.__rng.random())
            phase_z = 2.0 * np.pi * float(self.__rng.random())

        first_coordinates = _crossover_scalar_coordinates(
            pos1,
            parent1.box_dims,
            amplitude_y=amplitude_y,
            amplitude_z=amplitude_z,
            phase_y=phase_y,
            phase_z=phase_z,
        )
        second_coordinates = _crossover_scalar_coordinates(
            pos2,
            parent1.box_dims,
            amplitude_y=amplitude_y,
            amplitude_z=amplitude_z,
            phase_y=phase_y,
            phase_z=phase_z,
        )
        half_window = 0.25 * parent1.gb_thickness
        maximum_excursion = abs(amplitude_y) + abs(amplitude_z)
        lower = parent1.gb_plane_x - half_window + maximum_excursion
        upper = parent1.gb_plane_x + half_window - maximum_excursion
        if lower >= upper:
            raise CompositionAwareCrossoverError(
                "periodic crossover surface does not fit inside the GB cut window"
            )
        intervals = _admissible_crossover_intervals(
            pos1,
            pos2,
            first_coordinates,
            second_coordinates,
            lower=lower,
            upper=upper,
            species_ratio=first_formula,
        )
        if not intervals:
            raise CompositionAwareCrossoverError(
                "no positive-width formula-preserving crossover interval exists"
            )
        slice_pos = _sample_interval_by_width(intervals, self.__rng)
        mask1 = first_coordinates < slice_pos
        mask2 = second_coordinates >= slice_pos
        new_positions = np.hstack((pos1[mask1], pos2[mask2]))
        if labels1 is None:
            child_labels = None
        else:
            child_labels = np.hstack((labels1[mask1], labels2[mask2]))
        self.__set_candidate_labels(child_labels, len(new_positions))

        try:
            validate_formula_composition(new_positions, parent1.unit_cell)
        except CandidateAdmissibilityError as exc:
            raise CompositionAwareCrossoverError(
                f"internal crossover composition invariant failed: {exc}"
            ) from exc
        self.__last_crossover_provenance = (
            ("surface_mode", surface_mode),
            ("max_tilt_degrees", tilt),
            ("amplitude_y", float(amplitude_y)),
            ("amplitude_z", float(amplitude_z)),
            ("phase_y", float(phase_y)),
            ("phase_z", float(phase_z)),
            ("offset", float(slice_pos)),
        )

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

        if len(type_map) == 1:
            num_to_remove_dict = {1: num_to_remove}
        # determine both the number to remove of each type, and the probability of
        # removal
        elif keep_ratio:
            num_to_remove_dict = _get_stoichiometric_change(
                num_to_remove, parent.unit_cell.ratio
            )
            num_to_remove = sum(list(num_to_remove_dict.values()))
            central_type = min(num_to_remove_dict, key=num_to_remove_dict.get)
            cutoff = (
                parent.unit_cell.nn_distance(2) + parent.unit_cell.nn_distance(1)
            ) / 2
            neighbor_list = _create_neighbor_list(cutoff, positions)
            Delta = 0.05  # Bin size to calculate the fingerprint vector.
            Rmax = 15  # Max distance allowed to be a neighbor
            args_list = [
                (
                    atoms[atom_idx],
                    atoms[neighbor_list[atom_idx]],
                    parent.unit_cell.names(asint=True),
                    parent.unit_cell.a0,
                    len(parent.unit_cell.unit_cell),
                    Delta,
                    Rmax,
                )
                for idx, atom_idx in enumerate(gb_atom_indices)
            ]
            order = np.zeros(len(args_list))
            for i, args in enumerate(args_list):
                order[i] = _calculate_local_order(*args)

            # We want the probabilities to be inversely proportional to the order parameter.
            # Higher order parameters should be more "stable" against removal than low order
            # parameters. We give small probabilities to the higher order values just to
            # allow for variety in the calculations.
            probabilities = max(order) - order + min(order)
            probabilities = probabilities / np.sum(probabilities, dtype=float)
        else:
            # If we aren't worried about keeping the ratio, randomly assign atoms to be
            # removed to each type, summing up to num_to_remove.
            breaks = np.sort(
                np.random.choice(
                    range(1, num_to_remove), len(type_map) - 1, replace=False
                )
            )
            breaks = np.concatenate(([0], breaks, [num_to_remove]))
            values = np.diff(breaks)
            num_to_remove_dict = {
                i + 1: int(values[i]) for i in range(len(type_map))
            }

        if keep_ratio and len(type_map) > 1:
            type_mask = atoms[gb_atom_indices][:, 0] == central_type
            central_indices = gb_atom_indices[type_mask]
            central_probabilities = probabilities[type_mask]
            central_probabilities = (
                central_probabilities / np.sum(central_probabilities)
            )

            if len(central_indices) == 0:
                raise GBManipulatorValueError(
                    f"No atoms found for type {central_type} in the grain boundary."
                )

            central_num_to_remove = num_to_remove_dict[central_type]
            selected_central_indices = self.__rng.choice(
                central_indices,
                central_num_to_remove,
                replace=False,
                p=central_probabilities,
            )

            distances = {
                idx: np.full(len(neighbor_list[idx]), np.inf)
                for idx in selected_central_indices
            }
            for central_idx in selected_central_indices:
                neighbors = neighbor_list[central_idx]
                gb_neighbors = np.intersect1d(neighbors, gb_atom_indices)
                mask = np.isin(neighbors, gb_neighbors)
                distances[central_idx][mask] = np.linalg.norm(
                    positions[gb_neighbors] - positions[central_idx], axis=1
                )

            indices_to_remove = list(distances.keys())
            for atom_type, ratio in parent.unit_cell.ratio.items():
                if atom_type == central_type:
                    continue
                # type_mask = atoms[gb_atom_indices][:, 0] == atom_type
                # type_indices = gb_atom_indices[type_mask]
                for idx, dists in distances.items():
                    neighbor_indices = np.asarray(neighbor_list[idx])
                    gb_neighbor_indices = np.intersect1d(
                        neighbor_indices, gb_atom_indices)
                    mask = np.isin(neighbor_indices, gb_neighbor_indices)
                    type_mask = atoms[gb_neighbor_indices][:, 0] == atom_type
                    type_indices = neighbor_indices[mask][type_mask]
                    duplicates = [
                        i for i, el in enumerate(type_indices)
                        if el in indices_to_remove
                    ]

                    type_indices = list(set(type_indices) - set(duplicates))
                    if len(type_indices) < ratio:
                        raise GBManipulatorValueError(
                            f"Not enough neighbor atoms of type {atom_type} to remove."
                        )

                    # this really shouldn't happen, as this would indicate overlapping
                    # atoms
                    dists[dists < 1e-8] = 1e-8
                    type_probabilities = 1 / dists[mask][type_mask]
                    type_probabilities = type_probabilities / np.sum(type_probabilities)

                    type_idx_to_remove = self.__rng.choice(
                        type_indices, ratio, replace=False, p=type_probabilities
                    )

                    indices_to_remove.extend(type_idx_to_remove)

        else:  # keep_ratio == False or len(type_map) == 1
            indices_to_remove = []
            for atom_type, num in num_to_remove_dict.items():
                type_indices = gb_atom_indices[
                    atoms[gb_atom_indices][:, 0] == atom_type
                ]
                type_idx_to_remove = self.__rng.choice(type_indices, num, replace=False)
                indices_to_remove.extend(type_idx_to_remove)

        if not len(indices_to_remove) == num_to_remove:
            raise GBManipulatorValueError("")
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
        def Delaunay_approach(
            gb_atoms: np.ndarray,
            atom_radius: float,
            num_to_insert: int
        ) -> np.ndarray:
            """
            Delaunay triangulation approach for inserting atoms. Potential insertion
            sites are the circumcenters of the tetrahedra.

            :param gb_atoms: Array of atom positions where we are considering inserting
                new atoms.
            :param atom_radius: The radius of an atom.
            :param num_to_insert: The number of atoms to insert.
            :return: The sites at which new atoms are inserted.
            """
            # First we need to duplicate the gb_atoms in the y and z directions to
            # account for PBCs.
            min_bounds = np.min(gb_atoms, axis=0)
            max_bounds = np.max(gb_atoms, axis=0)
            Lx, Ly, Lz = max_bounds - min_bounds
            tiles = [(dy, dz) for dy in [-1, 0, 1] for dz in [-1, 0, 1]]
            replicas = []
            original_indices = []
            for dy, dz in tiles:
                shift = np.zeros_like(gb_atoms)
                shift[:, 1] = dy * Ly
                shift[:, 2] = dz * Lz
                replicas.append(gb_atoms + shift)
                original_indices.extend(np.arange(len(gb_atoms)))
            tiled = np.vstack(replicas)
            original_indices = np.array(original_indices)

            # Delaunay triangulation approach
            tri = Delaunay(tiled)
            # ijk is for the 3x3 transformation matrix triangulation.transform[:, :3, :]
            # ik is for the offset vector triangulation.transform[:, 3, :], and ij is
            # the resulting circumcenter coordinates
            circumcenters = -np.einsum(
                "ijk,ik->ij",
                tri.transform[:, :3, :],
                tri.transform[:, 3, :]
            )
            # Wrap circumcenters back into the original bounds
            circumcenters[:, 1] = np.mod(
                circumcenters[:, 1] - min_bounds[1], Ly) + min_bounds[1]
            circumcenters[:, 2] = np.mod(
                circumcenters[:, 2] - min_bounds[2], Lz) + min_bounds[2]

            original_indices_simplices = original_indices[tri.simplices]

            # Bounds check
            in_bounds = np.all((circumcenters >= min_bounds) & (
                circumcenters <= max_bounds), axis=1)
            mask = in_bounds

            # Volume check
            # Calculating the volume may occasionally fail if the points are collinear,
            # so we catch the warning so users are not concerned.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                simplices = gb_atoms[original_indices_simplices]
                A, B, C, D = simplices[:, 0], simplices[:,
                                                        1], simplices[:, 2], simplices[:, 3]
                volumes = np.abs(
                    np.einsum("ij,ij->i", np.cross(B - A, C - A), D - A)) / 6.0
            volume_threshold = 1e-3
            volume_mask = (volumes > volume_threshold * np.median(volumes)) & ~np.isnan(
                circumcenters).any(axis=1)
            mask &= volume_mask

            # Convex hull check
            hull_vertices = set(ConvexHull(gb_atoms).vertices)
            simplex_mask = ~np.any(np.isin(tri.simplices, list(hull_vertices)), axis=1)
            mask &= simplex_mask

            valid_circumcenters = circumcenters[mask]
            valid_simplices = original_indices_simplices[mask, 0]
            sphere_radii = np.linalg.norm(
                gb_atoms[valid_simplices] - valid_circumcenters, axis=1)
            interstitial_radii = sphere_radii - atom_radius
            interstitial_radii -= np.min(interstitial_radii)  # make everything >= 0
            probabilities = interstitial_radii / np.sum(interstitial_radii)
            probabilities = probabilities / np.sum(probabilities)  # normalize
            assert abs(1 - np.sum(probabilities)
                       ) < 1e-8, "Probabilities are not normalized!"
            num_sites = len(circumcenters)

            if num_to_insert is None:
                num_to_insert = int(fill_fraction * num_sites)

            if num_to_insert == 0:
                warnings.warn("Calculated fraction of atoms to insert is 0: "
                              f"int({fill_fraction}*{len(gb_atoms)}) = 0"
                              )

            return valid_circumcenters, probabilities

        def grid_approach(
            gb_atoms: np.ndarray,
            atom_radius: float,
            num_to_insert: int,
        ) -> np.ndarray:
            """
            Grid approach for inserting atoms. Potential insertion sites are on a 1x1x1
            Angstrom grid where sites must be at least *atom_radius* away.

            :param gb_atoms: Array of atom positions where we are considering inserting
                new atoms.
            :param atom_radius: The radius of an atom.
            :param num_to_insert: The number of atoms to insert.

            :return: The sites at which new atoms are inserted.
            """
            # Grid approach
            max_x, max_y, max_z = gb_atoms.max(axis=0)
            min_x, min_y, min_z = gb_atoms.min(axis=0)
            X, Y, Z = np.meshgrid(
                np.arange(np.floor(min_x), np.ceil(max_x) + 1),
                np.arange(np.floor(min_y), np.ceil(max_y) + 1),
                np.arange(np.floor(min_z), np.ceil(max_z) + 1),
                indexing="ij"
            )
            sites = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
            GB_tree = KDTree(gb_atoms)
            sites_tree = KDTree(sites)
            indices_to_remove = GB_tree.query_ball_tree(sites_tree, atom_radius)
            indices_to_remove = list(set(
                [i for sublist in indices_to_remove for i in sublist]))
            filtered_sites = np.delete(sites, indices_to_remove, axis=0)

            distances, _ = GB_tree.query(filtered_sites, k=1)
            probabilities = distances / np.sum(distances)
            probabilities = probabilities / np.sum(probabilities)  # normalize
            assert abs(1 - np.sum(probabilities)
                       ) < 1e-8, "Probabilities are not normalized!"
            num_sites = len(filtered_sites)

            if num_to_insert is None:
                num_to_insert = int(fill_fraction * num_sites)

            if num_to_insert == 0:
                warnings.warn("Calculated fraction of atoms to insert is 0: "
                              f"int({fill_fraction}*{len(gb_atoms)}) = 0"
                              )

            return filtered_sites, probabilities

            indices = self.__rng.choice(num_sites,
                                        num_to_insert,
                                        replace=False,
                                        p=probabilities
                                        )
            return filtered_sites[indices]

            raise GBManipulatorValueError(
                "fill_fraction or num_to_insert must be specified.")

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
            raise GBManipulatorValueError
            (f"Invalid value for fill_fraction ({fill_fraction=}). Must be 0 < "
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

        if len(type_map) == 1:
            num_to_insert_dict = {1: num_to_insert}
        elif keep_ratio:
            num_to_insert_dict = _get_stoichiometric_change(
                num_to_insert, parent.unit_cell.ratio
            )
            num_to_insert = sum(list(num_to_insert_dict.values()))
            central_type = min(num_to_insert_dict, key=num_to_insert_dict.get)
        else:  # random insertion of random types
            breaks = np.sort(
                np.random.choice(
                    range(1, num_to_insert), len(type_map) - 1, replace=False
                )
            )
            breaks = np.concatenate([[0], breaks, [num_to_insert]])
            values = np.diff(breaks)
            num_to_insert_dict = {
                i + 1: int(values[i]) for i in range(len(type_map))
            }

        # Calculate the insertion sites using the specified approach.
        if method == "delaunay":
            possible_sites, probabilities = Delaunay_approach(
                gb_atoms[:, 1:], parent.unit_cell.radius, num_to_insert)
        elif method == "grid":
            possible_sites, probabilities = grid_approach(
                gb_atoms[:, 1:], parent.unit_cell.radius, num_to_insert)
        else:
            raise GBManipulatorValueError(f"Unrecognized insert_atoms method: {method}")

        if keep_ratio and len(type_map) > 1:
            central_num_to_insert = num_to_insert_dict[central_type]
            selected_central_indices = self.__rng.choice(
                list(range(len(possible_sites))),
                central_num_to_insert,
                replace=False,
                p=probabilities
            )
            cutoff = (
                parent.unit_cell.nn_distance(2) + parent.unit_cell.nn_distance(1)
            ) / 2.0
            possible_sites_neighbor_list = _create_neighbor_list(cutoff, possible_sites)

            atoms_to_add = {
                type_map[i]: [] if type_map[i] != central_type else selected_central_indices for i in type_map.keys()}
            for atom_type, ratio in parent.unit_cell.ratio.items():
                if atom_type == central_type:
                    continue
                for idx in selected_central_indices:
                    neighbors = possible_sites_neighbor_list[idx]
                    already_assigned = {idx for v in atoms_to_add.values() for idx in v}
                    # Only consider the indices that have not already been assigned
                    available_neighbors = list(set(neighbors) - already_assigned)
                    if len(available_neighbors) < ratio:
                        raise GBManipulatorValueError(
                            "Not enough sites to insert atoms into."
                        )
                    partial_probabilities = probabilities[available_neighbors]
                    partial_probabilities = partial_probabilities / \
                        np.sum(partial_probabilities)
                    selected_neighbor_offsets = self.__rng.choice(
                        list(range(len(available_neighbors))), ratio, replace=False,
                        p=partial_probabilities
                    )
                    selected_indices = [
                        available_neighbors[offset]
                        for offset in selected_neighbor_offsets
                    ]
                    atoms_to_add[atom_type].extend(selected_indices)
        else:
            atoms_to_add = {}
            site_indices = list(range(len(possible_sites)))
            for atom_type, num in num_to_insert_dict.items():
                available_indices = list(
                    set(site_indices) - set(np.array(atoms_to_add.values()).flatten()))
                type_idx_to_insert = self.__rng.choice(
                    available_indices, num, replace=False, p=probabilities
                )
                atoms_to_add[atom_type] = type_idx_to_insert

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
        num_children: int = 1,
        subtract_displacement: bool = False,
    ) -> np.ndarray:
        """
        Displace atoms along soft phonon modes.

        :param threshold: Maximum displacement of atoms allowed, optional, defaults to 1.5
            times the ideal bond length.
        :param mesh_size: Keyword argument. Specifies the size of the mesh for
            identifying unique q points. Optional. Defaults to 4.
        :param num_q: Keyword argument. Specifies the number of unique q points to use
            when calculating the dynamical matrix and determining the displacements.
            Optional. Defaults to 50.
        :param num_children: Keyword argument. Specifies the number of children to
            create from the parent structure. Optional. Defaults to 1.
        :param subtract_displacement: Keyword argument. Flag for subtracting, rather
            than adding the displacements from the eigenvectors to the original
            positions. Optional. Defaults to False (adds the displacements).
        :return: *num_children* grain boundary structures.
        """
        if threshold is not None and threshold < 0:
            raise GBManipulatorValueError("d_max must be a positive float value.")
        if mesh_size < 1:
            raise GBManipulatorValueError("mesh_size must be >= 1.")
        if num_q < 1:
            raise GBManipulatorValueError("num_q must be >= 1.")
        if num_children < 1:
            raise GBManipulatorValueError("num_children must be >= 1.")
        parent = self.__parents[0]
        atoms = Atom.as_array(parent.whole_system)
        positions = atoms[:, 1:]

        ideal_bonds = parent.unit_cell.ideal_bond_lengths
        # TODO: justify the scaling factor. USPEX uses 1.5
        if not threshold:
            threshold = 1.5 * max(ideal_bonds.values())
        cutoff = 1.5 * max(ideal_bonds.values())
        neighbor_list = _create_neighbor_list(cutoff, positions[:, 1:])
        neighbor_list_typed = List()
        for neighbor in neighbor_list:
            neighbor_list_typed.append(List(neighbor))
        hardness = _calculate_bond_hardness(parent, neighbor_list, ideal_bonds)
        # spglib defines a structure by a tuple of (basis_vectors, atom_positions,
        # atom_types). basis_vectors is a 3x3 array of the basis vectors of the crystal,
        # atom_positions is the positions of the atoms in the unit cell in fractional
        # coordinates, and atom_types is the type of each atom indicated in
        # atom_positions.
        structure = (
            parent.unit_cell.primitive,
            parent.unit_cell.positions(),
            parent.unit_cell.types()
        )
        mesh = [mesh_size, mesh_size, mesh_size]  # mesh of q vectors
        # Gets all symmetrically distinct q vectors, including time reversal symmetry
        _, grid = spg.get_ir_reciprocal_mesh(mesh, structure)
        unique_q_points = grid / np.array(mesh, dtype=float)

        # sort the q vectors by magnitude
        q_magnitudes = np.linalg.norm(unique_q_points, axis=1)
        sorted_indices = np.argsort(q_magnitudes)
        unique_q_points = unique_q_points[sorted_indices]

        if len(unique_q_points) < num_q:
            warnings.warn(
                f"Fewer q_points generated than desired: {len(unique_q_points)} < "
                f"{num_q}. Recommended to increase mesh size.")

        n_atoms = len(parent.gb_indices)

        sparse_threshold = 10000

        # initialize the arrays to save the eigenvalues (frequencies) and eigenvectors
        # (displacements)
        freqs = np.zeros((num_q, num_children))
        disps = np.zeros((num_q, num_children, 3 * n_atoms))

        # For each unique q point, calculate the dynamical matrix and the associated
        # eigenvalues and eigenvectors.
        for i, q_vec in enumerate(unique_q_points[:num_q]):
            dynamical_matrix = _calculate_dynamical_matrix(
                hardness, positions, parent.gb_indices, neighbor_list_typed, q_vec)
            if 3 * n_atoms <= sparse_threshold:
                freq_vals, disp_vals = np.linalg.eigh(dynamical_matrix)
            else:
                sparse_matrix = sps.csc_matrix(dynamical_matrix)
                # scipy.sparse.linalg.eigsh can only calculate a small subset of the
                # eigenvalues and eigenvectors of a sparse matrix. Therefore, if the
                # number of children specified (which specifies how many eigenvectors we
                # need) is larger than 3 * n_atoms - 1, we cannot use this method, and
                # would need to fall back to calculating the eigenvalues using a dense
                # matrix, but that might be prohibitively expensive if we have reached
                # this point. TODO: Will need testing.
                if num_children >= 3 * n_atoms - 1 != num_children:
                    raise GBManipulatorValueError(
                        "Cannot generate the specified number of children.")
                freq_vals, disp_vals = sps.linalg.eigsh(
                    sparse_matrix, k=num_children, which="SA")
            freqs[i] = freq_vals[:num_children]
            # The eigvec associated with the Nth eigfreq for the ith q vector is saved
            # in the (start + N)th index
            disps[i, :, :] = np.real(disp_vals)[:, :num_children].T

        # Now that we have all of the frequencies for a variety of q points, we can
        # identify the N largest instabilities and use the associated displacements to
        # create the N child structures. We first filter out the frequencies at or near
        # 0, as these are associated with translational or rotational (acoustic) modes

        # TODO: Look into combining the eigenvectors of the multiple q points. AiVA suggests that weighted averages or using principle component analysis might work well in this regard.
        non_acoustic_indices = np.where(~np.isclose(freqs, 0))

        # TODO: Look into further filtering this so we only consider unique displacements. Do equivalent eigenvalues results in the same eigenvectors for different q values? What about within the same q vector?
        filtered_freqs = freqs[non_acoustic_indices]
        # We want the softest modes, which have the largest negative eigenvalues
        sorted_filtered_freq_indices = np.argsort(filtered_freqs)
        # indexing order: q_point, eigenvector number, eigenvector
        saved_disps = disps[non_acoustic_indices[0][sorted_filtered_freq_indices],
                            non_acoustic_indices[1][sorted_filtered_freq_indices], :]

        # We are going to be creating num_children separate systems based on the
        # eigen displacements. We initialize this here.
        pos = np.zeros((num_children, *positions.shape))

        # minimum allowable distance before atoms are "too close"
        d_min = 2 * parent.unit_cell.radius

        # Here we precompute the neighbor distances for each atom pair, subtracting off
        # the minimum allowable distance between the atoms.
        precomputed_distances = np.zeros(len(parent.gb_indices))
        for i, atom_idx in enumerate(parent.gb_indices):
            neighbors = neighbor_list[atom_idx]
            neighbor_positions = positions[neighbors]
            dists = np.linalg.norm(positions[atom_idx] - neighbor_positions, axis=1)
            precomputed_distances[i] = np.min(dists) - d_min

        # We now need to perform the displacement. We check to make sure that the
        # displacement does not cause atoms to overlap. We do this for each child that
        # we want to generate from this analysis.
        for mode_index in range(num_children):
            pos[mode_index] = np.copy(positions)
            disp_vector = saved_disps[mode_index].reshape(-1, 3)
            disp_magnitude = np.linalg.norm(disp_vector, axis=1)

            if np.any(disp_magnitude == 0):
                continue

            # Any True value here is a possible overlap between two atoms after the
            # displacement suggested in disp_vector.
            overlap_condition = precomputed_distances < disp_magnitude
            # disp_magnitude / disp_magnitude
            safe_displacements = np.ones_like(disp_magnitude)
            if np.any(overlap_condition):
                overlapped_atoms = precomputed_distances[overlap_condition]
                overlap_disps = disp_magnitude[overlap_condition]
                safe_displacements[overlap_condition] = overlapped_atoms / overlap_disps

            adjusted_displacements = disp_vector * safe_displacements[:, None]
            pos[mode_index, parent.gb_indices] = positions[parent.gb_indices] + \
                adjusted_displacements * (-1 if subtract_displacement else 1)

            non_gb_indices = np.setdiff1d(
                np.arange(positions.shape[0]), parent.gb_indices)

            pos[mode_index, non_gb_indices] = positions[non_gb_indices]

        structured_pos = []
        for child in pos:
            structured_p = np.zeros((len(atoms)), dtype=Atom.atom_dtype)
            structured_p["name"] = parent.whole_system["name"]
            structured_p["x"] = child[:, 0]
            structured_p["y"] = child[:, 1]
            structured_p["z"] = child[:, 2]
            structured_pos.append(structured_p)
        return structured_pos

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
