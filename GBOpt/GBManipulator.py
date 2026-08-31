# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Manipulate grain-boundary structures while preserving interface geometry.

This module owns in-memory structural transformations. External file ownership,
calculator evaluation, and optimizer policy do not belong here.
"""

from __future__ import annotations

import copy as copy_module
import multiprocessing as mp
import warnings
from dataclasses import dataclass
from itertools import combinations_with_replacement, product
from numbers import Real
from os.path import isfile

import numpy as np
import scipy.sparse as sps
import spglib as spg
from numba import float64, jit, prange
from numba.typed import List
from scipy.spatial import Delaunay, KDTree, QhullError

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
from GBOpt.FileGrainOwnership import (
    LammpsAtomData,
    LammpsDataError,
    read_lammps_data_file,
    read_lammps_dump_file,
)
from GBOpt.GBMaker import GBMaker
from GBOpt.GrainOwnership import (
    LEFT_GRAIN_LABEL,
    RIGHT_GRAIN_LABEL,
    GrainOwnership,
    GrainOwnershipError,
)
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


def _translate_inplane(
    atoms: np.ndarray,
    *,
    dy: float,
    dz: float,
    box_dims: np.ndarray,
    inplane_periodic: tuple[bool, bool],
    tolerance: float,
) -> np.ndarray:
    """Return atoms translated under explicit y/z boundary conditions.

    :param atoms: Structured atom rows to translate.
    :param dy: Keyword argument, required. y displacement in angstroms.
    :param dz: Keyword argument, required. z displacement in angstroms.
    :param box_dims: Keyword argument, required. Finite 3 by 2 Cartesian box bounds.
    :param inplane_periodic: Keyword argument, required. y/z periodicity flags.
    :param tolerance: Keyword argument, required. Coordinate tolerance in angstroms.
    :return: Translated structured atom rows.
    :raises GBManipulatorValueError: If box geometry is invalid or a nonperiodic
        displacement leaves the box.
    """
    updated = np.array(atoms, copy=True)
    for axis_name, displacement, is_periodic, axis_index in zip(
        ("y", "z"),
        (dy, dz),
        inplane_periodic,
        (1, 2),
        strict=True,
    ):
        lower = float(box_dims[axis_index, 0])
        upper = float(box_dims[axis_index, 1])
        width = upper - lower
        if not np.isfinite(lower) or not np.isfinite(upper) or width <= 0.0:
            raise GBManipulatorValueError(
                f"The {axis_name} box interval must be finite and have positive width"
            )
        translated = updated[axis_name] + displacement
        if is_periodic:
            updated[axis_name] = np.mod(translated - lower, width) + lower
            continue
        if np.any(translated < lower - tolerance) or np.any(translated >= upper):
            raise GBManipulatorValueError(
                f"d{axis_name} moves atoms outside the nonperiodic half-open "
                f"{axis_name} interval [{lower}, {upper})"
            )
        updated[axis_name] = translated
    return updated


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


def _cycle_half_open(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    shift: float,
    tolerance: float,
) -> np.ndarray:
    """Cycle coordinates through a finite half-open interval.

    :param values: Coordinate values to cycle.
    :param lower: Keyword argument, required. Inclusive interval lower bound.
    :param upper: Keyword argument, required. Exclusive interval upper bound.
    :param shift: Keyword argument, required. Cyclic displacement in angstroms.
    :param tolerance: Keyword argument, required. Coordinate tolerance in angstroms.
    :return: Cycled coordinate copy.
    :raises GBManipulatorValueError: If interval geometry is invalid.
    """
    width = upper - lower
    if not np.isfinite(lower) or not np.isfinite(upper) or width <= 0.0:
        raise GBManipulatorValueError(
            "A termination-cycling interval must be finite and have positive width"
        )
    canonical_shift = float(np.mod(shift, width))
    if np.isclose(canonical_shift, 0.0, atol=tolerance, rtol=0.0):
        return np.array(values, copy=True)
    wrapped = lower + np.mod(values + canonical_shift - lower, width)
    wrapped[np.isclose(wrapped, upper, atol=tolerance, rtol=0.0)] = lower
    return wrapped


@dataclass(frozen=True, slots=True, init=False)
class InterfaceCandidate:
    """Immutable atom rows and interface geometry for composable manipulation.

    Grain labels are local to this in-memory candidate. They are not persistent atom
    identifiers and do not define an external-file ownership format.
    """

    _atoms: np.ndarray
    _box_dims: np.ndarray
    gb_plane_x: float
    _left_grain_x_bounds: np.ndarray
    _right_grain_x_bounds: np.ndarray
    _grain_labels: np.ndarray
    inplane_periodic: tuple[bool, bool]
    normal_topology: BoundaryNormalTopology
    coordinate_tolerance: float
    interface_separation: float

    def __init__(
        self,
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
    ) -> None:
        """Initialize validated immutable candidate state.

        :param atoms: Keyword argument, required. Structured atom rows.
        :param box_dims: Keyword argument, required. Finite 3 by 2 box array.
        :param gb_plane_x: Keyword argument, required. Interface-gap midpoint.
        :param left_grain_x_bounds: Keyword argument, required. Left physical grain
            interval.
        :param right_grain_x_bounds: Keyword argument, required. Right physical grain
            interval.
        :param grain_labels: Keyword argument, required. Left/right labels aligned with
            ``atoms``.
        :param inplane_periodic: Keyword argument, required. y/z periodicity flags.
        :param normal_topology: Keyword argument, required. Boundary-normal topology.
        :param coordinate_tolerance: Keyword argument, required. Coordinate tolerance in
            angstroms.
        :param interface_separation: Keyword argument, optional, defaults to ``0.0``.
            Inserted central separation in angstroms, optional, defaults to ``0.0``.
        :raises GBManipulatorValueError: If candidate state is malformed or internally
            inconsistent.
        """
        structured = np.asarray(atoms)
        required_fields = {"name", "x", "y", "z"}
        if (
            structured.ndim != 1
            or structured.dtype.names is None
            or not required_fields.issubset(structured.dtype.names)
        ):
            raise GBManipulatorValueError(
                "InterfaceCandidate atoms must be a one-dimensional structured array "
                "containing name, x, y, and z fields"
            )
        labels = _normalize_grain_labels(grain_labels, expected_count=structured.size)
        box = _strict_float_array("box_dims", box_dims, shape=(3, 2))
        left_bounds = _strict_float_array(
            "left_grain_x_bounds",
            left_grain_x_bounds,
            shape=(2,)
        )
        right_bounds = _strict_float_array(
            "right_grain_x_bounds",
            right_grain_x_bounds,
            shape=(2,)
        )
        plane = _validate_finite_real("gb_plane_x", gb_plane_x)
        tolerance = _validate_finite_real("coordinate_tolerance", coordinate_tolerance)
        separation = _validate_finite_real("interface_separation", interface_separation)
        if tolerance <= 0.0:
            raise GBManipulatorValueError("coordinate_tolerance must be positive")
        if separation < 0.0:
            raise GBManipulatorValueError("interface_separation must be nonnegative")
        if np.any(box[:, 0] >= box[:, 1]):
            raise GBManipulatorValueError(
                "InterfaceCandidate box bounds must be strictly ordered"
            )
        if not box[0, 0] < plane < box[0, 1]:
            raise GBManipulatorValueError(
                "InterfaceCandidate gb_plane_x must lie strictly inside the x box"
            )
        if left_bounds[0] >= left_bounds[1] or right_bounds[0] >= right_bounds[1]:
            raise GBManipulatorValueError(
                "Physical grain bounds must be strictly ordered"
            )
        if (
            left_bounds[0] < box[0, 0] - tolerance
            or right_bounds[1] > box[0, 1] + tolerance
            or left_bounds[1] > plane + tolerance
            or right_bounds[0] < plane - tolerance
            or left_bounds[1] > right_bounds[0] + tolerance
        ):
            raise GBManipulatorValueError(
                "Physical grain bounds must lie inside the box on their respective "
                "sides of gb_plane_x without overlapping"
            )
        periodic = _normalize_inplane_periodic(inplane_periodic)
        try:
            topology = normalize_boundary_normal_topology(normal_topology)
        except ValueError as exc:
            raise GBManipulatorValueError(str(exc)) from exc

        for axis_index, axis_name in enumerate(("x", "y", "z")):
            coordinates = np.asarray(structured[axis_name], dtype=float)
            if not np.all(np.isfinite(coordinates)):
                raise GBManipulatorValueError(
                    "InterfaceCandidate atom coordinates must be finite"
                )
            lower = float(box[axis_index, 0])
            upper = float(box[axis_index, 1])
            if np.any(coordinates < lower - tolerance) or np.any(coordinates >= upper):
                raise GBManipulatorValueError(
                    f"InterfaceCandidate atoms must lie inside the half-open "
                    f"{axis_name} box"
                )

        left_x = np.asarray(structured["x"][labels == LEFT_GRAIN_LABEL], dtype=float)
        right_x = np.asarray(structured["x"][labels == RIGHT_GRAIN_LABEL], dtype=float)
        if (
            np.any(left_x < left_bounds[0] - tolerance)
            or np.any(left_x >= left_bounds[1])
            or np.any(right_x < right_bounds[0] - tolerance)
            or np.any(right_x >= right_bounds[1])
        ):
            raise GBManipulatorValueError(
                "Candidate atoms must lie inside their labeled physical grain bounds"
            )

        object.__setattr__(self, "_atoms", _readonly_copy(structured))
        object.__setattr__(self, "_box_dims", _readonly_copy(box, dtype=float))
        object.__setattr__(self, "gb_plane_x", plane)
        object.__setattr__(
            self, "_left_grain_x_bounds", _readonly_copy(left_bounds, dtype=float)
        )
        object.__setattr__(
            self, "_right_grain_x_bounds", _readonly_copy(right_bounds, dtype=float)
        )
        object.__setattr__(
            self, "_grain_labels", _readonly_copy(labels, dtype=np.int8)
        )
        object.__setattr__(self, "inplane_periodic", periodic)
        object.__setattr__(self, "normal_topology", topology)
        object.__setattr__(self, "coordinate_tolerance", tolerance)
        object.__setattr__(self, "interface_separation", separation)

    @property
    def atoms(self) -> np.ndarray:
        """Defensive read-only atom-array copy."""
        return _readonly_copy(self._atoms)

    @property
    def box_dims(self) -> np.ndarray:
        """Defensive read-only box copy."""
        return _readonly_copy(self._box_dims, dtype=float)

    @property
    def left_grain_x_bounds(self) -> np.ndarray:
        """Defensive read-only left-grain interval copy."""
        return _readonly_copy(self._left_grain_x_bounds, dtype=float)

    @property
    def right_grain_x_bounds(self) -> np.ndarray:
        """Defensive read-only right-grain interval copy."""
        return _readonly_copy(self._right_grain_x_bounds, dtype=float)

    @property
    def grain_labels(self) -> np.ndarray:
        """Defensive read-only grain-label copy."""
        return _readonly_copy(self._grain_labels, dtype=np.int8)

    @property
    def periodic_outer_x_interface(self) -> bool:
        """Whether the outer x faces form a second interface."""
        return self.normal_topology.periodic_outer_x_interface


@dataclass(frozen=True, slots=True)
class _SeparatedInterfaceGeometry:
    """Geometry produced by applying interface separation.

    Stores the updated simulation box, grain-boundary plane, and physical x bounds of
    both grains after a topology-aware separation operation. The box dimensions are
    defensively copied and exposed as a read-only NumPy array.
    """
    box_dims: np.ndarray
    gb_plane_x: float
    left_grain_x_bounds: tuple[float, float]
    right_grain_x_bounds: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "box_dims", _readonly_copy(self.box_dims, dtype=float))


def _separated_interface_geometry(
    *,
    box_dims: np.ndarray,
    gb_plane_x: float,
    left_grain_x_bounds: np.ndarray,
    right_grain_x_bounds: np.ndarray,
    interface_separation: float,
    normal_topology: BoundaryNormalTopology,
    coordinate_tolerance: float,
) -> _SeparatedInterfaceGeometry:
    """Calculate topology-aware geometry after opening the central interface.

    The input grain bounds must be contiguous at ``gb_plane_x`` before separation. The
    left grain remains fixed, while the right grain and its physical x bounds are
    shifted in the positive x direction by ``interface_separation``. The returned
    grain-boundary plane lies at the midpoint of the resulting central gap.

    For a periodic bicrystal, both grains must initially span the complete x box without
    vacuum. The x box length is increased by twice the requested separation so that
    equal gaps are introduced at the central interface and across the periodic outer x
    boundary.

    For a single-interface slab, at least one free-surface or vacuum interval must
    already exist along x. The upper x box bound is increased by the requested
    separation, preserving the existing outer vacuum intervals while opening only the
    central grain boundary.

    :param box_dims: Keyword argument, required. Simulation-box bounds, in Angstroms,
        with one ``(lower, upper)`` row per Cartesian axis.
    :param gb_plane_x: Keyword argument, required. Initial x coordinate, in Angstroms,
        of the contiguous grain-boundary plane.
    :param left_grain_x_bounds: Keyword argument, required. Physical lower and upper x
        bounds of the left grain, in Angstroms.
    :param right_grain_x_bounds: Keyword argument, required. Physical lower and upper x
        bounds of the right grain, in Angstroms.
    :param interface_separation: Keyword argument, required. Distance, in Angstroms, by
        which the right grain is displaced in the positive x direction.
    :param normal_topology: Keyword argument, required. Physical topology along the
        grain-boundary normal.
    :param coordinate_tolerance: Keyword argument, required. Absolute coordinate
        tolerance, in Angstroms, used when comparing grain and box boundaries.
    :return: Updated box bounds, central-interface plane, and physical x bounds of both
        grains.
    :raises GBManipulatorValueError: If the initial grain bounds are not contiguous, the
        physical grain bounds are inconsistent with the specified topology, a slab has
        no free-surface or vacuum interval, or the boundary-normal topology is unknown.
    """
    box = np.asarray(box_dims, dtype=float)
    left = np.asarray(left_grain_x_bounds, dtype=float)
    right = np.asarray(right_grain_x_bounds, dtype=float)
    plane = float(gb_plane_x)
    separation = float(interface_separation)
    tolerance = float(coordinate_tolerance)
    xlo = float(box[0, 0])
    xhi = float(box[0, 1])

    if not np.isclose(left[1], plane, atol=tolerance, rtol=0.0) or not np.isclose(
        right[0], plane, atol=tolerance, rtol=0.0
    ):
        raise GBManipulatorValueError(
            "Interface separation requires initially contiguous grain bounds"
        )

    new_box = np.array(box, copy=True)
    if normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL:
        if not np.isclose(left[0], xlo, atol=tolerance, rtol=0.0) or not np.isclose(
            right[1], xhi, atol=tolerance, rtol=0.0
        ):
            raise GBManipulatorValueError(
                "Periodic separation requires zero-vacuum physical grain bounds"
            )
        new_box[0, 1] = xhi + 2.0 * separation
    elif normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB:
        left_vacuum = float(left[0] - xlo)
        right_vacuum = float(xhi - right[1])
        if left_vacuum < -tolerance or right_vacuum < -tolerance:
            raise GBManipulatorValueError(
                "Slab physical grain bounds must lie inside the x box"
            )
        if left_vacuum <= tolerance and right_vacuum <= tolerance:
            raise GBManipulatorValueError(
                "Slab separation requires a free-surface or vacuum interval"
            )
        new_box[0, 1] = xhi + separation
    else:
        raise GBManipulatorValueError(
            "Interface separation requires known boundary-normal topology"
        )

    return _SeparatedInterfaceGeometry(
        box_dims=new_box,
        gb_plane_x=plane + separation / 2.0,
        left_grain_x_bounds=(float(left[0]), float(left[1])),
        right_grain_x_bounds=(
            float(right[0] + separation),
            float(right[1] + separation),
        ),
    )


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
    """


class ParentFileMissingDataError(ParentError):
    """
    Exception raised when data is missing from a snapshot that is otherwise formatted
    correctly.
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
    __num_to_name = {val: key for key, val in Atom._numbers.items()}

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
        return InterfaceCandidate(
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

        :param system_file: Filename of the atom structure file. Currently allowed
            formats: LAMMPS dump file, LAMMPS input file.
        :param unit_cell: Nominal unit cell of the bulk structure.
        :param gb_thickness: Thickness of the GB region, given in angstroms.
        :param type_dict: Conversion from type number to type name, optional. Note that
            if this is not provided and the snapshot does not indicate the atom names,
            atom names are assumed started from "H".
        :raises ParentValueError: Exception raised if unit_cell is not passed in or the
            file format of the file is unrecognized, or the file has less than 10 lines.
        :raises ParentFileNotFoundError: Exception raised if the specified file is not
            found.
        """

        if not unit_cell:
            raise ParentValueError("Unit cell must be specified for files")
        self.__unit_cell = unit_cell
        self.__gb_thickness = gb_thickness
        self.__inplane_periodic = (True, True)
        self.__coordinate_tolerance = 1.0e-10
        self.__grain_ownership = None
        self.__initial_atom_ids = None
        self.__normal_topology = BoundaryNormalTopology.UNKNOWN
        if not isfile(system_file):
            raise ParentFileNotFoundError(f"{system_file} does not exist.")
        # We need to first identify what type of file it is. Since filenames can be just
        # about anything, we do this by checking the first few lines of the file.
        head = []
        try:
            # The 10 here is arbitrary. We may need to look into making this more robust.
            with open(system_file) as f:
                head = [next(f) for _ in range(10)]
        except StopIteration as e:
            raise ParentValueError(
                f"Unable to determine format of {system_file}. File too short. {e}")

        keywords = {
            self.__init_from_lammps_dump: [
                "ITEM: TIMESTEP",
                "ITEM: NUMBER OF ATOMS",
                "ITEM: BOX BOUNDS",
                "ITEM: ATOMS",
            ],
            self.__init_from_lammps_input: [
                "atoms",
                "bonds",
                "angles",
                "dihedrals",
                "impropers",
                "atom types",
                "bond types",
                "angle types",
                "dihedral types",
                "improper types",
                "xlo xhi",
                "ylo yhi",
                "zlo zhi",
                "xy xz yz",
                "avec",
                "bvec",
                "cvec",
                "abc origin",
            ]
        }

        for method, file_keywords in keywords.items():
            if any(keyword in line for keyword in file_keywords for line in head):
                if grain_ownership is None:
                    warnings.warn(
                        "File-backed Parent initialization without explicit grain "
                        "ownership is deprecated because gb_plane_x and grain "
                        "membership must be inferred from coordinates. Supply "
                        "grain_ownership with explicit interface metadata instead.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                method(
                    system_file,
                    unit_cell,
                    gb_thickness,
                    type_dict,
                    grain_ownership,
                )
                break
        else:
            raise ParentValueError(f"Unknown file format for {system_file}")

        if self.__grain_ownership is None:
            self.__left_grain_x_bounds = np.array(
                [self.__box_dims[0, 0], self.__gb_plane_x], dtype=float
            )
            self.__right_grain_x_bounds = np.array(
                [self.__gb_plane_x, self.__box_dims[0, 1]], dtype=float
            )

    def __init_from_owned_snapshot(
        self,
        snapshot: LammpsAtomData,
        grain_ownership: GrainOwnership,
    ) -> None:
        """Restore a file-backed parent from explicit ownership and parsed rows.

        :param snapshot: Parsed LAMMPS snapshot in file-row order.
        :param grain_ownership: Explicit ownership keyed by serialization-local IDs.
        :raises GrainOwnershipError: If IDs, geometry, or topology are inconsistent.
        """
        aligned = grain_ownership.aligned_to(snapshot.atom_ids)
        file_ids = snapshot.atom_ids
        order = np.argsort(file_ids, kind="stable")
        sorted_ids = file_ids[order]
        ownership = aligned.aligned_to(sorted_ids)
        atoms = snapshot.atoms[order]
        box_dims = snapshot.box_dims
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
        if snapshot.boundary_periodic is not None:
            expected = (
                ownership.periodic_outer_x_interface,
                *ownership.inplane_periodic,
            )
            if snapshot.boundary_periodic != expected:
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

    def __init_from_lammps_dump(
        self,
        system_file: str,
        unit_cell: UnitCell,
        gb_thickness: float,
        type_dict: dict | None,
        grain_ownership: GrainOwnership | None = None,
    ) -> None:
        """
        Method for initializing the Parent using a LAMMPS dump file.

        :param system_file: Filename of the dump file.
        :param unit_cell: Nominal unit cell of the bulk structure.
        :param gb_thickness: Thickness of the GB region, given in angstroms.
        :param type_dict: Conversion from type number to type name, optional. Note that
            if this is not provided and the snapshot does not indicate the atom names,
            atom names are assumed started from "H".
        :param file_keywords: List of keywords used to identify different sections of
            the file.
        :raises ParentCorruptedFileError: Exception raised if the file is not formatted
            correctly.
        :raises ParentFileMissingDataError: Exception raised if the file is otherwise
            formatted correctly, but is missing required data.
        """
        if grain_ownership is not None:
            try:
                snapshot = read_lammps_dump_file(
                    system_file,
                    type_dict=type_dict,
                )
                self.__init_from_owned_snapshot(snapshot, grain_ownership)
            except (LammpsDataError, GrainOwnershipError) as exc:
                raise ParentValueError(
                    f"invalid explicit ownership for {system_file}"
                ) from exc
            return

        skip_rows = 0
        with open(system_file) as f:
            line = f.readline()
            skip_rows += 1
            # skip to the box bounds
            while not line.startswith("ITEM: BOX BOUNDS"):
                line = f.readline()
                skip_rows += 1
                if not line:
                    raise ParentCorruptedFileError(
                        f"Box bounds not found in {system_file}")
            skip_rows += 3
            if len(line.split()) == 6:  # orthogonal box
                x_dims = [float(i) for i in f.readline().split()]
                y_dims = [float(i) for i in f.readline().split()]
                z_dims = [float(i) for i in f.readline().split()]
            elif len(line.split()) == 9:  # triclinic box, restricted format
                xline = f.readline().split()
                yline = f.readline().split()
                zline = f.readline().split()
                x_dims, _ = ([float(i) for i in xline[0:2]], float(xline[2]))
                y_dims, _ = ([float(i) for i in yline[0:2]], float(yline[2]))
                z_dims, _ = ([float(i) for i in zline[0:2]], float(zline[2]))
            elif len(line.split()) == 8:  # triclinic box, general format
                xline = f.readline().split()
                yline = f.readline().split()
                zline = f.readline().split()
                origin = np.empty((3,))
                A, origin[0] = (np.array([float(i)
                                for i in xline[0:3]]), float(xline[3]))
                B, origin[1] = (np.array([float(i)
                                for i in yline[0:3]]), float(yline[3]))
                C, origin[2] = (np.array([float(i)
                                for i in zline[0:3]]), float(zline[3]))

                a = np.array([np.linalg.norm(A), 0, 0])
                Ahat = A / a[0]
                b = np.array([np.dot(B, Ahat), np.cross(Ahat, B), 0])
                AxB = np.cross(A, B)
                AxBhat = AxB/np.linalg.norm(AxB)
                c = np.array([np.dot(C, Ahat), np.dot(
                    C, np.cross(AxBhat, Ahat)), np.abs(np.dot(C, AxBhat))])

                x_dims = [origin[0], origin[0] + a[0]]
                y_dims = [origin[1], origin[1] + a[1] + b[1]]
                z_dims = [origin[2], origin[2] + c[2]]
            else:
                raise ParentCorruptedFileError(
                    f"Box bounds corrupted in {system_file}")
            if not (x_dims or y_dims or z_dims) or len(x_dims) != 2 or \
                    len(y_dims) != 2 or len(z_dims) != 2:
                raise ParentCorruptedFileError(
                    f"Box bounds corrupted in {system_file}")
            self.__box_dims = np.array([x_dims, y_dims, z_dims])
            self.__x_dim = x_dims[1] - x_dims[0]
            self.__y_dim = y_dims[1] - y_dims[0]
            self.__z_dim = z_dims[1] - z_dims[0]
            # TODO: Need a more robust calculation of where the GB is located. This calculation is duplicated.
            grain_cutoff = (x_dims[1] - x_dims[0]) / 2 + x_dims[0]
            line = f.readline()
            skip_rows += 1
            while not line.startswith("ITEM: ATOMS"):
                line = f.readline()
                skip_rows += 1
                if not line:
                    raise ParentCorruptedFileError(
                        f"Atoms not found in {system_file}")
            atom_attributes = line.split()[2:]
            required_attributes = ["type", "x", "y", "z"]

            if not all(i in atom_attributes for i in required_attributes):
                raise ParentFileMissingDataError(
                    f"One or more required attributes are missing.\n"
                    f"Required: {required_attributes}, "
                    f"available: {atom_attributes}")
            required_attribute_indices = {attr: atom_attributes.index(
                attr) for attr in required_attributes}

            typelabel_in_attrs = "typelabel" in atom_attributes
            if typelabel_in_attrs:
                required_attribute_indices["typelabel"] = atom_attributes.index(
                    "typelabel")
            col_indices = [required_attribute_indices["typelabel"] if typelabel_in_attrs else required_attribute_indices["type"],
                           required_attribute_indices["x"], required_attribute_indices["y"], required_attribute_indices["z"]]

            id_to_name = {}
            if type_dict:
                if all(isinstance(key, int) and isinstance(val, str) for key, val in type_dict.items()):
                    id_to_name = dict(type_dict)
                elif all(isinstance(key, str) and isinstance(val, int) for key, val in type_dict.items()):
                    id_to_name = {val: key for key, val in type_dict.items()}
                else:
                    raise ParentValueError(
                        "type_dict must be a dict[str, int] or dict[int, str]."
                    )

            def convert_type(value):
                if typelabel_in_attrs:
                    return value
                if id_to_name:
                    type_id = int(value)
                    if type_id not in id_to_name:
                        raise ParentFileMissingDataError(
                            f"Type id {type_id} not found in type mapping."
                        )
                    return id_to_name[type_id]
                return self.__num_to_name[int(value)]
            max_rows = 0
            line = f.readline()  # read the next line to move the file pointer ahead.
            while not line.startswith("ITEM"):
                line = f.readline()
                max_rows += 1
                if not line:
                    break

        self.__whole_system = np.loadtxt(system_file, skiprows=skip_rows, max_rows=max_rows, converters={
            col_indices[0]: convert_type}, usecols=tuple(col_indices), dtype=Atom.atom_dtype)
        mask = self.__whole_system["x"] < grain_cutoff
        self.__left_grain = self.__whole_system[mask]
        self.__right_grain = self.__whole_system[~mask]
        self.__gb_plane_x = (
            max(self.__left_grain["x"]) + min(self.__right_grain["x"])) / 2

    def __init_from_lammps_input(
        self,
        system_file: str,
        unit_cell: UnitCell,
        gb_thickness: float,
        type_dict: dict | None,
        grain_ownership: GrainOwnership | None = None,
    ) -> None:
        """
        Method for initializing the Parent using a LAMMPS input file.

        :param system_file: Filename of the LAMMPS input file.
        :param unit_cell: Nominal unit cell of the bulk structure.
        :param gb_thickness: Thickness of the GB region, given in angstroms.
        :param type_dict: Conversion from type number to type name, optional. Note that
            if this is not provided and the snapshot does not indicate the atom names,
            atom names are assumed started from "H".
        :param file_keywords: List of keywords used to identify different sections of
            the file.
        :raises ParentCorruptedFileError: Exception raised if the file is not formatted
            correctly.
        :raises ParentFileMissingDataError: Exception raised if the file is otherwise
            formatted correctly, but is missing required data.
        """
        n_atoms = n_types = 0
        x_dims = y_dims = z_dims = []
        name_to_id = {}
        id_to_name = {}
        if type_dict:
            if all(isinstance(key, str) and isinstance(val, int) for key, val in type_dict.items()):
                name_to_id = dict(type_dict)
                id_to_name = {val: key for key, val in type_dict.items()}
            elif all(isinstance(key, int) and isinstance(val, str) for key, val in type_dict.items()):
                id_to_name = dict(type_dict)
                name_to_id = {val: key for key, val in type_dict.items()}
            else:
                raise ParentValueError(
                    "type_dict must be a dict[str, int] or dict[int, str]."
                )
        skiprows = 0

        if grain_ownership is not None:
            try:
                snapshot = read_lammps_data_file(
                    system_file,
                    type_dict=type_dict,
                )
                self.__init_from_owned_snapshot(snapshot, grain_ownership)
            except (LammpsDataError, GrainOwnershipError) as exc:
                raise ParentValueError(
                    f"invalid explicit ownership for {system_file}"
                ) from exc
            return

        with open(system_file) as f:
            lines = iter(f)
            # Skip header and blank lines
            next(lines)
            next(lines)
            skiprows += 2

            for line in lines:
                skiprows += 1
                line = line.strip()

                if line.startswith("Atoms"):
                    next(lines)  # Skip the blank line after "Atoms"
                    skiprows += 1
                    break

                line_sp = line.split()

                if "atoms" in line:
                    n_atoms = int(line_sp[0])
                elif "atom types" in line:
                    n_types = int(line_sp[0])
                elif "xlo xhi" in line:
                    x_dims = [float(line_sp[0]), float(line_sp[1])]
                elif "ylo yhi" in line:
                    y_dims = [float(line_sp[0]), float(line_sp[1])]
                elif "zlo zhi" in line:
                    z_dims = [float(line_sp[0]), float(line_sp[1])]
                elif "xy xz yz" in line:
                    tilt = [float(line_sp[0]), float(line_sp[1]), float(line_sp[2])]
                    self.__tilt = tilt
                elif line == "Atom Type Labels":
                    next(lines)  # Skip the blank line before the data
                    skiprows += 1
                    num_labels = 0

                    for label_line in lines:
                        skiprows += 1
                        label_line = label_line.strip().split()
                        if not label_line:
                            break
                        type_id = int(label_line[0])
                        type_name = label_line[1]
                        name_to_id[type_name] = type_id
                        id_to_name[type_id] = type_name
                        num_labels += 1

                    if num_labels != n_types:
                        raise ParentCorruptedFileError(
                            "Number of labels does not equal number of atom types."
                        )

        def convert_type(value):
            if isinstance(value, bytes):
                value = value.decode()
            value = value.strip()
            try:
                type_id = int(value)
                if id_to_name:
                    if type_id not in id_to_name:
                        raise ParentFileMissingDataError(
                            f"Type id {type_id} not found in type mapping."
                        )
                    return id_to_name[type_id]
                return self.__num_to_name[type_id]
            except ValueError:
                return value
        # We now have to make some assumptions about how the data is actually formatted.
        # Here, we assume the following:
        #  column 2: atom type (numeric, if "Atom Type Labels" not found previously, else string)
        #  column 3: x position
        #  column 4: y position
        #  column 5: z position
        self.__box_dims = np.array([x_dims, y_dims, z_dims])
        self.__x_dim = x_dims[1] - x_dims[0]
        self.__y_dim = y_dims[1] - y_dims[0]
        self.__z_dim = z_dims[1] - z_dims[0]
        # TODO: Need a more robust calculation of where the GB is located. This calculation is duplicated.
        grain_cutoff = (x_dims[1] - x_dims[0]) / 2 + x_dims[0]
        self.__whole_system = np.loadtxt(
            system_file,
            skiprows=skiprows,
            max_rows=n_atoms,
            converters={1: convert_type},
            usecols=[1, 2, 3, 4],
            dtype=Atom.atom_dtype,
        )
        mask = self.__whole_system["x"] < grain_cutoff
        self.__left_grain = self.__whole_system[mask]
        self.__right_grain = self.__whole_system[~mask]
        self.__gb_plane_x = (
            max(self.__left_grain["x"]) + min(self.__right_grain["x"])) / 2

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
    :param int NB: Number of atoms of type B in the nominal unit cell.
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


@jit(nopython=True, cache=True)
def _calculate_local_order(atom, neighs, unit_cell_types, unit_cell_a0, N, Delta, Rmax):
    """
    Calculates the local order parameter following Lyakhov *et al.*, Computer Phys.
    Comm. 181 (2010) 1623-1632 (Eq. 5).

    :param np.ndarray atom: Atom we are calculating the local order for.
    :param np.ndarray neighs: Neighbors of ``atom``.
    :param np.ndarray unit_cell_types: The types of the atoms in the unit cell.
    :param float unit_cell_a0: The lattice parameter.
    :param int N: The number of atoms in the unit cell.
    :param float Delta: Bin size to calculate the fingerprint vector.
    :param float Rmax: Maximum distance from ``atom`` to consider as a neighbor to
        ``atom`` in angstroms.
    :return: The local order parameter for ``atom`` based on its neighbors.
    """
    local_sum = 0
    atom_types = np.unique(unit_cell_types)
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


def _create_periodic_image_cloud(
    rcut: float,
    positions: np.ndarray,
    box_dims: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return periodic images and their source metadata for a neighbor cutoff.

    Periodic axes are replicated far enough that every image which could lie within
    ``rcut`` of a position in the primary box is represented. Nonperiodic axes are not
    replicated.

    :param rcut: Maximum neighbor distance in angstroms.
    :param positions: Cartesian atom or site positions with shape ``(N, 3)``.
    :param box_dims: Simulation-box bounds with shape ``(3, 2)``.
    :param periodic: Periodicity along x, y, and z.
    :return: Image positions, source indices, and integer image offsets.
    :raises GBManipulatorValueError: If the cutoff or periodic box geometry is invalid.
    """
    rcut = _validate_finite_real("rcut", rcut)
    if rcut <= 0.0:
        raise GBManipulatorValueError("rcut must be greater than zero")

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise GBManipulatorValueError("positions must have shape (N, 3)")
    if not np.all(np.isfinite(positions)):
        raise GBManipulatorValueError("positions must contain only finite values")

    box_dims = np.asarray(box_dims, dtype=float)
    if box_dims.shape != (3, 2) or not np.all(np.isfinite(box_dims)):
        raise GBManipulatorValueError(
            "box_dims must be a finite array with shape (3, 2)"
        )

    if len(periodic) != 3 or not all(
        isinstance(flag, (bool, np.bool_))
        for flag in periodic
    ):
        raise GBManipulatorValueError(
            "periodic must contain exactly three Boolean values"
        )

    widths = box_dims[:, 1] - box_dims[:, 0]

    image_ranges = []
    for axis, (width, is_periodic) in enumerate(zip(widths, periodic, strict=True)):
        if is_periodic:
            if width <= 0.0:
                raise GBManipulatorValueError(
                    f"periodic box width for axis {axis} must be greater than zero"
                )

            num_images = int(np.ceil(rcut / width))
            image_ranges.append(
                range(-num_images, num_images + 1)
            )
        else:
            image_ranges.append(range(1))

    offsets = np.asarray(list(product(*image_ranges)), dtype=np.intp)
    translations = offsets * widths

    num_positions = len(positions)

    image_positions = (
        positions[np.newaxis, :, :]
        + translations[:, np.newaxis, :]
    ).reshape(-1, 3)

    source_indices = np.tile(np.arange(num_positions, dtype=np.intp), len(offsets))

    image_offsets = np.repeat(offsets, num_positions, axis=0)

    return image_positions, source_indices, image_offsets


def _create_periodic_image_neighborhoods(
    rcut: float,
    atoms: np.ndarray,
    center_indices: np.ndarray,
    box_dims: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> list[np.ndarray]:
    """Return periodic-image neighborhoods for selected atoms.

    The returned arrays contain ``type, x, y, z`` rows for every periodic image within
    ``rcut``. Only the center atom in the unshifted image is excluded. Translated images
    of the same physical atom remain because each image contributes separately to a
    radial fingerprint.

    :param rcut: Maximum neighbor distance in angstroms.
    :param atoms: Numeric atom array with columns ``type, x, y, z``.
    :param center_indices: Indices of atoms whose neighborhoods are required.
    :param box_dims: Simulation-box bounds with shape ``(3, 2)``.
    :param periodic: Periodicity along x, y, and z.
    :return: One periodic-image neighborhood array for each requested center.
    :raises GBManipulatorValueError: If atom data or center indices are invalid.
    """
    atoms = np.asarray(atoms)
    if atoms.ndim != 2 or atoms.shape[1] != 4:
        raise GBManipulatorValueError("atoms must have shape (N, 4)")

    positions = np.asarray(atoms[:, 1:], dtype=float)
    atom_types = np.asarray(atoms[:, 0])

    center_indices = np.asarray(center_indices, dtype=np.intp)

    if np.any(center_indices < 0) or np.any(center_indices >= len(atoms)):
        raise GBManipulatorValueError("center_indices contains an out-of-range index")

    image_positions, source_indices, image_offsets = _create_periodic_image_cloud(
        rcut,
        positions,
        box_dims,
        periodic,
    )

    image_types = np.tile(atom_types, len(image_positions) // len(atoms))

    tree = KDTree(image_positions)

    neighborhoods = []

    for center_idx in center_indices:
        matches = np.asarray(
            tree.query_ball_point(positions[center_idx], r=rcut),
            dtype=np.intp,
        )

        # Exclude only the unshifted center. Translated images of the same physical atom
        # are legitimate radial neighbors.
        central_self = (
            (source_indices[matches] == center_idx)
            & np.all(image_offsets[matches] == 0, axis=1,)
        )

        matches = matches[~central_self]

        neighborhoods.append(
            np.column_stack((image_types[matches], image_positions[matches]))
        )

    return neighborhoods


def _create_periodic_neighbor_list(
    rcut: float,
    positions: np.ndarray,
    center_indices: np.ndarray,
    box_dims: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return unique physical neighbors and minimum-image distances.

    Unlike :func:`_create_periodic_image_neighborhoods`, this helper collapses all
    periodic images of the same source position to one physical neighbor and retains its
    shortest image distance. Every image of the center itself is excluded.

    :param rcut: Maximum neighbor distance in angstroms.
    :param positions: Cartesian atom or candidate-site positions with shape ``(N, 3)``.
    :param center_indices: Indices of positions whose neighbors are required.
    :param box_dims: Simulation-box bounds with shape ``(3, 2)``.
    :param periodic: Periodicity along x, y, and z.
    :return: Parallel lists of neighbor source-index arrays and distance arrays. Within
        each pair, ``indices[k]`` corresponds to ``distances[k]``.
    :raises GBManipulatorValueError: If center indices are invalid.
    """
    positions = np.asarray(positions, dtype=float)

    center_indices = np.asarray(center_indices, dtype=np.intp)

    if np.any(center_indices < 0) or np.any(center_indices >= len(positions)):
        raise GBManipulatorValueError(
            "center_indices contains an out-of-range index"
        )

    image_positions, source_indices, _ = _create_periodic_image_cloud(
        rcut,
        positions,
        box_dims,
        periodic,
    )

    tree = KDTree(image_positions)

    neighbor_indices = []
    neighbor_distances = []

    for center_idx in center_indices:
        matches = np.asarray(
            tree.query_ball_point(positions[center_idx], r=rcut),
            dtype=np.intp,
        )

        sources = source_indices[matches]

        # A translated image of the center is still the same physical atom, so all
        # center-source images are excluded for physical-neighbor selection.
        keep = sources != center_idx
        matches = matches[keep]
        sources = sources[keep]

        if len(matches) == 0:
            neighbor_indices.append(np.empty(0, dtype=np.intp))
            neighbor_distances.append(np.empty(0, dtype=float))
            continue

        distances = np.linalg.norm(
            image_positions[matches] - positions[center_idx],
            axis=1,
        )

        # Sort first by source index and then by distance. The first entry for each
        # source is therefore its minimum-image representative.
        order = np.lexsort((distances, sources))

        sorted_sources = sources[order]
        sorted_distances = distances[order]

        first_for_source = np.concatenate(
            (
                np.array([True]),
                sorted_sources[1:]
                != sorted_sources[:-1],
            )
        )

        neighbor_indices.append(sorted_sources[first_for_source])
        neighbor_distances.append(sorted_distances[first_for_source])

    return neighbor_indices, neighbor_distances


def _tile_periodic_positions_once(
    positions: np.ndarray,
    box_dims: np.ndarray,
    periodic: tuple[bool, bool, bool],
) -> np.ndarray:
    """Tile positions by one adjacent image along each periodic axis.

    :param positions: Cartesian positions with shape ``(N, 3)``.
    :param box_dims: Simulation-box bounds with shape ``(3, 2)``.
    :param periodic: Periodicity along x, y, and z.
    :return: Positions in the primary cell and its required adjacent images.
    """
    positions = np.asarray(positions, dtype=float)
    box_dims = np.asarray(box_dims, dtype=float)
    widths = box_dims[:, 1] - box_dims[:, 0]

    image_ranges = [(-1, 0, 1) if is_periodic else (0,) for is_periodic in periodic]
    offsets = np.asarray(list(product(*image_ranges)), dtype=np.intp)
    translations = offsets * widths

    return (
        positions[np.newaxis, :, :] + translations[:, np.newaxis, :]
    ).reshape(-1, 3)


def _wrap_periodic_positions(
    positions: np.ndarray,
    box_dims: np.ndarray,
    periodic: tuple[bool, bool, bool],
    tolerance: float,
) -> np.ndarray:
    """Wrap Cartesian positions into the primary cell along periodic axes.

    :param positions: Cartesian positions with shape ``(N, 3)``.
    :param box_dims: Simulation-box bounds with shape ``(3, 2)``.
    :param periodic: Periodicity along x, y, and z.
    :param tolerance: Coordinate tolerance in angstroms.
    :return: A wrapped copy of the positions.
    """
    wrapped = np.array(positions, dtype=float, copy=True)
    box_dims = np.asarray(box_dims, dtype=float)

    for axis, is_periodic in enumerate(periodic):
        if not is_periodic:
            continue

        lower, upper = box_dims[axis]
        width = upper - lower
        wrapped[:, axis] = np.mod(wrapped[:, axis] - lower, width) + lower

        near_upper = upper - wrapped[:, axis] <= tolerance
        wrapped[near_upper, axis] = lower

    return wrapped


def _minimum_periodic_distances(
    query_positions: np.ndarray,
    reference_positions: np.ndarray,
    box_dims: np.ndarray,
    periodic: tuple[bool, bool, bool],
    tolerance: float,
) -> np.ndarray:
    """Return minimum-image distance from each query to any reference position.

    :param query_positions: Cartesian query positions with shape ``(N, 3)``.
    :param reference_positions: Cartesian reference positions with shape ``(M, 3)``.
    :param box_dims: Simulation-box bounds with shape ``(3, 2)``.
    :param periodic: Periodicity along x, y, and z.
    :param tolerance: Coordinate tolerance in angstroms.
    :return: Minimum distance for each query position.
    """
    query_positions = np.asarray(query_positions, dtype=float)
    reference_positions = np.asarray(reference_positions, dtype=float)

    if len(query_positions) == 0:
        return np.empty(0, dtype=float)

    wrapped_queries = _wrap_periodic_positions(
        query_positions, box_dims, periodic, tolerance
    )
    wrapped_references = _wrap_periodic_positions(
        reference_positions, box_dims, periodic, tolerance
    )
    tiled_references = _tile_periodic_positions_once(
        wrapped_references, box_dims, periodic
    )

    distances, _ = KDTree(tiled_references).query(wrapped_queries, k=1)
    return np.asarray(distances, dtype=float)


def _deduplicate_positions(positions: np.ndarray, tolerance: float) -> np.ndarray:
    """Return deterministically ordered positions unique within a tolerance.

    :param positions: Cartesian positions with shape ``(N, 3)``.
    :param tolerance: Coordinate equivalence tolerance in angstroms.
    :return: Deterministically ordered unique positions.
    """
    positions = np.asarray(positions, dtype=float)

    if len(positions) == 0:
        return positions.copy()

    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    positions = positions[order]

    if tolerance <= 0.0:
        return np.unique(positions, axis=0)

    keys = np.rint(positions / tolerance)
    _, first_indices = np.unique(keys, axis=0, return_index=True)

    return positions[np.sort(first_indices)]


def _tetrahedron_circumcenters(tetrahedra: np.ndarray) -> np.ndarray:
    """Calculate Cartesian circumcenters of nondegenerate tetrahedra.

    :param tetrahedra: Tetrahedron vertices with shape ``(N, 4, 3)``.
    :return: Cartesian circumcenters with shape ``(N, 3)``.
    :raises np.linalg.LinAlgError: If a tetrahedron is singular.
    """
    tetrahedra = np.asarray(tetrahedra, dtype=float)

    origins = tetrahedra[:, 0]
    edges = tetrahedra[:, 1:] - origins[:, np.newaxis, :]
    rhs = np.einsum("ijk,ijk->ij", edges, edges)

    relative_centers = np.linalg.solve(2.0 * edges, rhs[..., np.newaxis])[..., 0]

    return origins + relative_centers


def _delaunay_insertion_sites(
    gb_positions: np.ndarray,
    reference_positions: np.ndarray,
    atom_radius: float,
    box_dims: np.ndarray,
    periodic: tuple[bool, bool, bool],
    coordinate_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate insertion sites from periodic Delaunay tetrahedra.

    Candidate sites are circumcenters of nondegenerate tetrahedra. Only circumcenters
    lying in the primary GB insertion domain are retained, and candidates must have
    sufficient minimum-image clearance from every atom in the full structure.

    :param gb_positions: Positions used to construct the GB triangulation.
    :param reference_positions: Positions of all atoms used for clearance checking.
    :param atom_radius: Required minimum atom-site distance in angstroms.
    :param box_dims: Simulation-box bounds with shape ``(3, 2)``.
    :param periodic: Periodicity along x, y, and z.
    :param coordinate_tolerance: Coordinate tolerance in angstroms.
    :return: Candidate insertion sites and their normalized probabilities.
    :raises GBManipulatorValueError: If no valid insertion sites can be generated.
    """
    gb_positions = np.asarray(gb_positions, dtype=float)
    box_dims = np.asarray(box_dims, dtype=float)

    tiled_positions = _tile_periodic_positions_once(
        gb_positions, box_dims, periodic
    )

    try:
        triangulation = Delaunay(tiled_positions)
    except QhullError as exc:
        raise GBManipulatorValueError(
            "Unable to construct Delaunay insertion-site triangulation."
        ) from exc

    tetrahedra = tiled_positions[triangulation.simplices]

    origins = tetrahedra[:, 0]
    edges = tetrahedra[:, 1:] - origins[:, np.newaxis, :]
    volumes = np.abs(np.linalg.det(edges)) / 6.0

    positive_volumes = volumes[volumes > 0.0]
    if len(positive_volumes) == 0:
        raise GBManipulatorValueError(
            "Delaunay triangulation contains no nondegenerate tetrahedra."
        )

    # Retain the existing relative small-volume rejection policy, but calculate
    # volumes from the actual periodic-image tetrahedra.
    volume_threshold = 1e-3 * np.median(positive_volumes)
    valid_tetrahedra = tetrahedra[volumes > volume_threshold]

    try:
        circumcenters = _tetrahedron_circumcenters(valid_tetrahedra)
    except np.linalg.LinAlgError as exc:
        raise GBManipulatorValueError(
            "Unable to calculate Delaunay circumcenters."
        ) from exc

    finite_mask = np.all(np.isfinite(circumcenters), axis=1)
    circumcenters = circumcenters[finite_mask]

    if len(circumcenters) == 0:
        raise GBManipulatorValueError(
            "Delaunay triangulation produced no finite circumcenters."
        )

    # The GB atom extent defines the nonperiodic insertion domain. Explicit box bounds
    # define periodic dimensions.
    site_bounds = np.column_stack(
        (np.min(gb_positions, axis=0), np.max(gb_positions, axis=0))
    )
    for axis, is_periodic in enumerate(periodic):
        if is_periodic:
            site_bounds[axis] = box_dims[axis]

    # Select the representative whose unwrapped circumcenter lies in the primary domain.
    # Do this before wrapping so outer tiled-hull tetrahedra do not map back into the
    # primary cell as artificial candidates.
    in_bounds = np.ones(len(circumcenters), dtype=bool)

    for axis, is_periodic in enumerate(periodic):
        lower, upper = site_bounds[axis]

        if is_periodic:
            in_bounds &= circumcenters[:, axis] >= lower - coordinate_tolerance
            in_bounds &= circumcenters[:, axis] < upper + coordinate_tolerance
        else:
            in_bounds &= circumcenters[:, axis] >= lower - coordinate_tolerance
            in_bounds &= circumcenters[:, axis] <= upper + coordinate_tolerance

    circumcenters = circumcenters[in_bounds]
    circumcenters = _wrap_periodic_positions(
        circumcenters, box_dims, periodic, coordinate_tolerance
    )
    circumcenters = _deduplicate_positions(
        circumcenters, coordinate_tolerance
    )

    if len(circumcenters) == 0:
        raise GBManipulatorValueError(
            "Delaunay triangulation produced no insertion sites in the GB region."
        )

    distances = _minimum_periodic_distances(
        circumcenters,
        reference_positions,
        box_dims,
        periodic,
        coordinate_tolerance,
    )

    clearances = distances - atom_radius
    valid_sites = clearances > coordinate_tolerance

    sites = circumcenters[valid_sites]
    clearances = clearances[valid_sites]

    if len(sites) == 0:
        raise GBManipulatorValueError(
            "No Delaunay insertion sites have sufficient atomic clearance."
        )

    weight_sum = np.sum(clearances)
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise GBManipulatorValueError(
            "Unable to construct Delaunay insertion-site probabilities."
        )

    probabilities = clearances / weight_sum
    return sites, probabilities


def _grid_insertion_sites(
    gb_positions: np.ndarray,
    reference_positions: np.ndarray,
    atom_radius: float,
    box_dims: np.ndarray,
    periodic: tuple[bool, bool, bool],
    coordinate_tolerance: float,
    spacing: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate insertion sites on a regular Cartesian grid.

    :param gb_positions: Positions defining the GB insertion region.
    :param reference_positions: Positions of all atoms used for clearance checking.
    :param atom_radius: Required minimum atom-site distance in angstroms.
    :param box_dims: Simulation-box bounds with shape ``(3, 2)``.
    :param periodic: Periodicity along x, y, and z.
    :param coordinate_tolerance: Coordinate tolerance in angstroms.
    :param spacing: Grid spacing in angstroms.
    :return: Candidate insertion sites and their normalized probabilities.
    :raises GBManipulatorValueError: If no valid insertion sites can be generated.
    """
    gb_positions = np.asarray(gb_positions, dtype=float)
    box_dims = np.asarray(box_dims, dtype=float)

    site_bounds = np.column_stack(
        (np.min(gb_positions, axis=0), np.max(gb_positions, axis=0))
    )
    for axis, is_periodic in enumerate(periodic):
        if is_periodic:
            site_bounds[axis] = box_dims[axis]

    axes = []

    for axis, is_periodic in enumerate(periodic):
        lower, upper = site_bounds[axis]
        width = upper - lower

        count = int(np.floor(width / spacing))
        values = lower + spacing * np.arange(count + 1, dtype=float)

        if is_periodic:
            values = values[values < upper - coordinate_tolerance]
        else:
            values = values[values <= upper + coordinate_tolerance]

        axes.append(values)

    if any(len(values) == 0 for values in axes):
        raise GBManipulatorValueError(
            "GB insertion region is too small for the configured grid spacing."
        )

    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    sites = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    distances = _minimum_periodic_distances(
        sites,
        reference_positions,
        box_dims,
        periodic,
        coordinate_tolerance,
    )

    valid_sites = distances >= atom_radius - coordinate_tolerance
    sites = sites[valid_sites]
    distances = distances[valid_sites]

    if len(sites) == 0:
        raise GBManipulatorValueError(
            "No grid insertion sites have sufficient atomic clearance."
        )

    weight_sum = np.sum(distances)
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise GBManipulatorValueError(
            "Unable to construct grid insertion-site probabilities."
        )

    probabilities = distances / weight_sum
    return sites, probabilities


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


def _stoichiometric_unit_size(ratio: dict[int, int]) -> int:
    """Return the number of atoms in one stoichiometric unit.

    :param ratio: Number of atoms of each type in one stoichiometric unit.
    :return: Total number of atoms in one stoichiometric unit.
    """
    return sum(ratio.values())


def _floor_stoichiometric_count(n_atoms: int, ratio: dict[int, int]) -> int:
    """Round an atom count down to one compatible with a stoichiometric ratio.

    :param n_atoms: Maximum number of atoms to modify.
    :param ratio: Number of atoms of each type in one stoichiometric unit.
    :return: Largest compatible atom count not greater than *n_atoms*.
    """
    unit_size = _stoichiometric_unit_size(ratio)
    return n_atoms - n_atoms % unit_size


def _get_stoichiometric_change(
    n_atoms: int,
    ratio: dict[int, int],
) -> dict[int, int]:
    """Return per-type counts for an exact stoichiometric atom change.

    :param n_atoms: Total number of atoms to modify.
    :param ratio: Number of atoms of each type in one stoichiometric unit.
    :return: Number of atoms of each type to modify.
    :raises GBManipulatorValueError: If ``n_atoms`` cannot preserve the requested
        stoichiometric ratio.
    """
    unit_size = _stoichiometric_unit_size(ratio)

    if n_atoms % unit_size != 0:
        raise GBManipulatorValueError(
            f"Cannot modify exactly {n_atoms} atoms while preserving the "
            f"stoichiometric ratio {ratio}. The atom count must be a multiple of "
            f"{unit_size}."
        )

    n_units = n_atoms // unit_size
    return {atom_type: count * n_units for atom_type, count in ratio.items()}


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
        parent = self.__parents[0]
        if parent.normal_topology is BoundaryNormalTopology.UNKNOWN:
            raise GBManipulatorValueError(
                "a parent candidate requires known boundary-normal topology"
            )
        labels = parent.grain_labels
        if labels is None:
            labels = self.__concatenated_labels(parent, len(parent.whole_system))
        self.__set_candidate_labels(labels, len(parent.whole_system))
        return parent._to_interface_candidate(parent.whole_system, labels)

    def translate_right_grain(
        self,
        dy: float,
        dz: float,
        *,
        dx: float = 0.0,
    ) -> np.ndarray:
        """Rigidly translate the right grain.

        :param dy: Displacement in y in angstroms.
        :param dz: Displacement in z in angstroms.
        :param dx: Keyword argument, optional, defaults to ``0.0``. Displacement in x in
            angstroms.
        :return: Left-grain rows followed by translated right-grain rows.
        :raises GBManipulatorValueError: If a displacement is invalid or moves atoms
            outside a supported interval.
        """
        dx = _validate_finite_real("dx", dx)
        dy = _validate_finite_real("dy", dy)
        dz = _validate_finite_real("dz", dz)

        if not self.__one_parent:
            warnings.warn(
                "grain translation only occurring based on parent 1",
                UserWarning,
                stacklevel=2,
            )

        parent = self.__parents[0]
        updated_right = np.array(parent.right_grain, copy=True)
        tolerance = float(parent.coordinate_tolerance)
        x_lower, x_upper = parent.right_grain_x_bounds

        translated_x = updated_right["x"] + dx
        # Explicit ownership is persistent state. A relaxed atom may legitimately cross
        # the nominal interface plane without changing grains, so a pure in-plane
        # registry translation must not reject that state just because a right-owned
        # atom currently lies outside the original physical x interval. Preserve the
        # existing x-translation safety check whenever x is actually displaced.
        if abs(dx) > tolerance and (
            np.any(translated_x < x_lower - tolerance)
            or np.any(translated_x >= x_upper)
        ):
            raise GBManipulatorValueError(
                "dx moves one or more right-grain atoms outside the supported "
                f"half-open x interval [{x_lower}, {x_upper})"
            )

        updated_right["x"] = translated_x
        updated_right = _translate_inplane(
            updated_right,
            dy=dy,
            dz=dz,
            box_dims=parent.box_dims,
            inplane_periodic=parent.inplane_periodic,
            tolerance=tolerance,
        )

        candidate = np.hstack((parent.left_grain, updated_right))
        labels = self.__concatenated_labels(parent, len(candidate))
        self.__set_candidate_labels(labels, len(candidate))
        return candidate

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

        atoms = self.translate_right_grain(dy, dz, dx=dx)
        labels = self.__concatenated_labels(parent, len(atoms))
        return parent._to_interface_candidate(atoms, labels)

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
        left_phase_shift = _validate_finite_real(
            "left_phase_shift",
            left_phase_shift,
        )
        right_phase_shift = _validate_finite_real(
            "right_phase_shift",
            right_phase_shift,
        )
        right_dy = _validate_finite_real("right_dy", right_dy)
        right_dz = _validate_finite_real("right_dz", right_dz)

        if not self.__one_parent:
            raise GBManipulatorValueError(
                "termination cycling requires exactly one parent"
            )

        parent = self.__parents[0]
        topology = parent.normal_topology
        if topology is BoundaryNormalTopology.UNKNOWN:
            raise GBManipulatorValueError(
                "termination cycling requires known boundary-normal topology"
            )

        tolerance = float(parent.coordinate_tolerance)
        box = np.asarray(parent.box_dims, dtype=float)
        if (
            box.shape != (3, 2)
            or not np.all(np.isfinite(box))
            or np.any(box[:, 0] >= box[:, 1])
        ):
            raise GBManipulatorValueError(
                "termination cycling requires finite strictly ordered box bounds"
            )

        left_bounds = np.asarray(parent.left_grain_x_bounds, dtype=float)
        right_bounds = np.asarray(parent.right_grain_x_bounds, dtype=float)
        plane = float(parent.gb_plane_x)

        if (
            left_bounds.shape != (2,)
            or right_bounds.shape != (2,)
            or not np.all(np.isfinite(left_bounds))
            or not np.all(np.isfinite(right_bounds))
            or left_bounds[0] >= left_bounds[1]
            or right_bounds[0] >= right_bounds[1]
            or not np.isclose(
                left_bounds[1],
                plane,
                atol=tolerance,
                rtol=0.0,
            )
            or not np.isclose(
                right_bounds[0],
                plane,
                atol=tolerance,
                rtol=0.0,
            )
        ):
            raise GBManipulatorValueError(
                "termination cycling requires contiguous valid physical grain bounds"
            )

        xlo = float(box[0, 0])
        xhi = float(box[0, 1])

        if topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL:
            if (
                not np.isclose(
                    left_bounds[0],
                    xlo,
                    atol=tolerance,
                    rtol=0.0,
                )
                or not np.isclose(
                    right_bounds[1],
                    xhi,
                    atol=tolerance,
                    rtol=0.0,
                )
            ):
                raise GBManipulatorValueError(
                    "seriodic termination cycling requires zero-vacuum grain bounds"
                )
        elif topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB:
            left_vacuum = float(left_bounds[0] - xlo)
            right_vacuum = float(xhi - right_bounds[1])

            if left_vacuum < -tolerance or right_vacuum < -tolerance:
                raise GBManipulatorValueError(
                    "slab grain bounds must lie inside the x box"
                )
            if left_vacuum <= tolerance and right_vacuum <= tolerance:
                raise GBManipulatorValueError(
                    "slab termination cycling requires a free-surface or vacuum "
                    "interval"
                )

        left_x = np.asarray(parent.left_grain["x"], dtype=float)
        right_x = np.asarray(parent.right_grain["x"], dtype=float)

        if (
            not np.all(np.isfinite(left_x))
            or not np.all(np.isfinite(right_x))
            or np.any(left_x < left_bounds[0] - tolerance)
            or np.any(left_x >= left_bounds[1] + tolerance)
            or np.any(right_x < right_bounds[0] - tolerance)
            or np.any(right_x >= right_bounds[1] + tolerance)
        ):
            raise GBManipulatorValueError(
                "Parent atoms do not lie inside their physical grain bounds"
            )

        updated_left = np.array(parent.left_grain, copy=True)
        updated_right = np.array(parent.right_grain, copy=True)

        updated_left["x"] = _cycle_half_open(
            updated_left["x"],
            lower=float(left_bounds[0]),
            upper=float(left_bounds[1]),
            shift=left_phase_shift,
            tolerance=tolerance,
        )
        updated_right["x"] = _cycle_half_open(
            updated_right["x"],
            lower=float(right_bounds[0]),
            upper=float(right_bounds[1]),
            shift=right_phase_shift,
            tolerance=tolerance,
        )

        updated_right = _translate_inplane(
            updated_right,
            dy=right_dy,
            dz=right_dz,
            box_dims=box,
            inplane_periodic=parent.inplane_periodic,
            tolerance=tolerance,
        )

        return np.hstack((updated_left, updated_right))

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
        if not isinstance(candidate, InterfaceCandidate):
            raise GBManipulatorValueError("candidate must be an InterfaceCandidate")
        separation = _validate_finite_real(
            "interface_separation", interface_separation
        )
        if separation < 0.0:
            raise GBManipulatorValueError("interface_separation must be nonnegative")
        parent = self.__parents[0]
        tolerance = candidate.coordinate_tolerance
        if not np.isclose(
            candidate.interface_separation, 0.0, atol=tolerance, rtol=0.0
        ):
            raise GBManipulatorValueError(
                "interface separation cannot be reapplied to a separated candidate"
            )
        if candidate.normal_topology is not parent.normal_topology:
            raise GBManipulatorValueError(
                "candidate topology does not match the manipulator parent"
            )
        if candidate.normal_topology is BoundaryNormalTopology.UNKNOWN:
            raise GBManipulatorValueError(
                "interface separation requires known boundary-normal topology."
            )
        if candidate.inplane_periodic != parent.inplane_periodic:
            raise GBManipulatorValueError(
                "candidate in-plane periodicity does not match the parent"
            )
        if not np.allclose(
            candidate.box_dims, parent.box_dims, atol=tolerance, rtol=0.0
        ) or not np.isclose(
            candidate.gb_plane_x, parent.gb_plane_x, atol=tolerance, rtol=0.0
        ):
            raise GBManipulatorValueError(
                "interface separation requires fixed-cell geometry from this parent"
            )
        if not np.allclose(
            candidate.left_grain_x_bounds,
            parent.left_grain_x_bounds,
            atol=tolerance,
            rtol=0.0,
        ) or not np.allclose(
            candidate.right_grain_x_bounds,
            parent.right_grain_x_bounds,
            atol=tolerance,
            rtol=0.0,
        ):
            raise GBManipulatorValueError(
                "candidate physical grain bounds do not match the parent"
            )

        geometry = _separated_interface_geometry(
            box_dims=candidate.box_dims,
            gb_plane_x=candidate.gb_plane_x,
            left_grain_x_bounds=candidate.left_grain_x_bounds,
            right_grain_x_bounds=candidate.right_grain_x_bounds,
            interface_separation=separation,
            normal_topology=candidate.normal_topology,
            coordinate_tolerance=tolerance,
        )
        atoms = candidate.atoms
        labels = candidate.grain_labels
        shifted = np.array(atoms, copy=True)
        shifted["x"][labels == RIGHT_GRAIN_LABEL] += separation
        return InterfaceCandidate(
            atoms=shifted,
            box_dims=geometry.box_dims,
            gb_plane_x=geometry.gb_plane_x,
            left_grain_x_bounds=geometry.left_grain_x_bounds,
            right_grain_x_bounds=geometry.right_grain_x_bounds,
            grain_labels=labels,
            inplane_periodic=candidate.inplane_periodic,
            normal_topology=candidate.normal_topology,
            coordinate_tolerance=tolerance,
            interface_separation=separation,
        )

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
        """Remove atoms from the grain boundary region.

        Atom-removal probabilities are based on the local order parameter method of
        Lyakhov *et al.*, Computer Phys. Comm. 181 (2010) 1623-1632.

        Exactly one of ``gb_fraction`` and ``num_to_remove`` should be specified. Both
        parameters describe the total number of atoms removed, not the number of
        stoichiometric formula units.

        When ``keep_ratio`` is true, an explicit ``num_to_remove`` must be compatible
        with the unit-cell stoichiometric ratio. A count calculated from ``gb_fraction``
        is rounded down to the nearest compatible atom count.

        :param gb_fraction: Keyword argument. Fraction of atoms in the GB region to
            remove. Must satisfy ``0 < gb_fraction <= 0.25``.
        :param num_to_remove: Keyword argument. Total number of atoms to remove. Must be
            at least one and no greater than 25% of the atoms in the GB region. When
            ``keep_ratio`` is ``True``, this count must permit exact preservation of the
            unit cell stoichiometry.
        :param keep_ratio: Keyword argument, defaults to ``True``. Whether to preserve
            the unit cell stoichiometric ratio.
        :param return_positions: Keyword argument, optional, defaults to False. Whether
            to return the removed atoms together with the modified structure.
        :return: Atom positions after atom removal, optionally together with the removed
            atoms.
        """
        if (gb_fraction is None) == (num_to_remove is None):
            raise GBManipulatorValueError(
                "Exactly one of gb_fraction and num_to_remove must be specified."
            )

        if not self.__one_parent:
            warnings.warn("Atom removal only occurring based on parent 1.")

        parent = self.__parents[0]

        atoms = Atom.as_array(parent.whole_system, type_map=parent.unit_cell.type_map)
        gb_atoms = Atom.as_array(parent.gb_atoms, type_map=parent.unit_cell.type_map)
        gb_atom_indices = np.asarray(parent.gb_indices, dtype=np.intp)
        type_map = parent.unit_cell.type_map
        positions = atoms[:, 1:]

        if gb_fraction is not None and (gb_fraction <= 0 or gb_fraction > 0.25):
            raise GBManipulatorValueError(
                f"Invalid value for gb_fraction ({gb_fraction=}). Must be 0 < "
                "gb_fraction <= 0.25"
            )

        max_atoms_to_remove = int(0.25 * len(gb_atoms))

        if num_to_remove is not None and (
            num_to_remove < 1 or num_to_remove > max_atoms_to_remove
        ):
            raise GBManipulatorValueError(
                "Invalid num_to_remove value. Must be >= 1, and no greater than 25% of "
                "the total number of atoms in the GB region."
            )

        fractional_request = num_to_remove is None

        if fractional_request:
            num_to_remove = int(gb_fraction * len(gb_atoms))

            if keep_ratio and len(type_map) > 1:
                requested_count = num_to_remove
                num_to_remove = _floor_stoichiometric_count(
                    requested_count,
                    parent.unit_cell.ratio,
                )

                if num_to_remove != requested_count and num_to_remove > 0:
                    warnings.warn(
                        f"gb_fraction corresponds to {requested_count} atoms, but "
                        "preserving stoichiometry requires a multiple of "
                        f"{_stoichiometric_unit_size(parent.unit_cell.ratio)}. "
                        f"Removing {num_to_remove} atoms instead.",
                        stacklevel=2,
                    )

        if num_to_remove == 0:
            warnings.warn(
                "The requested removal corresponds to zero atoms after applying "
                "stoichiometric constraints.",
                stacklevel=2,
            )
            self.__set_candidate_labels(
                getattr(parent, "grain_labels", None),
                len(parent.whole_system),
            )
            return atoms

        if len(type_map) == 1:
            num_to_remove_dict = {1: num_to_remove}

        elif keep_ratio:
            num_to_remove_dict = _get_stoichiometric_change(
                num_to_remove,
                parent.unit_cell.ratio,
            )
            central_type = min(num_to_remove_dict, key=num_to_remove_dict.get)

            # The Lyakhov fingerprint uses the complete radial neighborhood out to Rmax.
            # This neighborhood is deliberately distinct from the short-range chemical
            # partner neighborhood below.
            Delta = 0.05
            Rmax = 15.0

            local_order_neighborhoods = (
                _create_periodic_image_neighborhoods(
                    Rmax,
                    atoms,
                    gb_atom_indices,
                    parent.box_dims,
                    periodic=(
                        False,
                        *parent.inplane_periodic,
                    ),
                )
            )

            args_list = [
                (
                    atoms[atom_idx],
                    local_order_neighborhoods[idx],
                    parent.unit_cell.names(asint=True),
                    parent.unit_cell.a0,
                    len(parent.unit_cell.unit_cell),
                    Delta,
                    Rmax,
                )
                for idx, atom_idx in enumerate(gb_atom_indices)
            ]

            order = np.empty(len(args_list))

            for i, args in enumerate(args_list):
                order[i] = (_calculate_local_order(*args))

            # Higher-order environments should be more resistant to removal, while
            # retaining a small probability of selection.
            probabilities = (max(order) - order + min(order))
            probabilities = probabilities / np.sum(probabilities, dtype=float)

        else:
            # If we aren't worried about keeping the ratio, randomly assign atoms to be
            # removed to each type, summing up to num_to_remove.
            breaks = np.sort(
                self.__rng.choice(
                    range(1, num_to_remove),
                    len(type_map) - 1,
                    replace=False,
                )
            )

            breaks = np.concatenate(([0], breaks, [num_to_remove]))
            values = np.diff(breaks)
            num_to_remove_dict = {i + 1: int(values[i]) for i in range(len(type_map))}

        if keep_ratio and len(type_map) > 1:
            type_mask = atoms[gb_atom_indices][:, 0] == central_type

            central_indices = gb_atom_indices[type_mask]

            if len(central_indices) == 0:
                raise GBManipulatorValueError(
                    f"No atoms found for type {central_type} in the grain boundary."
                )

            central_probabilities = probabilities[type_mask]
            central_probabilities /= np.sum(central_probabilities)
            central_num_to_remove = num_to_remove_dict[central_type]

            selected_central_indices = (
                self.__rng.choice(
                    central_indices,
                    central_num_to_remove,
                    replace=False,
                    p=central_probabilities,
                )
            )

            # This neighborhood is used only to choose atoms participating in a
            # stoichiometrically coupled removal. Each physical atom appears once at its
            # minimum-image distance.
            removal_cutoff = (
                parent.unit_cell.nn_distance(2)
                + parent.unit_cell.nn_distance(1)
            ) / 2

            (
                removal_neighbor_indices,
                removal_neighbor_distances,
            ) = _create_periodic_neighbor_list(
                removal_cutoff,
                positions,
                selected_central_indices,
                parent.box_dims,
                periodic=(
                    False,
                    *parent.inplane_periodic,
                ),
            )

            indices_to_remove = list(
                np.asarray(selected_central_indices, dtype=np.intp)
            )

            for atom_type, ratio in parent.unit_cell.ratio.items():
                if atom_type == central_type:
                    continue

                for (
                    neighbor_indices,
                    neighbor_distances,
                ) in zip(
                    removal_neighbor_indices,
                    removal_neighbor_distances,
                    strict=True,
                ):
                    gb_mask = np.isin(neighbor_indices, gb_atom_indices)

                    candidate_indices = neighbor_indices[gb_mask]
                    candidate_distances = neighbor_distances[gb_mask]

                    atom_type_mask = (atoms[candidate_indices][:, 0] == atom_type)
                    candidate_indices = (candidate_indices[atom_type_mask])
                    candidate_distances = candidate_distances[atom_type_mask]

                    available_mask = ~np.isin(candidate_indices, indices_to_remove)
                    type_indices = candidate_indices[available_mask]
                    type_distances = candidate_distances[available_mask]

                    if len(type_indices) < ratio:
                        raise GBManipulatorValueError(
                            f"Not enough neighbor atoms of type {atom_type} to remove."
                        )

                    # Distances below this tolerance would indicate overlapping atoms.
                    type_distances = np.maximum(type_distances, 1e-8)
                    type_probabilities = 1.0 / type_distances
                    type_probabilities /= np.sum(type_probabilities)
                    type_idx_to_remove = (
                        self.__rng.choice(
                            type_indices,
                            ratio,
                            replace=False,
                            p=type_probabilities,
                        )
                    )

                    indices_to_remove.extend(type_idx_to_remove)

        else:
            indices_to_remove = []

            for atom_type, num in (num_to_remove_dict.items()):
                type_indices = (
                    gb_atom_indices[atoms[gb_atom_indices][:, 0] == atom_type]
                )

                type_idx_to_remove = (
                    self.__rng.choice(
                        type_indices,
                        num,
                        replace=False,
                    )
                )

                indices_to_remove.extend(type_idx_to_remove)

        if len(indices_to_remove) != num_to_remove:
            raise GBManipulatorValueError("")

        pos = np.delete(parent.whole_system, indices_to_remove, axis=0)

        labels = getattr(parent, "grain_labels", None)
        retained_labels = (
            None
            if labels is None
            else np.delete(
                labels,
                indices_to_remove,
                axis=0,
            )
        )

        self.__set_candidate_labels(
            retained_labels,
            len(pos),
        )

        if return_positions:
            return (
                pos,
                parent.whole_system[
                    indices_to_remove
                ],
            )

        return pos

    def insert_atoms(
        self,
        *,
        fill_fraction: float | None = None,
        num_to_insert: int | None = None,
        method: str = "delaunay",
        keep_ratio: bool = True,
        return_positions: bool = False,
    ) -> np.ndarray:
        """Insert atoms at available sites in the grain boundary region.

        Candidate insertion sites are generated using Delaunay tetrahedral circumcenters
        or a regular Cartesian grid.

        Exactly one of ``fill_fraction`` and ``num_to_insert`` should be specified.
        ``num_to_insert`` always refers to the total number of atoms inserted.
        ``fill_fraction`` refers to the fraction of valid generated insertion sites to
        occupy. When ``keep_ratio`` is true, the resulting atom count is rounded down to
        the nearest count compatible with the unit cell stoichiometric ratio.

        :param fill_fraction: Keyword argument, defaults to ``None``. Fraction of valid
            generated insertion sites to occupy, subject to the maximum insertion limit
            of 25% of the original GB atom population. Must satisfy ``0 < fill_fraction
            <= 0.25``.
        :param num_to_insert: Keyword argument, defaults to ``None``. Total number of
            atoms to insert. Must be at least one and no greater than 25% of the atoms
            in the original GB region. When ``keep_ratio`` is ``True``, this count must
            permit exact preservation of the unit cell stoichiometry.
        :param method: Keyword argument, optional, defaults to "delaunay". Method used
            to generate candidate sites. Must be either "delaunay" or "grid."
        :param keep_ratio: Keyword argument, optional, defaults to True. Whether to
            preserve the unit cell stoichiometric ratio.
        :param return_positions: Keyword argument, optional, defaults to False. Whether
            to return the inserted atoms together with the modified structure.
        :raises GBManipulatorValueError: Exception raised if an invalid method is
            specified.
        :return: Atom positions after atom insertion, optionally together with the
            inserted atoms.
        """
        if (fill_fraction is None) == (num_to_insert is None):
            raise GBManipulatorValueError(
                "Exactly one of fill_fraction and num_to_insert must be specified."
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

        max_atoms_to_insert = int(0.25 * len(gb_atoms))

        if num_to_insert is not None and (
            num_to_insert < 1 or num_to_insert > max_atoms_to_insert
        ):
            raise GBManipulatorValueError(
                "Invalid num_to_insert value. Must be >= 1, and no greater than 25% of "
                "the atoms in the GB region.")

        if num_to_insert is not None and keep_ratio and len(type_map) > 1:
            # Validate before doing the potentially expensive site generation.
            _get_stoichiometric_change(num_to_insert, parent.unit_cell.ratio)

        periodic = (False, *parent.inplane_periodic)

        # Calculate the insertion sites using the specified approach.
        if method == "delaunay":
            possible_sites, probabilities = _delaunay_insertion_sites(
                gb_atoms[:, 1:],
                atoms[:, 1:],
                parent.unit_cell.radius,
                parent.box_dims,
                periodic,
                parent.coordinate_tolerance,
            )
        elif method == "grid":
            possible_sites, probabilities = _grid_insertion_sites(
                gb_atoms[:, 1:],
                atoms[:, 1:],
                parent.unit_cell.radius,
                parent.box_dims,
                periodic,
                parent.coordinate_tolerance,
            )
        else:
            raise GBManipulatorValueError(f"Unrecognized insert_atoms method: {method}")

        fractional_request = num_to_insert is None

        if fractional_request:
            requested_count = int(fill_fraction * len(possible_sites))
            num_to_insert = min(requested_count, max_atoms_to_insert)

            if num_to_insert != requested_count and num_to_insert > 0:
                warnings.warn(
                    f"fill_fraction corresponds to {requested_count} insertion sites, but "
                    f"the 25% GB-region limit permits at most {max_atoms_to_insert} atoms. "
                    f"Inserting {num_to_insert} atoms instead.",
                    stacklevel=2,
                )

            if keep_ratio and len(type_map) > 1:
                pre_stoichiometry_count = num_to_insert
                num_to_insert = _floor_stoichiometric_count(
                    requested_count,
                    parent.unit_cell.ratio
                )

                if num_to_insert != pre_stoichiometry_count and num_to_insert > 0:
                    warnings.warn(
                        f"Inserting {pre_stoichiometry_count} atoms would not preserve "
                        "stoichiometry, which requires a multiple of "
                        f"{_stoichiometry_unit_size(parent.unit_cell.ratio)}. Inserting "
                        f"{num_to_insert} atoms instead.",
                        stacklavel=2
                    )

        if num_to_insert == 0:
            warnings.warn(
                (
                    "The requested insertion corresponds to zero atoms after applying "
                    "fraction and stoichiometric constraints."
                ),
                stacklevel=2,
            )
            self.__set_candidate_labels(
                getattr(parent, "grain_labels", None),
                len(parent.whole_system),
            )
            return atoms

        if num_to_insert > max_atoms_to_insert:
            raise GBManipulatorValueError(
                "Calculated insertion count exceeds 25% of the atoms in the GB region."
            )

        if num_to_insert > len(possible_sites):
            raise GBManipulatorValueError(
                "Not enough valid insertion sites for the requested atom count."
            )

        if len(type_map) == 1:
            num_to_insert_dict = {1: num_to_insert}
        elif keep_ratio:
            num_to_insert_dict = _get_stoichiometric_change(
                num_to_insert,
                parent.unit_cell.ratio
            )
            central_type = min(num_to_insert_dict, key=num_to_insert_dict.get)
        else:  # random insertion of random types
            breaks = np.sort(
                self.__rng.choice(
                    range(1, num_to_insert),
                    len(type_map) - 1,
                    replace=False
                )
            )
            breaks = np.concatenate([[0], breaks, [num_to_insert]])
            values = np.diff(breaks)
            num_to_insert_dict = {
                i + 1: int(values[i])
                for i in range(len(type_map))
            }

        if keep_ratio and len(type_map) > 1:
            central_num_to_insert = num_to_insert_dict[central_type]

            selected_central_indices = (
                self.__rng.choice(
                    np.arange(len(possible_sites), dtype=np.intp),
                    central_num_to_insert,
                    replace=False,
                    p=probabilities,
                )
            )

            cutoff = (
                parent.unit_cell.nn_distance(2)
                + parent.unit_cell.nn_distance(1)
            ) / 2.0

            possible_sites_neighbor_list, _ = _create_periodic_neighbor_list(
                cutoff,
                possible_sites,
                selected_central_indices,
                parent.box_dims,
                periodic=(False, *parent.inplane_periodic),
            )

            atoms_to_add = {
                type_map[i]: (
                    []
                    if type_map[i] != central_type
                    else list(np.asarray(selected_central_indices, dtype=np.intp))
                )
                for i in type_map.keys()
            }

            for atom_type, ratio in parent.unit_cell.ratio.items():
                if atom_type == central_type:
                    continue

                for neighbors in possible_sites_neighbor_list:
                    already_assigned = {
                        idx
                        for values in atoms_to_add.values()
                        for idx in values
                    }

                    # Keep the helper's deterministic neighbor ordering while excluding
                    # sites already assigned to another inserted atom.
                    available_neighbors = np.asarray(
                        [idx for idx in neighbors if idx not in already_assigned],
                        dtype=np.intp,
                    )

                    if len(available_neighbors) < ratio:
                        raise GBManipulatorValueError(
                            "Not enough sites to insert atoms into."
                        )

                    partial_probabilities = probabilities[available_neighbors]
                    partial_probabilities /= np.sum(partial_probabilities)

                    selected_indices = (
                        self.__rng.choice(
                            available_neighbors,
                            ratio,
                            replace=False,
                            p=partial_probabilities,
                        )
                    )

                    atoms_to_add[atom_type].extend(selected_indices)

        else:
            atoms_to_add = {}

            site_indices = list(range(len(possible_sites)))

            for atom_type, num in (num_to_insert_dict.items()):
                available_indices = list(
                    set(site_indices)
                    - set(np.array(atoms_to_add.values()).flatten())
                )

                type_idx_to_insert = (
                    self.__rng.choice(
                        available_indices,
                        num,
                        replace=False,
                        p=probabilities,
                    )
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
