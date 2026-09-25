# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Two-parent formula-preserving slice-and-merge crossover as a built-in operation.

``SliceAndMerge`` slices each parent along an x-normal or gently tilted periodic-wave
surface and concatenates the low-x side of the first parent with the high-x side of
the second, subject to explicit preflight compatibility validation (matching ownership
mode, boundary topology, affine-equivalent physical grain geometry, and identical
normalized unit-cell formula) and a formula-preserving cut-offset search.

Like ``GBOpt.manipulation.density``/``soft_mode``, ``InterfaceCandidate`` carries no
unit-cell information, so this operation requires ``unit_cell`` (a two-element
sequence, one per parent in parent order) and ``gb_thickness`` (a single value, applied
only to the first parent) as explicit ``context.params``.

The actual crossover math (``crossover_slice_and_merge``) takes duck-typed
parent-geometry objects -- anything exposing ``whole_system``/``box_dims``/
``gb_plane_x``/``gb_thickness``/``left_grain_x_bounds``/``right_grain_x_bounds``/
``grain_labels``/``inplane_periodic``/``normal_topology``/``coordinate_tolerance``/
``unit_cell`` -- rather than an ``InterfaceCandidate``, and its ``rng`` parameter is
likewise duck-typed (only ``.random()`` is called). This lets a caller supply a
possibly-``None`` ``grain_labels`` for a parent without persistent ownership, and a
possibly-non-Generator random source, directly, with no adapter and no
``ManipulationContext``/``InterfaceCandidate`` construction -- both of which
unconditionally require real ``InterfaceCandidate``/``np.random.Generator`` values.
``SliceAndMerge.execute`` instead wraps each ``context.parents[i]`` (always an
``InterfaceCandidate``, so ``grain_labels`` is always a real array) in a small local
adapter alongside its ``unit_cell``, and always receives a real ``context.rng``.
"""

from __future__ import annotations

from numbers import Real
from typing import Protocol

import numpy as np

from GBOpt._candidate_admissibility import (
    CandidateAdmissibilityError,
    composition_delta_is_formula_multiple,
    validate_formula_composition,
)
from GBOpt.interface import InterfaceCandidate
from GBOpt.interface.types import (
    InterfaceCandidateTypeError,
    InterfaceCandidateValueError,
)
from GBOpt.manipulation.types import (
    ManipulationArityError,
    ManipulationCapabilityError,
    ManipulationCompatibilityError,
    ManipulationConfigurationError,
    ManipulationContext,
    ManipulationResult,
)


class CrossoverParent(Protocol):
    """Structural contract for one parent side of a slice-and-merge crossover.

    Satisfied by ``GBManipulator.Parent`` as-is (every attribute below is already one
    of its properties) and by this module's own ``_CandidateCrossoverParent`` adapter
    around an ``InterfaceCandidate``.
    """

    @property
    def whole_system(self) -> np.ndarray: ...

    @property
    def box_dims(self) -> np.ndarray: ...

    @property
    def gb_plane_x(self) -> float: ...

    @property
    def gb_thickness(self) -> float: ...

    @property
    def left_grain_x_bounds(self) -> np.ndarray: ...

    @property
    def right_grain_x_bounds(self) -> np.ndarray: ...

    @property
    def grain_labels(self) -> np.ndarray | None: ...

    @property
    def inplane_periodic(self) -> tuple[bool, bool]: ...

    @property
    def normal_topology(self) -> object: ...

    @property
    def coordinate_tolerance(self) -> float: ...

    @property
    def unit_cell(self) -> object: ...


def _validate_finite_real(name: str, value: object) -> float:
    """Return ``value`` as a finite float.

    :param name: Input name used in validation messages.
    :param value: Candidate finite real scalar.
    :return: Validated Python ``float``.
    :raises TypeError: If ``value`` is Boolean or non-real.
    :raises ManipulationConfigurationError: If ``value`` is non-finite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real value.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ManipulationConfigurationError(f"{name} must be a finite real value.")
    return normalized


def _require(params, key):
    """Return ``params[key]``, raising ``ManipulationConfigurationError`` if absent."""
    if key not in params:
        raise ManipulationConfigurationError(f"{key} is a required parameter")
    return params[key]


def _remap_axis_values(
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


def _rescale_atoms(
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
        rescaled[axis_name] = _remap_axis_values(
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
    rng,
) -> float:
    """Sample uniformly over the union of positive-width intervals.

    :param intervals: Positive-width ordered intervals.
    :param rng: Duck-typed random source exposing ``.random()``.
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


def crossover_slice_and_merge(
    first: CrossoverParent,
    second: CrossoverParent,
    *,
    surface_mode: str,
    max_tilt_degrees: float,
    rng,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    """Return a formula-preserving child sliced from two parents.

    This is the pure computational core shared by ``SliceAndMerge.execute`` and
    ``GBManipulator.slice_and_merge``, taking duck-typed parent-geometry objects (see
    ``CrossoverParent``) and returning plain data rather than an ``InterfaceCandidate``,
    deliberately never constructing one: a parent without persistent grain ownership
    (``grain_labels is None``) has no labels to preserve, and this function's own
    preflight topology/geometry-equivalence checks below are correspondingly skipped for
    that case -- only composition (formula) compatibility is still enforced
    unconditionally. ``SliceAndMerge.execute`` always supplies parents with real
    ``grain_labels`` (via ``InterfaceCandidate``), so those checks always run there.

    :param first: First parent (child atoms with a crossover offset below the sampled
        cut are drawn from here).
    :param second: Second parent (child atoms at or above the sampled cut are drawn
        from here, remapped into ``first``'s box when its box differs).
    :param surface_mode: Keyword argument, required. ``"normal_plane"`` uses a plane
        parallel to yz; ``"periodic_wave"`` uses a smooth sinusoidal surface, continuous
        across the y/z periodic boundaries, whose combined maximum local tilt is bounded
        by ``max_tilt_degrees``.
    :param max_tilt_degrees: Keyword argument, required. Maximum combined local tilt for
        ``"periodic_wave"``, in degrees; must satisfy ``0 <= value < 90``.
    :param rng: Keyword argument, required. Duck-typed random source exposing
        ``.random()`` only.
    :return: A 3-tuple of the child's structured atom rows, its aligned grain labels (or
        ``None`` when neither parent has persistent ownership), and a mapping of the
        concrete crossover parameters used.
    :raises TypeError: If ``max_tilt_degrees`` is Boolean or non-real.
    :raises ManipulationConfigurationError: If ``surface_mode`` is not
        ``"normal_plane"``/``"periodic_wave"``, or ``max_tilt_degrees`` is out of range.
    :raises ManipulationCompatibilityError: If the parents use different ownership
        modes, mismatched boundary topology, or non-affine-equivalent physical grain
        geometry.
    :raises ManipulationCapabilityError: If the parents' unit cells use different
        normalized formula vectors, either parent's atoms are not an exact formula
        multiple, or no positive-width formula-preserving crossover interval exists.
    """
    if surface_mode not in {"normal_plane", "periodic_wave"}:
        raise ManipulationConfigurationError(
            "surface_mode must be 'normal_plane' or 'periodic_wave'"
        )
    tilt = _validate_finite_real("max_tilt_degrees", max_tilt_degrees)
    if tilt < 0.0 or tilt >= 90.0:
        raise ManipulationConfigurationError(
            "max_tilt_degrees must satisfy 0 <= value < 90"
        )

    labels1 = first.grain_labels
    labels2 = second.grain_labels
    if (labels1 is None) != (labels2 is None):
        raise ManipulationCompatibilityError(
            "slice_and_merge requires both parents to use the same ownership mode"
        )
    if labels1 is not None:
        tolerance = max(first.coordinate_tolerance, second.coordinate_tolerance)
        if (
            first.inplane_periodic != second.inplane_periodic
            or first.normal_topology is not second.normal_topology
        ):
            raise ManipulationCompatibilityError(
                "owned crossover requires matching boundary topology"
            )
        mapped_plane = _remap_axis_values(
            second.gb_plane_x,
            second.box_dims[0],
            first.box_dims[0],
        )
        mapped_left_bounds = _remap_axis_values(
            second.left_grain_x_bounds,
            second.box_dims[0],
            first.box_dims[0],
        )
        mapped_right_bounds = _remap_axis_values(
            second.right_grain_x_bounds,
            second.box_dims[0],
            first.box_dims[0],
        )
        if (
            not np.isclose(
                first.gb_plane_x,
                mapped_plane,
                atol=tolerance,
                rtol=0.0,
            )
            or not np.allclose(
                first.left_grain_x_bounds,
                mapped_left_bounds,
                atol=tolerance,
                rtol=0.0,
            )
            or not np.allclose(
                first.right_grain_x_bounds,
                mapped_right_bounds,
                atol=tolerance,
                rtol=0.0,
            )
        ):
            raise ManipulationCompatibilityError(
                "owned crossover requires affine-equivalent physical grain "
                "geometry"
            )
    pos1 = first.whole_system
    pos2 = second.whole_system
    if labels1 is not None and not np.allclose(
        first.box_dims,
        second.box_dims,
        atol=tolerance,
        rtol=0.0,
    ):
        pos2 = _rescale_atoms(
            pos2,
            second.box_dims,
            first.box_dims,
        )

    try:
        validate_formula_composition(pos1, first.unit_cell)
        validate_formula_composition(pos2, second.unit_cell)
    except CandidateAdmissibilityError as exc:
        raise ManipulationCapabilityError(str(exc)) from exc
    first_formula = first.unit_cell.formula_ratio
    second_formula = second.unit_cell.formula_ratio
    if first_formula != second_formula:
        raise ManipulationCapabilityError(
            "crossover parents use different normalized formula vectors"
        )

    amplitude_y = 0.0
    amplitude_z = 0.0
    phase_y = 0.0
    phase_z = 0.0
    if surface_mode == "periodic_wave" and tilt > 0.0:
        maximum_slope = np.tan(np.deg2rad(tilt))
        slope_radius = maximum_slope * np.sqrt(float(rng.random()))
        slope_angle = 2.0 * np.pi * float(rng.random())
        slope_y = slope_radius * np.cos(slope_angle)
        slope_z = slope_radius * np.sin(slope_angle)
        y_length = float(np.ptp(first.box_dims[1]))
        z_length = float(np.ptp(first.box_dims[2]))
        amplitude_y = slope_y * y_length / (2.0 * np.pi)
        amplitude_z = slope_z * z_length / (2.0 * np.pi)
        phase_y = 2.0 * np.pi * float(rng.random())
        phase_z = 2.0 * np.pi * float(rng.random())

    first_coordinates = _crossover_scalar_coordinates(
        pos1,
        first.box_dims,
        amplitude_y=amplitude_y,
        amplitude_z=amplitude_z,
        phase_y=phase_y,
        phase_z=phase_z,
    )
    second_coordinates = _crossover_scalar_coordinates(
        pos2,
        first.box_dims,
        amplitude_y=amplitude_y,
        amplitude_z=amplitude_z,
        phase_y=phase_y,
        phase_z=phase_z,
    )
    half_window = 0.25 * first.gb_thickness
    maximum_excursion = abs(amplitude_y) + abs(amplitude_z)
    lower = first.gb_plane_x - half_window + maximum_excursion
    upper = first.gb_plane_x + half_window - maximum_excursion
    if lower >= upper:
        raise ManipulationCapabilityError(
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
        raise ManipulationCapabilityError(
            "no positive-width formula-preserving crossover interval exists"
        )
    slice_pos = _sample_interval_by_width(intervals, rng)
    mask1 = first_coordinates < slice_pos
    mask2 = second_coordinates >= slice_pos
    new_positions = np.hstack((pos1[mask1], pos2[mask2]))
    if labels1 is None:
        child_labels = None
    else:
        child_labels = np.hstack((labels1[mask1], labels2[mask2]))

    try:
        validate_formula_composition(new_positions, first.unit_cell)
    except CandidateAdmissibilityError as exc:
        raise ManipulationCapabilityError(
            f"internal crossover composition invariant failed: {exc}"
        ) from exc

    provenance = {
        "surface_mode": surface_mode,
        "max_tilt_degrees": tilt,
        "amplitude_y": float(amplitude_y),
        "amplitude_z": float(amplitude_z),
        "phase_y": float(phase_y),
        "phase_z": float(phase_z),
        "offset": float(slice_pos),
    }
    return new_positions, child_labels, provenance


class _CandidateCrossoverParent:
    """Adapt an ``InterfaceCandidate`` and its extra scalars to ``CrossoverParent``.

    ``InterfaceCandidate`` always carries real ``grain_labels`` (never ``None``), so a
    ``SliceAndMerge`` child is always fully labeled, regardless of whether the
    manipulator that produced its parent candidates tracks persistent grain ownership.
    """

    __slots__ = ("_candidate", "_gb_thickness", "_unit_cell")

    def __init__(
        self, candidate: InterfaceCandidate, unit_cell: object, gb_thickness: float
    ) -> None:
        self._candidate = candidate
        self._unit_cell = unit_cell
        self._gb_thickness = gb_thickness

    @property
    def whole_system(self) -> np.ndarray:
        return self._candidate.atoms

    @property
    def box_dims(self) -> np.ndarray:
        return self._candidate.box_dims

    @property
    def gb_plane_x(self) -> float:
        return self._candidate.gb_plane_x

    @property
    def gb_thickness(self) -> float:
        return self._gb_thickness

    @property
    def left_grain_x_bounds(self) -> np.ndarray:
        return self._candidate.left_grain_x_bounds

    @property
    def right_grain_x_bounds(self) -> np.ndarray:
        return self._candidate.right_grain_x_bounds

    @property
    def grain_labels(self) -> np.ndarray:
        return self._candidate.grain_labels

    @property
    def inplane_periodic(self) -> tuple[bool, bool]:
        return self._candidate.inplane_periodic

    @property
    def normal_topology(self) -> object:
        return self._candidate.normal_topology

    @property
    def coordinate_tolerance(self) -> float:
        return self._candidate.coordinate_tolerance

    @property
    def unit_cell(self) -> object:
        return self._unit_cell


class SliceAndMerge:
    """Slice two parents along a crossover surface and merge them into one child."""

    @property
    def name(self) -> str:
        """Stable operation name used for registry lookup and lineage metadata."""
        return "slice_and_merge"

    @property
    def arity(self) -> int:
        """Number of parent candidates this operation requires."""
        return 2

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        """Slice and merge ``context.parents[0]``/``context.parents[1]``.

        Unlike every other built-in operation in this package, this method validates
        its own arity explicitly (``len(context.parents) == 2``) rather than trusting
        the caller, because ``SliceAndMerge`` is reachable through
        ``GBOpt.manipulation``'s public surface (e.g. a hand-built
        ``ManipulationContext``) independently of ``GBManipulator.apply()``'s generic
        arity check, and a two-parent crossover run against the wrong parent count has
        no other sensible failure mode to fall back on.

        :param context: Validated input whose ``params`` supply ``unit_cell`` (a
            two-element sequence, one per parent in parent order) and ``gb_thickness``
            (required; applied to the first parent only), ``surface_mode`` (optional,
            defaults to ``"normal_plane"``), and ``max_tilt_degrees`` (optional,
            defaults to ``5.0``).
        :return: A single merged child candidate.
        :raises ManipulationArityError: If ``context`` does not carry exactly two
            parents.
        :raises ManipulationConfigurationError: If a required parameter is missing or
            malformed.
        :raises ManipulationCompatibilityError: If the parents use different ownership
            modes, mismatched boundary topology, or non-affine-equivalent physical
            grain geometry.
        :raises ManipulationCapabilityError: If the parents' unit cells are
            incompatible or no positive-width formula-preserving crossover interval
            exists.
        """
        if len(context.parents) != 2:
            raise ManipulationArityError(
                f"{self.name!r} requires 2 parent(s); received "
                f"{len(context.parents)}"
            )
        unit_cells = _require(context.params, "unit_cell")
        if len(unit_cells) != 2:
            raise ManipulationConfigurationError(
                "unit_cell must supply exactly one unit cell per parent"
            )
        gb_thickness = float(_require(context.params, "gb_thickness"))
        surface_mode = context.params.get("surface_mode", "normal_plane")
        max_tilt_degrees = context.params.get("max_tilt_degrees", 5.0)

        first = _CandidateCrossoverParent(
            context.parents[0], unit_cells[0], gb_thickness
        )
        second = _CandidateCrossoverParent(
            context.parents[1], unit_cells[1], gb_thickness
        )
        new_atoms, child_labels, provenance = crossover_slice_and_merge(
            first,
            second,
            surface_mode=surface_mode,
            max_tilt_degrees=max_tilt_degrees,
            rng=context.rng,
        )
        try:
            child = InterfaceCandidate(
                atoms=new_atoms,
                box_dims=context.parents[0].box_dims,
                gb_plane_x=context.parents[0].gb_plane_x,
                left_grain_x_bounds=context.parents[0].left_grain_x_bounds,
                right_grain_x_bounds=context.parents[0].right_grain_x_bounds,
                grain_labels=child_labels,
                inplane_periodic=context.parents[0].inplane_periodic,
                normal_topology=context.parents[0].normal_topology,
                coordinate_tolerance=context.parents[0].coordinate_tolerance,
                interface_separation=context.parents[0].interface_separation,
            )
        except InterfaceCandidateTypeError as exc:
            raise TypeError(str(exc)) from exc
        except InterfaceCandidateValueError as exc:
            raise ManipulationConfigurationError(str(exc)) from exc

        return ManipulationResult(
            children=(child,),
            parameters=provenance,
            lineage={
                "operation": self.name,
                "parent_count": 2,
            },
        )


__all__ = ["CrossoverParent", "SliceAndMerge", "crossover_slice_and_merge"]
