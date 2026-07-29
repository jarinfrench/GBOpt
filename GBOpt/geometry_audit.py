# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Deterministic geometry diagnostics for periodic bicrystal structures.

The audit operates on already-generated left- and right-grain atom arrays. It does not
modify coordinates, select a termination, or decide a relative grain translation. Its
purpose is to quantify interface gaps, close contacts, and periodic duplicate sites so
construction defects can be identified without changing generation behavior.

The legacy audit entry point retains its periodic-x, in-plane-y/z behavior. Reusable
primitives in this module additionally support arbitrary interface axes and mixed
periodic/fixed boundary conditions for strict ``BicrystalState`` validation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


class GeometryAuditError(ValueError):
    """Raised when geometry-audit inputs are malformed or inconsistent."""


@dataclass(frozen=True, slots=True)
class InterfaceGapStatistics:
    """Binned local separation statistics for one bicrystal interface.

    :param minimum_angstrom: Minimum local gap over bins populated by both grains.
    :param median_angstrom: Median local gap over bins populated by both grains.
    :param percentile_95_angstrom: 95th-percentile local gap.
    :param maximum_angstrom: Maximum local gap.
    :param range_angstrom: Maximum minus minimum local gap.
    :param empty_left_bin_fraction: Fraction of bins with no left-grain atom.
    :param empty_right_bin_fraction: Fraction of bins with no right-grain atom.
    :param valid_bin_fraction: Fraction of bins populated by both grains.
    :param valid_bins: Number of bins populated by both grains.
    :param total_bins: Total number of in-plane bins.
    """

    minimum_angstrom: float | None
    median_angstrom: float | None
    percentile_95_angstrom: float | None
    maximum_angstrom: float | None
    range_angstrom: float | None
    empty_left_bin_fraction: float
    empty_right_bin_fraction: float
    valid_bin_fraction: float
    valid_bins: int
    total_bins: int


@dataclass(frozen=True, slots=True)
class NearestNeighborDiagnostics:
    """Minimum-distance and periodic-duplicate diagnostics.

    :param left_internal_min_angstrom: Minimum left-grain internal distance with y/z
        periodicity and no x wrapping.
    :param right_internal_min_angstrom: Minimum right-grain internal distance with y/z
        periodicity and no x wrapping.
    :param central_cross_min_angstrom: Minimum left/right distance across the central
        interface with y/z periodicity and no x wrapping.
    :param periodic_cross_min_angstrom: Minimum right/left distance across the periodic
        x box boundary with y/z periodicity.
    :param periodic_duplicate_count: Number of unique whole-system atom pairs separated
        by no more than the duplicate tolerance under full x/y/z periodicity.
    """

    left_internal_min_angstrom: float | None
    right_internal_min_angstrom: float | None
    central_cross_min_angstrom: float | None
    periodic_cross_min_angstrom: float | None
    periodic_duplicate_count: int


@dataclass(frozen=True, slots=True)
class GeometryAuditThresholds:
    """Conservative warning thresholds for initial campaign classification.

    Raw metrics remain available independently of this policy. The defaults are meant
    to flag obvious channels, voids, overlaps, and duplicates without rejecting a
    generated structure.

    :param max_empty_bin_fraction: Maximum tolerated empty fraction on either side of
        either interface.
    :param max_gap_range_bulk_factor: Maximum local-gap range as a multiple of the
        minimum bulk nearest-neighbor distance.
    :param max_gap_tail_bulk_factor: Maximum ``p95 - median`` gap tail as a multiple of
        the minimum bulk nearest-neighbor distance.
    :param min_cross_distance_bulk_factor: Minimum cross-interface distance as a
        multiple of the minimum bulk nearest-neighbor distance.
    :param duplicate_tolerance_angstrom: Distance at or below which a periodic atom pair
        is classified as a duplicate.
    """

    max_empty_bin_fraction: float = 0.25
    max_gap_range_bulk_factor: float = 2.0
    max_gap_tail_bulk_factor: float = 1.0
    min_cross_distance_bulk_factor: float = 0.45
    duplicate_tolerance_angstrom: float = 1.0e-6

    def __post_init__(self) -> None:
        """Validate threshold values."""
        for name, value in (
            ("max_empty_bin_fraction", self.max_empty_bin_fraction),
            ("max_gap_range_bulk_factor", self.max_gap_range_bulk_factor),
            ("max_gap_tail_bulk_factor", self.max_gap_tail_bulk_factor),
            ("min_cross_distance_bulk_factor", self.min_cross_distance_bulk_factor),
            ("duplicate_tolerance_angstrom", self.duplicate_tolerance_angstrom),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                raise GeometryAuditError(f"{name} must be a finite non-negative float.")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise GeometryAuditError(f"{name} must be a finite non-negative float.")
        if self.max_empty_bin_fraction > 1.0:
            raise GeometryAuditError("max_empty_bin_fraction must not exceed 1.0.")


@dataclass(frozen=True, slots=True)
class GeometryAuditResult:
    """Complete deterministic geometry audit for one periodic bicrystal.

    :param status: ``"ok"``, ``"suspicious"``, or ``"invalid"``.
    :param reasons: Stable machine-readable classification reasons.
    :param central_interface: Local-gap statistics at the central left/right interface.
    :param periodic_interface: Local-gap statistics at the periodic x box boundary.
    :param nearest_neighbors: Minimum-distance and duplicate diagnostics.
    :param bins_y: Number of reduced-coordinate bins along y.
    :param bins_z: Number of reduced-coordinate bins along z.
    :param bulk_reference_distance_angstrom: Minimum finite internal-grain distance used
        to scale warning thresholds.
    :param central_plane_x_angstrom: Supplied central-interface plane coordinate.
    :param thresholds: Classification policy used for this result.
    """

    status: str
    reasons: tuple[str, ...]
    central_interface: InterfaceGapStatistics
    periodic_interface: InterfaceGapStatistics
    nearest_neighbors: NearestNeighborDiagnostics
    bins_y: int
    bins_z: int
    bulk_reference_distance_angstrom: float | None
    central_plane_x_angstrom: float
    thresholds: GeometryAuditThresholds

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable nested dictionary."""
        return asdict(self)


def _positions(atoms: Any, name: str) -> np.ndarray:
    """Return atom coordinates as a finite ``(N, 3)`` float array."""
    if atoms is None:
        raise GeometryAuditError(f"{name} must not be None.")

    arr = np.asarray(atoms)
    if arr.ndim == 2 and arr.shape[1] == 3 and arr.dtype.names is None:
        positions = np.asarray(arr, dtype=np.float64)
    elif arr.dtype.names is not None and {"x", "y", "z"}.issubset(arr.dtype.names):
        positions = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(
            np.float64,
            copy=False,
        )
    else:
        raise GeometryAuditError(
            f"{name} must be an (N, 3) coordinate array or a structured array with "
            "x, y, and z fields."
        )

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise GeometryAuditError(f"{name} coordinates must have shape (N, 3).")
    if len(positions) == 0:
        raise GeometryAuditError(f"{name} must contain at least one atom.")
    if not np.all(np.isfinite(positions)):
        raise GeometryAuditError(f"{name} contains non-finite coordinates.")
    return np.array(positions, dtype=np.float64, copy=True)


def _validated_box(box_dims: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return orthorhombic box lower bounds and positive lengths."""
    try:
        box = np.asarray(box_dims, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GeometryAuditError("box_dims must be a finite 3 by 2 array.") from exc
    if box.shape != (3, 2):
        raise GeometryAuditError(f"box_dims must have shape (3, 2); got {box.shape}.")
    if not np.all(np.isfinite(box)):
        raise GeometryAuditError("box_dims contains non-finite values.")
    lower = box[:, 0]
    lengths = box[:, 1] - box[:, 0]
    if np.any(lengths <= 0.0):
        raise GeometryAuditError("box_dims upper bounds must exceed lower bounds.")
    return lower, lengths


def _validated_bins(bins: tuple[int, int]) -> tuple[int, int]:
    """Return validated positive y/z bin counts."""
    if not isinstance(bins, tuple) or len(bins) != 2:
        raise GeometryAuditError("bins must be a two-item tuple (bins_y, bins_z).")
    normalized: list[int] = []
    for axis_name, value in zip(("bins_y", "bins_z"), bins, strict=True):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise GeometryAuditError(f"{axis_name} must be a positive integer.")
        integer = int(value)
        if integer <= 0:
            raise GeometryAuditError(f"{axis_name} must be a positive integer.")
        normalized.append(integer)
    return normalized[0], normalized[1]


def _wrap_periodic(values: np.ndarray, lower: float, length: float) -> np.ndarray:
    """Wrap coordinates into the half-open interval ``[0, length)``.

    ``numpy.mod`` can round a value infinitesimally below ``lower`` to exactly
    ``length``.  ``scipy.spatial.cKDTree`` rejects that endpoint when ``boxsize`` is
    supplied, so clamp any rounded endpoint to the largest representable value below
    the box length.
    """
    wrapped = np.mod(np.asarray(values, dtype=np.float64) - lower, length)
    upper_inside = np.nextafter(float(length), 0.0)
    return np.clip(wrapped, 0.0, upper_inside)


def wrap_periodic_coordinates(
    positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    periodic_axes: tuple[int, ...],
) -> np.ndarray:
    """Return coordinates wrapped only along the declared periodic axes."""
    result = np.array(positions, dtype=np.float64, copy=True)
    for axis in periodic_axes:
        result[:, axis] = _wrap_periodic(
            result[:, axis], float(lower[axis]), float(lengths[axis])
        )
    return result


def mixed_boundary_tree_coordinates(
    positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    periodic_axes: tuple[int, ...],
    *,
    shared_fixed_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform coordinates for ``cKDTree`` with mixed boundary conditions.

    ``cKDTree`` accepts a periodic box on every axis or none. Fixed axes are therefore
    embedded in a box more than twice their occupied span, making the periodic image
    farther away than the direct separation. Callers comparing two sets should pass
    shared fixed bounds derived from their union.
    """
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise GeometryAuditError("positions must have shape (N, 3).")
    periodic = frozenset(int(axis) for axis in periodic_axes)
    if not periodic.issubset({0, 1, 2}):
        raise GeometryAuditError("periodic_axes may contain only 0, 1, and 2.")
    transformed = np.array(array, copy=True)
    boxsize = np.empty(3, dtype=np.float64)
    if shared_fixed_bounds is None:
        fixed_min = np.min(array, axis=0) if len(array) else np.array(lower, copy=True)
        fixed_max = np.max(array, axis=0) if len(array) else np.array(lower, copy=True)
    else:
        fixed_min, fixed_max = shared_fixed_bounds
    for axis in range(3):
        if axis in periodic:
            transformed[:, axis] = _wrap_periodic(
                transformed[:, axis], float(lower[axis]), float(lengths[axis])
            )
            boxsize[axis] = float(lengths[axis])
        else:
            span = max(float(fixed_max[axis] - fixed_min[axis]), 0.0)
            padding = max(1.0, 0.05 * max(span, 1.0))
            transformed[:, axis] = transformed[:, axis] - float(fixed_min[axis]) + padding
            boxsize[axis] = 2.0 * span + 3.0 * padding
    return transformed, boxsize


def same_distance_summary(
    positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    periodic_axes: tuple[int, ...],
) -> float | None:
    """Return the minimum distinct-site distance under mixed boundaries."""
    if len(positions) < 2:
        return None
    transformed, boxsize = mixed_boundary_tree_coordinates(
        positions, lower, lengths, periodic_axes
    )
    tree = cKDTree(transformed, boxsize=boxsize)
    distances = tree.query(transformed, k=2, workers=1)[0][:, 1]
    finite = distances[np.isfinite(distances)]
    return None if finite.size == 0 else float(np.min(finite))


def cross_distance_summary(
    query_positions: np.ndarray,
    target_positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    periodic_axes: tuple[int, ...],
) -> float | None:
    """Return the minimum cross-set distance under mixed boundaries."""
    if len(query_positions) == 0 or len(target_positions) == 0:
        return None
    combined = np.vstack((query_positions, target_positions))
    bounds = (np.min(combined, axis=0), np.max(combined, axis=0))
    query, boxsize = mixed_boundary_tree_coordinates(
        query_positions, lower, lengths, periodic_axes, shared_fixed_bounds=bounds
    )
    target, _ = mixed_boundary_tree_coordinates(
        target_positions, lower, lengths, periodic_axes, shared_fixed_bounds=bounds
    )
    distances = cKDTree(target, boxsize=boxsize).query(query, k=1, workers=1)[0]
    finite = distances[np.isfinite(distances)]
    return None if finite.size == 0 else float(np.min(finite))


def periodic_duplicate_pairs(
    positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    periodic_axes: tuple[int, ...],
    tolerance: float,
) -> tuple[tuple[int, int], ...]:
    """Return stable unique index pairs within ``tolerance`` under mixed boundaries."""
    if len(positions) < 2:
        return ()
    transformed, boxsize = mixed_boundary_tree_coordinates(
        positions, lower, lengths, periodic_axes
    )
    pairs = cKDTree(transformed, boxsize=boxsize).query_pairs(
        float(tolerance), output_type="set"
    )
    return tuple(sorted((int(i), int(j)) for i, j in pairs))


def _generic_bin_indices(
    positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    tangent_axes: tuple[int, int],
    bins: tuple[int, int],
    periodic_axes: tuple[int, ...],
) -> np.ndarray:
    indices: list[np.ndarray] = []
    periodic = frozenset(periodic_axes)
    for axis, count in zip(tangent_axes, bins, strict=True):
        if axis in periodic:
            reduced = (
                np.mod(positions[:, axis] - lower[axis], lengths[axis])
                / lengths[axis]
            )
        else:
            reduced = (positions[:, axis] - lower[axis]) / lengths[axis]
            reduced = np.clip(reduced, 0.0, np.nextafter(1.0, 0.0))
        values = np.floor(reduced * count).astype(np.int64)
        np.minimum(values, count - 1, out=values)
        indices.append(values)
    return indices[0] * bins[1] + indices[1]


def summarize_interface_gaps(
    minus_positions: np.ndarray,
    plus_positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    *,
    axis: int,
    normal_sign: int,
    bins: tuple[int, int],
    plus_normal_shift: float = 0.0,
    periodic_axes: tuple[int, ...] = (0, 1, 2),
) -> InterfaceGapStatistics:
    """Return local per-bin gap metrics for an oriented physical interface.

    ``plus_normal_shift`` unfolds the plus grain for a periodic-boundary interface.
    It is zero for an interior interface and normally one signed box length for a
    periodic interface.
    """
    if axis not in (0, 1, 2) or normal_sign not in (-1, 1):
        raise GeometryAuditError("axis and normal_sign are invalid.")
    bins = _validated_bins(bins)
    tangent = tuple(candidate for candidate in range(3) if candidate != axis)
    total = bins[0] * bins[1]
    minus_bins = _generic_bin_indices(
        minus_positions, lower, lengths, tangent, bins, periodic_axes
    )
    plus_bins = _generic_bin_indices(
        plus_positions, lower, lengths, tangent, bins, periodic_axes
    )
    minus_surface = np.full(total, -np.inf, dtype=np.float64)
    plus_surface = np.full(total, np.inf, dtype=np.float64)
    minus_oriented = normal_sign * minus_positions[:, axis]
    plus_oriented = normal_sign * (plus_positions[:, axis] + plus_normal_shift)
    np.maximum.at(minus_surface, minus_bins, minus_oriented)
    np.minimum.at(plus_surface, plus_bins, plus_oriented)
    minus_present = np.isfinite(minus_surface)
    plus_present = np.isfinite(plus_surface)
    return _summarize_gaps(
        plus_surface - minus_surface, minus_present, plus_present
    )


def automatic_interface_bins(
    lengths: np.ndarray,
    axis: int,
    bulk_reference_distance: float | None,
    *,
    min_bins_per_axis: int = 1,
    max_bins_per_axis: int = 64,
    count_mode: str = "nearest",
) -> tuple[int, int]:
    """Choose deterministic approximately isotropic bins tangent to an interface."""
    tangent = tuple(candidate for candidate in range(3) if candidate != axis)
    if bulk_reference_distance is None or bulk_reference_distance <= 0.0:
        target = min(float(lengths[tangent[0]]), float(lengths[tangent[1]])) / 8.0
    else:
        target = 2.0 * float(bulk_reference_distance)
    target = max(target, np.finfo(np.float64).eps)

    def count(length: float) -> int:
        estimate_float = length / target
        estimate = (
            int(math.ceil(estimate_float))
            if count_mode == "ceil"
            else int(math.floor(estimate_float + 0.5))
        )
        return min(max(estimate, min_bins_per_axis), max_bins_per_axis)
    if count_mode not in {"ceil", "nearest"}:
        raise GeometryAuditError("count_mode must be 'ceil' or 'nearest'.")
    return count(float(lengths[tangent[0]])), count(float(lengths[tangent[1]]))


def _minimum_internal_distance(
    positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
) -> float | None:
    """Return minimum same-grain distance with y/z periodicity only."""
    return same_distance_summary(positions, lower, lengths, (1, 2))


def _minimum_cross_distance(
    query_positions: np.ndarray,
    target_positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
) -> float | None:
    """Return minimum cross-set distance with y/z periodicity only."""
    return cross_distance_summary(
        query_positions, target_positions, lower, lengths, (1, 2)
    )


def _periodic_duplicate_count(
    positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    tolerance: float,
) -> int:
    """Count unique whole-system periodic atom pairs within ``tolerance``."""
    return len(
        periodic_duplicate_pairs(
            positions, lower, lengths, (0, 1, 2), tolerance
        )
    )


def _bin_indices(
    positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    bins_y: int,
    bins_z: int,
) -> np.ndarray:
    """Return flattened reduced-coordinate y/z bin indices."""
    reduced_y = np.mod(positions[:, 1] - lower[1], lengths[1]) / lengths[1]
    reduced_z = np.mod(positions[:, 2] - lower[2], lengths[2]) / lengths[2]
    index_y = np.floor(reduced_y * bins_y).astype(np.int64)
    index_z = np.floor(reduced_z * bins_z).astype(np.int64)
    np.minimum(index_y, bins_y - 1, out=index_y)
    np.minimum(index_z, bins_z - 1, out=index_z)
    return index_y * bins_z + index_z


def _summarize_gaps(
    gaps: np.ndarray,
    left_present: np.ndarray,
    right_present: np.ndarray,
) -> InterfaceGapStatistics:
    """Build interface statistics from per-bin surfaces and gaps."""
    total_bins = int(len(gaps))
    valid = left_present & right_present & np.isfinite(gaps)
    valid_gaps = gaps[valid]
    if valid_gaps.size == 0:
        minimum = median = percentile_95 = maximum = gap_range = None
    else:
        minimum = float(np.min(valid_gaps))
        median = float(np.median(valid_gaps))
        percentile_95 = float(np.percentile(valid_gaps, 95.0))
        maximum = float(np.max(valid_gaps))
        gap_range = float(maximum - minimum)
    return InterfaceGapStatistics(
        minimum_angstrom=minimum,
        median_angstrom=median,
        percentile_95_angstrom=percentile_95,
        maximum_angstrom=maximum,
        range_angstrom=gap_range,
        empty_left_bin_fraction=float(np.mean(~left_present)),
        empty_right_bin_fraction=float(np.mean(~right_present)),
        valid_bin_fraction=float(np.mean(valid)),
        valid_bins=int(np.count_nonzero(valid)),
        total_bins=total_bins,
    )


def _interface_gap_statistics(
    left: np.ndarray,
    right: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    bins_y: int,
    bins_z: int,
    *,
    periodic_x_interface: bool,
) -> InterfaceGapStatistics:
    """Return binned local gaps for the central or periodic x interface."""
    total_bins = bins_y * bins_z
    left_bins = _bin_indices(left, lower, lengths, bins_y, bins_z)
    right_bins = _bin_indices(right, lower, lengths, bins_y, bins_z)

    if periodic_x_interface:
        left_surface = np.full(total_bins, np.inf, dtype=np.float64)
        right_surface = np.full(total_bins, -np.inf, dtype=np.float64)
        np.minimum.at(left_surface, left_bins, left[:, 0])
        np.maximum.at(right_surface, right_bins, right[:, 0])
        left_present = np.isfinite(left_surface)
        right_present = np.isfinite(right_surface)
        gaps = (lower[0] + lengths[0] - right_surface) + (
            left_surface - lower[0]
        )
    else:
        left_surface = np.full(total_bins, -np.inf, dtype=np.float64)
        right_surface = np.full(total_bins, np.inf, dtype=np.float64)
        np.maximum.at(left_surface, left_bins, left[:, 0])
        np.minimum.at(right_surface, right_bins, right[:, 0])
        left_present = np.isfinite(left_surface)
        right_present = np.isfinite(right_surface)
        gaps = right_surface - left_surface

    return _summarize_gaps(gaps, left_present, right_present)


def _automatic_bins(
    lengths: np.ndarray,
    bulk_reference_distance: float | None,
    *,
    min_bins_per_axis: int,
    max_bins_per_axis: int,
) -> tuple[int, int]:
    """Choose approximately isotropic physical bins from the bulk spacing."""
    return automatic_interface_bins(
        lengths,
        0,
        bulk_reference_distance,
        min_bins_per_axis=min_bins_per_axis,
        max_bins_per_axis=max_bins_per_axis,
        count_mode="ceil",
    )


def _bulk_reference(*distances: float | None) -> float | None:
    """Return the smallest finite positive internal-grain distance."""
    finite = [
        float(value)
        for value in distances
        if value is not None and math.isfinite(value) and value > 0.0
    ]
    return None if not finite else min(finite)


def _classify(
    central: InterfaceGapStatistics,
    periodic: InterfaceGapStatistics,
    nearest: NearestNeighborDiagnostics,
    bulk_reference: float | None,
    thresholds: GeometryAuditThresholds,
) -> tuple[str, tuple[str, ...]]:
    """Classify raw metrics without rejecting construction."""
    reasons: list[str] = []
    invalid = False

    for label, stats in (("central", central), ("periodic", periodic)):
        if stats.valid_bins == 0:
            reasons.append(f"{label}_interface_has_no_valid_bins")
            invalid = True
            continue
        if stats.empty_left_bin_fraction > thresholds.max_empty_bin_fraction:
            reasons.append(f"{label}_interface_excess_empty_left_bins")
        if stats.empty_right_bin_fraction > thresholds.max_empty_bin_fraction:
            reasons.append(f"{label}_interface_excess_empty_right_bins")

        if bulk_reference is not None:
            if (
                stats.range_angstrom is not None
                and stats.range_angstrom
                > thresholds.max_gap_range_bulk_factor * bulk_reference
            ):
                reasons.append(f"{label}_interface_large_gap_range")
            if (
                stats.percentile_95_angstrom is not None
                and stats.median_angstrom is not None
                and stats.percentile_95_angstrom - stats.median_angstrom
                > thresholds.max_gap_tail_bulk_factor * bulk_reference
            ):
                reasons.append(f"{label}_interface_heavy_gap_tail")

    if nearest.periodic_duplicate_count > 0:
        reasons.append("periodic_duplicate_sites")

    if bulk_reference is not None:
        minimum_cross = thresholds.min_cross_distance_bulk_factor * bulk_reference
        if (
            nearest.central_cross_min_angstrom is not None
            and nearest.central_cross_min_angstrom < minimum_cross
        ):
            reasons.append("central_interface_severe_overlap")
        if (
            nearest.periodic_cross_min_angstrom is not None
            and nearest.periodic_cross_min_angstrom < minimum_cross
        ):
            reasons.append("periodic_interface_severe_overlap")

    deduplicated = tuple(dict.fromkeys(reasons))
    if invalid:
        return "invalid", deduplicated
    if deduplicated:
        return "suspicious", deduplicated
    return "ok", ()


def audit_bicrystal_geometry(
    left_atoms: Any,
    right_atoms: Any,
    box_dims: Any,
    *,
    central_plane_x: float,
    bins: tuple[int, int] | None = None,
    min_bins_per_axis: int = 1,
    max_bins_per_axis: int = 64,
    thresholds: GeometryAuditThresholds | None = None,
) -> GeometryAuditResult:
    """Audit both interfaces, nearest-neighbor distances, and duplicate sites.

    This function is observational: it never modifies atom arrays and a suspicious
    result does not raise. Malformed inputs raise ``GeometryAuditError``.

    :param left_atoms: Left-grain structured atom array or ``(N, 3)`` coordinates.
    :param right_atoms: Right-grain structured atom array or ``(N, 3)`` coordinates.
    :param box_dims: Orthorhombic bounds as ``[[xlo, xhi], [ylo, yhi], [zlo, zhi]]``.
    :param central_plane_x: Lab-frame x coordinate of the central interface. Keyword
        argument, required.
    :param bins: Optional explicit ``(bins_y, bins_z)``. When omitted, approximately
        isotropic physical bins are chosen from the bulk nearest-neighbor distance.
    :param min_bins_per_axis: Lower bound for automatically selected bins. Keyword
        argument, optional, defaults to ``1``.
    :param max_bins_per_axis: Upper bound for automatically selected bins. Keyword
        argument, optional, defaults to ``64``.
    :param thresholds: Optional classification policy. Keyword argument, optional,
        defaults to ``GeometryAuditThresholds()``.
    :return: Complete raw metrics and non-failing classification.
    :raises GeometryAuditError: If atoms, box bounds, plane coordinate, bins, or
        thresholds are invalid.
    """
    left = _positions(left_atoms, "left_atoms")
    right = _positions(right_atoms, "right_atoms")
    lower, lengths = _validated_box(box_dims)

    if isinstance(central_plane_x, (bool, np.bool_)):
        raise GeometryAuditError("central_plane_x must be finite and inside the x box.")
    try:
        central_plane = float(central_plane_x)
    except (TypeError, ValueError) as exc:
        raise GeometryAuditError(
            "central_plane_x must be finite and inside the x box."
        ) from exc
    if (
        not math.isfinite(central_plane)
        or central_plane <= lower[0]
        or central_plane >= lower[0] + lengths[0]
    ):
        raise GeometryAuditError("central_plane_x must be finite and inside the x box.")

    for name, value in (
        ("min_bins_per_axis", min_bins_per_axis),
        ("max_bins_per_axis", max_bins_per_axis),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise GeometryAuditError(f"{name} must be a positive integer.")
        if int(value) <= 0:
            raise GeometryAuditError(f"{name} must be a positive integer.")
    min_bins_per_axis = int(min_bins_per_axis)
    max_bins_per_axis = int(max_bins_per_axis)
    if min_bins_per_axis > max_bins_per_axis:
        raise GeometryAuditError(
            "min_bins_per_axis must not exceed max_bins_per_axis."
        )

    policy = GeometryAuditThresholds() if thresholds is None else thresholds
    if not isinstance(policy, GeometryAuditThresholds):
        raise GeometryAuditError(
            "thresholds must be a GeometryAuditThresholds instance."
        )

    left_internal = _minimum_internal_distance(left, lower, lengths)
    right_internal = _minimum_internal_distance(right, lower, lengths)
    bulk_reference = _bulk_reference(left_internal, right_internal)

    if bins is None:
        bins_y, bins_z = _automatic_bins(
            lengths,
            bulk_reference,
            min_bins_per_axis=min_bins_per_axis,
            max_bins_per_axis=max_bins_per_axis,
        )
    else:
        bins_y, bins_z = _validated_bins(bins)

    central = _interface_gap_statistics(
        left,
        right,
        lower,
        lengths,
        bins_y,
        bins_z,
        periodic_x_interface=False,
    )
    periodic = _interface_gap_statistics(
        left,
        right,
        lower,
        lengths,
        bins_y,
        bins_z,
        periodic_x_interface=True,
    )

    central_cross = _minimum_cross_distance(left, right, lower, lengths)
    shifted_left = np.array(left, copy=True)
    shifted_left[:, 0] += lengths[0]
    periodic_cross = _minimum_cross_distance(right, shifted_left, lower, lengths)
    all_positions = np.vstack((left, right))
    duplicate_count = _periodic_duplicate_count(
        all_positions,
        lower,
        lengths,
        policy.duplicate_tolerance_angstrom,
    )
    nearest = NearestNeighborDiagnostics(
        left_internal_min_angstrom=left_internal,
        right_internal_min_angstrom=right_internal,
        central_cross_min_angstrom=central_cross,
        periodic_cross_min_angstrom=periodic_cross,
        periodic_duplicate_count=duplicate_count,
    )

    status, reasons = _classify(
        central,
        periodic,
        nearest,
        bulk_reference,
        policy,
    )
    return GeometryAuditResult(
        status=status,
        reasons=reasons,
        central_interface=central,
        periodic_interface=periodic,
        nearest_neighbors=nearest,
        bins_y=bins_y,
        bins_z=bins_z,
        bulk_reference_distance_angstrom=bulk_reference,
        central_plane_x_angstrom=central_plane,
        thresholds=policy,
    )


__all__ = [
    "GeometryAuditError",
    "GeometryAuditResult",
    "GeometryAuditThresholds",
    "InterfaceGapStatistics",
    "NearestNeighborDiagnostics",
    "automatic_interface_bins",
    "cross_distance_summary",
    "mixed_boundary_tree_coordinates",
    "periodic_duplicate_pairs",
    "same_distance_summary",
    "summarize_interface_gaps",
    "wrap_periodic_coordinates",
    "audit_bicrystal_geometry",
]
