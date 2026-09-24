# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Grain-local termination cycling as a built-in ``Manipulation`` operation.

``GrainTerminationCycle`` cycles each grain independently through its finite physical x
interval. The left grain remains fixed in-plane, while the right grain may also be
translated along the periodic in-plane directions. For a periodic bicrystal, the
physical grain bounds must span the complete x box so that the central and outer
periodic interfaces remain consistent. For a single-interface slab, the physical grain
bounds must lie inside the x box with at least one free-surface or vacuum interval;
cycling a complete finite grain changes both its GB-facing and free-surface
terminations, so those terminations remain coupled. Both topologies are handled by this
one operation, matching the legacy method's own topology branching.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
from GBOpt.interface import InterfaceCandidate
from GBOpt.interface.types import (
    InterfaceCandidateTypeError,
    InterfaceCandidateValueError,
)
from GBOpt.manipulation.types import (
    ManipulationCapabilityError,
    ManipulationConfigurationError,
    ManipulationContext,
    ManipulationResult,
)


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
    :raises ManipulationCapabilityError: If box geometry is invalid or a nonperiodic
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
            raise ManipulationCapabilityError(
                f"The {axis_name} box interval must be finite and have positive width"
            )
        translated = updated[axis_name] + displacement
        if is_periodic:
            updated[axis_name] = np.mod(translated - lower, width) + lower
            continue
        if np.any(translated < lower - tolerance) or np.any(translated >= upper):
            raise ManipulationCapabilityError(
                f"d{axis_name} moves atoms outside the nonperiodic half-open "
                f"{axis_name} interval [{lower}, {upper})"
            )
        updated[axis_name] = translated
    return updated


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
    :raises ManipulationCapabilityError: If interval geometry is invalid.
    """
    width = upper - lower
    if not np.isfinite(lower) or not np.isfinite(upper) or width <= 0.0:
        raise ManipulationCapabilityError(
            "A termination-cycling interval must be finite and have positive width"
        )
    canonical_shift = float(np.mod(shift, width))
    if np.isclose(canonical_shift, 0.0, atol=tolerance, rtol=0.0):
        return np.array(values, copy=True)
    wrapped = lower + np.mod(values + canonical_shift - lower, width)
    wrapped[np.isclose(wrapped, upper, atol=tolerance, rtol=0.0)] = lower
    return wrapped


def _build_candidate(parent: InterfaceCandidate, atoms: np.ndarray) -> InterfaceCandidate:
    """Construct a candidate reusing ``parent``'s geometry with updated ``atoms``.

    :param parent: Parent candidate supplying every geometry field but atom positions.
    :param atoms: Full, left-then-right-ordered structured atom rows.
    :return: Complete immutable candidate.
    :raises ManipulationConfigurationError: If the resulting candidate is malformed.
    """
    labels = np.hstack(
        (
            np.full(
                int(np.count_nonzero(parent.grain_labels == LEFT_GRAIN_LABEL)),
                LEFT_GRAIN_LABEL,
                dtype=np.int8,
            ),
            np.full(
                int(np.count_nonzero(parent.grain_labels == RIGHT_GRAIN_LABEL)),
                RIGHT_GRAIN_LABEL,
                dtype=np.int8,
            ),
        )
    )
    try:
        return InterfaceCandidate(
            atoms=atoms,
            box_dims=parent.box_dims,
            gb_plane_x=parent.gb_plane_x,
            left_grain_x_bounds=parent.left_grain_x_bounds,
            right_grain_x_bounds=parent.right_grain_x_bounds,
            grain_labels=labels,
            inplane_periodic=parent.inplane_periodic,
            normal_topology=parent.normal_topology,
            coordinate_tolerance=parent.coordinate_tolerance,
            interface_separation=parent.interface_separation,
        )
    except InterfaceCandidateTypeError as exc:
        raise TypeError(str(exc)) from exc
    except InterfaceCandidateValueError as exc:
        raise ManipulationConfigurationError(str(exc)) from exc


def _validate_box_and_bounds(parent: InterfaceCandidate, *, tolerance: float) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, float
]:
    """Validate and return box, left/right bounds, and the gb-plane for cycling.

    :raises ManipulationCapabilityError: If box or grain-bound geometry is invalid.
    """
    box = np.asarray(parent.box_dims, dtype=float)
    if (
        box.shape != (3, 2)
        or not np.all(np.isfinite(box))
        or np.any(box[:, 0] >= box[:, 1])
    ):
        raise ManipulationCapabilityError(
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
        or not np.isclose(left_bounds[1], plane, atol=tolerance, rtol=0.0)
        or not np.isclose(right_bounds[0], plane, atol=tolerance, rtol=0.0)
    ):
        raise ManipulationCapabilityError(
            "termination cycling requires contiguous valid physical grain bounds"
        )
    return box, left_bounds, right_bounds, plane


def _validate_topology_bounds(
    topology: BoundaryNormalTopology,
    *,
    box: np.ndarray,
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    tolerance: float,
) -> None:
    """Validate grain bounds are consistent with ``topology``.

    :raises ManipulationCapabilityError: If the bounds do not match the topology's
        vacuum requirements.
    """
    xlo = float(box[0, 0])
    xhi = float(box[0, 1])

    if topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL:
        if not np.isclose(
            left_bounds[0], xlo, atol=tolerance, rtol=0.0
        ) or not np.isclose(right_bounds[1], xhi, atol=tolerance, rtol=0.0):
            raise ManipulationCapabilityError(
                "seriodic termination cycling requires zero-vacuum grain bounds"
            )
    elif topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB:
        left_vacuum = float(left_bounds[0] - xlo)
        right_vacuum = float(xhi - right_bounds[1])

        if left_vacuum < -tolerance or right_vacuum < -tolerance:
            raise ManipulationCapabilityError(
                "slab grain bounds must lie inside the x box"
            )
        if left_vacuum <= tolerance and right_vacuum <= tolerance:
            raise ManipulationCapabilityError(
                "slab termination cycling requires a free-surface or vacuum interval"
            )


class GrainTerminationCycle:
    """Cycle grain-local terminations for a periodic bicrystal or slab."""

    @property
    def name(self) -> str:
        """Stable operation name used for registry lookup and lineage metadata."""
        return "grain_termination_cycle"

    @property
    def arity(self) -> int:
        """Number of parent candidates this operation requires."""
        return 1

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        """Cycle ``context.parents[0]``'s grain-local terminations.

        :param context: Validated input whose ``params`` may supply
            ``left_phase_shift``, ``right_phase_shift``, ``right_dy``, and
            ``right_dz`` (all optional, default ``0.0``).
        :return: A single termination-cycled child candidate.
        :raises ManipulationConfigurationError: If a displacement is non-finite.
        :raises ManipulationCapabilityError: If the topology, geometry, or atom
            positions are invalid for cycling.
        """
        parent = context.parents[0]
        left_phase_shift = _validate_finite_real(
            "left_phase_shift", context.params.get("left_phase_shift", 0.0)
        )
        right_phase_shift = _validate_finite_real(
            "right_phase_shift", context.params.get("right_phase_shift", 0.0)
        )
        right_dy = _validate_finite_real(
            "right_dy", context.params.get("right_dy", 0.0)
        )
        right_dz = _validate_finite_real(
            "right_dz", context.params.get("right_dz", 0.0)
        )

        topology = parent.normal_topology
        if topology is BoundaryNormalTopology.UNKNOWN:
            raise ManipulationCapabilityError(
                "termination cycling requires known boundary-normal topology"
            )

        tolerance = float(parent.coordinate_tolerance)
        box, left_bounds, right_bounds, _plane = _validate_box_and_bounds(
            parent, tolerance=tolerance
        )
        _validate_topology_bounds(
            topology,
            box=box,
            left_bounds=left_bounds,
            right_bounds=right_bounds,
            tolerance=tolerance,
        )

        labels = parent.grain_labels
        atoms = parent.atoms
        left = atoms[labels == LEFT_GRAIN_LABEL]
        right = atoms[labels == RIGHT_GRAIN_LABEL]
        left_x = np.asarray(left["x"], dtype=float)
        right_x = np.asarray(right["x"], dtype=float)

        if (
            not np.all(np.isfinite(left_x))
            or not np.all(np.isfinite(right_x))
            or np.any(left_x < left_bounds[0] - tolerance)
            or np.any(left_x >= left_bounds[1] + tolerance)
            or np.any(right_x < right_bounds[0] - tolerance)
            or np.any(right_x >= right_bounds[1] + tolerance)
        ):
            raise ManipulationCapabilityError(
                "Parent atoms do not lie inside their physical grain bounds"
            )

        updated_left = np.array(left, copy=True)
        updated_right = np.array(right, copy=True)

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

        child = _build_candidate(parent, np.hstack((updated_left, updated_right)))
        return ManipulationResult(
            children=(child,),
            parameters={
                "left_phase_shift": left_phase_shift,
                "right_phase_shift": right_phase_shift,
                "right_dy": right_dy,
                "right_dz": right_dz,
            },
            lineage={"operation": self.name, "parent_count": 1},
        )


__all__ = ["GrainTerminationCycle"]
