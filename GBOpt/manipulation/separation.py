# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Topology-aware interface separation as a built-in ``Manipulation`` operation.

``InterfaceSeparation`` inserts an empty interval between the two grains of an
already-constructed candidate, validated against the manipulator's current parent.
Periodic bicrystals expand the x box by twice the requested separation so both the
central and outer periodic interfaces gain the same spacing; slabs expand by the
requested separation while preserving both outer vacuum widths. Unlike
``RightGrainTranslation``/``GrainTerminationCycle``, the candidate being transformed is
not the parent itself but an explicit ``candidate`` parameter -- ordinarily the output
of an earlier translation or termination-cycling step -- validated for fixed-cell
geometry agreement with the supplied parent before separation is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GrainOwnership import RIGHT_GRAIN_LABEL
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


def _readonly_copy(values: np.ndarray, *, dtype=None) -> np.ndarray:
    """Return a defensive read-only array copy."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


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
    :raises ManipulationCapabilityError: If the initial grain bounds are not contiguous,
        the physical grain bounds are inconsistent with the specified topology, a slab
        has no free-surface or vacuum interval, or the boundary-normal topology is
        unknown.
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
        raise ManipulationCapabilityError(
            "Interface separation requires initially contiguous grain bounds"
        )

    new_box = np.array(box, copy=True)
    if normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL:
        if not np.isclose(left[0], xlo, atol=tolerance, rtol=0.0) or not np.isclose(
            right[1], xhi, atol=tolerance, rtol=0.0
        ):
            raise ManipulationCapabilityError(
                "Periodic separation requires zero-vacuum physical grain bounds"
            )
        new_box[0, 1] = xhi + 2.0 * separation
    elif normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB:
        left_vacuum = float(left[0] - xlo)
        right_vacuum = float(xhi - right[1])
        if left_vacuum < -tolerance or right_vacuum < -tolerance:
            raise ManipulationCapabilityError(
                "Slab physical grain bounds must lie inside the x box"
            )
        if left_vacuum <= tolerance and right_vacuum <= tolerance:
            raise ManipulationCapabilityError(
                "Slab separation requires a free-surface or vacuum interval"
            )
        new_box[0, 1] = xhi + separation
    else:
        raise ManipulationCapabilityError(
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


class InterfaceSeparation:
    """Insert a topology-aware empty interval between the two grains of a candidate."""

    @property
    def name(self) -> str:
        """Stable operation name used for registry lookup and lineage metadata."""
        return "interface_separation"

    @property
    def arity(self) -> int:
        """Number of parent candidates this operation requires."""
        return 1

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        """Separate the candidate supplied in ``context.params["candidate"]``.

        :param context: Validated input. ``context.parents[0]`` supplies the reference
            fixed-cell geometry the candidate must match. ``params`` must supply
            ``candidate`` (an ``InterfaceCandidate``, ordinarily produced by an earlier
            translation or termination-cycling step) and ``interface_separation`` (a
            nonnegative float).
        :return: A single separated child candidate.
        :raises ManipulationConfigurationError: If ``candidate`` is not an
            ``InterfaceCandidate`` or ``interface_separation`` is missing, non-finite,
            or negative.
        :raises ManipulationCapabilityError: If the candidate's topology, provenance, or
            geometry is incompatible with separation.
        """
        parent = context.parents[0]
        candidate = context.params.get("candidate")
        if not isinstance(candidate, InterfaceCandidate):
            raise ManipulationConfigurationError(
                "candidate must be an InterfaceCandidate"
            )
        if "interface_separation" not in context.params:
            raise ManipulationConfigurationError(
                "interface_separation is a required parameter"
            )
        separation = _validate_finite_real(
            "interface_separation", context.params["interface_separation"]
        )
        if separation < 0.0:
            raise ManipulationConfigurationError(
                "interface_separation must be nonnegative"
            )

        tolerance = candidate.coordinate_tolerance
        if not np.isclose(
            candidate.interface_separation, 0.0, atol=tolerance, rtol=0.0
        ):
            raise ManipulationCapabilityError(
                "interface separation cannot be reapplied to a separated candidate"
            )
        if candidate.normal_topology is not parent.normal_topology:
            raise ManipulationCapabilityError(
                "candidate topology does not match the manipulator parent"
            )
        if candidate.normal_topology is BoundaryNormalTopology.UNKNOWN:
            raise ManipulationCapabilityError(
                "interface separation requires known boundary-normal topology."
            )
        if candidate.inplane_periodic != parent.inplane_periodic:
            raise ManipulationCapabilityError(
                "candidate in-plane periodicity does not match the parent"
            )
        if not np.allclose(
            candidate.box_dims, parent.box_dims, atol=tolerance, rtol=0.0
        ) or not np.isclose(
            candidate.gb_plane_x, parent.gb_plane_x, atol=tolerance, rtol=0.0
        ):
            raise ManipulationCapabilityError(
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
            raise ManipulationCapabilityError(
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

        try:
            child = InterfaceCandidate(
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
        except InterfaceCandidateTypeError as exc:
            raise TypeError(str(exc)) from exc
        except InterfaceCandidateValueError as exc:
            raise ManipulationConfigurationError(str(exc)) from exc

        return ManipulationResult(
            children=(child,),
            parameters={"interface_separation": separation},
            lineage={"operation": self.name, "parent_count": 1},
        )


__all__ = ["InterfaceSeparation"]
