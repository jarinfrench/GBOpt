# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Rigid right-grain translation as a built-in ``Manipulation`` operation.

``RightGrainTranslation`` moves the single parent's right grain (as flagged by
``GBOpt.GrainOwnership.RIGHT_GRAIN_LABEL``) by an explicit x/y/z displacement. Each
in-plane axis wraps or rejects the displacement according to the parent's own
``inplane_periodic`` flags; a nonperiodic x displacement that would move any right-grain
atom outside the parent's supported half-open x interval is rejected. The left grain,
simulation box, grain-boundary plane, physical grain bounds, and every other geometry
field of the produced candidate are unchanged from the parent.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

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


def right_grain_translation_atoms(
    *,
    left_atoms: np.ndarray,
    right_atoms: np.ndarray,
    right_grain_x_bounds: tuple[float, float] | np.ndarray,
    box_dims: np.ndarray,
    inplane_periodic: tuple[bool, bool],
    coordinate_tolerance: float,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """Return ``left_atoms`` followed by ``right_atoms`` rigidly translated.

    This is the pure computational core shared by ``RightGrainTranslation.execute``
    and ``GBManipulator.translate_right_grain``. It takes plain grain arrays and
    geometry rather than an ``InterfaceCandidate`` and returns a plain structured
    array, deliberately never constructing or validating against an
    ``InterfaceCandidate``: explicit ownership is persistent state, and a relaxed
    right-grain atom may legitimately already lie -- or be translated to lie --
    outside ``right_grain_x_bounds`` without changing grains, which
    ``InterfaceCandidate.__init__``'s atoms-in-bounds check would reject. The
    x-translation safety check below is preserved exactly whenever x is actually
    displaced.

    :param left_atoms: Keyword argument, required. Left-grain structured atom rows.
    :param right_atoms: Keyword argument, required. Right-grain structured atom rows.
    :param right_grain_x_bounds: Keyword argument, required. Right grain's physical x
        interval.
    :param box_dims: Keyword argument, required. Finite 3 by 2 Cartesian box bounds.
    :param inplane_periodic: Keyword argument, required. y/z periodicity flags.
    :param coordinate_tolerance: Keyword argument, required. Coordinate tolerance in
        angstroms.
    :param dx: Keyword argument, required. Displacement in x in angstroms.
    :param dy: Keyword argument, required. Displacement in y in angstroms.
    :param dz: Keyword argument, required. Displacement in z in angstroms.
    :return: Left-grain rows followed by translated right-grain rows.
    :raises ManipulationCapabilityError: If the displacement moves atoms outside a
        supported interval.
    """
    tolerance = float(coordinate_tolerance)
    right = np.array(right_atoms, copy=True)
    x_lower, x_upper = right_grain_x_bounds

    translated_x = right["x"] + dx
    if abs(dx) > tolerance and (
        np.any(translated_x < x_lower - tolerance) or np.any(translated_x >= x_upper)
    ):
        raise ManipulationCapabilityError(
            "dx moves one or more right-grain atoms outside the supported "
            f"half-open x interval [{x_lower}, {x_upper})"
        )

    right["x"] = translated_x
    right = _translate_inplane(
        right,
        dy=dy,
        dz=dz,
        box_dims=box_dims,
        inplane_periodic=inplane_periodic,
        tolerance=tolerance,
    )
    return np.hstack((left_atoms, right))


class RightGrainTranslation:
    """Rigidly translate a single parent's right grain."""

    @property
    def name(self) -> str:
        """Stable operation name used for registry lookup and lineage metadata."""
        return "right_grain_translation"

    @property
    def arity(self) -> int:
        """Number of parent candidates this operation requires."""
        return 1

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        """Translate ``context.parents[0]``'s right grain.

        :param context: Validated input whose ``params`` supply ``dy`` and ``dz``
            (required) and ``dx`` (optional, defaults to ``0.0``).
        :return: A single translated child candidate. Unlike
            ``right_grain_translation_atoms``, the produced candidate is validated
            by ``InterfaceCandidate``, so a displacement that leaves a right-grain
            atom outside ``context.parents[0]``'s physical grain bounds surfaces as a
            ``ManipulationConfigurationError`` here even where the pure computation
            would have allowed it.
        :raises ManipulationConfigurationError: If a required parameter is missing, a
            displacement is non-finite, or the resulting candidate is malformed.
        :raises ManipulationCapabilityError: If the displacement moves atoms outside a
            supported interval.
        """
        parent = context.parents[0]
        if "dy" not in context.params or "dz" not in context.params:
            raise ManipulationConfigurationError(
                "right_grain_translation requires dy and dz parameters"
            )
        dx = _validate_finite_real("dx", context.params.get("dx", 0.0))
        dy = _validate_finite_real("dy", context.params["dy"])
        dz = _validate_finite_real("dz", context.params["dz"])

        labels = parent.grain_labels
        parent_atoms = parent.atoms
        atoms = right_grain_translation_atoms(
            left_atoms=parent_atoms[labels == LEFT_GRAIN_LABEL],
            right_atoms=parent_atoms[labels == RIGHT_GRAIN_LABEL],
            right_grain_x_bounds=parent.right_grain_x_bounds,
            box_dims=parent.box_dims,
            inplane_periodic=parent.inplane_periodic,
            coordinate_tolerance=parent.coordinate_tolerance,
            dx=dx,
            dy=dy,
            dz=dz,
        )
        child = _build_candidate(parent, atoms)
        return ManipulationResult(
            children=(child,),
            parameters={"dx": dx, "dy": dy, "dz": dz},
            lineage={"operation": self.name, "parent_count": 1},
        )


__all__ = ["RightGrainTranslation"]
