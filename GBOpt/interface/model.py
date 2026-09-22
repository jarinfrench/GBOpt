# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define the neutral immutable interface-candidate domain model.

This module owns ``InterfaceCandidate``, the atom rows and interface geometry shared by
composable manipulation, evaluation, and checkpoint code. It consumes the neutral
``GrainOwnership`` label constants and ``BoundaryTopology`` vocabulary. External file
syntax, calculator execution, and optimizer selection policy do not belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from GBOpt.BoundaryTopology import (
    BoundaryNormalTopology,
    normalize_boundary_normal_topology,
)
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
from GBOpt.interface.types import (
    InterfaceCandidateTypeError,
    InterfaceCandidateValueError,
)


def _validate_finite_real(name: str, value: object) -> float:
    """Return ``value`` as a finite float.

    :param name: Input name used in validation messages.
    :param value: Candidate finite real scalar.
    :return: Validated Python ``float``.
    :raises InterfaceCandidateValueError: If ``value`` is non-finite.
    :raises InterfaceCandidateTypeError: If ``value`` is Boolean or non-real.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InterfaceCandidateTypeError(f"{name} must be a finite real value.")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise InterfaceCandidateValueError(f"{name} must be a finite real value.")
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
    :raises InterfaceCandidateValueError: If shape or finiteness is invalid.
    :raises InterfaceCandidateTypeError: If type is invalid.
    """
    raw = np.asarray(values, dtype=object)
    if raw.shape != shape:
        raise InterfaceCandidateValueError(
            f"{name} must have shape {shape}; got {raw.shape}.")
    normalized = np.empty(shape, dtype=float)
    for index in np.ndindex(shape):
        normalized[index] = _validate_finite_real(f"{name}{index}", raw[index])
    return normalized


def _normalize_inplane_periodic(value: object) -> tuple[bool, bool]:
    """Return strict y/z periodicity flags.

    :param value: Two Boolean periodicity flags.
    :return: Normalized ``(periodic_y, periodic_z)`` tuple.
    :raises InterfaceCandidateValueError: If the input is malformed.
    :raises InterfaceCandidateTypeError: If the input is coercive.
    """
    if not isinstance(value, (tuple, list)):
        raise InterfaceCandidateTypeError(
            "inplane_periodic must contain exactly two Boolean values"
        )
    if len(value) != 2:
        raise InterfaceCandidateValueError(
            "inplane_periodic must contain exactly two Boolean values"
        )
    normalized = []
    for axis_name, flag in zip(("y", "z"), value, strict=True):
        if not isinstance(flag, (bool, np.bool_)):
            raise InterfaceCandidateTypeError(
                f"{axis_name}-axis periodicity must be Boolean"
            )
        normalized.append(bool(flag))
    return normalized[0], normalized[1]


def _normalize_grain_labels(labels: object, *, expected_count: int) -> np.ndarray:
    """Return a read-only array of strict left/right labels.

    :param labels: Candidate-aligned grain labels.
    :param expected_count: Keyword argument, required. Required number of labels.
    :return: Read-only ``int8`` labels.
    :raises InterfaceCandidateValueError: If labels are malformed or omit a grain.
    :raises InterfaceCandidateTypeError: If labels are not integers.
    """
    raw = np.asarray(labels)
    if raw.ndim != 1 or raw.size != expected_count:
        raise InterfaceCandidateValueError(
            "grain_labels length must equal the candidate atom count"
        )
    if raw.dtype.kind not in ("i", "u"):
        raise InterfaceCandidateTypeError(
            "grain_labels must use an integer left/right label dtype"
        )
    if not np.all(np.isin(raw, (LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL))):
        raise InterfaceCandidateValueError(
            "grain_labels must contain only left and right labels"
        )
    result = np.array(raw, dtype=np.int8, copy=True)
    if result.size and (
        not np.any(result == LEFT_GRAIN_LABEL)
        or not np.any(result == RIGHT_GRAIN_LABEL)
    ):
        raise InterfaceCandidateValueError(
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
        :raises InterfaceCandidateValueError: If candidate state is malformed or
            internally inconsistent.
        """
        structured = np.asarray(atoms)
        required_fields = {"name", "x", "y", "z"}
        if (
            structured.ndim != 1
            or structured.dtype.names is None
            or not required_fields.issubset(structured.dtype.names)
        ):
            raise InterfaceCandidateValueError(
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
            raise InterfaceCandidateValueError("coordinate_tolerance must be positive")
        if separation < 0.0:
            raise InterfaceCandidateValueError(
                "interface_separation must be nonnegative")
        if np.any(box[:, 0] >= box[:, 1]):
            raise InterfaceCandidateValueError(
                "InterfaceCandidate box bounds must be strictly ordered"
            )
        if not box[0, 0] < plane < box[0, 1]:
            raise InterfaceCandidateValueError(
                "InterfaceCandidate gb_plane_x must lie strictly inside the x box"
            )
        if left_bounds[0] >= left_bounds[1] or right_bounds[0] >= right_bounds[1]:
            raise InterfaceCandidateValueError(
                "Physical grain bounds must be strictly ordered"
            )
        if (
            left_bounds[0] < box[0, 0] - tolerance
            or right_bounds[1] > box[0, 1] + tolerance
            or left_bounds[1] > plane + tolerance
            or right_bounds[0] < plane - tolerance
            or left_bounds[1] > right_bounds[0] + tolerance
        ):
            raise InterfaceCandidateValueError(
                "Physical grain bounds must lie inside the box on their respective "
                "sides of gb_plane_x without overlapping"
            )
        periodic = _normalize_inplane_periodic(inplane_periodic)
        try:
            topology = normalize_boundary_normal_topology(normal_topology)
        except ValueError as exc:
            raise InterfaceCandidateValueError(str(exc)) from exc

        for axis_index, axis_name in enumerate(("x", "y", "z")):
            coordinates = np.asarray(structured[axis_name], dtype=float)
            if not np.all(np.isfinite(coordinates)):
                raise InterfaceCandidateValueError(
                    "InterfaceCandidate atom coordinates must be finite"
                )
            lower = float(box[axis_index, 0])
            upper = float(box[axis_index, 1])
            if np.any(coordinates < lower - tolerance) or np.any(coordinates >= upper):
                raise InterfaceCandidateValueError(
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
            raise InterfaceCandidateValueError(
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


__all__ = ["InterfaceCandidate"]
