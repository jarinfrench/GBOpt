# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared data types and exceptions for GBMaker construction-state contracts.

Contains the exception hierarchy and immutable construction-state dataclasses used
across the ``gbmaker`` package as later roadmap issues (R04 through R09) decompose
``GBOpt.GBMaker``. Types here describe normalized build configuration, resolved
boundary input, per-grain material identity, orientation state, per-axis strain
accommodation, box-dimension planning, per-grain build requests and results, and the
final assembled bicrystal. No boundary resolution, supercell construction, or geometry
arithmetic belongs here; this module is a pure data-definition layer. Existing
``BoundarySpec``, ``BoundaryEmbedding``, ``RationalBasis``, and crystallography CSL/PQ
representations are referenced, not duplicated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from GBOpt.BoundarySpec import BoundaryEmbedding
from GBOpt.BoundaryTopology import BoundaryNormalTopology


class GBMakerConstructionError(Exception):
    """Base for construction-state contract errors in the ``gbmaker`` package."""


class GBMakerConstructionValueError(GBMakerConstructionError, ValueError):
    """Invalid value supplied to a construction-state contract."""


class GBMakerConstructionTypeError(GBMakerConstructionError, TypeError):
    """Invalid type supplied to a construction-state contract."""


BoundaryMode: TypeAlias = Literal["exact", "prefer_exact", "approximate"]
StrainGrainPolicy: TypeAlias = Literal["both", "left", "right"]
GrainSide: TypeAlias = Literal["left", "right"]

_VALID_BOUNDARY_MODES = frozenset({"exact", "prefer_exact", "approximate"})
_VALID_STRAIN_GRAIN = frozenset({"both", "left", "right"})
_VALID_GRAIN_SIDES = frozenset({"left", "right"})


def _require_positive_float(value: object, name: str) -> float:
    """Normalize a finite positive non-Boolean real scalar.

    :param value: Candidate real value.
    :param name: Field name for diagnostics.
    :return: Python float.
    :raises GBMakerConstructionValueError: If the value is not a finite positive real
        scalar.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise GBMakerConstructionValueError(f"{name} must be a non-Boolean real scalar")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized <= 0:
        raise GBMakerConstructionValueError(f"{name} must be a finite positive value")
    return normalized


def _require_nonnegative_float(value: object, name: str) -> float:
    """Normalize a finite non-negative non-Boolean real scalar.

    :param value: Candidate real value.
    :param name: Field name for diagnostics.
    :return: Python float.
    :raises GBMakerConstructionValueError: If the value is not a finite non-negative
        real scalar.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise GBMakerConstructionValueError(f"{name} must be a non-Boolean real scalar")
    normalized = float(value)
    if not np.isfinite(normalized) or normalized < 0:
        raise GBMakerConstructionValueError(f"{name} must be finite and non-negative")
    return normalized


def _require_positive_int(value: object, name: str) -> int:
    """Normalize a positive non-Boolean integer.

    :param value: Candidate integer value.
    :param name: Field name for diagnostics.
    :return: Python integer.
    :raises GBMakerConstructionValueError: If the value is not a positive non-Boolean
        integer.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise GBMakerConstructionValueError(f"{name} must be a non-Boolean integer")
    normalized = int(value)
    if normalized <= 0:
        raise GBMakerConstructionValueError(f"{name} must be positive")
    return normalized


def _require_bool(value: object, name: str) -> bool:
    """Normalize a Boolean scalar.

    :param value: Candidate Boolean value.
    :param name: Field name for diagnostics.
    :return: Python bool.
    :raises GBMakerConstructionValueError: If the value is not a Boolean scalar.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise GBMakerConstructionValueError(f"{name} must be a bool")
    return bool(value)


def _require_nonempty_string(value: object, name: str) -> str:
    """Normalize a non-empty string field.

    :param value: Candidate string value.
    :param name: Field name for diagnostics.
    :return: Validated string.
    :raises GBMakerConstructionValueError: If the value is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise GBMakerConstructionValueError(f"{name} must be a non-empty string")
    return value


def _require_atom_types(value: object) -> str | tuple[str, ...]:
    """Normalize an atom-type string or tuple of atom-type strings.

    :param value: Candidate atom-type value.
    :return: Validated string or tuple of non-empty strings.
    :raises GBMakerConstructionValueError: If the value is not a non-empty string, or a
        tuple of non-empty strings.
    """
    if isinstance(value, str):
        return _require_nonempty_string(value, "atom_types")
    if isinstance(value, tuple):
        return tuple(
            _require_nonempty_string(item, f"atom_types[{index}]")
            for index, item in enumerate(value)
        )
    raise GBMakerConstructionValueError(
        "atom_types must be a non-empty string or a tuple of non-empty strings"
    )


def _require_repeat_factor(value: object) -> tuple[int, int]:
    """Normalize an in-plane repeat factor to a ``(y, z)`` positive integer pair.

    :param value: Candidate repeat-factor value: a single positive integer applied to
        both axes, or a two-value sequence applied to y and z respectively.
    :return: ``(y, z)`` positive integer repeat pair.
    :raises GBMakerConstructionValueError: If the value is not a positive integer or a
        two-value sequence of positive integers.
    """
    if isinstance(value, (bool, np.bool_)):
        raise GBMakerConstructionValueError("repeat_factor must not be a bool")
    if isinstance(value, Integral):
        repeat = _require_positive_int(value, "repeat_factor")
        return (repeat, repeat)
    try:
        y_value, z_value = value
    except (TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(
            "repeat_factor must be a positive integer or a two-value sequence of "
            "positive integers"
        ) from exc
    return (
        _require_positive_int(y_value, "repeat_factor[0]"),
        _require_positive_int(z_value, "repeat_factor[1]"),
    )


def _require_strain_grain(value: object) -> StrainGrainPolicy:
    """Normalize an in-plane strain-grain policy string.

    :param value: Candidate strain-grain policy.
    :return: One of ``"both"``, ``"left"``, or ``"right"``.
    :raises GBMakerConstructionValueError: If the value is not a supported policy.
    """
    if value not in _VALID_STRAIN_GRAIN:
        expected = ", ".join(sorted(repr(item) for item in _VALID_STRAIN_GRAIN))
        raise GBMakerConstructionValueError(
            f"strain_grain must be one of {expected}; got {value!r}"
        )
    return value


def _require_boundary_mode(value: object) -> BoundaryMode:
    """Normalize a construction-mode string.

    :param value: Candidate construction mode.
    :return: One of ``"exact"``, ``"prefer_exact"``, or ``"approximate"``.
    :raises GBMakerConstructionValueError: If the value is not a supported mode.
    """
    if value not in _VALID_BOUNDARY_MODES:
        expected = ", ".join(sorted(repr(item) for item in _VALID_BOUNDARY_MODES))
        raise GBMakerConstructionValueError(
            f"mode must be one of {expected}; got {value!r}"
        )
    return value


def _require_grain_side(value: object) -> GrainSide:
    """Normalize a grain-side string.

    :param value: Candidate grain side.
    :return: One of ``"left"`` or ``"right"``.
    :raises GBMakerConstructionValueError: If the value is not a supported grain side.
    """
    if value not in _VALID_GRAIN_SIDES:
        expected = ", ".join(sorted(repr(item) for item in _VALID_GRAIN_SIDES))
        raise GBMakerConstructionValueError(
            f"grain_side must be one of {expected}; got {value!r}"
        )
    return value


def _readonly_float_matrix(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return an owned read-only finite float array with a required shape.

    :param value: Array-like input.
    :param shape: Required NumPy shape.
    :param name: Field name for diagnostics.
    :return: Read-only floating-point array.
    :raises GBMakerConstructionValueError: If conversion, shape, or finiteness
        validation fails.
    """
    try:
        arr = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(f"{name} must be a real array-like") from exc
    if arr.shape != shape:
        raise GBMakerConstructionValueError(f"{name} must have shape {shape}; got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise GBMakerConstructionValueError(f"{name} must contain only finite values")
    arr.setflags(write=False)
    return arr


def _readonly_box_dims(value: object, name: str = "box_dims") -> np.ndarray:
    """Return an owned read-only 3 by 2 box-bounds array with ``lo < hi`` per axis.

    :param value: Candidate 3 by 2 ``[lo, hi]`` box-bounds array-like.
    :param name: Field name for diagnostics.
    :return: Read-only 3 by 2 floating-point box-bounds array.
    :raises GBMakerConstructionValueError: If shape, finiteness, or ordering validation
        fails.
    """
    arr = _readonly_float_matrix(value, (3, 2), name)
    if np.any(arr[:, 1] <= arr[:, 0]):
        raise GBMakerConstructionValueError(f"{name} upper bounds must exceed lower bounds")
    return arr


@dataclass(frozen=True, slots=True)
class MaterialState:
    """Crystal identity used to build unit cells, independent of orientation.

    :param a0: Lattice parameter (Angstroms).
    :param structure: Crystal structure name. Supported values are ``"fcc"``,
        ``"bcc"``, ``"sc"``, ``"diamond"``, ``"fluorite"``, ``"rocksalt"``, and
        ``"zincblende"``.
    :param atom_types: Atom type string or tuple of atom type strings accepted by
        ``UnitCell``.
    """

    a0: float
    structure: str
    atom_types: str | tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and freeze material-identity fields."""
        object.__setattr__(self, "a0", _require_positive_float(self.a0, "a0"))
        object.__setattr__(
            self, "structure", _require_nonempty_string(self.structure, "structure")
        )
        object.__setattr__(self, "atom_types", _require_atom_types(self.atom_types))


@dataclass(frozen=True, slots=True)
class GBBuildConfig:
    """Normalized top-level ``GBMaker`` construction configuration.

    :param material: Crystal identity shared by both grains.
    :param gb_thickness: Width of the grain-boundary region (Angstroms).
    :param repeat_factor: Positive in-plane repeat counts ``(y, z)``.
    :param x_dim_min: Minimum size of one grain along x (Angstroms).
    :param vacuum: Vacuum thickness around the grains along x (Angstroms).
    :param interaction_distance: Maximum atom interaction distance (Angstroms).
    :param gb_id: Grain-boundary identifier.
    :param epsilon: Numerical tolerance used for geometric comparisons.
    :param mismatch_tol: Maximum relative in-plane mismatch permitted by the
        commensurate-repeat search, or ``None`` to disable mismatch accommodation.
    :param mismatch_max_cells: Maximum repeat count searched per axis when mismatch
        accommodation is active.
    :param strain_grain: In-plane strain policy used when mismatch accommodation is
        active. Ignored when ``mismatch_tol`` is ``None``.
    """

    material: MaterialState
    gb_thickness: float
    repeat_factor: tuple[int, int] = (2, 2)
    x_dim_min: float = 50.0
    vacuum: float = 10.0
    interaction_distance: float = 15.0
    gb_id: int = 1
    epsilon: float = 1e-10
    mismatch_tol: float | None = None
    mismatch_max_cells: int = 50
    strain_grain: StrainGrainPolicy = "both"

    def __post_init__(self) -> None:
        """Validate and freeze the normalized build configuration."""
        if not isinstance(self.material, MaterialState):
            raise GBMakerConstructionTypeError("material must be a MaterialState")

        object.__setattr__(
            self, "gb_thickness", _require_nonnegative_float(self.gb_thickness, "gb_thickness")
        )
        object.__setattr__(
            self, "repeat_factor", _require_repeat_factor(self.repeat_factor)
        )
        object.__setattr__(
            self, "x_dim_min", _require_positive_float(self.x_dim_min, "x_dim_min")
        )
        object.__setattr__(self, "vacuum", _require_nonnegative_float(self.vacuum, "vacuum"))
        object.__setattr__(
            self,
            "interaction_distance",
            _require_positive_float(self.interaction_distance, "interaction_distance"),
        )
        object.__setattr__(self, "gb_id", _require_positive_int(self.gb_id, "gb_id"))
        object.__setattr__(self, "epsilon", _require_positive_float(self.epsilon, "epsilon"))
        object.__setattr__(
            self,
            "mismatch_tol",
            None
            if self.mismatch_tol is None
            else _require_positive_float(self.mismatch_tol, "mismatch_tol"),
        )
        object.__setattr__(
            self,
            "mismatch_max_cells",
            _require_positive_int(self.mismatch_max_cells, "mismatch_max_cells"),
        )
        object.__setattr__(self, "strain_grain", _require_strain_grain(self.strain_grain))


@dataclass(frozen=True, slots=True)
class ResolvedBoundaryInput:
    """Boundary embedding resolved from a boundary spec under a construction mode.

    :param embedding: Canonical ``BoundaryEmbedding`` produced by the boundary-spec
        adapter for the requested mode.
    :param mode: Construction mode used to resolve the embedding: ``"exact"``,
        ``"prefer_exact"``, or ``"approximate"``.
    :param max_primitive_area_index: Exact-cell primitive-reconstruction area-index
        limit applied while resolving the embedding.
    :param max_pq_determinant: Exact-cell P/Q determinant limit applied while resolving
        the embedding.
    """

    embedding: BoundaryEmbedding
    mode: BoundaryMode
    max_primitive_area_index: int
    max_pq_determinant: int

    def __post_init__(self) -> None:
        """Validate and freeze resolved-boundary-input fields."""
        if not isinstance(self.embedding, BoundaryEmbedding):
            raise GBMakerConstructionTypeError("embedding must be a BoundaryEmbedding")
        object.__setattr__(self, "mode", _require_boundary_mode(self.mode))
        object.__setattr__(
            self,
            "max_primitive_area_index",
            _require_positive_int(
                self.max_primitive_area_index, "max_primitive_area_index"
            ),
        )
        object.__setattr__(
            self,
            "max_pq_determinant",
            _require_positive_int(self.max_pq_determinant, "max_pq_determinant"),
        )


@dataclass(frozen=True, slots=True)
class OrientationState:
    """Per-grain orientation state derived from a resolved boundary embedding.

    :param embedding: Canonical boundary embedding carrying left/right rotations and,
        for exact construction paths, integer P/Q orientation matrices.
    :param inplane_periodic: Per-axis in-plane periodicity flags ``(y, z)``.
    :param normal_topology: Physical topology along the grain-boundary normal.
    """

    embedding: BoundaryEmbedding
    inplane_periodic: tuple[bool, bool] = (True, True)
    normal_topology: BoundaryNormalTopology = BoundaryNormalTopology.PERIODIC_BICRYSTAL

    def __post_init__(self) -> None:
        """Validate and freeze orientation-state fields."""
        if not isinstance(self.embedding, BoundaryEmbedding):
            raise GBMakerConstructionTypeError("embedding must be a BoundaryEmbedding")
        try:
            y_periodic, z_periodic = self.inplane_periodic
        except (TypeError, ValueError) as exc:
            raise GBMakerConstructionValueError(
                "inplane_periodic must be a two-value sequence of bools"
            ) from exc
        object.__setattr__(
            self,
            "inplane_periodic",
            (
                _require_bool(y_periodic, "inplane_periodic[0]"),
                _require_bool(z_periodic, "inplane_periodic[1]"),
            ),
        )
        if not isinstance(self.normal_topology, BoundaryNormalTopology):
            raise GBMakerConstructionTypeError(
                "normal_topology must be a BoundaryNormalTopology"
            )


@dataclass(frozen=True, slots=True)
class AxisAccommodation:
    """Integer repeat pair and lab-axis scale factors for one in-plane axis.

    Produced by the commensurate-period search for a single in-plane axis, y or z, when
    mismatch accommodation is requested.

    :param left_repeats: Number of left-grain unit-cell repeats along this axis.
    :param right_repeats: Number of right-grain unit-cell repeats along this axis.
    :param left_unstrained_length: Unstrained left-grain slab length along this axis,
        equal to ``left_repeats`` times the left-grain period (Angstroms).
    :param right_unstrained_length: Unstrained right-grain slab length along this axis,
        equal to ``right_repeats`` times the right-grain period (Angstroms).
    :param box_length: Shared simulation box length along this axis (Angstroms). Chosen
        from the unstrained lengths according to the ``strain_grain`` policy.
    :param left_scale: Factor by which left-grain atom coordinates are scaled along this
        axis to fit the shared box, equal to ``box_length / left_unstrained_length``.
    :param right_scale: Factor by which right-grain atom coordinates are scaled along
        this axis to fit the shared box, equal to ``box_length /
        right_unstrained_length``.
    :param mismatch: Relative mismatch before scaling, computed as ``abs(l1 - l2) /
        max(l1, l2)``.
    """

    left_repeats: int
    right_repeats: int
    left_unstrained_length: float
    right_unstrained_length: float
    box_length: float
    left_scale: float
    right_scale: float
    mismatch: float

    def __post_init__(self) -> None:
        """Validate and freeze per-axis accommodation fields."""
        object.__setattr__(
            self, "left_repeats", _require_positive_int(self.left_repeats, "left_repeats")
        )
        object.__setattr__(
            self,
            "right_repeats",
            _require_positive_int(self.right_repeats, "right_repeats"),
        )
        object.__setattr__(
            self,
            "left_unstrained_length",
            _require_positive_float(
                self.left_unstrained_length, "left_unstrained_length"
            ),
        )
        object.__setattr__(
            self,
            "right_unstrained_length",
            _require_positive_float(
                self.right_unstrained_length, "right_unstrained_length"
            ),
        )
        object.__setattr__(
            self, "box_length", _require_positive_float(self.box_length, "box_length")
        )
        object.__setattr__(
            self, "left_scale", _require_positive_float(self.left_scale, "left_scale")
        )
        object.__setattr__(
            self, "right_scale", _require_positive_float(self.right_scale, "right_scale")
        )
        object.__setattr__(
            self, "mismatch", _require_nonnegative_float(self.mismatch, "mismatch")
        )

    def resized(self, factor: int) -> AxisAccommodation:
        """Return this accommodation with repeat counts and lengths multiplied.

        The repeat counts, unstrained lengths, and shared box length are multiplied by
        ``factor``. Coordinate scale factors and mismatch are unchanged because the
        relative strain state is unchanged.

        :param factor: Positive integer multiplier for the repeat counts and axis
            lengths.
        :return: Resized strain accommodation for the same axis.
        :raises GBMakerConstructionValueError: If ``factor`` is boolean, non-integral,
            or less than one.
        """
        factor = _require_positive_int(factor, "factor")
        return AxisAccommodation(
            left_repeats=self.left_repeats * factor,
            right_repeats=self.right_repeats * factor,
            left_unstrained_length=self.left_unstrained_length * factor,
            right_unstrained_length=self.right_unstrained_length * factor,
            box_length=self.box_length * factor,
            left_scale=self.left_scale,
            right_scale=self.right_scale,
            mismatch=self.mismatch,
        )


@dataclass(frozen=True, slots=True)
class DimensionPlan:
    """Planned simulation-box dimensions and per-axis in-plane accommodation.

    :param box_dims: Read-only 3 by 2 array of ``[lo, hi]`` box bounds (Angstroms), rows
        ordered x, y, z.
    :param vacuum_thickness: Vacuum thickness applied along x (Angstroms).
    :param normal_topology: Physical topology along the grain-boundary normal.
    :param periodic_spacing: Read-only mapping of periodic-distance metadata keyed by
        axis or direction name (Angstroms).
    :param accommodation: Read-only mapping from in-plane axis name (``"y"`` or
        ``"z"``) to its commensurate-repeat accommodation. Empty when mismatch
        accommodation is inactive.
    """

    box_dims: NDArray[np.floating] = field(repr=False)
    vacuum_thickness: float
    normal_topology: BoundaryNormalTopology
    periodic_spacing: Mapping[str, float] = field(default_factory=dict, repr=False)
    accommodation: Mapping[str, AxisAccommodation] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        """Validate and freeze dimension-plan fields."""
        object.__setattr__(self, "box_dims", _readonly_box_dims(self.box_dims))
        object.__setattr__(
            self,
            "vacuum_thickness",
            _require_nonnegative_float(self.vacuum_thickness, "vacuum_thickness"),
        )
        if not isinstance(self.normal_topology, BoundaryNormalTopology):
            raise GBMakerConstructionTypeError(
                "normal_topology must be a BoundaryNormalTopology"
            )

        if not isinstance(self.periodic_spacing, Mapping):
            raise GBMakerConstructionTypeError("periodic_spacing must be a mapping")
        spacing = {
            _require_nonempty_string(key, "periodic_spacing key"): _require_positive_float(
                value, f"periodic_spacing[{key!r}]"
            )
            for key, value in self.periodic_spacing.items()
        }
        object.__setattr__(self, "periodic_spacing", MappingProxyType(spacing))

        if not isinstance(self.accommodation, Mapping):
            raise GBMakerConstructionTypeError("accommodation must be a mapping")
        accommodation: dict[str, AxisAccommodation] = {}
        for axis, value in self.accommodation.items():
            if axis not in ("y", "z"):
                raise GBMakerConstructionValueError(
                    f"accommodation keys must be 'y' or 'z'; got {axis!r}"
                )
            if not isinstance(value, AxisAccommodation):
                raise GBMakerConstructionTypeError(
                    f"accommodation[{axis!r}] must be an AxisAccommodation"
                )
            accommodation[axis] = value
        object.__setattr__(self, "accommodation", MappingProxyType(accommodation))


@dataclass(frozen=True, slots=True)
class GrainBuildRequest:
    """Inputs needed to build one grain's atoms for a bicrystal.

    :param material: Crystal identity shared by both grains.
    :param orientation: 3 by 3 orientation matrix for this grain: an exact integer P/Q
        matrix on exact construction paths, or a floating-point rotation matrix on
        approximate paths.
    :param grain_side: Which grain this request builds, ``"left"`` or ``"right"``.
    :param x_length: Equalized x-slab thickness for this grain (Angstroms).
    :param box_dims: Shared simulation-box bounds this grain's atoms must fit within.
    :param exact: Whether ``orientation`` is an exact integer matrix rather than a
        floating-point rotation.
    """

    material: MaterialState
    orientation: NDArray = field(repr=False)
    grain_side: GrainSide
    x_length: float
    box_dims: NDArray[np.floating] = field(repr=False)
    exact: bool = False

    def __post_init__(self) -> None:
        """Validate and freeze per-grain build-request fields."""
        if not isinstance(self.material, MaterialState):
            raise GBMakerConstructionTypeError("material must be a MaterialState")

        object.__setattr__(self, "exact", _require_bool(self.exact, "exact"))
        dtype = int if self.exact else float
        try:
            orientation = np.array(self.orientation, dtype=dtype, copy=True)
        except (TypeError, ValueError) as exc:
            raise GBMakerConstructionValueError(
                "orientation must be a 3 by 3 real array-like"
            ) from exc
        if orientation.shape != (3, 3):
            raise GBMakerConstructionValueError(
                f"orientation must have shape (3, 3); got {orientation.shape}"
            )
        orientation.setflags(write=False)
        object.__setattr__(self, "orientation", orientation)

        object.__setattr__(self, "grain_side", _require_grain_side(self.grain_side))
        object.__setattr__(
            self, "x_length", _require_positive_float(self.x_length, "x_length")
        )
        object.__setattr__(self, "box_dims", _readonly_box_dims(self.box_dims))


@dataclass(frozen=True, slots=True)
class GrainBuildResult:
    """Generated grain atoms with conventional-cell origin metadata.

    Carries the result of one grain's build path through trimming, clipping, wrapping,
    and deduplication operations that must preserve complete conventional-cell origins.

    ``atoms`` and ``origin_ids`` are parallel one-dimensional arrays. Each atom has one
    origin identifier, and atoms sharing an origin identifier belong to the same
    generated conventional-cell origin group. ``basis_size`` gives the expected number
    of atoms in each complete origin group.

    The dataclass is frozen to prevent rebinding the result fields, but the underlying
    NumPy arrays remain mutable because the generated atom arrays are later assigned
    into bicrystal-assembly state and may be modified by downstream geometry
    operations.

    :param grain_side: Which grain this result was built for, ``"left"`` or
        ``"right"``.
    :param atoms: Structured atom array for the generated grain after build-path
        selection and filtering.
    :param origin_ids: Integer array parallel to ``atoms``. Each value identifies the
        generated conventional-cell origin that produced the corresponding atom.
    :param basis_size: Number of atoms generated per conventional-cell origin.
        Complete-origin filtering assumes retained atom groups have this size.
    """

    grain_side: GrainSide
    atoms: np.ndarray = field(repr=False)
    origin_ids: np.ndarray = field(repr=False)
    basis_size: int

    def __post_init__(self) -> None:
        """Validate grain-build-result shape and identity fields."""
        object.__setattr__(self, "grain_side", _require_grain_side(self.grain_side))
        atoms = np.asarray(self.atoms)
        if atoms.ndim != 1:
            raise GBMakerConstructionValueError("atoms must be a one-dimensional array")
        origin_ids = np.asarray(self.origin_ids)
        if origin_ids.shape != atoms.shape:
            raise GBMakerConstructionValueError(
                "origin_ids must be parallel to atoms"
            )
        object.__setattr__(
            self, "basis_size", _require_positive_int(self.basis_size, "basis_size")
        )


@dataclass(frozen=True, slots=True)
class BicrystalResult:
    """Fully assembled bicrystal produced by grain construction and merging.

    :param atoms: Structured atom array for the combined left and right grains.
    :param box_dims: Read-only 3 by 2 simulation-box bounds (Angstroms).
    :param normal_topology: Physical topology along the grain-boundary normal.
    :param gb_id: Grain-boundary identifier carried onto the assembled result.
    """

    atoms: np.ndarray = field(repr=False)
    box_dims: NDArray[np.floating] = field(repr=False)
    normal_topology: BoundaryNormalTopology
    gb_id: int

    def __post_init__(self) -> None:
        """Validate and freeze bicrystal-result fields."""
        atoms = np.asarray(self.atoms)
        if atoms.ndim != 1:
            raise GBMakerConstructionValueError("atoms must be a one-dimensional array")
        object.__setattr__(self, "box_dims", _readonly_box_dims(self.box_dims))
        if not isinstance(self.normal_topology, BoundaryNormalTopology):
            raise GBMakerConstructionTypeError(
                "normal_topology must be a BoundaryNormalTopology"
            )
        object.__setattr__(self, "gb_id", _require_positive_int(self.gb_id, "gb_id"))


__all__ = [
    "GBMakerConstructionError",
    "GBMakerConstructionValueError",
    "GBMakerConstructionTypeError",
    "BoundaryMode",
    "StrainGrainPolicy",
    "GrainSide",
    "MaterialState",
    "GBBuildConfig",
    "ResolvedBoundaryInput",
    "OrientationState",
    "AxisAccommodation",
    "DimensionPlan",
    "GrainBuildRequest",
    "GrainBuildResult",
    "BicrystalResult",
]
