# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Generation-time bicrystal state, topology descriptors, and serialization.

The state model in this module is deliberately limited to clean boundary generation.
It preserves constructed atom and grain identity, simulation-box bounds, boundary
conditions, active grain-boundary interfaces, external surfaces and spatial regions,
relative translation, termination identifiers, and deterministic construction
provenance. It is not an optimizer checkpoint or an atom-lineage framework.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

import numpy as np


LEFT_GRAIN_ID = 0
RIGHT_GRAIN_ID = 1
STATE_SCHEMA_VERSION = 1
TRANSLATION_CONVENTION = (
    "relative_translation_lab is the cumulative displacement of the right grain "
    "relative to the left grain in lab-frame Angstroms; positive components move "
    "the right grain along the corresponding positive lab axis"
)

BoundaryCondition: TypeAlias = Literal["periodic", "fixed"]
BicrystalTopology: TypeAlias = Literal[
    "periodic_bicrystal",
    "single_interface_slab",
]
InterfaceLocation: TypeAlias = Literal["interior", "periodic_boundary"]
RegionKind: TypeAlias = Literal["vacuum", "fixed", "buffer"]

_VALID_BOUNDARY_CONDITIONS = frozenset({"periodic", "fixed"})
_VALID_TOPOLOGIES = frozenset({"periodic_bicrystal", "single_interface_slab"})
_VALID_INTERFACE_LOCATIONS = frozenset({"interior", "periodic_boundary"})
_VALID_REGION_KINDS = frozenset({"vacuum", "fixed", "buffer"})
_REQUIRED_ATOM_FIELDS = ("name", "x", "y", "z")
_COORDINATE_FIELDS = ("x", "y", "z")
# Legacy float-path rotations and periodic wrapping can leave coordinates a few
# 1e-7 Angstrom beyond a nominal box face. Treat sub-microangstrom excursions
# as numerical roundoff while preserving the original coordinates unchanged.
_COORDINATE_TOLERANCE = 1.0e-6


class BicrystalStateError(Exception):
    """Base class for generation-time bicrystal state failures."""


class BicrystalStateTypeError(BicrystalStateError, TypeError):
    """Raised when a state field has an invalid type."""


class BicrystalStateValueError(BicrystalStateError, ValueError):
    """Raised when a state field has an invalid value or inconsistent topology."""


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise BicrystalStateTypeError(f"{name} must be a string; got {value!r}.")
    if not value:
        raise BicrystalStateValueError(f"{name} must not be empty.")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise BicrystalStateTypeError(f"{name} must be a finite float; got {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BicrystalStateTypeError(
            f"{name} must be a finite float; got {value!r}."
        ) from exc
    if not math.isfinite(result):
        raise BicrystalStateValueError(
            f"{name} must be finite; got {result!r}."
        )
    return result


def _axis(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise BicrystalStateTypeError(f"{name} must be 0, 1, or 2; got {value!r}.")
    result = int(value)
    if result not in (0, 1, 2):
        raise BicrystalStateValueError(
            f"{name} must be 0, 1, or 2; got {result!r}."
        )
    return result


def _grain_id(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise BicrystalStateTypeError(
            f"{name} must be a grain ID; got {value!r}."
        )
    result = int(value)
    if result not in (LEFT_GRAIN_ID, RIGHT_GRAIN_ID):
        raise BicrystalStateValueError(
            f"{name} must be {LEFT_GRAIN_ID} or {RIGHT_GRAIN_ID}; got {result}."
        )
    return result


def _grain_id_tuple(values: object, name: str, *, allow_empty: bool) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise BicrystalStateTypeError(f"{name} must be a sequence of grain IDs.")
    try:
        normalized = tuple(
            _grain_id(value, f"{name}[{index}]")
            for index, value in enumerate(values)  # type: ignore[arg-type]
        )
    except TypeError as exc:
        raise BicrystalStateTypeError(
            f"{name} must be a sequence of grain IDs."
        ) from exc
    if not allow_empty and not normalized:
        raise BicrystalStateValueError(f"{name} must not be empty.")
    if len(set(normalized)) != len(normalized):
        raise BicrystalStateValueError(f"{name} must not contain duplicates.")
    return normalized


def _normal_vector(
    values: object,
    name: str,
    *,
    axis: int,
) -> tuple[float, float, float]:
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BicrystalStateTypeError(
            f"{name} must be a three-component vector."
        ) from exc
    if len(raw) != 3:
        raise BicrystalStateValueError(
            f"{name} must have three components; got {len(raw)}."
        )
    normal = tuple(_finite_float(value, f"{name}[{index}]") for index, value in enumerate(raw))
    expected_nonzero = normal[axis]
    if not math.isclose(abs(expected_nonzero), 1.0, abs_tol=1.0e-12, rel_tol=0.0):
        raise BicrystalStateValueError(
            f"{name} must be a unit normal aligned with axis {axis}; got {normal}."
        )
    for index, component in enumerate(normal):
        if index != axis and not math.isclose(
            component,
            0.0,
            abs_tol=1.0e-12,
            rel_tol=0.0,
        ):
            raise BicrystalStateValueError(
                f"{name} must be aligned with axis {axis}; got {normal}."
            )
    return normal  # type: ignore[return-value]


def _freeze_json(value: Any, name: str = "metadata") -> Any:
    """Normalize a JSON-compatible value and freeze nested containers."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _freeze_json(value.item(), name)
    if isinstance(value, np.ndarray):
        return _freeze_json(value.tolist(), name)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BicrystalStateValueError(
                f"{name} contains a non-finite float: {value!r}."
            )
        return float(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise BicrystalStateTypeError(
                    f"{name} keys must be strings; got {key!r}."
                )
            normalized[key] = _freeze_json(value[key], f"{name}.{key}")
        return MappingProxyType(normalized)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise BicrystalStateTypeError(
        f"{name} must contain only JSON-compatible values; got {type(value).__name__}."
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _thaw_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _frozen_array(values: object, *, dtype: Any | None = None) -> np.ndarray:
    try:
        array = np.array(values, dtype=dtype, copy=True)
    except (TypeError, ValueError) as exc:
        raise BicrystalStateTypeError(
            f"Value cannot be converted to a NumPy array: {exc}"
        ) from exc
    array.setflags(write=False)
    return array


def _exact_integer_vector(values: object, name: str, length: int) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.shape != (length,):
        raise BicrystalStateValueError(
            f"{name} must have shape ({length},); got {raw.shape}."
        )
    normalized = np.empty(length, dtype=np.int64)
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise BicrystalStateTypeError(
                f"{name}[{index}] must be an integer; got {value!r}."
            )
        normalized[index] = int(value)
    normalized.setflags(write=False)
    return normalized


@dataclass(frozen=True, slots=True)
class InterfaceDescriptor:
    """One active physical grain-boundary interface.

    ``minus_grain_id`` and ``plus_grain_id`` are ordered along ``normal_lab``. For a
    periodic x-boundary, ``position`` is the lower box face and
    ``periodic_partner_position`` is the upper face. Crossing from the minus grain at
    the upper face to the plus grain at the lower face follows ``normal_lab``.
    """

    interface_id: str
    axis: int
    location: InterfaceLocation
    position: float
    minus_grain_id: int
    plus_grain_id: int
    normal_lab: tuple[float, float, float]
    periodic_partner_position: float | None = None

    def __post_init__(self) -> None:
        interface_id = _nonempty_string(self.interface_id, "interface_id")
        axis = _axis(self.axis, "axis")
        if self.location not in _VALID_INTERFACE_LOCATIONS:
            raise BicrystalStateValueError(
                "location must be 'interior' or 'periodic_boundary'; "
                f"got {self.location!r}."
            )
        position = _finite_float(self.position, "position")
        minus_grain_id = _grain_id(self.minus_grain_id, "minus_grain_id")
        plus_grain_id = _grain_id(self.plus_grain_id, "plus_grain_id")
        if minus_grain_id == plus_grain_id:
            raise BicrystalStateValueError(
                "An interface must separate two different grains."
            )
        normal_lab = _normal_vector(self.normal_lab, "normal_lab", axis=axis)

        partner = self.periodic_partner_position
        if self.location == "periodic_boundary":
            if partner is None:
                raise BicrystalStateValueError(
                    "periodic_boundary interfaces require periodic_partner_position."
                )
            partner = _finite_float(partner, "periodic_partner_position")
            if math.isclose(partner, position, abs_tol=0.0, rel_tol=0.0):
                raise BicrystalStateValueError(
                    "Periodic interface positions must be distinct."
                )
        elif partner is not None:
            raise BicrystalStateValueError(
                "Interior interfaces must not define periodic_partner_position."
            )

        object.__setattr__(self, "interface_id", interface_id)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "minus_grain_id", minus_grain_id)
        object.__setattr__(self, "plus_grain_id", plus_grain_id)
        object.__setattr__(self, "normal_lab", normal_lab)
        object.__setattr__(self, "periodic_partner_position", partner)


@dataclass(frozen=True, slots=True)
class SurfaceDescriptor:
    """One external surface bounding a nonperiodic part of the bicrystal."""

    surface_id: str
    axis: int
    position: float
    outward_normal_lab: tuple[float, float, float]
    grain_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        surface_id = _nonempty_string(self.surface_id, "surface_id")
        axis = _axis(self.axis, "axis")
        position = _finite_float(self.position, "position")
        normal = _normal_vector(
            self.outward_normal_lab,
            "outward_normal_lab",
            axis=axis,
        )
        grain_ids = _grain_id_tuple(
            self.grain_ids,
            "grain_ids",
            allow_empty=False,
        )
        object.__setattr__(self, "surface_id", surface_id)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "outward_normal_lab", normal)
        object.__setattr__(self, "grain_ids", grain_ids)


@dataclass(frozen=True, slots=True)
class RegionDescriptor:
    """One intentional vacuum, fixed, or buffer interval along a box axis."""

    region_id: str
    kind: RegionKind
    axis: int
    lower: float
    upper: float
    grain_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        region_id = _nonempty_string(self.region_id, "region_id")
        if self.kind not in _VALID_REGION_KINDS:
            raise BicrystalStateValueError(
                f"kind must be one of {sorted(_VALID_REGION_KINDS)}; got {self.kind!r}."
            )
        axis = _axis(self.axis, "axis")
        lower = _finite_float(self.lower, "lower")
        upper = _finite_float(self.upper, "upper")
        if upper <= lower:
            raise BicrystalStateValueError(
                f"Region upper bound must exceed lower bound; got [{lower}, {upper}]."
            )
        grain_ids = _grain_id_tuple(
            self.grain_ids,
            "grain_ids",
            allow_empty=True,
        )
        object.__setattr__(self, "region_id", region_id)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "grain_ids", grain_ids)


@dataclass(frozen=True, slots=True, eq=False)
class BicrystalState:
    """Immutable generation-time state for one constructed bicrystal seed.

    Atom IDs are stable one-based identifiers. Grain ID ``0`` denotes the left grain
    and grain ID ``1`` denotes the right grain. Relative translation always describes
    the right grain's cumulative lab-frame displacement relative to the left grain.
    """

    atoms: np.ndarray
    box_dims: np.ndarray
    topology: BicrystalTopology
    boundary_conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    atom_ids: np.ndarray
    grain_ids: np.ndarray
    interfaces: tuple[InterfaceDescriptor, ...]
    external_surfaces: tuple[SurfaceDescriptor, ...] = ()
    vacuum_regions: tuple[RegionDescriptor, ...] = ()
    fixed_regions: tuple[RegionDescriptor, ...] = ()
    buffer_regions: tuple[RegionDescriptor, ...] = ()
    relative_translation_lab: tuple[float, float, float] = (0.0, 0.0, 0.0)
    termination_ids: tuple[int, int] | None = (0, 0)
    moving_grain_id: int = RIGHT_GRAIN_ID
    translation_convention: str = TRANSLATION_CONVENTION
    metadata: Mapping[str, object] = MappingProxyType({})
    schema_version: int = STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        atoms = self._validated_atoms(self.atoms)
        box_dims = self._validated_box(self.box_dims)
        atom_count = len(atoms)
        atom_ids = _exact_integer_vector(self.atom_ids, "atom_ids", atom_count)
        grain_ids = _exact_integer_vector(self.grain_ids, "grain_ids", atom_count)

        if atom_count == 0:
            raise BicrystalStateValueError("BicrystalState.atoms must not be empty.")
        if np.any(atom_ids <= 0):
            raise BicrystalStateValueError("atom_ids must be positive one-based IDs.")
        if len(np.unique(atom_ids)) != atom_count:
            raise BicrystalStateValueError("atom_ids must be unique.")
        invalid_grains = set(int(value) for value in np.unique(grain_ids)) - {
            LEFT_GRAIN_ID,
            RIGHT_GRAIN_ID,
        }
        if invalid_grains:
            raise BicrystalStateValueError(
                f"grain_ids contain unsupported values: {sorted(invalid_grains)}."
            )
        if set(int(value) for value in np.unique(grain_ids)) != {
            LEFT_GRAIN_ID,
            RIGHT_GRAIN_ID,
        }:
            raise BicrystalStateValueError(
                "A bicrystal state must contain atoms from both left and right grains."
            )

        if self.topology not in _VALID_TOPOLOGIES:
            raise BicrystalStateValueError(
                f"topology must be one of {sorted(_VALID_TOPOLOGIES)}; "
                f"got {self.topology!r}."
            )
        boundary_conditions = tuple(self.boundary_conditions)
        if len(boundary_conditions) != 3:
            raise BicrystalStateValueError(
                "boundary_conditions must contain exactly three axis entries."
            )
        for axis, condition in enumerate(boundary_conditions):
            if condition not in _VALID_BOUNDARY_CONDITIONS:
                raise BicrystalStateValueError(
                    f"boundary_conditions[{axis}] must be one of "
                    f"{sorted(_VALID_BOUNDARY_CONDITIONS)}; got {condition!r}."
                )

        interfaces = tuple(self.interfaces)
        external_surfaces = tuple(self.external_surfaces)
        vacuum_regions = tuple(self.vacuum_regions)
        fixed_regions = tuple(self.fixed_regions)
        buffer_regions = tuple(self.buffer_regions)
        self._validate_unique_descriptor_ids(
            interfaces,
            external_surfaces,
            vacuum_regions,
            fixed_regions,
            buffer_regions,
        )
        self._validate_descriptor_bounds(
            box_dims,
            interfaces,
            external_surfaces,
            vacuum_regions,
            fixed_regions,
            buffer_regions,
        )
        self._validate_topology(
            self.topology,
            boundary_conditions,
            interfaces,
            external_surfaces,
            vacuum_regions,
        )
        self._validate_atom_bounds(atoms, box_dims)

        translation = tuple(
            _finite_float(value, f"relative_translation_lab[{index}]")
            for index, value in enumerate(self.relative_translation_lab)
        )
        if len(translation) != 3:
            raise BicrystalStateValueError(
                "relative_translation_lab must contain exactly three components."
            )

        termination_ids = self.termination_ids
        if termination_ids is not None:
            raw_termination = tuple(termination_ids)
            if len(raw_termination) != 2:
                raise BicrystalStateValueError(
                    "termination_ids must contain exactly two entries."
                )
            normalized_termination: list[int] = []
            for index, value in enumerate(raw_termination):
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value,
                    (int, np.integer),
                ):
                    raise BicrystalStateTypeError(
                        f"termination_ids[{index}] must be an integer."
                    )
                integer = int(value)
                if integer < 0:
                    raise BicrystalStateValueError(
                        "termination_ids must be nonnegative."
                    )
                normalized_termination.append(integer)
            termination_ids = tuple(normalized_termination)  # type: ignore[assignment]

        moving_grain_id = _grain_id(self.moving_grain_id, "moving_grain_id")
        if moving_grain_id != RIGHT_GRAIN_ID:
            raise BicrystalStateValueError(
                "Generation-time relative translation is defined for the right grain "
                f"(grain ID {RIGHT_GRAIN_ID})."
            )
        translation_convention = _nonempty_string(
            self.translation_convention,
            "translation_convention",
        )
        if translation_convention != TRANSLATION_CONVENTION:
            raise BicrystalStateValueError(
                "translation_convention does not match the generation-time state "
                "contract."
            )
        if isinstance(self.schema_version, (bool, np.bool_)) or not isinstance(
            self.schema_version,
            (int, np.integer),
        ):
            raise BicrystalStateTypeError("schema_version must be an integer.")
        schema_version = int(self.schema_version)
        if schema_version != STATE_SCHEMA_VERSION:
            raise BicrystalStateValueError(
                f"Unsupported schema_version={schema_version}; expected "
                f"{STATE_SCHEMA_VERSION}."
            )

        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise BicrystalStateTypeError("metadata must be a mapping.")

        object.__setattr__(self, "atoms", atoms)
        object.__setattr__(self, "box_dims", box_dims)
        object.__setattr__(self, "boundary_conditions", boundary_conditions)
        object.__setattr__(self, "atom_ids", atom_ids)
        object.__setattr__(self, "grain_ids", grain_ids)
        object.__setattr__(self, "interfaces", interfaces)
        object.__setattr__(self, "external_surfaces", external_surfaces)
        object.__setattr__(self, "vacuum_regions", vacuum_regions)
        object.__setattr__(self, "fixed_regions", fixed_regions)
        object.__setattr__(self, "buffer_regions", buffer_regions)
        object.__setattr__(self, "relative_translation_lab", translation)
        object.__setattr__(self, "termination_ids", termination_ids)
        object.__setattr__(self, "moving_grain_id", moving_grain_id)
        object.__setattr__(self, "translation_convention", translation_convention)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "schema_version", schema_version)

    @staticmethod
    def _validated_atoms(values: object) -> np.ndarray:
        atoms = _frozen_array(values)
        if atoms.ndim != 1:
            raise BicrystalStateValueError(
                f"atoms must be a one-dimensional structured array; got {atoms.shape}."
            )
        names = atoms.dtype.names
        if names is None:
            raise BicrystalStateTypeError("atoms must be a structured NumPy array.")
        missing = [field for field in _REQUIRED_ATOM_FIELDS if field not in names]
        if missing:
            raise BicrystalStateValueError(
                "atoms is missing required field(s): " + ", ".join(missing)
            )
        for coordinate in _COORDINATE_FIELDS:
            try:
                values_float = np.asarray(atoms[coordinate], dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise BicrystalStateTypeError(
                    f"atoms[{coordinate!r}] must be numeric."
                ) from exc
            if not np.all(np.isfinite(values_float)):
                raise BicrystalStateValueError(
                    f"atoms[{coordinate!r}] contains non-finite coordinates."
                )
        return atoms

    @staticmethod
    def _validated_box(values: object) -> np.ndarray:
        box = _frozen_array(values, dtype=np.float64)
        if box.shape != (3, 2):
            raise BicrystalStateValueError(
                f"box_dims must have shape (3, 2); got {box.shape}."
            )
        if not np.all(np.isfinite(box)):
            raise BicrystalStateValueError("box_dims contains non-finite values.")
        if np.any(box[:, 1] <= box[:, 0]):
            raise BicrystalStateValueError(
                "Every box upper bound must exceed its lower bound."
            )
        return box

    @staticmethod
    def _validate_atom_bounds(atoms: np.ndarray, box: np.ndarray) -> None:
        for axis, coordinate in enumerate(_COORDINATE_FIELDS):
            values = np.asarray(atoms[coordinate], dtype=np.float64)
            lower, upper = box[axis]
            outside = (
                (values < lower - _COORDINATE_TOLERANCE)
                | (values > upper + _COORDINATE_TOLERANCE)
            )
            if np.any(outside):
                raise BicrystalStateValueError(
                    f"atoms[{coordinate!r}] extends outside box axis {axis}: "
                    f"[{float(np.min(values))}, {float(np.max(values))}] versus "
                    f"[{lower}, {upper}]."
                )

    @staticmethod
    def _validate_unique_descriptor_ids(*groups: tuple[object, ...]) -> None:
        identifiers: list[str] = []
        for group in groups:
            for descriptor in group:
                for field_name in ("interface_id", "surface_id", "region_id"):
                    if hasattr(descriptor, field_name):
                        identifiers.append(str(getattr(descriptor, field_name)))
                        break
        if len(identifiers) != len(set(identifiers)):
            raise BicrystalStateValueError(
                "Interface, surface, and region descriptor IDs must be unique."
            )

    @staticmethod
    def _validate_descriptor_bounds(
        box: np.ndarray,
        interfaces: tuple[InterfaceDescriptor, ...],
        surfaces: tuple[SurfaceDescriptor, ...],
        vacuum_regions: tuple[RegionDescriptor, ...],
        fixed_regions: tuple[RegionDescriptor, ...],
        buffer_regions: tuple[RegionDescriptor, ...],
    ) -> None:
        for interface in interfaces:
            lower, upper = box[interface.axis]
            if not (
                lower - _COORDINATE_TOLERANCE
                <= interface.position
                <= upper + _COORDINATE_TOLERANCE
            ):
                raise BicrystalStateValueError(
                    f"Interface {interface.interface_id!r} lies outside its box axis."
                )
            partner = interface.periodic_partner_position
            if partner is not None and not (
                lower - _COORDINATE_TOLERANCE
                <= partner
                <= upper + _COORDINATE_TOLERANCE
            ):
                raise BicrystalStateValueError(
                    f"Interface {interface.interface_id!r} periodic partner lies "
                    "outside its box axis."
                )
        for surface in surfaces:
            lower, upper = box[surface.axis]
            if not (
                lower - _COORDINATE_TOLERANCE
                <= surface.position
                <= upper + _COORDINATE_TOLERANCE
            ):
                raise BicrystalStateValueError(
                    f"Surface {surface.surface_id!r} lies outside its box axis."
                )
        for region in (*vacuum_regions, *fixed_regions, *buffer_regions):
            lower, upper = box[region.axis]
            if (
                region.lower < lower - _COORDINATE_TOLERANCE
                or region.upper > upper + _COORDINATE_TOLERANCE
            ):
                raise BicrystalStateValueError(
                    f"Region {region.region_id!r} lies outside its box axis."
                )

    @staticmethod
    def _validate_topology(
        topology: str,
        boundary_conditions: tuple[str, ...],
        interfaces: tuple[InterfaceDescriptor, ...],
        surfaces: tuple[SurfaceDescriptor, ...],
        vacuum_regions: tuple[RegionDescriptor, ...],
    ) -> None:
        interior = [item for item in interfaces if item.location == "interior"]
        periodic = [
            item for item in interfaces if item.location == "periodic_boundary"
        ]
        if topology == "periodic_bicrystal":
            if boundary_conditions[0] != "periodic":
                raise BicrystalStateValueError(
                    "periodic_bicrystal requires a periodic normal axis."
                )
            if len(interfaces) != 2 or len(interior) != 1 or len(periodic) != 1:
                raise BicrystalStateValueError(
                    "periodic_bicrystal requires one interior and one periodic "
                    "grain-boundary interface."
                )
            x_surfaces = [surface for surface in surfaces if surface.axis == 0]
            x_vacuum = [region for region in vacuum_regions if region.axis == 0]
            if x_surfaces or x_vacuum:
                raise BicrystalStateValueError(
                    "periodic_bicrystal must not define normal-axis external surfaces "
                    "or vacuum regions."
                )
        else:
            if boundary_conditions[0] == "periodic":
                raise BicrystalStateValueError(
                    "single_interface_slab requires a nonperiodic normal axis."
                )
            if len(interfaces) != 1 or len(interior) != 1 or periodic:
                raise BicrystalStateValueError(
                    "single_interface_slab requires exactly one interior physical GB."
                )
            x_surfaces = [surface for surface in surfaces if surface.axis == 0]
            if len(x_surfaces) < 2:
                raise BicrystalStateValueError(
                    "single_interface_slab requires at least two external x surfaces."
                )

        for surface in surfaces:
            if boundary_conditions[surface.axis] == "periodic":
                raise BicrystalStateValueError(
                    f"Surface {surface.surface_id!r} lies on periodic axis "
                    f"{surface.axis}."
                )
        for region in vacuum_regions:
            if boundary_conditions[region.axis] == "periodic":
                raise BicrystalStateValueError(
                    f"Vacuum region {region.region_id!r} lies on periodic axis "
                    f"{region.axis}."
                )
        for region in vacuum_regions:
            if region.kind != "vacuum":
                raise BicrystalStateValueError(
                    "vacuum_regions may contain only kind='vacuum' descriptors."
                )

    def with_atoms(self, atoms: np.ndarray) -> BicrystalState:
        """Return a new state with replacement atoms and unchanged identity metadata."""
        return replace(self, atoms=atoms)

    @property
    def structure_hash(self) -> str:
        """Return a deterministic hash of geometry, identity, topology, and placement."""
        payload = {
            "schema_version": self.schema_version,
            "array_hashes": self._array_hashes(),
            "topology": self.topology,
            "boundary_conditions": self.boundary_conditions,
            "interfaces": [asdict(item) for item in self.interfaces],
            "external_surfaces": [asdict(item) for item in self.external_surfaces],
            "vacuum_regions": [asdict(item) for item in self.vacuum_regions],
            "fixed_regions": [asdict(item) for item in self.fixed_regions],
            "buffer_regions": [asdict(item) for item in self.buffer_regions],
            "relative_translation_lab": self.relative_translation_lab,
            "termination_ids": self.termination_ids,
            "moving_grain_id": self.moving_grain_id,
            "translation_convention": self.translation_convention,
        }
        return _sha256_bytes(_canonical_json_bytes(payload))

    @property
    def state_hash(self) -> str:
        """Return a deterministic hash including construction metadata/provenance."""
        payload = {
            "structure_hash": self.structure_hash,
            "metadata": self.metadata,
        }
        return _sha256_bytes(_canonical_json_bytes(payload))

    def _array_hashes(self) -> dict[str, str]:
        return {
            "atoms": _sha256_bytes(_npy_bytes(self.atoms)),
            "box_dims": _sha256_bytes(_npy_bytes(self.box_dims)),
            "atom_ids": _sha256_bytes(_npy_bytes(self.atom_ids)),
            "grain_ids": _sha256_bytes(_npy_bytes(self.grain_ids)),
        }

    def manifest(self) -> dict[str, Any]:
        """Return the deterministic JSON-compatible state manifest."""
        arrays = {
            "atoms": "atoms.npy",
            "box_dims": "box_dims.npy",
            "atom_ids": "atom_ids.npy",
            "grain_ids": "grain_ids.npy",
        }
        return {
            "schema_version": self.schema_version,
            "topology": self.topology,
            "boundary_conditions": list(self.boundary_conditions),
            "interfaces": [asdict(item) for item in self.interfaces],
            "external_surfaces": [asdict(item) for item in self.external_surfaces],
            "vacuum_regions": [asdict(item) for item in self.vacuum_regions],
            "fixed_regions": [asdict(item) for item in self.fixed_regions],
            "buffer_regions": [asdict(item) for item in self.buffer_regions],
            "relative_translation_lab": list(self.relative_translation_lab),
            "termination_ids": (
                None if self.termination_ids is None else list(self.termination_ids)
            ),
            "moving_grain_id": self.moving_grain_id,
            "translation_convention": self.translation_convention,
            "metadata": _thaw_json(self.metadata),
            "arrays": arrays,
            "array_hashes": self._array_hashes(),
            "structure_hash": self.structure_hash,
            "state_hash": self.state_hash,
        }

    def save(self, directory: str | Path) -> None:
        """Write deterministic NPY arrays and a canonical JSON manifest."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        array_payloads = {
            "atoms.npy": _npy_bytes(self.atoms),
            "box_dims.npy": _npy_bytes(self.box_dims),
            "atom_ids.npy": _npy_bytes(self.atom_ids),
            "grain_ids.npy": _npy_bytes(self.grain_ids),
        }
        for filename, payload in array_payloads.items():
            (target / filename).write_bytes(payload)
        manifest_bytes = _canonical_json_bytes(self.manifest()) + b"\n"
        (target / "state.json").write_bytes(manifest_bytes)

    @classmethod
    def load(cls, directory: str | Path) -> BicrystalState:
        """Load a state directory and verify every retained content hash."""
        source = Path(directory)
        try:
            manifest = json.loads((source / "state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BicrystalStateValueError(
                f"Cannot read bicrystal state manifest from {source}: {exc}"
            ) from exc

        arrays: dict[str, np.ndarray] = {}
        for name, filename in manifest.get("arrays", {}).items():
            path = source / filename
            try:
                payload = path.read_bytes()
                arrays[name] = np.load(io.BytesIO(payload), allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise BicrystalStateValueError(
                    f"Cannot read state array {path}: {exc}"
                ) from exc
            expected = manifest.get("array_hashes", {}).get(name)
            actual = _sha256_bytes(payload)
            if expected != actual:
                raise BicrystalStateValueError(
                    f"State array hash mismatch for {name}: expected {expected}, "
                    f"got {actual}."
                )

        try:
            state = cls(
                atoms=arrays["atoms"],
                box_dims=arrays["box_dims"],
                topology=manifest["topology"],
                boundary_conditions=tuple(manifest["boundary_conditions"]),
                atom_ids=arrays["atom_ids"],
                grain_ids=arrays["grain_ids"],
                interfaces=tuple(
                    InterfaceDescriptor(**item) for item in manifest["interfaces"]
                ),
                external_surfaces=tuple(
                    SurfaceDescriptor(**item)
                    for item in manifest["external_surfaces"]
                ),
                vacuum_regions=tuple(
                    RegionDescriptor(**item) for item in manifest["vacuum_regions"]
                ),
                fixed_regions=tuple(
                    RegionDescriptor(**item) for item in manifest["fixed_regions"]
                ),
                buffer_regions=tuple(
                    RegionDescriptor(**item) for item in manifest["buffer_regions"]
                ),
                relative_translation_lab=tuple(
                    manifest["relative_translation_lab"]
                ),
                termination_ids=(
                    None
                    if manifest["termination_ids"] is None
                    else tuple(manifest["termination_ids"])
                ),
                moving_grain_id=manifest["moving_grain_id"],
                translation_convention=manifest["translation_convention"],
                metadata=manifest["metadata"],
                schema_version=manifest["schema_version"],
            )
        except (KeyError, TypeError) as exc:
            raise BicrystalStateValueError(
                f"State manifest is incomplete or malformed: {exc}"
            ) from exc

        if state.structure_hash != manifest.get("structure_hash"):
            raise BicrystalStateValueError("State structure_hash verification failed.")
        if state.state_hash != manifest.get("state_hash"):
            raise BicrystalStateValueError("State state_hash verification failed.")
        return state


__all__ = [
    "LEFT_GRAIN_ID",
    "RIGHT_GRAIN_ID",
    "STATE_SCHEMA_VERSION",
    "TRANSLATION_CONVENTION",
    "BoundaryCondition",
    "BicrystalTopology",
    "BicrystalStateError",
    "BicrystalStateTypeError",
    "BicrystalStateValueError",
    "InterfaceDescriptor",
    "SurfaceDescriptor",
    "RegionDescriptor",
    "BicrystalState",
]
