# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Strict, deterministic, topology-aware bicrystal feasibility validation.

The validator in this module consumes :class:`GBOpt.BicrystalState.BicrystalState`
and is deliberately observational. It records raw contact, local-gap, surface, vacuum,
fixed-region, and buffer-region metrics without modifying coordinates or selecting a
translation, termination, or energy-ranked candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence, TypeAlias

import numpy as np
from scipy.spatial import cKDTree

from GBOpt.BicrystalState import BicrystalState, InterfaceDescriptor, RegionDescriptor
from GBOpt.geometry_audit import (
    GeometryAuditError,
    InterfaceGapStatistics,
    automatic_interface_bins,
    mixed_boundary_tree_coordinates,
    periodic_duplicate_pairs,
    same_distance_summary,
    summarize_interface_gaps,
)

FeasibilityStatus: TypeAlias = Literal["invalid", "infeasible", "warning", "feasible"]
_STATUS_RANK: dict[str, int] = {
    "feasible": 0,
    "warning": 1,
    "infeasible": 2,
    "invalid": 3,
}
_VALID_STATUSES = frozenset(_STATUS_RANK)


class GeometryValidationError(ValueError):
    """Raised when a feasibility policy or explicit override is malformed."""


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise GeometryValidationError(f"{name} must be a finite non-negative float.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryValidationError(
            f"{name} must be a finite non-negative float."
        ) from exc
    if not math.isfinite(number) or number < 0.0:
        raise GeometryValidationError(f"{name} must be a finite non-negative float.")
    return number


def _fraction(value: object, name: str) -> float:
    number = _finite_nonnegative(value, name)
    if number > 1.0:
        raise GeometryValidationError(f"{name} must not exceed 1.0.")
    return number


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GeometryValidationError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer <= 0:
        raise GeometryValidationError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise GeometryValidationError(f"{name} must be a non-negative integer.")
    integer = int(value)
    if integer < 0:
        raise GeometryValidationError(f"{name} must be a non-negative integer.")
    return integer


def _species(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GeometryValidationError(f"{name} must be a non-empty species string.")
    return value


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    pair = sorted(
        (_species(first, "species[0]"), _species(second, "species[1]"))
    )
    return pair[0], pair[1]


def _canonical_payload(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_payload(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SpeciesPairThresholds:
    """Explicit duplicate, hard-contact, and warning thresholds for one pair."""

    species: tuple[str, str]
    duplicate_angstrom: float
    hard_minimum_angstrom: float
    warning_minimum_angstrom: float

    def __post_init__(self) -> None:
        try:
            first, second = self.species
        except (TypeError, ValueError) as exc:
            raise GeometryValidationError(
                "species must contain exactly two species strings."
            ) from exc
        pair = _canonical_pair(first, second)
        duplicate = _finite_nonnegative(self.duplicate_angstrom, "duplicate_angstrom")
        hard = _finite_nonnegative(self.hard_minimum_angstrom, "hard_minimum_angstrom")
        warning = _finite_nonnegative(
            self.warning_minimum_angstrom, "warning_minimum_angstrom"
        )
        if not duplicate <= hard <= warning:
            raise GeometryValidationError(
                "Pair thresholds must satisfy duplicate <= hard minimum <= warning minimum."
            )
        object.__setattr__(self, "species", pair)
        object.__setattr__(self, "duplicate_angstrom", duplicate)
        object.__setattr__(self, "hard_minimum_angstrom", hard)
        object.__setattr__(self, "warning_minimum_angstrom", warning)


@dataclass(frozen=True, slots=True)
class ContactPolicy:
    """Species-aware contact policy with deterministic bulk-neighbor fallback."""

    pair_thresholds: tuple[SpeciesPairThresholds, ...] = ()
    duplicate_tolerance_angstrom: float = 1.0e-6
    hard_minimum_bulk_factor: float = 0.45
    warning_minimum_bulk_factor: float = 0.60

    def __post_init__(self) -> None:
        duplicate = _finite_nonnegative(
            self.duplicate_tolerance_angstrom, "duplicate_tolerance_angstrom"
        )
        hard = _finite_nonnegative(
            self.hard_minimum_bulk_factor, "hard_minimum_bulk_factor"
        )
        warning = _finite_nonnegative(
            self.warning_minimum_bulk_factor, "warning_minimum_bulk_factor"
        )
        if hard > warning:
            raise GeometryValidationError(
                "hard_minimum_bulk_factor must not exceed warning_minimum_bulk_factor."
            )
        normalized = tuple(sorted(tuple(self.pair_thresholds), key=lambda item: item.species))
        if any(not isinstance(item, SpeciesPairThresholds) for item in normalized):
            raise GeometryValidationError(
                "pair_thresholds must contain SpeciesPairThresholds instances."
            )
        pairs = [item.species for item in normalized]
        if len(pairs) != len(set(pairs)):
            raise GeometryValidationError("pair_thresholds contains a duplicate species pair.")
        object.__setattr__(self, "pair_thresholds", normalized)
        object.__setattr__(self, "duplicate_tolerance_angstrom", duplicate)
        object.__setattr__(self, "hard_minimum_bulk_factor", hard)
        object.__setattr__(self, "warning_minimum_bulk_factor", warning)

    def explicit_thresholds(
        self, pair: tuple[str, str]
    ) -> SpeciesPairThresholds | None:
        """Return the explicit entry for a canonical species pair, when present."""
        canonical = _canonical_pair(*pair)
        return next(
            (item for item in self.pair_thresholds if item.species == canonical), None
        )


@dataclass(frozen=True, slots=True)
class VoidPolicy:
    """Hard and warning limits for local per-interface void metrics."""

    hard_max_empty_bin_fraction: float = 0.25
    warning_max_empty_bin_fraction: float = 0.10
    hard_max_gap_range_bulk_factor: float = 2.0
    warning_max_gap_range_bulk_factor: float = 1.0
    hard_max_p95_bulk_factor: float = 4.0
    warning_max_p95_bulk_factor: float = 3.0
    min_bins_per_axis: int = 1
    max_bins_per_axis: int = 64

    def __post_init__(self) -> None:
        hard_empty = _fraction(
            self.hard_max_empty_bin_fraction, "hard_max_empty_bin_fraction"
        )
        warning_empty = _fraction(
            self.warning_max_empty_bin_fraction, "warning_max_empty_bin_fraction"
        )
        hard_range = _finite_nonnegative(
            self.hard_max_gap_range_bulk_factor, "hard_max_gap_range_bulk_factor"
        )
        warning_range = _finite_nonnegative(
            self.warning_max_gap_range_bulk_factor,
            "warning_max_gap_range_bulk_factor",
        )
        hard_p95 = _finite_nonnegative(
            self.hard_max_p95_bulk_factor, "hard_max_p95_bulk_factor"
        )
        warning_p95 = _finite_nonnegative(
            self.warning_max_p95_bulk_factor, "warning_max_p95_bulk_factor"
        )
        minimum = _positive_int(self.min_bins_per_axis, "min_bins_per_axis")
        maximum = _positive_int(self.max_bins_per_axis, "max_bins_per_axis")
        if minimum > maximum:
            raise GeometryValidationError(
                "min_bins_per_axis must not exceed max_bins_per_axis."
            )
        for warning_value, hard_value, name in (
            (warning_empty, hard_empty, "empty-bin fractions"),
            (warning_range, hard_range, "gap-range factors"),
            (warning_p95, hard_p95, "p95 factors"),
        ):
            if warning_value > hard_value:
                raise GeometryValidationError(
                    f"Warning {name} must not exceed hard {name}."
                )
        object.__setattr__(self, "hard_max_empty_bin_fraction", hard_empty)
        object.__setattr__(self, "warning_max_empty_bin_fraction", warning_empty)
        object.__setattr__(self, "hard_max_gap_range_bulk_factor", hard_range)
        object.__setattr__(self, "warning_max_gap_range_bulk_factor", warning_range)
        object.__setattr__(self, "hard_max_p95_bulk_factor", hard_p95)
        object.__setattr__(self, "warning_max_p95_bulk_factor", warning_p95)
        object.__setattr__(self, "min_bins_per_axis", minimum)
        object.__setattr__(self, "max_bins_per_axis", maximum)


@dataclass(frozen=True, slots=True)
class SlabPolicy:
    """Independent surface, vacuum, fixed-region, and buffer-region limits."""

    minimum_vacuum_thickness_angstrom: float = 0.0
    warning_vacuum_thickness_angstrom: float = 0.0
    minimum_surface_clearance_angstrom: float = 0.0
    warning_surface_clearance_angstrom: float = 0.0
    minimum_fixed_region_thickness_angstrom: float = 0.0
    warning_fixed_region_thickness_angstrom: float = 0.0
    minimum_buffer_thickness_angstrom: float = 0.0
    warning_buffer_thickness_angstrom: float = 0.0
    minimum_fixed_region_atoms: int = 0
    minimum_buffer_region_atoms: int = 0
    descriptor_tolerance_angstrom: float = 1.0e-8

    def __post_init__(self) -> None:
        values = {}
        for field in (
            "minimum_vacuum_thickness_angstrom",
            "warning_vacuum_thickness_angstrom",
            "minimum_surface_clearance_angstrom",
            "warning_surface_clearance_angstrom",
            "minimum_fixed_region_thickness_angstrom",
            "warning_fixed_region_thickness_angstrom",
            "minimum_buffer_thickness_angstrom",
            "warning_buffer_thickness_angstrom",
            "descriptor_tolerance_angstrom",
        ):
            values[field] = _finite_nonnegative(getattr(self, field), field)
        for minimum, warning, name in (
            (
                values["minimum_vacuum_thickness_angstrom"],
                values["warning_vacuum_thickness_angstrom"],
                "vacuum thickness",
            ),
            (
                values["minimum_surface_clearance_angstrom"],
                values["warning_surface_clearance_angstrom"],
                "surface clearance",
            ),
            (
                values["minimum_fixed_region_thickness_angstrom"],
                values["warning_fixed_region_thickness_angstrom"],
                "fixed-region thickness",
            ),
            (
                values["minimum_buffer_thickness_angstrom"],
                values["warning_buffer_thickness_angstrom"],
                "buffer thickness",
            ),
        ):
            if warning and warning < minimum:
                raise GeometryValidationError(
                    f"Warning {name} must be zero or at least the hard minimum."
                )
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "minimum_fixed_region_atoms",
            _nonnegative_int(self.minimum_fixed_region_atoms, "minimum_fixed_region_atoms"),
        )
        object.__setattr__(
            self,
            "minimum_buffer_region_atoms",
            _nonnegative_int(
                self.minimum_buffer_region_atoms, "minimum_buffer_region_atoms"
            ),
        )


@dataclass(frozen=True, slots=True)
class FeasibilityPolicy:
    """Complete immutable feasibility policy."""

    contact: ContactPolicy = ContactPolicy()
    void: VoidPolicy = VoidPolicy()
    slab: SlabPolicy = SlabPolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.contact, ContactPolicy):
            raise GeometryValidationError("contact must be a ContactPolicy instance.")
        if not isinstance(self.void, VoidPolicy):
            raise GeometryValidationError("void must be a VoidPolicy instance.")
        if not isinstance(self.slab, SlabPolicy):
            raise GeometryValidationError("slab must be a SlabPolicy instance.")

    @classmethod
    def from_unit_cell(
        cls,
        unit_cell: Any,
        *,
        contact: ContactPolicy | None = None,
        void: VoidPolicy | None = None,
        slab: SlabPolicy | None = None,
    ) -> "FeasibilityPolicy":
        """Build explicit species-pair thresholds from ``UnitCell`` bond lengths."""
        base = ContactPolicy() if contact is None else contact
        explicit = {item.species: item for item in base.pair_thresholds}
        type_map = dict(unit_cell.type_map)
        inverse = {int(index): str(name) for name, index in type_map.items()}
        for raw_pair, raw_distance in dict(unit_cell.ideal_bond_lengths).items():
            try:
                pair = _canonical_pair(inverse[int(raw_pair[0])], inverse[int(raw_pair[1])])
            except (KeyError, TypeError, ValueError):
                continue
            if pair in explicit:
                continue
            distance = _finite_nonnegative(raw_distance, f"ideal_bond_lengths[{raw_pair!r}]")
            explicit[pair] = SpeciesPairThresholds(
                species=pair,
                duplicate_angstrom=base.duplicate_tolerance_angstrom,
                hard_minimum_angstrom=base.hard_minimum_bulk_factor * distance,
                warning_minimum_angstrom=base.warning_minimum_bulk_factor * distance,
            )
        return cls(
            contact=ContactPolicy(
                pair_thresholds=tuple(explicit.values()),
                duplicate_tolerance_angstrom=base.duplicate_tolerance_angstrom,
                hard_minimum_bulk_factor=base.hard_minimum_bulk_factor,
                warning_minimum_bulk_factor=base.warning_minimum_bulk_factor,
            ),
            void=VoidPolicy() if void is None else void,
            slab=SlabPolicy() if slab is None else slab,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible policy dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Return canonical compact JSON."""
        return _canonical_payload(self.to_dict())

    @property
    def policy_hash(self) -> str:
        """Return the SHA-256 hash of canonical policy JSON."""
        return _sha256_payload(self.to_dict())


@dataclass(frozen=True, slots=True)
class FeasibilityOverride:
    """Reason-bearing expert override of a non-invalid raw decision."""

    status: FeasibilityStatus
    reason: str

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES or self.status == "invalid":
            raise GeometryValidationError(
                "Override status must be 'infeasible', 'warning', or 'feasible'."
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise GeometryValidationError("An override requires a non-empty reason.")
        object.__setattr__(self, "reason", self.reason.strip())


@dataclass(frozen=True, slots=True)
class ValidationReason:
    """One stable machine-readable reason contributing to a decision."""

    status: FeasibilityStatus
    code: str
    descriptor_id: str | None = None
    species: tuple[str, str] | None = None
    observed: float | int | None = None
    threshold: float | int | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES or self.status == "feasible":
            raise GeometryValidationError("Reason status must be invalid, infeasible, or warning.")
        if not isinstance(self.code, str) or not self.code:
            raise GeometryValidationError("Reason code must be a non-empty string.")
        if self.species is not None:
            object.__setattr__(self, "species", _canonical_pair(*self.species))


@dataclass(frozen=True, slots=True)
class ResolvedContactThresholds:
    """Threshold values and provenance selected for one observed species pair."""

    species: tuple[str, str]
    duplicate_angstrom: float
    hard_minimum_angstrom: float
    warning_minimum_angstrom: float
    source: str
    reference_distance_angstrom: float | None


@dataclass(frozen=True, slots=True)
class SpeciesPairContactMetrics:
    """Raw cross-interface contact metrics for one species pair."""

    species: tuple[str, str]
    minimum_distance_angstrom: float | None
    duplicate_count: int
    hard_contact_count: int
    warning_contact_count: int
    thresholds: ResolvedContactThresholds | None


@dataclass(frozen=True, slots=True)
class DuplicatePairMetrics:
    """One duplicate representative found under declared periodic axes."""

    atom_ids: tuple[int, int]
    species: tuple[str, str]
    grain_ids: tuple[int, int]
    distance_angstrom: float


@dataclass(frozen=True, slots=True)
class InterfaceFeasibilityMetrics:
    """Raw measurements for one physical GB interface."""

    interface_id: str
    axis: int
    location: str
    bins: tuple[int, int]
    gap_statistics: InterfaceGapStatistics
    bulk_reference_distance_angstrom: float | None
    contacts: tuple[SpeciesPairContactMetrics, ...]


@dataclass(frozen=True, slots=True)
class SurfaceValidationMetrics:
    """Raw measurements for one external slab surface."""

    surface_id: str
    axis: int
    nearest_inward_clearance_angstrom: float | None
    outward_atom_count: int
    considered_atom_count: int


@dataclass(frozen=True, slots=True)
class RegionValidationMetrics:
    """Raw measurements for one vacuum, fixed, or buffer descriptor."""

    region_id: str
    declared_kind: str
    axis: int
    thickness_angstrom: float
    atom_count: int
    species_counts: tuple[tuple[str, int], ...]
    grain_counts: tuple[tuple[int, int], ...]
    undeclared_grain_atom_count: int


@dataclass(frozen=True, slots=True)
class SlabValidationMetrics:
    """Raw surface and region metrics for a slab; absent for periodic bicrystals."""

    surfaces: tuple[SurfaceValidationMetrics, ...]
    vacuum_regions: tuple[RegionValidationMetrics, ...]
    fixed_regions: tuple[RegionValidationMetrics, ...]
    buffer_regions: tuple[RegionValidationMetrics, ...]


@dataclass(frozen=True, slots=True)
class BicrystalFeasibilityReport:
    """Complete deterministic validation result with raw and effective decisions."""

    raw_status: FeasibilityStatus
    status: FeasibilityStatus
    reasons: tuple[ValidationReason, ...]
    raw_reasons: tuple[ValidationReason, ...]
    duplicate_pairs: tuple[DuplicatePairMetrics, ...]
    interfaces: tuple[InterfaceFeasibilityMetrics, ...]
    slab: SlabValidationMetrics | None
    structure_hash: str
    state_hash: str
    policy: FeasibilityPolicy
    override: FeasibilityOverride | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible nested dictionary."""
        payload = asdict(self)
        payload["policy_hash"] = self.policy.policy_hash
        payload["report_hash"] = self.report_hash
        return payload

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "raw_status": self.raw_status,
            "status": self.status,
            "reasons": [asdict(item) for item in self.reasons],
            "raw_reasons": [asdict(item) for item in self.raw_reasons],
            "duplicate_pairs": [asdict(item) for item in self.duplicate_pairs],
            "interfaces": [asdict(item) for item in self.interfaces],
            "slab": None if self.slab is None else asdict(self.slab),
            "structure_hash": self.structure_hash,
            "state_hash": self.state_hash,
            "policy": self.policy.to_dict(),
            "override": None if self.override is None else asdict(self.override),
        }

    def to_json(self) -> str:
        """Return canonical compact JSON including the report hash."""
        return _canonical_payload(self.to_dict())

    @property
    def report_hash(self) -> str:
        """Return the SHA-256 hash of report contents excluding the hash field."""
        return _sha256_payload(self._hash_payload())


@dataclass(frozen=True, slots=True)
class _AtomView:
    positions: np.ndarray
    species: np.ndarray
    atom_ids: np.ndarray
    grain_ids: np.ndarray


def _atom_view(state: BicrystalState) -> _AtomView:
    atoms = state.atoms
    positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"])).astype(
        np.float64, copy=False
    )
    if not np.all(np.isfinite(positions)):
        raise GeometryAuditError("BicrystalState contains non-finite coordinates.")
    species = np.asarray(atoms["name"]).astype(str)
    return _AtomView(
        positions=np.array(positions, copy=True),
        species=np.array(species, copy=True),
        atom_ids=np.asarray(state.atom_ids, dtype=np.int64),
        grain_ids=np.asarray(state.grain_ids, dtype=np.int64),
    )


def _periodic_axes(state: BicrystalState, *, exclude_axis: int | None = None) -> tuple[int, ...]:
    return tuple(
        axis
        for axis, condition in enumerate(state.boundary_conditions)
        if condition == "periodic" and axis != exclude_axis
    )


def _distance_between(
    first: np.ndarray,
    second: np.ndarray,
    lengths: np.ndarray,
    periodic_axes: Sequence[int],
) -> float:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    for axis in periodic_axes:
        delta[axis] -= np.rint(delta[axis] / lengths[axis]) * lengths[axis]
    return float(np.linalg.norm(delta))


def _topology_reasons(state: BicrystalState) -> list[ValidationReason]:
    """Return deterministic reasons for descriptor contradictions."""
    reasons: list[ValidationReason] = []
    lower = state.box_dims[:, 0]
    upper = state.box_dims[:, 1]
    tolerance = 1.0e-8
    for interface in state.interfaces:
        if interface.location != "periodic_boundary":
            continue
        if state.boundary_conditions[interface.axis] != "periodic":
            reasons.append(
                ValidationReason(
                    "invalid",
                    "interface.periodic_on_fixed_axis",
                    descriptor_id=interface.interface_id,
                )
            )
        positions = sorted(
            (float(interface.position), float(interface.periodic_partner_position))
        )
        expected = sorted(
            (float(lower[interface.axis]), float(upper[interface.axis]))
        )
        if not np.allclose(positions, expected, rtol=0.0, atol=tolerance):
            reasons.append(
                ValidationReason(
                    "invalid",
                    "interface.periodic_positions_not_box_faces",
                    descriptor_id=interface.interface_id,
                )
            )

    if state.topology == "periodic_bicrystal":
        interior = next(
            (item for item in state.interfaces if item.location == "interior"),
            None,
        )
        periodic = next(
            (
                item
                for item in state.interfaces
                if item.location == "periodic_boundary"
            ),
            None,
        )
        if interior is not None and periodic is not None and (
            interior.plus_grain_id != periodic.minus_grain_id
            or periodic.plus_grain_id != interior.minus_grain_id
        ):
            reasons.append(
                ValidationReason(
                    "invalid", "topology.periodic_interface_grain_order"
                )
            )
        return reasons

    for region in state.fixed_regions:
        if region.kind != "fixed":
            reasons.append(
                ValidationReason(
                    "invalid",
                    "region.fixed_kind_mismatch",
                    descriptor_id=region.region_id,
                )
            )
    for region in state.buffer_regions:
        if region.kind != "buffer":
            reasons.append(
                ValidationReason(
                    "invalid",
                    "region.buffer_kind_mismatch",
                    descriptor_id=region.region_id,
                )
            )
    all_regions = (
        tuple(state.vacuum_regions)
        + tuple(state.fixed_regions)
        + tuple(state.buffer_regions)
    )
    for region in all_regions:
        if state.boundary_conditions[region.axis] == "periodic":
            reasons.append(
                ValidationReason(
                    "invalid",
                    "region.declared_on_periodic_axis",
                    descriptor_id=region.region_id,
                )
            )
    for index, first in enumerate(all_regions):
        for second in all_regions[index + 1 :]:
            if first.axis != second.axis:
                continue
            overlap = min(first.upper, second.upper) - max(
                first.lower, second.lower
            )
            if overlap <= tolerance:
                continue
            if "vacuum" in {first.kind, second.kind}:
                code = "region.vacuum_solid_overlap"
            else:
                code = "region.solid_region_overlap"
            reasons.append(
                ValidationReason(
                    "invalid",
                    code,
                    descriptor_id=f"{first.region_id}|{second.region_id}",
                    observed=float(overlap),
                    threshold=0.0,
                )
            )

    for vacuum in state.vacuum_regions:
        matching = [
            surface
            for surface in state.external_surfaces
            if surface.axis == vacuum.axis
            and (
                math.isclose(surface.position, vacuum.lower, abs_tol=tolerance)
                or math.isclose(surface.position, vacuum.upper, abs_tol=tolerance)
            )
        ]
        if not matching:
            reasons.append(
                ValidationReason(
                    "invalid",
                    "vacuum.missing_surface_boundary",
                    descriptor_id=vacuum.region_id,
                )
            )
            continue
        for surface in matching:
            sign = float(surface.outward_normal_lab[vacuum.axis])
            expected_sign = (
                1.0
                if math.isclose(
                    surface.position, vacuum.lower, abs_tol=tolerance
                )
                else -1.0
            )
            if sign != expected_sign:
                reasons.append(
                    ValidationReason(
                        "invalid",
                        "vacuum.surface_normal_inconsistent",
                        descriptor_id=surface.surface_id,
                    )
                )

    for interface in state.interfaces:
        same_axis_surfaces = sorted(
            surface.position
            for surface in state.external_surfaces
            if surface.axis == interface.axis
        )
        if len(same_axis_surfaces) >= 2 and not (
            same_axis_surfaces[0] + tolerance
            < interface.position
            < same_axis_surfaces[-1] - tolerance
        ):
            reasons.append(
                ValidationReason(
                    "invalid",
                    "interface.outside_slab_surfaces",
                    descriptor_id=interface.interface_id,
                )
            )
    return reasons


def _duplicate_metrics(
    state: BicrystalState,
    atoms: _AtomView,
    policy: ContactPolicy,
) -> tuple[DuplicatePairMetrics, ...]:
    lower = state.box_dims[:, 0]
    lengths = state.box_dims[:, 1] - lower
    periodic = _periodic_axes(state)
    search_tolerance = max(
        [policy.duplicate_tolerance_angstrom]
        + [item.duplicate_angstrom for item in policy.pair_thresholds]
    )
    pairs = periodic_duplicate_pairs(
        atoms.positions,
        lower,
        lengths,
        periodic,
        search_tolerance,
    )
    metrics: list[DuplicatePairMetrics] = []
    for first, second in pairs:
        pair = _canonical_pair(str(atoms.species[first]), str(atoms.species[second]))
        explicit = policy.explicit_thresholds(pair)
        tolerance = (
            policy.duplicate_tolerance_angstrom
            if explicit is None
            else explicit.duplicate_angstrom
        )
        distance = _distance_between(
            atoms.positions[first], atoms.positions[second], lengths, periodic
        )
        if distance <= tolerance:
            entries = sorted(
                (
                    (int(atoms.atom_ids[first]), int(atoms.grain_ids[first])),
                    (int(atoms.atom_ids[second]), int(atoms.grain_ids[second])),
                )
            )
            metrics.append(
                DuplicatePairMetrics(
                    atom_ids=(entries[0][0], entries[1][0]),
                    species=pair,
                    grain_ids=(entries[0][1], entries[1][1]),
                    distance_angstrom=distance,
                )
            )
    return tuple(sorted(metrics, key=lambda item: item.atom_ids))


def _internal_references(
    state: BicrystalState,
    atoms: _AtomView,
    interface: InterfaceDescriptor,
) -> tuple[dict[tuple[str, str], float], float | None]:
    lower = state.box_dims[:, 0]
    lengths = state.box_dims[:, 1] - lower
    periodic = _periodic_axes(state, exclude_axis=interface.axis)
    species_values = sorted(set(str(value) for value in atoms.species))
    references: dict[tuple[str, str], float] = {}
    for first_index, first in enumerate(species_values):
        for second in species_values[first_index:]:
            pair = (first, second)
            candidates: list[float] = []
            for grain_id in (interface.minus_grain_id, interface.plus_grain_id):
                grain_mask = atoms.grain_ids == grain_id
                first_positions = atoms.positions[grain_mask & (atoms.species == first)]
                second_positions = atoms.positions[grain_mask & (atoms.species == second)]
                if first == second:
                    value = same_distance_summary(
                        first_positions, lower, lengths, periodic
                    )
                else:
                    if len(first_positions) == 0 or len(second_positions) == 0:
                        value = None
                    else:
                        combined = np.vstack((first_positions, second_positions))
                        bounds = (np.min(combined, axis=0), np.max(combined, axis=0))
                        query, boxsize = mixed_boundary_tree_coordinates(
                            first_positions,
                            lower,
                            lengths,
                            periodic,
                            shared_fixed_bounds=bounds,
                        )
                        target, _ = mixed_boundary_tree_coordinates(
                            second_positions,
                            lower,
                            lengths,
                            periodic,
                            shared_fixed_bounds=bounds,
                        )
                        nearest = cKDTree(target, boxsize=boxsize).query(
                            query, k=1, workers=1
                        )[0]
                        finite = nearest[np.isfinite(nearest)]
                        value = None if finite.size == 0 else float(np.min(finite))
                if value is not None and value > 0.0:
                    candidates.append(value)
            if candidates:
                references[pair] = min(candidates)
    global_reference = min(references.values()) if references else None
    return references, global_reference


def _resolve_thresholds(
    pair: tuple[str, str],
    policy: ContactPolicy,
    internal: Mapping[tuple[str, str], float],
    global_reference: float | None,
) -> ResolvedContactThresholds | None:
    explicit = policy.explicit_thresholds(pair)
    if explicit is not None:
        return ResolvedContactThresholds(
            species=pair,
            duplicate_angstrom=explicit.duplicate_angstrom,
            hard_minimum_angstrom=explicit.hard_minimum_angstrom,
            warning_minimum_angstrom=explicit.warning_minimum_angstrom,
            source="explicit",
            reference_distance_angstrom=None,
        )
    reference = internal.get(pair)
    source = "internal_species_pair"
    if reference is None:
        reference = global_reference
        source = "internal_global"
    if reference is None:
        return None
    return ResolvedContactThresholds(
        species=pair,
        duplicate_angstrom=policy.duplicate_tolerance_angstrom,
        hard_minimum_angstrom=policy.hard_minimum_bulk_factor * reference,
        warning_minimum_angstrom=policy.warning_minimum_bulk_factor * reference,
        source=source,
        reference_distance_angstrom=reference,
    )


def _contact_counts(
    first_positions: np.ndarray,
    second_positions: np.ndarray,
    lower: np.ndarray,
    lengths: np.ndarray,
    periodic_axes: tuple[int, ...],
    thresholds: ResolvedContactThresholds,
) -> tuple[float | None, int, int, int]:
    if len(first_positions) == 0 or len(second_positions) == 0:
        return None, 0, 0, 0
    combined = np.vstack((first_positions, second_positions))
    bounds = (np.min(combined, axis=0), np.max(combined, axis=0))
    first, boxsize = mixed_boundary_tree_coordinates(
        first_positions, lower, lengths, periodic_axes, shared_fixed_bounds=bounds
    )
    second, _ = mixed_boundary_tree_coordinates(
        second_positions, lower, lengths, periodic_axes, shared_fixed_bounds=bounds
    )
    tree_first = cKDTree(first, boxsize=boxsize)
    tree_second = cKDTree(second, boxsize=boxsize)
    nearest = tree_second.query(first, k=1, workers=1)[0]
    finite = nearest[np.isfinite(nearest)]
    minimum = None if finite.size == 0 else float(np.min(finite))
    radius = thresholds.warning_minimum_angstrom
    if radius <= 0.0:
        return minimum, 0, 0, 0
    neighborhoods = tree_first.query_ball_tree(tree_second, r=radius)
    duplicate = hard = warning = 0
    for first_index, second_indices in enumerate(neighborhoods):
        if not second_indices:
            continue
        delta = second[np.asarray(second_indices)] - first[first_index]
        delta -= np.rint(delta / boxsize) * boxsize
        distances = np.linalg.norm(delta, axis=1)
        duplicate += int(np.count_nonzero(distances <= thresholds.duplicate_angstrom))
        hard += int(
            np.count_nonzero(
                (distances > thresholds.duplicate_angstrom)
                & (distances < thresholds.hard_minimum_angstrom)
            )
        )
        warning += int(
            np.count_nonzero(
                (distances >= thresholds.hard_minimum_angstrom)
                & (distances < thresholds.warning_minimum_angstrom)
            )
        )
    return minimum, duplicate, hard, warning


def _interface_metrics_and_reasons(
    state: BicrystalState,
    atoms: _AtomView,
    interface: InterfaceDescriptor,
    policy: FeasibilityPolicy,
) -> tuple[InterfaceFeasibilityMetrics, list[ValidationReason]]:
    reasons: list[ValidationReason] = []
    lower = state.box_dims[:, 0]
    lengths = state.box_dims[:, 1] - lower
    tangent_periodic = _periodic_axes(state, exclude_axis=interface.axis)
    minus_mask = atoms.grain_ids == interface.minus_grain_id
    plus_mask = atoms.grain_ids == interface.plus_grain_id
    minus_positions = np.array(atoms.positions[minus_mask], copy=True)
    plus_positions = np.array(atoms.positions[plus_mask], copy=True)
    normal_sign = 1 if interface.normal_lab[interface.axis] > 0.0 else -1
    plus_shift = (
        normal_sign * float(lengths[interface.axis])
        if interface.location == "periodic_boundary"
        else 0.0
    )
    plus_positions[:, interface.axis] += plus_shift

    internal, bulk_reference = _internal_references(state, atoms, interface)
    bins = automatic_interface_bins(
        lengths,
        interface.axis,
        bulk_reference,
        min_bins_per_axis=policy.void.min_bins_per_axis,
        max_bins_per_axis=policy.void.max_bins_per_axis,
    )
    gap = summarize_interface_gaps(
        minus_positions,
        plus_positions,
        lower,
        lengths,
        axis=interface.axis,
        normal_sign=normal_sign,
        bins=bins,
        periodic_axes=tangent_periodic,
    )
    if gap.valid_bins == 0:
        reasons.append(
            ValidationReason(
                "invalid",
                "interface.no_valid_local_gap_bins",
                descriptor_id=interface.interface_id,
            )
        )
    else:
        empty = max(gap.empty_left_bin_fraction, gap.empty_right_bin_fraction)
        if empty > policy.void.hard_max_empty_bin_fraction:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "interface.empty_bin_fraction_exceeded",
                    descriptor_id=interface.interface_id,
                    observed=empty,
                    threshold=policy.void.hard_max_empty_bin_fraction,
                )
            )
        elif empty > policy.void.warning_max_empty_bin_fraction:
            reasons.append(
                ValidationReason(
                    "warning",
                    "interface.empty_bin_fraction_warning",
                    descriptor_id=interface.interface_id,
                    observed=empty,
                    threshold=policy.void.warning_max_empty_bin_fraction,
                )
            )
        if bulk_reference is None:
            reasons.append(
                ValidationReason(
                    "invalid",
                    "interface.bulk_reference_unavailable",
                    descriptor_id=interface.interface_id,
                )
            )
        else:
            if gap.range_angstrom is not None:
                ratio = gap.range_angstrom / bulk_reference
                if ratio > policy.void.hard_max_gap_range_bulk_factor:
                    reasons.append(
                        ValidationReason(
                            "infeasible",
                            "interface.local_gap_range_exceeded",
                            descriptor_id=interface.interface_id,
                            observed=ratio,
                            threshold=policy.void.hard_max_gap_range_bulk_factor,
                        )
                    )
                elif ratio > policy.void.warning_max_gap_range_bulk_factor:
                    reasons.append(
                        ValidationReason(
                            "warning",
                            "interface.local_gap_range_warning",
                            descriptor_id=interface.interface_id,
                            observed=ratio,
                            threshold=policy.void.warning_max_gap_range_bulk_factor,
                        )
                    )
            if gap.percentile_95_angstrom is not None:
                ratio = gap.percentile_95_angstrom / bulk_reference
                if ratio > policy.void.hard_max_p95_bulk_factor:
                    reasons.append(
                        ValidationReason(
                            "infeasible",
                            "interface.local_gap_p95_exceeded",
                            descriptor_id=interface.interface_id,
                            observed=ratio,
                            threshold=policy.void.hard_max_p95_bulk_factor,
                        )
                    )
                elif ratio > policy.void.warning_max_p95_bulk_factor:
                    reasons.append(
                        ValidationReason(
                            "warning",
                            "interface.local_gap_p95_warning",
                            descriptor_id=interface.interface_id,
                            observed=ratio,
                            threshold=policy.void.warning_max_p95_bulk_factor,
                        )
                    )

    contacts: list[SpeciesPairContactMetrics] = []
    minus_species = atoms.species[minus_mask]
    plus_species = atoms.species[plus_mask]
    for first in sorted(set(str(value) for value in minus_species)):
        for second in sorted(set(str(value) for value in plus_species)):
            pair = _canonical_pair(first, second)
            thresholds = _resolve_thresholds(pair, policy.contact, internal, bulk_reference)
            if thresholds is None:
                contacts.append(
                    SpeciesPairContactMetrics(pair, None, 0, 0, 0, None)
                )
                continue
            first_positions = minus_positions[minus_species == first]
            second_positions = plus_positions[plus_species == second]
            minimum, duplicate, hard, warning = _contact_counts(
                first_positions,
                second_positions,
                lower,
                lengths,
                tangent_periodic,
                thresholds,
            )
            contacts.append(
                SpeciesPairContactMetrics(
                    species=pair,
                    minimum_distance_angstrom=minimum,
                    duplicate_count=duplicate,
                    hard_contact_count=hard,
                    warning_contact_count=warning,
                    thresholds=thresholds,
                )
            )
    # Canonical species pairs can occur twice when the two sides contain the same set.
    merged: dict[tuple[str, str], SpeciesPairContactMetrics] = {}
    for item in contacts:
        previous = merged.get(item.species)
        if previous is None:
            merged[item.species] = item
            continue
        minima = [
            value
            for value in (previous.minimum_distance_angstrom, item.minimum_distance_angstrom)
            if value is not None
        ]
        merged[item.species] = SpeciesPairContactMetrics(
            species=item.species,
            minimum_distance_angstrom=min(minima) if minima else None,
            duplicate_count=previous.duplicate_count + item.duplicate_count,
            hard_contact_count=previous.hard_contact_count + item.hard_contact_count,
            warning_contact_count=previous.warning_contact_count + item.warning_contact_count,
            thresholds=previous.thresholds or item.thresholds,
        )
    for pair in sorted(merged):
        item = merged[pair]
        thresholds = item.thresholds
        if thresholds is None:
            reasons.append(
                ValidationReason(
                    "invalid",
                    "interface.contact_reference_unavailable",
                    descriptor_id=interface.interface_id,
                    species=pair,
                )
            )
            continue
        if item.duplicate_count:
            reasons.append(
                ValidationReason(
                    "invalid",
                    "interface.cross_contact_duplicate",
                    descriptor_id=interface.interface_id,
                    species=pair,
                    observed=item.minimum_distance_angstrom,
                    threshold=thresholds.duplicate_angstrom,
                )
            )
        if item.hard_contact_count:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "interface.cross_contact_below_hard_minimum",
                    descriptor_id=interface.interface_id,
                    species=pair,
                    observed=item.minimum_distance_angstrom,
                    threshold=thresholds.hard_minimum_angstrom,
                )
            )
        elif item.warning_contact_count:
            reasons.append(
                ValidationReason(
                    "warning",
                    "interface.cross_contact_below_warning_minimum",
                    descriptor_id=interface.interface_id,
                    species=pair,
                    observed=item.minimum_distance_angstrom,
                    threshold=thresholds.warning_minimum_angstrom,
                )
            )
    return (
        InterfaceFeasibilityMetrics(
            interface_id=interface.interface_id,
            axis=interface.axis,
            location=interface.location,
            bins=bins,
            gap_statistics=gap,
            bulk_reference_distance_angstrom=bulk_reference,
            contacts=tuple(merged[pair] for pair in sorted(merged)),
        ),
        reasons,
    )


def _region_metrics(
    region: RegionDescriptor,
    atoms: _AtomView,
    tolerance: float,
) -> RegionValidationMetrics:
    coordinates = atoms.positions[:, region.axis]
    mask = (coordinates > region.lower + tolerance) & (coordinates < region.upper - tolerance)
    species_counts = tuple(
        sorted(
            (str(species), int(np.count_nonzero(mask & (atoms.species == species))))
            for species in sorted(set(str(value) for value in atoms.species[mask]))
        )
    )
    grain_counts = tuple(
        sorted(
            (int(grain), int(np.count_nonzero(mask & (atoms.grain_ids == grain))))
            for grain in sorted(set(int(value) for value in atoms.grain_ids[mask]))
        )
    )
    undeclared = 0
    if region.grain_ids:
        undeclared = int(
            np.count_nonzero(mask & ~np.isin(atoms.grain_ids, region.grain_ids))
        )
    return RegionValidationMetrics(
        region_id=region.region_id,
        declared_kind=region.kind,
        axis=region.axis,
        thickness_angstrom=float(region.upper - region.lower),
        atom_count=int(np.count_nonzero(mask)),
        species_counts=species_counts,
        grain_counts=grain_counts,
        undeclared_grain_atom_count=undeclared,
    )


def _slab_metrics_and_reasons(
    state: BicrystalState, atoms: _AtomView, policy: SlabPolicy
) -> tuple[SlabValidationMetrics | None, list[ValidationReason]]:
    if state.topology != "single_interface_slab":
        return None, []
    reasons: list[ValidationReason] = []
    tolerance = policy.descriptor_tolerance_angstrom
    surfaces: list[SurfaceValidationMetrics] = []
    for surface in state.external_surfaces:
        mask = np.isin(atoms.grain_ids, surface.grain_ids)
        sign = float(surface.outward_normal_lab[surface.axis])
        inward = -sign * (atoms.positions[mask, surface.axis] - surface.position)
        clearance = None if inward.size == 0 else float(np.min(inward))
        outward_count = int(np.count_nonzero(inward < -tolerance))
        surfaces.append(
            SurfaceValidationMetrics(
                surface_id=surface.surface_id,
                axis=surface.axis,
                nearest_inward_clearance_angstrom=clearance,
                outward_atom_count=outward_count,
                considered_atom_count=int(np.count_nonzero(mask)),
            )
        )
        if inward.size == 0:
            reasons.append(
                ValidationReason(
                    "invalid", "surface.no_declared_grain_atoms", descriptor_id=surface.surface_id
                )
            )
        elif outward_count:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "surface.atoms_outside_surface",
                    descriptor_id=surface.surface_id,
                    observed=outward_count,
                    threshold=0,
                )
            )
        elif clearance is not None and clearance < policy.minimum_surface_clearance_angstrom:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "surface.clearance_below_minimum",
                    descriptor_id=surface.surface_id,
                    observed=clearance,
                    threshold=policy.minimum_surface_clearance_angstrom,
                )
            )
        elif (
            clearance is not None
            and policy.warning_surface_clearance_angstrom
            and clearance < policy.warning_surface_clearance_angstrom
        ):
            reasons.append(
                ValidationReason(
                    "warning",
                    "surface.clearance_warning",
                    descriptor_id=surface.surface_id,
                    observed=clearance,
                    threshold=policy.warning_surface_clearance_angstrom,
                )
            )

    vacuum = tuple(_region_metrics(item, atoms, tolerance) for item in state.vacuum_regions)
    fixed = tuple(_region_metrics(item, atoms, tolerance) for item in state.fixed_regions)
    buffer = tuple(_region_metrics(item, atoms, tolerance) for item in state.buffer_regions)
    for item in vacuum:
        if item.atom_count:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "vacuum.contains_atoms",
                    descriptor_id=item.region_id,
                    observed=item.atom_count,
                    threshold=0,
                )
            )
        if item.thickness_angstrom < policy.minimum_vacuum_thickness_angstrom:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "vacuum.thickness_below_minimum",
                    descriptor_id=item.region_id,
                    observed=item.thickness_angstrom,
                    threshold=policy.minimum_vacuum_thickness_angstrom,
                )
            )
        elif (
            policy.warning_vacuum_thickness_angstrom
            and item.thickness_angstrom < policy.warning_vacuum_thickness_angstrom
        ):
            reasons.append(
                ValidationReason(
                    "warning",
                    "vacuum.thickness_warning",
                    descriptor_id=item.region_id,
                    observed=item.thickness_angstrom,
                    threshold=policy.warning_vacuum_thickness_angstrom,
                )
            )
    for item in fixed:
        if item.undeclared_grain_atom_count:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "fixed_region.contains_undeclared_grain",
                    descriptor_id=item.region_id,
                    observed=item.undeclared_grain_atom_count,
                    threshold=0,
                )
            )
        if item.atom_count < policy.minimum_fixed_region_atoms:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "fixed_region.population_below_minimum",
                    descriptor_id=item.region_id,
                    observed=item.atom_count,
                    threshold=policy.minimum_fixed_region_atoms,
                )
            )
        if item.thickness_angstrom < policy.minimum_fixed_region_thickness_angstrom:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "fixed_region.thickness_below_minimum",
                    descriptor_id=item.region_id,
                    observed=item.thickness_angstrom,
                    threshold=policy.minimum_fixed_region_thickness_angstrom,
                )
            )
        elif (
            policy.warning_fixed_region_thickness_angstrom
            and item.thickness_angstrom < policy.warning_fixed_region_thickness_angstrom
        ):
            reasons.append(
                ValidationReason(
                    "warning",
                    "fixed_region.thickness_warning",
                    descriptor_id=item.region_id,
                    observed=item.thickness_angstrom,
                    threshold=policy.warning_fixed_region_thickness_angstrom,
                )
            )
    for item in buffer:
        if item.undeclared_grain_atom_count:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "buffer_region.contains_undeclared_grain",
                    descriptor_id=item.region_id,
                    observed=item.undeclared_grain_atom_count,
                    threshold=0,
                )
            )
        if item.atom_count < policy.minimum_buffer_region_atoms:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "buffer_region.population_below_minimum",
                    descriptor_id=item.region_id,
                    observed=item.atom_count,
                    threshold=policy.minimum_buffer_region_atoms,
                )
            )
        if item.thickness_angstrom < policy.minimum_buffer_thickness_angstrom:
            reasons.append(
                ValidationReason(
                    "infeasible",
                    "buffer_region.thickness_below_minimum",
                    descriptor_id=item.region_id,
                    observed=item.thickness_angstrom,
                    threshold=policy.minimum_buffer_thickness_angstrom,
                )
            )
        elif (
            policy.warning_buffer_thickness_angstrom
            and item.thickness_angstrom < policy.warning_buffer_thickness_angstrom
        ):
            reasons.append(
                ValidationReason(
                    "warning",
                    "buffer_region.thickness_warning",
                    descriptor_id=item.region_id,
                    observed=item.thickness_angstrom,
                    threshold=policy.warning_buffer_thickness_angstrom,
                )
            )
    return (
        SlabValidationMetrics(
            surfaces=tuple(surfaces),
            vacuum_regions=vacuum,
            fixed_regions=fixed,
            buffer_regions=buffer,
        ),
        reasons,
    )


def _sort_reasons(reasons: Sequence[ValidationReason]) -> tuple[ValidationReason, ...]:
    unique: dict[tuple[Any, ...], ValidationReason] = {}
    for reason in reasons:
        key = (
            reason.status,
            reason.code,
            reason.descriptor_id,
            reason.species,
            reason.observed,
            reason.threshold,
            reason.message,
        )
        unique[key] = reason
    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                -_STATUS_RANK[item.status],
                item.descriptor_id or "",
                item.code,
                item.species or ("", ""),
            ),
        )
    )


def _status_from_reasons(reasons: Sequence[ValidationReason]) -> FeasibilityStatus:
    if not reasons:
        return "feasible"
    return max(
        (reason.status for reason in reasons),
        key=lambda item: _STATUS_RANK[item],
    )  # type: ignore[return-value]


def validate_bicrystal_state(
    state: BicrystalState,
    *,
    policy: FeasibilityPolicy | None = None,
    override: FeasibilityOverride | None = None,
) -> BicrystalFeasibilityReport:
    """Return a strict feasibility report for every physical interface and slab region.

    Validation is deterministic and read-only. Both interfaces of a periodic
    bicrystal are evaluated. A slab's declared vacuum is measured only by slab rules
    and is never included as a grain-boundary local gap.
    """
    if not isinstance(state, BicrystalState):
        raise GeometryValidationError("state must be a BicrystalState instance.")
    selected = FeasibilityPolicy() if policy is None else policy
    if not isinstance(selected, FeasibilityPolicy):
        raise GeometryValidationError("policy must be a FeasibilityPolicy instance.")
    if override is not None and not isinstance(override, FeasibilityOverride):
        raise GeometryValidationError("override must be a FeasibilityOverride instance.")

    structure_hash = state.structure_hash
    reasons = _topology_reasons(state)
    duplicates: tuple[DuplicatePairMetrics, ...] = ()
    interface_metrics: list[InterfaceFeasibilityMetrics] = []
    slab: SlabValidationMetrics | None = None
    try:
        atoms = _atom_view(state)
        duplicates = _duplicate_metrics(state, atoms, selected.contact)
        if duplicates:
            reasons.append(
                ValidationReason(
                    "invalid",
                    "structure.periodic_duplicate_representatives",
                    observed=len(duplicates),
                    threshold=0,
                )
            )
        for interface in state.interfaces:
            metrics, interface_reasons = _interface_metrics_and_reasons(
                state, atoms, interface, selected
            )
            interface_metrics.append(metrics)
            reasons.extend(interface_reasons)
        slab, slab_reasons = _slab_metrics_and_reasons(
            state, atoms, selected.slab
        )
        reasons.extend(slab_reasons)
    except (GeometryAuditError, ValueError, FloatingPointError) as exc:
        reasons.append(
            ValidationReason(
                "invalid",
                "validation.measurement_failure",
                message=f"{type(exc).__name__}: {exc}",
            )
        )

    raw_reasons = _sort_reasons(reasons)
    raw_status = _status_from_reasons(raw_reasons)
    effective_status: FeasibilityStatus = raw_status
    effective_reasons = raw_reasons
    if override is not None:
        if raw_status == "invalid":
            raise GeometryValidationError("An invalid result cannot be overridden.")
        effective_status = override.status
        effective_reasons = _sort_reasons(
            (*raw_reasons, ValidationReason("warning", "override.applied", message=override.reason))
        )
    if state.structure_hash != structure_hash:
        raise RuntimeError("Feasibility validation modified the BicrystalState structure.")
    return BicrystalFeasibilityReport(
        raw_status=raw_status,
        status=effective_status,
        reasons=effective_reasons,
        raw_reasons=raw_reasons,
        duplicate_pairs=duplicates,
        interfaces=tuple(interface_metrics),
        slab=slab,
        structure_hash=structure_hash,
        state_hash=state.state_hash,
        policy=selected,
        override=override,
    )


__all__ = [
    "BicrystalFeasibilityReport",
    "ContactPolicy",
    "DuplicatePairMetrics",
    "FeasibilityOverride",
    "FeasibilityPolicy",
    "FeasibilityStatus",
    "GeometryValidationError",
    "InterfaceFeasibilityMetrics",
    "RegionValidationMetrics",
    "ResolvedContactThresholds",
    "SlabPolicy",
    "SlabValidationMetrics",
    "SpeciesPairContactMetrics",
    "SpeciesPairThresholds",
    "SurfaceValidationMetrics",
    "ValidationReason",
    "VoidPolicy",
    "validate_bicrystal_state",
]
