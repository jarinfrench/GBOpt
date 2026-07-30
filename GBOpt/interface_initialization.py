# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Deterministic rigid-translation initialization for bicrystal seeds.

This module composes the topology-aware :func:`translate_grain` primitive with
strict :func:`validate_bicrystal_state` classification.  It does not evaluate
energy, relax structures, enumerate terminations, or participate in optimizer
checkpointing.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, TypeAlias

import numpy as np

from GBOpt.BicrystalState import (
    BicrystalState,
    BicrystalStateError,
    translate_grain,
)
from GBOpt.geometry_validation import (
    BicrystalFeasibilityReport,
    FeasibilityOverride,
    FeasibilityPolicy,
    FeasibilityStatus,
    GeometryValidationError,
    validate_bicrystal_state,
)


INITIALIZATION_SCHEMA_VERSION = 1

CandidateKind: TypeAlias = Literal["zero", "in_plane", "normal", "combined"]
AttemptDisposition: TypeAlias = Literal[
    "retained",
    "rejected_status",
    "periodic_equivalent",
    "duplicate_structure",
    "translation_error",
    "validation_error",
]
InitializationStatus: TypeAlias = Literal[
    "zero_translation_accepted",
    "translated_seeds_retained",
    "seed_limit_reached",
    "translation_domain_exhausted",
    "invalid_input",
]

_VALID_CANDIDATE_KINDS = frozenset({"zero", "in_plane", "normal", "combined"})
_VALID_DISPOSITIONS = frozenset(
    {
        "retained",
        "rejected_status",
        "periodic_equivalent",
        "duplicate_structure",
        "translation_error",
        "validation_error",
    }
)
_VALID_FEASIBILITY_STATUSES = frozenset(
    {"invalid", "infeasible", "warning", "feasible"}
)
_VALID_RESULT_STATUSES = frozenset(
    {
        "zero_translation_accepted",
        "translated_seeds_retained",
        "seed_limit_reached",
        "translation_domain_exhausted",
        "invalid_input",
    }
)


class InterfaceInitializationError(ValueError):
    """Raised when a translation-initialization configuration is malformed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise InterfaceInitializationError(f"{name} must be a finite float.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InterfaceInitializationError(
            f"{name} must be a finite float; got {value!r}."
        ) from exc
    if not math.isfinite(result):
        raise InterfaceInitializationError(
            f"{name} must be finite; got {result!r}."
        )
    return 0.0 if result == 0.0 else result


def _axis(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise InterfaceInitializationError(
            f"{name} must be an integer axis 0, 1, or 2."
        )
    result = int(value)
    if result not in (0, 1, 2):
        raise InterfaceInitializationError(
            f"{name} must be 0, 1, or 2; got {result}."
        )
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise InterfaceInitializationError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise InterfaceInitializationError(f"{name} must be positive; got {result}.")
    return result


def _float_tuple(
    values: object,
    name: str,
    *,
    allow_empty: bool = False,
    reject_duplicates: bool = True,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise InterfaceInitializationError(
            f"{name} must be a sequence of finite floats."
        )
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise InterfaceInitializationError(
            f"{name} must be a sequence of finite floats."
        ) from exc
    if not raw and not allow_empty:
        raise InterfaceInitializationError(f"{name} must not be empty.")
    normalized = tuple(
        _finite_float(value, f"{name}[{index}]")
        for index, value in enumerate(raw)
    )
    if reject_duplicates and len(set(normalized)) != len(normalized):
        raise InterfaceInitializationError(f"{name} must not contain duplicates.")
    return normalized


def _vector(values: object, name: str) -> tuple[float, float, float]:
    normalized = _float_tuple(values, name, reject_duplicates=False)
    if len(normalized) != 3:
        raise InterfaceInitializationError(
            f"{name} must contain exactly three components; got {len(normalized)}."
        )
    return normalized  # type: ignore[return-value]


def _derived_axes(state: BicrystalState) -> tuple[int, tuple[int, int]]:
    axes = {interface.axis for interface in state.interfaces}
    if len(axes) != 1:
        raise InterfaceInitializationError(
            "All physical interfaces must share one explicit normal axis."
        )
    normal_axis = next(iter(axes))
    in_plane_axes = tuple(axis for axis in range(3) if axis != normal_axis)
    if len(in_plane_axes) != 2:  # defensive against future state schemas
        raise InterfaceInitializationError(
            "The interface topology does not define exactly two in-plane axes."
        )
    return normal_axis, in_plane_axes  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class TranslationCandidate:
    """One ordered lab-frame displacement generated by a resolved domain."""

    order: int
    kind: CandidateKind
    displacement_lab: tuple[float, float, float]

    def __post_init__(self) -> None:
        if isinstance(self.order, (bool, np.bool_)) or not isinstance(
            self.order, (int, np.integer)
        ):
            raise InterfaceInitializationError("candidate order must be an integer.")
        if int(self.order) < 0:
            raise InterfaceInitializationError("candidate order must be nonnegative.")
        if self.kind not in _VALID_CANDIDATE_KINDS:
            raise InterfaceInitializationError(
                f"Unsupported candidate kind {self.kind!r}."
            )
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(
            self,
            "displacement_lab",
            _vector(self.displacement_lab, "displacement_lab"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "kind": self.kind,
            "displacement_lab": list(self.displacement_lab),
        }


@dataclass(frozen=True, slots=True)
class CartesianTranslationDomain:
    """Finite Cartesian in-plane grid plus a bounded set of normal offsets.

    ``in_plane_components`` contains one finite component sequence for each
    resolved in-plane lab axis.  The sequences are treated as sets and sorted by
    the documented candidate-ordering rule; their input order does not rank
    candidates.  ``normal_offsets`` is similarly ordered by increasing absolute
    magnitude, with negative preceding positive at equal magnitude.

    Axis fields may be omitted and are then resolved from the state's interface
    descriptors.  When supplied, they must exactly match those descriptors.
    """

    in_plane_components: tuple[tuple[float, ...], tuple[float, ...]]
    normal_offsets: tuple[float, ...] = (0.0,)
    normal_axis: int | None = None
    in_plane_axes: tuple[int, int] | None = None
    schema_version: int = INITIALIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.in_plane_components, (str, bytes)):
            raise InterfaceInitializationError(
                "in_plane_components must contain two component sequences."
            )
        try:
            raw_components = tuple(self.in_plane_components)
        except TypeError as exc:
            raise InterfaceInitializationError(
                "in_plane_components must contain two component sequences."
            ) from exc
        if len(raw_components) != 2:
            raise InterfaceInitializationError(
                "in_plane_components must contain exactly two component sequences."
            )
        components = tuple(
            tuple(sorted(_float_tuple(values, f"in_plane_components[{index}]")))
            for index, values in enumerate(raw_components)
        )
        normal_offsets = tuple(
            sorted(_float_tuple(self.normal_offsets, "normal_offsets"))
        )

        if (self.normal_axis is None) != (self.in_plane_axes is None):
            raise InterfaceInitializationError(
                "normal_axis and in_plane_axes must either both be omitted "
                "or both be supplied."
            )
        normal_axis: int | None = None
        in_plane_axes: tuple[int, int] | None = None
        if self.normal_axis is not None:
            normal_axis = _axis(self.normal_axis, "normal_axis")
            try:
                raw_axes = tuple(self.in_plane_axes or ())
            except TypeError as exc:
                raise InterfaceInitializationError(
                    "in_plane_axes must contain two integer axes."
                ) from exc
            if len(raw_axes) != 2:
                raise InterfaceInitializationError(
                    "in_plane_axes must contain exactly two axes."
                )
            in_plane_axes = tuple(
                _axis(value, f"in_plane_axes[{index}]")
                for index, value in enumerate(raw_axes)
            )  # type: ignore[assignment]
            if tuple(sorted((*in_plane_axes, normal_axis))) != (0, 1, 2):
                raise InterfaceInitializationError(
                    "normal_axis and in_plane_axes must be a permutation of (0, 1, 2)."
                )
            expected = tuple(axis for axis in range(3) if axis != normal_axis)
            if in_plane_axes != expected:
                raise InterfaceInitializationError(
                    "in_plane_axes must be in increasing lab-axis order."
                )

        if isinstance(self.schema_version, (bool, np.bool_)) or not isinstance(
            self.schema_version, (int, np.integer)
        ):
            raise InterfaceInitializationError("schema_version must be an integer.")
        if int(self.schema_version) != INITIALIZATION_SCHEMA_VERSION:
            raise InterfaceInitializationError(
                f"Unsupported schema_version={self.schema_version}; expected "
                f"{INITIALIZATION_SCHEMA_VERSION}."
            )

        object.__setattr__(self, "in_plane_components", components)
        object.__setattr__(self, "normal_offsets", normal_offsets)
        object.__setattr__(self, "normal_axis", normal_axis)
        object.__setattr__(self, "in_plane_axes", in_plane_axes)
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def resolve_for(self, state: BicrystalState) -> "CartesianTranslationDomain":
        """Return an axis-explicit domain consistent with ``state`` topology."""
        if not isinstance(state, BicrystalState):
            raise InterfaceInitializationError("state must be a BicrystalState instance.")
        normal_axis, in_plane_axes = _derived_axes(state)
        if self.normal_axis is not None and self.normal_axis != normal_axis:
            raise InterfaceInitializationError(
                f"normal_axis={self.normal_axis} conflicts with interface axis {normal_axis}."
            )
        if self.in_plane_axes is not None and self.in_plane_axes != in_plane_axes:
            raise InterfaceInitializationError(
                f"in_plane_axes={self.in_plane_axes} conflict with topology-derived "
                f"axes {in_plane_axes}."
            )
        return CartesianTranslationDomain(
            in_plane_components=self.in_plane_components,
            normal_offsets=self.normal_offsets,
            normal_axis=normal_axis,
            in_plane_axes=in_plane_axes,
        )

    def ordered_candidates(self) -> tuple[TranslationCandidate, ...]:
        """Return candidates in the Phase 6 reproducibility order.

        The domain must be axis-explicit.  Ordering is zero, nonzero in-plane
        shifts by squared norm then lab-vector lexicographic order, pure normal
        offsets by absolute magnitude with negative first, and combined shifts
        in in-plane-major/normal-minor nested order.
        """
        if self.normal_axis is None or self.in_plane_axes is None:
            raise InterfaceInitializationError(
                "Resolve the translation domain against a BicrystalState first."
            )

        in_plane: list[tuple[float, float, float]] = []
        first_axis, second_axis = self.in_plane_axes
        for first, second in product(*self.in_plane_components):
            vector = [0.0, 0.0, 0.0]
            vector[first_axis] = first
            vector[second_axis] = second
            displacement = tuple(vector)
            if displacement != (0.0, 0.0, 0.0):
                in_plane.append(displacement)  # type: ignore[arg-type]
        in_plane.sort(
            key=lambda vector: (
                sum(component * component for component in vector),
                vector,
            )
        )

        offsets = [value for value in self.normal_offsets if value != 0.0]
        offsets.sort(key=lambda value: (abs(value), 0 if value < 0.0 else 1, value))

        normal: list[tuple[float, float, float]] = []
        for offset in offsets:
            vector = [0.0, 0.0, 0.0]
            vector[self.normal_axis] = offset
            normal.append(tuple(vector))  # type: ignore[arg-type]

        combined: list[tuple[float, float, float]] = []
        for in_plane_vector in in_plane:
            for offset in offsets:
                vector = list(in_plane_vector)
                vector[self.normal_axis] = offset
                combined.append(tuple(vector))  # type: ignore[arg-type]

        ordered: list[tuple[CandidateKind, tuple[float, float, float]]] = [
            ("zero", (0.0, 0.0, 0.0)),
            *(("in_plane", vector) for vector in in_plane),
            *(("normal", vector) for vector in normal),
            *(("combined", vector) for vector in combined),
        ]
        return tuple(
            TranslationCandidate(order=index, kind=kind, displacement_lab=vector)
            for index, (kind, vector) in enumerate(ordered)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "in_plane_components": [list(values) for values in self.in_plane_components],
            "normal_offsets": list(self.normal_offsets),
            "normal_axis": self.normal_axis,
            "in_plane_axes": (
                None if self.in_plane_axes is None else list(self.in_plane_axes)
            ),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def domain_hash(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class TranslationAttempt:
    """Complete deterministic record for one attempted domain displacement."""

    candidate: TranslationCandidate
    canonical_displacement_lab: tuple[float, float, float]
    disposition: AttemptDisposition
    validation_status: FeasibilityStatus | None
    structure_hash: str | None
    state_hash: str | None
    report: BicrystalFeasibilityReport | None
    rejection_reasons: tuple[str, ...] = ()
    error_message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, TranslationCandidate):
            raise InterfaceInitializationError(
                "candidate must be a TranslationCandidate."
            )
        if self.disposition not in _VALID_DISPOSITIONS:
            raise InterfaceInitializationError(
                f"Unsupported attempt disposition {self.disposition!r}."
            )
        canonical = _vector(
            self.canonical_displacement_lab, "canonical_displacement_lab"
        )
        if (
            self.validation_status is not None
            and self.validation_status not in _VALID_FEASIBILITY_STATUSES
        ):
            raise InterfaceInitializationError(
                f"Unsupported validation status {self.validation_status!r}."
            )
        if self.report is not None and not isinstance(
            self.report, BicrystalFeasibilityReport
        ):
            raise InterfaceInitializationError(
                "report must be a BicrystalFeasibilityReport or None."
            )
        if self.report is not None:
            if self.validation_status != self.report.status:
                raise InterfaceInitializationError(
                    "validation_status must match report.status."
                )
            if self.structure_hash != self.report.structure_hash:
                raise InterfaceInitializationError(
                    "structure_hash must match report.structure_hash."
                )
            if self.state_hash != self.report.state_hash:
                raise InterfaceInitializationError(
                    "state_hash must match report.state_hash."
                )
        elif self.validation_status is not None:
            raise InterfaceInitializationError(
                "validation_status requires a feasibility report."
            )
        if self.disposition in {"translation_error", "validation_error"}:
            if self.report is not None:
                raise InterfaceInitializationError(
                    "Error attempts must not carry a feasibility report."
                )
        elif self.report is None:
            raise InterfaceInitializationError(
                "Classified attempts require a feasibility report."
            )
        if self.disposition == "retained" and self.validation_status not in {
            "feasible",
            "warning",
        }:
            raise InterfaceInitializationError(
                "Retained attempts must be feasible or warning."
            )
        reasons = tuple(str(reason) for reason in self.rejection_reasons)
        if any(not reason for reason in reasons):
            raise InterfaceInitializationError(
                "rejection_reasons must contain non-empty reason codes."
            )
        if self.disposition == "retained" and reasons:
            raise InterfaceInitializationError(
                "Retained attempts must not contain rejection reasons."
            )
        if not isinstance(self.error_message, str):
            raise InterfaceInitializationError("error_message must be a string.")
        object.__setattr__(self, "canonical_displacement_lab", canonical)
        object.__setattr__(self, "rejection_reasons", reasons)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "canonical_displacement_lab": list(self.canonical_displacement_lab),
            "disposition": self.disposition,
            "validation_status": self.validation_status,
            "structure_hash": self.structure_hash,
            "state_hash": self.state_hash,
            "report": None if self.report is None else self.report.to_dict(),
            "rejection_reasons": list(self.rejection_reasons),
            "error_message": self.error_message,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["attempt_hash"] = self.attempt_hash
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def attempt_hash(self) -> str:
        return _sha256(self._hash_payload())


@dataclass(frozen=True, slots=True)
class TranslationSeed:
    """One retained state with its strict report and reconstruction inputs."""

    candidate: TranslationCandidate
    canonical_displacement_lab: tuple[float, float, float]
    state: BicrystalState
    report: BicrystalFeasibilityReport

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, TranslationCandidate):
            raise InterfaceInitializationError("candidate must be a TranslationCandidate.")
        if not isinstance(self.state, BicrystalState):
            raise InterfaceInitializationError("state must be a BicrystalState.")
        if not isinstance(self.report, BicrystalFeasibilityReport):
            raise InterfaceInitializationError(
                "report must be a BicrystalFeasibilityReport."
            )
        if self.report.structure_hash != self.state.structure_hash:
            raise InterfaceInitializationError(
                "report and retained state structure hashes do not match."
            )
        if self.report.state_hash != self.state.state_hash:
            raise InterfaceInitializationError(
                "report and retained state hashes do not match."
            )
        if self.report.status not in {"feasible", "warning"}:
            raise InterfaceInitializationError(
                "A retained seed must have feasible or warning status."
            )
        object.__setattr__(
            self,
            "canonical_displacement_lab",
            _vector(self.canonical_displacement_lab, "canonical_displacement_lab"),
        )

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "canonical_displacement_lab": list(self.canonical_displacement_lab),
            "structure_hash": self.state.structure_hash,
            "state_hash": self.state.state_hash,
            "report_hash": self.report.report_hash,
            "state_manifest": self.state.manifest(),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["report"] = self.report.to_dict()
        payload["seed_hash"] = self.seed_hash
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def seed_hash(self) -> str:
        return _sha256(self._hash_payload())


@dataclass(frozen=True, slots=True)
class TranslationSearchResult:
    """Deterministic outcome of one finite rigid-translation seed search."""

    status: InitializationStatus
    attempts: tuple[TranslationAttempt, ...]
    seeds: tuple[TranslationSeed, ...]
    max_seeds: int | None
    retain_warnings: bool
    source_structure_hash: str | None
    source_state_hash: str | None
    translation_domain: CartesianTranslationDomain | None
    seed_limit_reached: bool
    domain_exhausted: bool
    invalid_reasons: tuple[str, ...] = ()
    phase7_handoff: str | None = None
    schema_version: int = INITIALIZATION_SCHEMA_VERSION
    feasibility_override: FeasibilityOverride | None = None

    def __post_init__(self) -> None:
        attempts = tuple(self.attempts)
        seeds = tuple(self.seeds)
        invalid_reasons = tuple(str(reason) for reason in self.invalid_reasons)
        if self.status not in _VALID_RESULT_STATUSES:
            raise InterfaceInitializationError(
                f"Unsupported result status {self.status!r}."
            )
        if any(not isinstance(item, TranslationAttempt) for item in attempts):
            raise InterfaceInitializationError(
                "attempts must contain TranslationAttempt instances."
            )
        if any(not isinstance(item, TranslationSeed) for item in seeds):
            raise InterfaceInitializationError(
                "seeds must contain TranslationSeed instances."
            )
        if any(not reason for reason in invalid_reasons):
            raise InterfaceInitializationError(
                "invalid_reasons must contain non-empty strings."
            )
        if not isinstance(self.retain_warnings, bool):
            raise InterfaceInitializationError("retain_warnings must be a bool.")
        if self.feasibility_override is not None and not isinstance(
            self.feasibility_override, FeasibilityOverride
        ):
            raise InterfaceInitializationError(
                "feasibility_override must be a FeasibilityOverride or None."
            )
        if not isinstance(self.seed_limit_reached, bool) or not isinstance(
            self.domain_exhausted, bool
        ):
            raise InterfaceInitializationError(
                "seed_limit_reached and domain_exhausted must be bool values."
            )
        if self.max_seeds is not None:
            _positive_int(self.max_seeds, "max_seeds")
            if len(seeds) > self.max_seeds:
                raise InterfaceInitializationError(
                    "Retained seed count exceeds max_seeds."
                )
        if self.translation_domain is not None and not isinstance(
            self.translation_domain, CartesianTranslationDomain
        ):
            raise InterfaceInitializationError(
                "translation_domain must be a CartesianTranslationDomain or None."
            )
        if self.seed_limit_reached and self.domain_exhausted:
            raise InterfaceInitializationError(
                "A search cannot stop at the seed limit and exhaust the domain."
            )
        if self.status == "invalid_input":
            if seeds or self.seed_limit_reached or self.domain_exhausted:
                raise InterfaceInitializationError(
                    "invalid_input cannot retain seeds or report normal completion."
                )
            if not invalid_reasons:
                raise InterfaceInitializationError(
                    "invalid_input requires at least one reason."
                )
        else:
            if self.translation_domain is None:
                raise InterfaceInitializationError(
                    "Completed searches require a resolved translation domain."
                )
            if (
                self.translation_domain.normal_axis is None
                or self.translation_domain.in_plane_axes is None
            ):
                raise InterfaceInitializationError(
                    "Completed searches require an axis-explicit translation domain."
                )
            if self.max_seeds is None:
                raise InterfaceInitializationError(
                    "Completed searches require max_seeds."
                )
            if self.source_structure_hash is None or self.source_state_hash is None:
                raise InterfaceInitializationError(
                    "Completed searches require source hashes."
                )
            if invalid_reasons:
                raise InterfaceInitializationError(
                    "Completed searches must not carry invalid-input reasons."
                )
        if self.status == "seed_limit_reached":
            if not self.seed_limit_reached or self.max_seeds != len(seeds):
                raise InterfaceInitializationError(
                    "seed_limit_reached requires exactly max_seeds retained seeds."
                )
        elif self.seed_limit_reached:
            raise InterfaceInitializationError(
                "seed_limit_reached flag requires matching result status."
            )
        if self.status == "translation_domain_exhausted":
            if seeds or not self.domain_exhausted:
                raise InterfaceInitializationError(
                    "translation_domain_exhausted requires no retained seeds "
                    "and an exhausted domain."
                )
            if self.phase7_handoff != "termination_enumeration_required":
                raise InterfaceInitializationError(
                    "Exhaustion must carry the Phase 7 termination handoff."
                )
        elif self.phase7_handoff is not None:
            raise InterfaceInitializationError(
                "Only translation-domain exhaustion carries a Phase 7 handoff."
            )
        if self.status == "zero_translation_accepted":
            if not self.domain_exhausted or len(seeds) != 1:
                raise InterfaceInitializationError(
                    "zero_translation_accepted requires one seed and domain exhaustion."
                )
            if seeds[0].candidate.kind != "zero":
                raise InterfaceInitializationError(
                    "zero_translation_accepted requires the identity seed."
                )
        if self.status == "translated_seeds_retained":
            if not self.domain_exhausted or not any(
                seed.candidate.kind != "zero" for seed in seeds
            ):
                raise InterfaceInitializationError(
                    "translated_seeds_retained requires an exhausted domain and "
                    "at least one translated seed."
                )
        structure_hashes = [seed.state.structure_hash for seed in seeds]
        if len(structure_hashes) != len(set(structure_hashes)):
            raise InterfaceInitializationError(
                "Retained seeds must have unique structure hashes."
            )
        canonical = [seed.canonical_displacement_lab for seed in seeds]
        if len(canonical) != len(set(canonical)):
            raise InterfaceInitializationError(
                "Retained seeds must have unique canonical displacements."
            )
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "invalid_reasons", invalid_reasons)

    @property
    def zero_translation_accepted(self) -> bool:
        return bool(self.seeds and self.seeds[0].candidate.kind == "zero")

    @property
    def translated_seed_count(self) -> int:
        return sum(seed.candidate.kind != "zero" for seed in self.seeds)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "seeds": [seed.to_dict() for seed in self.seeds],
            "max_seeds": self.max_seeds,
            "retain_warnings": self.retain_warnings,
            "source_structure_hash": self.source_structure_hash,
            "source_state_hash": self.source_state_hash,
            "translation_domain": (
                None if self.translation_domain is None else self.translation_domain.to_dict()
            ),
            "domain_hash": (
                None
                if self.translation_domain is None
                else self.translation_domain.domain_hash
            ),
            "seed_limit_reached": self.seed_limit_reached,
            "domain_exhausted": self.domain_exhausted,
            "zero_translation_accepted": self.zero_translation_accepted,
            "translated_seed_count": self.translated_seed_count,
            "invalid_reasons": list(self.invalid_reasons),
            "phase7_handoff": self.phase7_handoff,
            "feasibility_override": (
                None
                if self.feasibility_override is None
                else {
                    "status": self.feasibility_override.status,
                    "reason": self.feasibility_override.reason,
                }
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._hash_payload()
        payload["result_hash"] = self.result_hash
        return payload

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def result_hash(self) -> str:
        return _sha256(self._hash_payload())


@dataclass(frozen=True, slots=True)
class InterfaceInitializer:
    """Strict deterministic initializer for one valid source bicrystal state."""

    state: BicrystalState
    feasibility_policy: FeasibilityPolicy
    retain_warnings: bool = False
    feasibility_override: FeasibilityOverride | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, BicrystalState):
            raise InterfaceInitializationError(
                "state must be a BicrystalState instance."
            )
        if not isinstance(self.feasibility_policy, FeasibilityPolicy):
            raise InterfaceInitializationError(
                "feasibility_policy must be a FeasibilityPolicy instance."
            )
        if not isinstance(self.retain_warnings, bool):
            raise InterfaceInitializationError("retain_warnings must be a bool.")
        if self.feasibility_override is not None and not isinstance(
            self.feasibility_override, FeasibilityOverride
        ):
            raise InterfaceInitializationError(
                "feasibility_override must be a FeasibilityOverride or None."
            )
        _derived_axes(self.state)

    def generate_translation_seeds(
        self,
        *,
        translation_domain: CartesianTranslationDomain,
        max_seeds: int,
    ) -> TranslationSearchResult:
        """Enumerate, classify, deduplicate, and retain acceptable placements."""
        try:
            limit = _positive_int(max_seeds, "max_seeds")
            if not isinstance(translation_domain, CartesianTranslationDomain):
                raise InterfaceInitializationError(
                    "translation_domain must be a CartesianTranslationDomain instance."
                )
            domain = translation_domain.resolve_for(self.state)
        except InterfaceInitializationError as exc:
            return _invalid_result(
                str(exc),
                state=self.state,
                retain_warnings=self.retain_warnings,
                max_seeds=None,
                feasibility_override=self.feasibility_override,
                domain=(
                    translation_domain
                    if isinstance(translation_domain, CartesianTranslationDomain)
                    else None
                ),
            )

        source_structure_hash = self.state.structure_hash
        source_state_hash = self.state.state_hash
        periodic_axes = tuple(
            axis
            for axis, condition in enumerate(self.state.boundary_conditions)
            if condition == "periodic"
        )
        lengths = tuple(
            float(self.state.box_dims[axis, 1] - self.state.box_dims[axis, 0])
            for axis in range(3)
        )
        accepted_statuses = {"feasible"}
        if self.retain_warnings:
            accepted_statuses.add("warning")

        attempts: list[TranslationAttempt] = []
        seeds: list[TranslationSeed] = []
        seen_canonical: set[tuple[float, float, float]] = set()
        seen_structures: set[str] = set()
        candidates = domain.ordered_candidates()

        for candidate in candidates:
            canonical = _canonical_displacement(
                candidate.displacement_lab,
                periodic_axes=periodic_axes,
                lengths=lengths,
            )
            try:
                if candidate.kind == "zero":
                    candidate_state = self.state
                else:
                    candidate_state = translate_grain(
                        self.state,
                        grain="right",
                        displacement=candidate.displacement_lab,
                        coordinates="lab",
                    )
            except (BicrystalStateError, ValueError, TypeError) as exc:
                attempts.append(
                    TranslationAttempt(
                        candidate=candidate,
                        canonical_displacement_lab=canonical,
                        disposition="translation_error",
                        validation_status=None,
                        structure_hash=None,
                        state_hash=None,
                        report=None,
                        rejection_reasons=("initializer.translation_error",),
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )
                seen_canonical.add(canonical)
                continue

            try:
                report = validate_bicrystal_state(
                    candidate_state,
                    policy=self.feasibility_policy,
                    override=self.feasibility_override,
                )
            except (GeometryValidationError, ValueError, TypeError) as exc:
                # The Phase 4 public API normally converts measurement failures into
                # an invalid report. Preserve a deterministic invalid-input outcome
                # if the validator itself rejects the source invocation.
                if candidate.kind == "zero":
                    return _invalid_result(
                        f"{type(exc).__name__}: {exc}",
                        state=self.state,
                        retain_warnings=self.retain_warnings,
                        max_seeds=limit,
                        domain=domain,
                        feasibility_override=self.feasibility_override,
                    )
                attempts.append(
                    TranslationAttempt(
                        candidate=candidate,
                        canonical_displacement_lab=canonical,
                        disposition="validation_error",
                        validation_status=None,
                        structure_hash=candidate_state.structure_hash,
                        state_hash=candidate_state.state_hash,
                        report=None,
                        rejection_reasons=("initializer.validation_error",),
                        error_message=f"{type(exc).__name__}: {exc}",
                    )
                )
                seen_canonical.add(canonical)
                seen_structures.add(candidate_state.structure_hash)
                continue

            if candidate.kind == "zero" and report.status == "invalid":
                invalid_attempt = TranslationAttempt(
                    candidate=candidate,
                    canonical_displacement_lab=canonical,
                    disposition="rejected_status",
                    validation_status=report.status,
                    structure_hash=candidate_state.structure_hash,
                    state_hash=candidate_state.state_hash,
                    report=report,
                    rejection_reasons=tuple(reason.code for reason in report.reasons),
                )
                return TranslationSearchResult(
                    status="invalid_input",
                    attempts=(invalid_attempt,),
                    seeds=(),
                    max_seeds=limit,
                    retain_warnings=self.retain_warnings,
                    feasibility_override=self.feasibility_override,
                    source_structure_hash=source_structure_hash,
                    source_state_hash=source_state_hash,
                    translation_domain=domain,
                    seed_limit_reached=False,
                    domain_exhausted=False,
                    invalid_reasons=("initializer.source_state_invalid",),
                )

            duplicate_reason: str | None = None
            disposition: AttemptDisposition
            if canonical in seen_canonical:
                disposition = "periodic_equivalent"
                duplicate_reason = "initializer.periodic_equivalent_translation"
            elif candidate_state.structure_hash in seen_structures:
                disposition = "duplicate_structure"
                duplicate_reason = "initializer.duplicate_structure"
            elif report.status not in accepted_statuses:
                disposition = "rejected_status"
            else:
                disposition = "retained"

            if duplicate_reason is not None:
                rejection_reasons = (duplicate_reason,)
            elif disposition == "rejected_status":
                rejection_reasons = tuple(reason.code for reason in report.reasons)
                if report.status == "warning" and not self.retain_warnings:
                    rejection_reasons = (
                        "initializer.warning_not_retainable",
                        *rejection_reasons,
                    )
            else:
                rejection_reasons = ()

            attempt = TranslationAttempt(
                candidate=candidate,
                canonical_displacement_lab=canonical,
                disposition=disposition,
                validation_status=report.status,
                structure_hash=candidate_state.structure_hash,
                state_hash=candidate_state.state_hash,
                report=report,
                rejection_reasons=rejection_reasons,
            )
            attempts.append(attempt)
            seen_canonical.add(canonical)
            seen_structures.add(candidate_state.structure_hash)

            if disposition == "retained":
                seeds.append(
                    TranslationSeed(
                        candidate=candidate,
                        canonical_displacement_lab=canonical,
                        state=candidate_state,
                        report=report,
                    )
                )
                if len(seeds) == limit:
                    if (
                        self.state.structure_hash != source_structure_hash
                        or self.state.state_hash != source_state_hash
                    ):
                        raise RuntimeError(
                            "Translation initialization modified the source state."
                        )
                    return TranslationSearchResult(
                        status="seed_limit_reached",
                        attempts=tuple(attempts),
                        seeds=tuple(seeds),
                        max_seeds=limit,
                        retain_warnings=self.retain_warnings,
                        feasibility_override=self.feasibility_override,
                        source_structure_hash=source_structure_hash,
                        source_state_hash=source_state_hash,
                        translation_domain=domain,
                        seed_limit_reached=True,
                        domain_exhausted=False,
                    )

        if (
            self.state.structure_hash != source_structure_hash
            or self.state.state_hash != source_state_hash
        ):
            raise RuntimeError(
                "Translation initialization modified the source state."
            )
        if not seeds:
            status: InitializationStatus = "translation_domain_exhausted"
            handoff = "termination_enumeration_required"
        elif any(seed.candidate.kind != "zero" for seed in seeds):
            status = "translated_seeds_retained"
            handoff = None
        else:
            status = "zero_translation_accepted"
            handoff = None
        return TranslationSearchResult(
            status=status,
            attempts=tuple(attempts),
            seeds=tuple(seeds),
            max_seeds=limit,
            retain_warnings=self.retain_warnings,
            feasibility_override=self.feasibility_override,
            source_structure_hash=source_structure_hash,
            source_state_hash=source_state_hash,
            translation_domain=domain,
            seed_limit_reached=False,
            domain_exhausted=True,
            phase7_handoff=handoff,
        )


def _canonical_displacement(
    displacement: tuple[float, float, float],
    *,
    periodic_axes: tuple[int, ...],
    lengths: tuple[float, float, float],
) -> tuple[float, float, float]:
    canonical = list(displacement)
    for axis in periodic_axes:
        value = float(np.remainder(canonical[axis], lengths[axis]))
        canonical[axis] = 0.0 if value == 0.0 else value
    return tuple(canonical)  # type: ignore[return-value]


def _invalid_result(
    reason: str,
    *,
    state: BicrystalState | None,
    retain_warnings: bool,
    max_seeds: int | None,
    domain: CartesianTranslationDomain | None,
    feasibility_override: FeasibilityOverride | None = None,
) -> TranslationSearchResult:
    return TranslationSearchResult(
        status="invalid_input",
        attempts=(),
        seeds=(),
        max_seeds=max_seeds,
        retain_warnings=retain_warnings,
        feasibility_override=feasibility_override,
        source_structure_hash=(None if state is None else state.structure_hash),
        source_state_hash=(None if state is None else state.state_hash),
        translation_domain=domain,
        seed_limit_reached=False,
        domain_exhausted=False,
        invalid_reasons=(reason,),
    )


def generate_translation_seeds(
    *,
    state: object,
    feasibility_policy: object,
    translation_domain: object,
    max_seeds: object,
    retain_warnings: object = False,
    feasibility_override: object = None,
) -> TranslationSearchResult:
    """Safe one-shot Phase 6 entry point returning ``invalid_input`` on bad input."""
    if not isinstance(retain_warnings, bool):
        return _invalid_result(
            "retain_warnings must be a bool.",
            state=state if isinstance(state, BicrystalState) else None,
            retain_warnings=False,
            max_seeds=None,
            domain=(
                translation_domain
                if isinstance(translation_domain, CartesianTranslationDomain)
                else None
            ),
        )
    if feasibility_override is not None and not isinstance(
        feasibility_override, FeasibilityOverride
    ):
        return _invalid_result(
            "feasibility_override must be a FeasibilityOverride or None.",
            state=state if isinstance(state, BicrystalState) else None,
            retain_warnings=retain_warnings,
            max_seeds=None,
            domain=(
                translation_domain
                if isinstance(translation_domain, CartesianTranslationDomain)
                else None
            ),
        )
    try:
        initializer = InterfaceInitializer(
            state=state,  # type: ignore[arg-type]
            feasibility_policy=feasibility_policy,  # type: ignore[arg-type]
            retain_warnings=retain_warnings,
            feasibility_override=feasibility_override,
        )
    except InterfaceInitializationError as exc:
        return _invalid_result(
            str(exc),
            state=state if isinstance(state, BicrystalState) else None,
            retain_warnings=retain_warnings,
            max_seeds=None,
            domain=(
                translation_domain
                if isinstance(translation_domain, CartesianTranslationDomain)
                else None
            ),
            feasibility_override=(
                feasibility_override
                if isinstance(feasibility_override, FeasibilityOverride)
                else None
            ),
        )
    return initializer.generate_translation_seeds(
        translation_domain=translation_domain,  # type: ignore[arg-type]
        max_seeds=max_seeds,  # type: ignore[arg-type]
    )


__all__ = [
    "INITIALIZATION_SCHEMA_VERSION",
    "AttemptDisposition",
    "CandidateKind",
    "CartesianTranslationDomain",
    "InitializationStatus",
    "InterfaceInitializationError",
    "InterfaceInitializer",
    "TranslationAttempt",
    "TranslationCandidate",
    "TranslationSearchResult",
    "TranslationSeed",
    "generate_translation_seeds",
]
