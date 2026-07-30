# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Deterministic exact-termination construction and seed initialization.

Phase 7 rebuilds each crystallographic termination through :class:`GBMaker`, validates
its zero-translation state with the Phase 4 validator, and delegates only unacceptable
but valid zero-translation states to the Phase 6 rigid-translation initializer.  It is
independent of energy evaluation, relaxation, stochastic sampling, and optimizers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, TypeAlias

import numpy as np

from GBOpt.BicrystalState import BicrystalState
from GBOpt.BoundarySpec import (
    CSLApproxSpec,
    CSLExactSpec,
    FiveDOFSpec,
    PQSpec,
)
from GBOpt.GBMaker import GBMaker
from GBOpt.geometry_validation import (
    BicrystalFeasibilityReport,
    FeasibilityOverride,
    FeasibilityPolicy,
    validate_bicrystal_state,
)
from GBOpt.interface_initialization import (
    CartesianTranslationDomain,
    TranslationSearchResult,
    generate_translation_seeds,
)
from GBOpt.termination import GrainTermination, TerminationPair


TERMINATION_INITIALIZATION_SCHEMA_VERSION = 1

TerminationDisposition: TypeAlias = Literal[
    "retained_zero",
    "retained_translated",
    "rejected_zero_status",
    "construction_error",
    "population_error",
    "duplicate_structure",
    "translation_exhausted",
    "invalid_zero_state",
    "validation_error",
]
TerminationSeedKind: TypeAlias = Literal[
    "default_zero",
    "nondefault_zero",
    "termination_plus_translation",
]
TerminationInitializationStatus: TypeAlias = Literal[
    "default_termination_accepted",
    "nondefault_termination_accepted",
    "termination_translated_seed_retained",
    "seed_limit_reached",
    "termination_translation_domain_exhausted",
    "invalid_input",
]

_VALID_DISPOSITIONS = frozenset(
    {
        "retained_zero",
        "retained_translated",
        "rejected_zero_status",
        "construction_error",
        "population_error",
        "duplicate_structure",
        "translation_exhausted",
        "invalid_zero_state",
        "validation_error",
    }
)
_VALID_STATUSES = frozenset(
    {
        "default_termination_accepted",
        "nondefault_termination_accepted",
        "termination_translated_seed_retained",
        "seed_limit_reached",
        "termination_translation_domain_exhausted",
        "invalid_input",
    }
)


class TerminationInitializationError(ValueError):
    """Raised when a Phase 7 termination initialization input is malformed."""


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


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TerminationInitializationError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise TerminationInitializationError(f"{name} must be positive; got {result}.")
    return result


def _finite_float(value: object, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TerminationInitializationError(f"{name} must be a finite float.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TerminationInitializationError(
            f"{name} must be a finite float; got {value!r}."
        ) from exc
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise TerminationInitializationError(
            f"{name} must be {qualifier}; got {result!r}."
        )
    return 0.0 if result == 0.0 else result


def _freeze_json(value: Any, name: str = "provenance") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TerminationInitializationError(
                f"{name} must not contain NaN or infinity."
            )
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TerminationInitializationError(
                    f"{name} mapping keys must be strings; got {key!r}."
                )
            items.append((key, _freeze_json(item, f"{name}.{key}")))
        return ("__mapping__", tuple(sorted(items)))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ("__sequence__", tuple(_freeze_json(item, name) for item in value))
    raise TerminationInitializationError(
        f"{name} must be JSON-compatible; got {type(value).__name__}."
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__mapping__":
        return {key: _thaw_json(item) for key, item in value[1]}
    if isinstance(value, tuple) and len(value) == 2 and value[0] == "__sequence__":
        return [_thaw_json(item) for item in value[1]]
    return value


def _boundary_to_dict(boundary: object) -> dict[str, Any]:
    if isinstance(boundary, PQSpec):
        return {
            "type": "PQSpec",
            "P": [list(row) for row in boundary.P],
            "Q": [list(row) for row in boundary.Q],
            "basis_mode": boundary.basis_mode,
        }
    if isinstance(boundary, CSLExactSpec):
        return {
            "type": "CSLExactSpec",
            "axis": list(boundary.axis),
            "plane": list(boundary.plane),
            "sigma": boundary.sigma,
            "quat": list(boundary.quat),
        }
    if isinstance(boundary, FiveDOFSpec):
        return {"type": "FiveDOFSpec", "params": list(boundary.params)}
    return {"type": type(boundary).__name__, "repr": repr(boundary)}


def _repeat_factor(value: object) -> tuple[int, int]:
    if isinstance(value, (bool, np.bool_)):
        raise TerminationInitializationError(
            "repeat_factor must be a positive integer or two positive integers."
        )
    if isinstance(value, (int, np.integer)):
        item = _positive_int(value, "repeat_factor")
        return (item, item)
    if isinstance(value, (str, bytes)):
        raise TerminationInitializationError(
            "repeat_factor must be a positive integer or two positive integers."
        )
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TerminationInitializationError(
            "repeat_factor must be a positive integer or two positive integers."
        ) from exc
    if len(raw) != 2:
        raise TerminationInitializationError(
            "repeat_factor sequence must contain exactly two entries."
        )
    return (
        _positive_int(raw[0], "repeat_factor[0]"),
        _positive_int(raw[1], "repeat_factor[1]"),
    )


def _boundary_conditions(value: object) -> tuple[str, str, str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise TerminationInitializationError(
            "boundary_conditions must be a three-entry sequence or None."
        )
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TerminationInitializationError(
            "boundary_conditions must be a three-entry sequence or None."
        ) from exc
    if len(raw) != 3 or any(item not in {"periodic", "fixed"} for item in raw):
        raise TerminationInitializationError(
            "boundary_conditions must contain three 'periodic'/'fixed' entries."
        )
    return raw  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ExactBoundaryReconstruction:
    """Immutable typed input sufficient to rebuild every exact termination variant."""

    a0: float
    structure: str
    atom_types: str | tuple[str, ...]
    boundary: PQSpec | CSLExactSpec | FiveDOFSpec
    max_primitive_area_index: int = 10000
    max_pq_determinant: int = 10000
    gb_thickness: float = 0.0
    repeat_factor: int | tuple[int, int] = 2
    x_dim_min: float = 50.0
    vacuum: float = 10.0
    interaction_distance: float = 15.0
    gb_id: int = 1
    mismatch_tol: float | None = None
    mismatch_max_cells: int = 50
    strain_grain: str = "both"
    topology: str | None = None
    boundary_conditions: tuple[str, str, str] | None = None
    provenance: Mapping[str, object] | None = None
    schema_version: int = TERMINATION_INITIALIZATION_SCHEMA_VERSION
    fixed_region_thickness: float = 0.0
    surface_buffer_thickness: float = 0.0

    def __post_init__(self) -> None:
        a0 = _finite_float(self.a0, "a0")
        if a0 <= 0:
            raise TerminationInitializationError("a0 must be positive.")
        if not isinstance(self.structure, str) or not self.structure:
            raise TerminationInitializationError("structure must be a non-empty string.")
        if isinstance(self.atom_types, str):
            atom_types = (self.atom_types,)
        else:
            try:
                atom_types = tuple(self.atom_types)
            except TypeError as exc:
                raise TerminationInitializationError(
                    "atom_types must be a string or sequence of strings."
                ) from exc
        if not atom_types or any(not isinstance(item, str) or not item for item in atom_types):
            raise TerminationInitializationError(
                "atom_types must contain non-empty strings."
            )
        if isinstance(self.boundary, CSLApproxSpec) or not isinstance(
            self.boundary, (PQSpec, CSLExactSpec, FiveDOFSpec)
        ):
            raise TerminationInitializationError(
                "boundary must be an exact-reconstructable PQSpec, CSLExactSpec, "
                "or exactly rationalizable FiveDOFSpec."
            )
        repeat = _repeat_factor(self.repeat_factor)
        max_primitive = _positive_int(
            self.max_primitive_area_index, "max_primitive_area_index"
        )
        max_pq = _positive_int(self.max_pq_determinant, "max_pq_determinant")
        gb_id = _positive_int(self.gb_id, "gb_id")
        mismatch_max = _positive_int(self.mismatch_max_cells, "mismatch_max_cells")
        gb_thickness = _finite_float(
            self.gb_thickness, "gb_thickness", nonnegative=True
        )
        x_dim_min = _finite_float(self.x_dim_min, "x_dim_min", nonnegative=True)
        vacuum = _finite_float(self.vacuum, "vacuum", nonnegative=True)
        fixed_region_thickness = _finite_float(
            self.fixed_region_thickness,
            "fixed_region_thickness",
            nonnegative=True,
        )
        surface_buffer_thickness = _finite_float(
            self.surface_buffer_thickness,
            "surface_buffer_thickness",
            nonnegative=True,
        )
        interaction = _finite_float(
            self.interaction_distance, "interaction_distance", nonnegative=True
        )
        mismatch_tol = (
            None
            if self.mismatch_tol is None
            else _finite_float(self.mismatch_tol, "mismatch_tol", nonnegative=True)
        )
        if self.strain_grain not in {"both", "left", "right"}:
            raise TerminationInitializationError(
                "strain_grain must be 'both', 'left', or 'right'."
            )
        if self.topology not in {None, "periodic_bicrystal", "single_interface_slab"}:
            raise TerminationInitializationError(
                "topology must be periodic_bicrystal, single_interface_slab, or None."
            )
        conditions = _boundary_conditions(self.boundary_conditions)
        frozen_provenance = _freeze_json(self.provenance or {})
        if self.schema_version != TERMINATION_INITIALIZATION_SCHEMA_VERSION:
            raise TerminationInitializationError(
                f"Unsupported reconstruction schema_version {self.schema_version!r}."
            )
        object.__setattr__(self, "a0", a0)
        object.__setattr__(self, "atom_types", atom_types)
        object.__setattr__(self, "repeat_factor", repeat)
        object.__setattr__(self, "max_primitive_area_index", max_primitive)
        object.__setattr__(self, "max_pq_determinant", max_pq)
        object.__setattr__(self, "gb_id", gb_id)
        object.__setattr__(self, "mismatch_max_cells", mismatch_max)
        object.__setattr__(self, "gb_thickness", gb_thickness)
        object.__setattr__(self, "x_dim_min", x_dim_min)
        object.__setattr__(self, "vacuum", vacuum)
        object.__setattr__(self, "fixed_region_thickness", fixed_region_thickness)
        object.__setattr__(self, "surface_buffer_thickness", surface_buffer_thickness)
        object.__setattr__(self, "interaction_distance", interaction)
        object.__setattr__(self, "mismatch_tol", mismatch_tol)
        object.__setattr__(self, "boundary_conditions", conditions)
        object.__setattr__(self, "provenance", frozen_provenance)

    def build(self, termination_pair: TerminationPair) -> GBMaker:
        """Rebuild one exact variant through ``GBMaker.from_boundary_spec``."""
        if not isinstance(termination_pair, TerminationPair):
            raise TerminationInitializationError(
                "termination_pair must be a TerminationPair."
            )
        return GBMaker.from_boundary_spec(
            self.a0,
            self.structure,
            self.atom_types,
            self.boundary,
            mode="exact",
            max_primitive_area_index=self.max_primitive_area_index,
            max_pq_determinant=self.max_pq_determinant,
            gb_thickness=self.gb_thickness,
            repeat_factor=self.repeat_factor,
            x_dim_min=self.x_dim_min,
            vacuum=self.vacuum,
            fixed_region_thickness=self.fixed_region_thickness,
            surface_buffer_thickness=self.surface_buffer_thickness,
            interaction_distance=self.interaction_distance,
            gb_id=self.gb_id,
            mismatch_tol=self.mismatch_tol,
            mismatch_max_cells=self.mismatch_max_cells,
            strain_grain=self.strain_grain,
            topology=self.topology,
            boundary_conditions=self.boundary_conditions,
            termination_pair=termination_pair,
            provenance=_thaw_json(self.provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "construction_mode": "exact",
            "a0": self.a0,
            "structure": self.structure,
            "atom_types": list(self.atom_types),
            "boundary": _boundary_to_dict(self.boundary),
            "max_primitive_area_index": self.max_primitive_area_index,
            "max_pq_determinant": self.max_pq_determinant,
            "gb_thickness": self.gb_thickness,
            "repeat_factor": list(self.repeat_factor),
            "x_dim_min": self.x_dim_min,
            "vacuum": self.vacuum,
            "fixed_region_thickness": self.fixed_region_thickness,
            "surface_buffer_thickness": self.surface_buffer_thickness,
            "interaction_distance": self.interaction_distance,
            "gb_id": self.gb_id,
            "mismatch_tol": self.mismatch_tol,
            "mismatch_max_cells": self.mismatch_max_cells,
            "strain_grain": self.strain_grain,
            "topology": self.topology,
            "boundary_conditions": (
                None
                if self.boundary_conditions is None
                else list(self.boundary_conditions)
            ),
            "provenance": _thaw_json(self.provenance),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def reconstruction_hash(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class TerminationCandidate:
    """One deterministically ordered canonical left/right termination pair."""

    order: int
    requested_pair: TerminationPair
    canonical_pair: TerminationPair

    def __post_init__(self) -> None:
        if isinstance(self.order, (bool, np.bool_)) or not isinstance(
            self.order, (int, np.integer)
        ) or int(self.order) < 0:
            raise TerminationInitializationError(
                "candidate order must be a nonnegative integer."
            )
        if not isinstance(self.requested_pair, TerminationPair) or not isinstance(
            self.canonical_pair, TerminationPair
        ):
            raise TerminationInitializationError(
                "candidate pairs must be TerminationPair instances."
            )
        object.__setattr__(self, "order", int(self.order))

    @property
    def is_default(self) -> bool:
        return self.canonical_pair.is_default

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "requested_pair": self.requested_pair.to_dict(),
            "canonical_pair": self.canonical_pair.to_dict(),
            "is_default": self.is_default,
        }


@dataclass(frozen=True, slots=True)
class TerminationDomain:
    """Explicit finite left/right exact termination phase domain.

    Input order is not a ranking.  Canonical enumeration is: default/default first;
    left-only nondefault pairs; right-only nondefault pairs; then pairs with both grains
    nondefault, ordered lexicographically by exact left and right phase.
    """

    left: tuple[GrainTermination, ...]
    right: tuple[GrainTermination, ...]
    schema_version: int = TERMINATION_INITIALIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        left = self._normalize(self.left, "left")
        right = self._normalize(self.right, "right")
        if self.schema_version != TERMINATION_INITIALIZATION_SCHEMA_VERSION:
            raise TerminationInitializationError(
                f"Unsupported termination-domain schema_version {self.schema_version!r}."
            )
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @staticmethod
    def _normalize(
        values: object, grain: Literal["left", "right"]
    ) -> tuple[GrainTermination, ...]:
        if isinstance(values, (str, bytes)):
            raise TerminationInitializationError(
                f"{grain} termination domain must be a non-empty sequence."
            )
        try:
            raw = tuple(values)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TerminationInitializationError(
                f"{grain} termination domain must be a non-empty sequence."
            ) from exc
        if not raw:
            raise TerminationInitializationError(
                f"{grain} termination domain must not be empty."
            )
        for item in raw:
            if not isinstance(item, GrainTermination) or item.grain != grain:
                raise TerminationInitializationError(
                    f"{grain} domain entries must be {grain} GrainTermination values."
                )
        phases = [item.phase for item in raw]
        if len(phases) != len(set(phases)):
            raise TerminationInitializationError(
                f"{grain} termination domain contains duplicate equivalent phases."
            )
        return tuple(sorted(raw, key=lambda item: (not item.is_default, item.phase)))

    @classmethod
    def from_gbmaker(cls, gbmaker: GBMaker) -> "TerminationDomain":
        if not isinstance(gbmaker, GBMaker) or not gbmaker.uses_exact_construction:
            raise TerminationInitializationError(
                "from_gbmaker requires an exact GBMaker construction."
            )
        left, right = gbmaker.available_termination_descriptors
        return cls(left=left, right=right)

    @classmethod
    def from_reconstruction(
        cls, reconstruction: ExactBoundaryReconstruction
    ) -> "TerminationDomain":
        if not isinstance(reconstruction, ExactBoundaryReconstruction):
            raise TerminationInitializationError(
                "reconstruction must be an ExactBoundaryReconstruction."
            )
        gbmaker = reconstruction.build(TerminationPair())
        return cls.from_gbmaker(gbmaker)

    def candidates(self) -> tuple[TerminationCandidate, ...]:
        pairs = [
            TerminationPair(left=left, right=right)
            for left, right in product(self.left, self.right)
        ]

        def key(pair: TerminationPair) -> tuple[Any, ...]:
            left_default = pair.left.is_default
            right_default = pair.right.is_default
            if left_default and right_default:
                return (0, 0, 0)
            if not left_default and right_default:
                return (1, 0, pair.left.phase)
            if left_default and not right_default:
                return (1, 1, pair.right.phase)
            return (2, pair.left.phase, pair.right.phase)

        ordered = sorted(pairs, key=key)
        return tuple(
            TerminationCandidate(order=index, requested_pair=pair, canonical_pair=pair)
            for index, pair in enumerate(ordered)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ordering": (
                "default_pair_then_single_nondefault_by_grain_then_"
                "double_nondefault_lexicographic"
            ),
            "left": [item.to_dict() for item in self.left],
            "right": [item.to_dict() for item in self.right],
            "candidates": [item.to_dict() for item in self.candidates()],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def domain_hash(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class DecoratedPopulationCheck:
    """Generic exact-basis population and stoichiometry audit for one reconstruction."""

    passed: bool
    decorated_population_complete: bool
    per_grain_stoichiometric: bool
    whole_system_stoichiometric: bool
    basis_counts: tuple[tuple[str, int], ...]
    left_counts: tuple[tuple[str, int], ...]
    right_counts: tuple[tuple[str, int], ...]
    whole_counts: tuple[tuple[str, int], ...]
    expected_left_atom_count: int
    expected_right_atom_count: int
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "decorated_population_complete": self.decorated_population_complete,
            "per_grain_stoichiometric": self.per_grain_stoichiometric,
            "whole_system_stoichiometric": self.whole_system_stoichiometric,
            "basis_counts": dict(self.basis_counts),
            "left_counts": dict(self.left_counts),
            "right_counts": dict(self.right_counts),
            "whole_counts": dict(self.whole_counts),
            "expected_left_atom_count": self.expected_left_atom_count,
            "expected_right_atom_count": self.expected_right_atom_count,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class TerminationSeed:
    """One retained exact termination or termination-plus-translation state."""

    kind: TerminationSeedKind
    candidate: TerminationCandidate
    termination_pair: TerminationPair
    applied_translation_lab: tuple[float, float, float]
    state: BicrystalState
    report: BicrystalFeasibilityReport
    population_check: DecoratedPopulationCheck
    nested_translation_result_hash: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "default_zero",
            "nondefault_zero",
            "termination_plus_translation",
        }:
            raise TerminationInitializationError(f"Unsupported seed kind {self.kind!r}.")
        if not isinstance(self.state, BicrystalState) or not isinstance(
            self.report, BicrystalFeasibilityReport
        ):
            raise TerminationInitializationError(
                "retained seed requires a BicrystalState and feasibility report."
            )
        if (
            self.report.structure_hash != self.state.structure_hash
            or self.report.state_hash != self.state.state_hash
        ):
            raise TerminationInitializationError(
                "retained state and feasibility report hashes do not match."
            )
        if self.report.status not in {"feasible", "warning"}:
            raise TerminationInitializationError(
                "retained seed report must be feasible or warning."
            )
        translation = tuple(float(value) for value in self.applied_translation_lab)
        if len(translation) != 3 or not all(math.isfinite(value) for value in translation):
            raise TerminationInitializationError(
                "applied_translation_lab must contain three finite values."
            )
        object.__setattr__(self, "applied_translation_lab", translation)

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "candidate": self.candidate.to_dict(),
            "termination_pair": self.termination_pair.to_dict(),
            "applied_translation_lab": list(self.applied_translation_lab),
            "structure_hash": self.state.structure_hash,
            "state_hash": self.state.state_hash,
            "report_hash": self.report.report_hash,
            "population_check": self.population_check.to_dict(),
            "nested_translation_result_hash": self.nested_translation_result_hash,
            "construction_provenance": self.state.manifest()["metadata"],
            "state_manifest": self.state.manifest(),
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._hash_payload()
        result["report"] = self.report.to_dict()
        result["seed_hash"] = self.seed_hash
        return result

    @property
    def seed_hash(self) -> str:
        return _sha256(self._hash_payload())


@dataclass(frozen=True, slots=True)
class TerminationAttempt:
    """Complete record of one requested/canonical termination reconstruction."""

    candidate: TerminationCandidate
    construction_status: Literal["constructed", "error"]
    disposition: TerminationDisposition
    population_check: DecoratedPopulationCheck | None
    zero_translation_report: BicrystalFeasibilityReport | None
    nested_translation_result: TranslationSearchResult | None
    structure_hash: str | None
    state_hash: str | None
    retained_seed_hashes: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    construction_error: str = ""
    validation_error: str = ""

    def __post_init__(self) -> None:
        if self.construction_status not in {"constructed", "error"}:
            raise TerminationInitializationError(
                f"Unsupported construction status {self.construction_status!r}."
            )
        if self.disposition not in _VALID_DISPOSITIONS:
            raise TerminationInitializationError(
                f"Unsupported termination disposition {self.disposition!r}."
            )
        if self.construction_status == "error":
            if not self.construction_error:
                raise TerminationInitializationError(
                    "construction error attempts require an error message."
                )
            if self.population_check is not None or self.zero_translation_report is not None:
                raise TerminationInitializationError(
                    "construction error attempts cannot carry constructed-state checks."
                )
        elif self.disposition == "validation_error":
            if self.population_check is None or self.zero_translation_report is not None:
                raise TerminationInitializationError(
                    "validation-error attempts require population checks and no report."
                )
            if not self.validation_error:
                raise TerminationInitializationError(
                    "validation-error attempts require an error message."
                )
        elif self.population_check is None or self.zero_translation_report is None:
            raise TerminationInitializationError(
                "constructed attempts require population and zero-translation reports."
            )
        object.__setattr__(self, "retained_seed_hashes", tuple(self.retained_seed_hashes))
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "construction_status": self.construction_status,
            "disposition": self.disposition,
            "population_check": (
                None if self.population_check is None else self.population_check.to_dict()
            ),
            "zero_translation_report": (
                None
                if self.zero_translation_report is None
                else self.zero_translation_report.to_dict()
            ),
            "nested_translation_result": (
                None
                if self.nested_translation_result is None
                else self.nested_translation_result.to_dict()
            ),
            "structure_hash": self.structure_hash,
            "state_hash": self.state_hash,
            "retained_seed_hashes": list(self.retained_seed_hashes),
            "rejection_reasons": list(self.rejection_reasons),
            "construction_error": self.construction_error,
            "validation_error": self.validation_error,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._hash_payload()
        result["attempt_hash"] = self.attempt_hash
        return result

    @property
    def attempt_hash(self) -> str:
        return _sha256(self._hash_payload())


@dataclass(frozen=True, slots=True)
class TerminationSearchResult:
    """Deterministic result of the complete finite Phase 7 search."""

    status: TerminationInitializationStatus
    reconstruction: ExactBoundaryReconstruction | None
    termination_domain: TerminationDomain | None
    translation_domain: CartesianTranslationDomain | None
    attempts: tuple[TerminationAttempt, ...]
    seeds: tuple[TerminationSeed, ...]
    max_seeds: int | None
    retain_warnings: bool
    seed_limit_reached: bool
    domain_exhausted: bool
    source_structure_hash: str | None = None
    source_state_hash: str | None = None
    invalid_reasons: tuple[str, ...] = ()
    schema_version: int = TERMINATION_INITIALIZATION_SCHEMA_VERSION
    feasibility_override: FeasibilityOverride | None = None

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            raise TerminationInitializationError(f"Unsupported result status {self.status!r}.")
        attempts = tuple(self.attempts)
        seeds = tuple(self.seeds)
        if len({seed.state.structure_hash for seed in seeds}) != len(seeds):
            raise TerminationInitializationError(
                "retained termination seeds must have unique structure hashes."
            )
        if self.status == "invalid_input":
            if seeds or not self.invalid_reasons:
                raise TerminationInitializationError(
                    "invalid_input requires reasons and cannot retain seeds."
                )
        elif (
            self.reconstruction is None
            or self.termination_domain is None
            or self.translation_domain is None
        ):
            raise TerminationInitializationError(
                "completed searches require reconstruction and both domains."
            )
        if self.status == "seed_limit_reached" and not self.seed_limit_reached:
            raise TerminationInitializationError(
                "seed_limit_reached status requires its flag."
            )
        if self.status == "termination_translation_domain_exhausted" and (
            seeds or not self.domain_exhausted
        ):
            raise TerminationInitializationError(
                "exhaustion requires no seeds and a fully exhausted domain."
            )
        if self.feasibility_override is not None and not isinstance(
            self.feasibility_override, FeasibilityOverride
        ):
            raise TerminationInitializationError(
                "feasibility_override must be a FeasibilityOverride or None."
            )
        object.__setattr__(self, "attempts", attempts)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "invalid_reasons", tuple(self.invalid_reasons))

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "reconstruction": (
                None if self.reconstruction is None else self.reconstruction.to_dict()
            ),
            "reconstruction_hash": (
                None
                if self.reconstruction is None
                else self.reconstruction.reconstruction_hash
            ),
            "termination_domain": (
                None if self.termination_domain is None else self.termination_domain.to_dict()
            ),
            "termination_domain_hash": (
                None
                if self.termination_domain is None
                else self.termination_domain.domain_hash
            ),
            "translation_domain": (
                None if self.translation_domain is None else self.translation_domain.to_dict()
            ),
            "translation_domain_hash": (
                None
                if self.translation_domain is None
                else self.translation_domain.domain_hash
            ),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "seeds": [seed.to_dict() for seed in self.seeds],
            "max_seeds": self.max_seeds,
            "retain_warnings": self.retain_warnings,
            "seed_limit_reached": self.seed_limit_reached,
            "domain_exhausted": self.domain_exhausted,
            "source_structure_hash": self.source_structure_hash,
            "source_state_hash": self.source_state_hash,
            "invalid_reasons": list(self.invalid_reasons),
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
        result = self._hash_payload()
        result["result_hash"] = self.result_hash
        return result

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def result_hash(self) -> str:
        return _sha256(self._hash_payload())


@dataclass(frozen=True, slots=True)
class TerminationInitializer:
    """Compose exact termination reconstruction, validation, and Phase 6 search."""

    reconstruction: ExactBoundaryReconstruction
    feasibility_policy: FeasibilityPolicy
    termination_domain: TerminationDomain
    translation_domain: CartesianTranslationDomain
    retain_warnings: bool = False
    feasibility_override: FeasibilityOverride | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.reconstruction, ExactBoundaryReconstruction):
            raise TerminationInitializationError(
                "reconstruction must be an ExactBoundaryReconstruction."
            )
        if not isinstance(self.feasibility_policy, FeasibilityPolicy):
            raise TerminationInitializationError(
                "feasibility_policy must be a FeasibilityPolicy."
            )
        if not isinstance(self.termination_domain, TerminationDomain):
            raise TerminationInitializationError(
                "termination_domain must be a TerminationDomain."
            )
        if not isinstance(self.translation_domain, CartesianTranslationDomain):
            raise TerminationInitializationError(
                "translation_domain must be a CartesianTranslationDomain."
            )
        if not isinstance(self.retain_warnings, bool):
            raise TerminationInitializationError("retain_warnings must be a bool.")
        if self.feasibility_override is not None and not isinstance(
            self.feasibility_override, FeasibilityOverride
        ):
            raise TerminationInitializationError(
                "feasibility_override must be a FeasibilityOverride or None."
            )

    def generate_seeds(self, *, max_seeds: object) -> TerminationSearchResult:
        limit = _positive_int(max_seeds, "max_seeds")
        try:
            source_gb = self.reconstruction.build(TerminationPair())
        except Exception as exc:
            return _invalid_result(
                f"source reconstruction failed: {type(exc).__name__}: {exc}",
                reconstruction=self.reconstruction,
                termination_domain=self.termination_domain,
                translation_domain=self.translation_domain,
                retain_warnings=self.retain_warnings,
                feasibility_override=self.feasibility_override,
            )
        if not source_gb.uses_exact_construction:
            return _invalid_result(
                "source reconstruction did not use the exact GBMaker path.",
                reconstruction=self.reconstruction,
                termination_domain=self.termination_domain,
                translation_domain=self.translation_domain,
                retain_warnings=self.retain_warnings,
                feasibility_override=self.feasibility_override,
            )
        source_state = source_gb.bicrystal_state
        if source_state.topology not in {"periodic_bicrystal", "single_interface_slab"}:
            return _invalid_result(
                f"unsupported topology {source_state.topology!r}.",
                reconstruction=self.reconstruction,
                termination_domain=self.termination_domain,
                translation_domain=self.translation_domain,
                retain_warnings=self.retain_warnings,
                feasibility_override=self.feasibility_override,
            )
        available_left, available_right = source_gb.available_termination_descriptors
        unsupported = [
            item.to_json()
            for item in (*self.termination_domain.left, *self.termination_domain.right)
            if item not in (available_left if item.grain == "left" else available_right)
        ]
        if unsupported:
            return _invalid_result(
                "termination domain contains unsupported exact cut phases: "
                + "; ".join(unsupported),
                reconstruction=self.reconstruction,
                termination_domain=self.termination_domain,
                translation_domain=self.translation_domain,
                retain_warnings=self.retain_warnings,
                feasibility_override=self.feasibility_override,
            )

        expected_counts = (len(source_gb.left_grain), len(source_gb.right_grain))
        attempts: list[TerminationAttempt] = []
        seeds: list[TerminationSeed] = []
        seen_reconstructed: set[str] = set()
        seen_retained: set[str] = set()
        accepted_statuses = {"feasible"}
        if self.retain_warnings:
            accepted_statuses.add("warning")

        for candidate in self.termination_domain.candidates():
            try:
                gbmaker = (
                    source_gb
                    if candidate.is_default
                    else self.reconstruction.build(candidate.canonical_pair)
                )
                if not gbmaker.uses_exact_construction:
                    raise TerminationInitializationError(
                        "candidate did not use the exact GBMaker path."
                    )
                state = gbmaker.bicrystal_state
            except Exception as exc:
                attempts.append(
                    TerminationAttempt(
                        candidate=candidate,
                        construction_status="error",
                        disposition="construction_error",
                        population_check=None,
                        zero_translation_report=None,
                        nested_translation_result=None,
                        structure_hash=None,
                        state_hash=None,
                        rejection_reasons=("termination.construction_error",),
                        construction_error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            population = _population_check(gbmaker, expected_counts)
            try:
                zero_report = validate_bicrystal_state(
                    state,
                    policy=self.feasibility_policy,
                    override=self.feasibility_override,
                )
            except Exception as exc:
                attempts.append(
                    TerminationAttempt(
                        candidate=candidate,
                        construction_status="constructed",
                        disposition="validation_error",
                        population_check=population,
                        zero_translation_report=None,
                        nested_translation_result=None,
                        structure_hash=state.structure_hash,
                        state_hash=state.state_hash,
                        rejection_reasons=("termination.validation_error",),
                        validation_error=f"{type(exc).__name__}: {exc}",
                    )
                )
                seen_reconstructed.add(state.structure_hash)
                continue
            structure_hash = state.structure_hash
            state_hash = state.state_hash

            if not population.passed:
                attempts.append(
                    TerminationAttempt(
                        candidate=candidate,
                        construction_status="constructed",
                        disposition="population_error",
                        population_check=population,
                        zero_translation_report=zero_report,
                        nested_translation_result=None,
                        structure_hash=structure_hash,
                        state_hash=state_hash,
                        rejection_reasons=population.reasons,
                    )
                )
                seen_reconstructed.add(structure_hash)
                continue

            if structure_hash in seen_reconstructed:
                attempts.append(
                    TerminationAttempt(
                        candidate=candidate,
                        construction_status="constructed",
                        disposition="duplicate_structure",
                        population_check=population,
                        zero_translation_report=zero_report,
                        nested_translation_result=None,
                        structure_hash=structure_hash,
                        state_hash=state_hash,
                        rejection_reasons=("termination.duplicate_reconstructed_structure",),
                    )
                )
                continue
            seen_reconstructed.add(structure_hash)

            retained_this_attempt: list[TerminationSeed] = []
            nested: TranslationSearchResult | None = None
            rejection_reasons: tuple[str, ...] = ()
            disposition: TerminationDisposition

            if zero_report.status in accepted_statuses:
                if structure_hash not in seen_retained:
                    kind: TerminationSeedKind = (
                        "default_zero" if candidate.is_default else "nondefault_zero"
                    )
                    retained_this_attempt.append(
                        TerminationSeed(
                            kind=kind,
                            candidate=candidate,
                            termination_pair=candidate.canonical_pair,
                            applied_translation_lab=(0.0, 0.0, 0.0),
                            state=state,
                            report=zero_report,
                            population_check=population,
                        )
                    )
                    disposition = "retained_zero"
                else:
                    disposition = "duplicate_structure"
                    rejection_reasons = ("termination.duplicate_final_structure",)
            elif zero_report.status == "invalid":
                disposition = "invalid_zero_state"
                rejection_reasons = tuple(reason.code for reason in zero_report.reasons)
            else:
                remaining = limit - len(seeds)
                nested = generate_translation_seeds(
                    state=state,
                    feasibility_policy=self.feasibility_policy,
                    translation_domain=self.translation_domain,
                    max_seeds=max(1, remaining),
                    retain_warnings=self.retain_warnings,
                    feasibility_override=self.feasibility_override,
                )
                for translated in nested.seeds:
                    translated_hash = translated.state.structure_hash
                    if translated_hash in seen_retained:
                        continue
                    retained_this_attempt.append(
                        TerminationSeed(
                            kind="termination_plus_translation",
                            candidate=candidate,
                            termination_pair=candidate.canonical_pair,
                            applied_translation_lab=translated.canonical_displacement_lab,
                            state=translated.state,
                            report=translated.report,
                            population_check=population,
                            nested_translation_result_hash=nested.result_hash,
                        )
                    )
                    if len(seeds) + len(retained_this_attempt) >= limit:
                        break
                if retained_this_attempt:
                    disposition = "retained_translated"
                else:
                    disposition = "translation_exhausted"
                    rejection_reasons = (
                        "termination.translation_domain_exhausted",
                        *tuple(
                            reason.code for reason in zero_report.reasons
                        ),
                    )

            for seed in retained_this_attempt:
                seeds.append(seed)
                seen_retained.add(seed.state.structure_hash)

            attempts.append(
                TerminationAttempt(
                    candidate=candidate,
                    construction_status="constructed",
                    disposition=disposition,
                    population_check=population,
                    zero_translation_report=zero_report,
                    nested_translation_result=nested,
                    structure_hash=structure_hash,
                    state_hash=state_hash,
                    retained_seed_hashes=tuple(
                        seed.seed_hash for seed in retained_this_attempt
                    ),
                    rejection_reasons=rejection_reasons,
                )
            )

            if len(seeds) >= limit:
                return TerminationSearchResult(
                    status="seed_limit_reached",
                    reconstruction=self.reconstruction,
                    termination_domain=self.termination_domain,
                    translation_domain=self.translation_domain,
                    attempts=tuple(attempts),
                    seeds=tuple(seeds[:limit]),
                    max_seeds=limit,
                    retain_warnings=self.retain_warnings,
                    feasibility_override=self.feasibility_override,
                    seed_limit_reached=True,
                    domain_exhausted=False,
                    source_structure_hash=source_state.structure_hash,
                    source_state_hash=source_state.state_hash,
                )

        if not seeds:
            status: TerminationInitializationStatus = (
                "termination_translation_domain_exhausted"
            )
        elif any(seed.kind == "termination_plus_translation" for seed in seeds):
            status = "termination_translated_seed_retained"
        elif any(seed.kind == "nondefault_zero" for seed in seeds):
            status = "nondefault_termination_accepted"
        else:
            status = "default_termination_accepted"
        return TerminationSearchResult(
            status=status,
            reconstruction=self.reconstruction,
            termination_domain=self.termination_domain,
            translation_domain=self.translation_domain,
            attempts=tuple(attempts),
            seeds=tuple(seeds),
            max_seeds=limit,
            retain_warnings=self.retain_warnings,
            feasibility_override=self.feasibility_override,
            seed_limit_reached=False,
            domain_exhausted=True,
            source_structure_hash=source_state.structure_hash,
            source_state_hash=source_state.state_hash,
        )


def _species_counts(atoms: np.ndarray) -> tuple[tuple[str, int], ...]:
    counter = Counter(str(name) for name in atoms["name"])
    return tuple(sorted(counter.items()))


def _is_stoichiometric(
    counts: tuple[tuple[str, int], ...],
    basis_counts: tuple[tuple[str, int], ...],
) -> bool:
    count_map = dict(counts)
    basis_map = dict(basis_counts)
    if set(count_map) != set(basis_map):
        return False
    multipliers = set()
    for species, basis_count in basis_map.items():
        count = count_map[species]
        if basis_count <= 0 or count % basis_count:
            return False
        multipliers.add(count // basis_count)
    return len(multipliers) == 1


def _population_check(
    gbmaker: GBMaker,
    expected_counts: tuple[int, int],
) -> DecoratedPopulationCheck:
    rational_basis = gbmaker.unit_cell.rational_basis
    if rational_basis is None:
        return DecoratedPopulationCheck(
            passed=False,
            decorated_population_complete=False,
            per_grain_stoichiometric=False,
            whole_system_stoichiometric=False,
            basis_counts=(),
            left_counts=_species_counts(gbmaker.left_grain),
            right_counts=_species_counts(gbmaker.right_grain),
            whole_counts=_species_counts(gbmaker.whole_system),
            expected_left_atom_count=expected_counts[0],
            expected_right_atom_count=expected_counts[1],
            reasons=("termination.missing_rational_basis",),
        )
    basis_counts = tuple(sorted(Counter(rational_basis.names).items()))
    left = _species_counts(gbmaker.left_grain)
    right = _species_counts(gbmaker.right_grain)
    whole = _species_counts(gbmaker.whole_system)
    complete = (
        len(gbmaker.left_grain) == expected_counts[0]
        and len(gbmaker.right_grain) == expected_counts[1]
        and len(gbmaker.whole_system) == sum(expected_counts)
    )
    per_grain = _is_stoichiometric(left, basis_counts) and _is_stoichiometric(
        right, basis_counts
    )
    whole_ok = _is_stoichiometric(whole, basis_counts)
    reasons = []
    if not complete:
        reasons.append("termination.decorated_population_incomplete")
    if not per_grain:
        reasons.append("termination.per_grain_stoichiometry_failed")
    if not whole_ok:
        reasons.append("termination.whole_system_stoichiometry_failed")
    return DecoratedPopulationCheck(
        passed=complete and per_grain and whole_ok,
        decorated_population_complete=complete,
        per_grain_stoichiometric=per_grain,
        whole_system_stoichiometric=whole_ok,
        basis_counts=basis_counts,
        left_counts=left,
        right_counts=right,
        whole_counts=whole,
        expected_left_atom_count=expected_counts[0],
        expected_right_atom_count=expected_counts[1],
        reasons=tuple(reasons),
    )


def check_decorated_population(
    gbmaker: GBMaker,
    expected_counts: tuple[int, int] | None = None,
) -> DecoratedPopulationCheck:
    """Return the exact rational-basis decorated-population audit for a construction.

    ``expected_counts`` defaults to the current left/right populations, which is
    appropriate for auditing the default exact construction. Phase 7 supplies the
    default counts explicitly when comparing nondefault terminations.
    """
    if not isinstance(gbmaker, GBMaker):
        raise TerminationInitializationError("gbmaker must be a GBMaker instance.")
    if expected_counts is None:
        expected_counts = (len(gbmaker.left_grain), len(gbmaker.right_grain))
    if (
        not isinstance(expected_counts, tuple)
        or len(expected_counts) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
            for value in expected_counts
        )
    ):
        raise TerminationInitializationError(
            "expected_counts must contain two nonnegative integers."
        )
    return _population_check(
        gbmaker,
        (int(expected_counts[0]), int(expected_counts[1])),
    )


def _invalid_result(
    reason: str,
    *,
    reconstruction: ExactBoundaryReconstruction | None,
    termination_domain: TerminationDomain | None,
    translation_domain: CartesianTranslationDomain | None,
    retain_warnings: bool,
    feasibility_override: FeasibilityOverride | None = None,
) -> TerminationSearchResult:
    return TerminationSearchResult(
        status="invalid_input",
        reconstruction=reconstruction,
        termination_domain=termination_domain,
        translation_domain=translation_domain,
        attempts=(),
        seeds=(),
        max_seeds=None,
        retain_warnings=retain_warnings,
        feasibility_override=feasibility_override,
        seed_limit_reached=False,
        domain_exhausted=False,
        invalid_reasons=(reason,),
    )


def generate_termination_seeds(
    *,
    reconstruction: object,
    feasibility_policy: object,
    termination_domain: object,
    translation_domain: object,
    max_seeds: object,
    retain_warnings: object = False,
    feasibility_override: object = None,
) -> TerminationSearchResult:
    """Safe one-shot Phase 7 entry point returning ``invalid_input`` on bad input."""
    if not isinstance(retain_warnings, bool):
        return _invalid_result(
            "retain_warnings must be a bool.",
            reconstruction=(
                reconstruction
                if isinstance(reconstruction, ExactBoundaryReconstruction)
                else None
            ),
            termination_domain=(
                termination_domain
                if isinstance(termination_domain, TerminationDomain)
                else None
            ),
            translation_domain=(
                translation_domain
                if isinstance(translation_domain, CartesianTranslationDomain)
                else None
            ),
            retain_warnings=False,
        )
    if feasibility_override is not None and not isinstance(
        feasibility_override, FeasibilityOverride
    ):
        return _invalid_result(
            "feasibility_override must be a FeasibilityOverride or None.",
            reconstruction=(
                reconstruction
                if isinstance(reconstruction, ExactBoundaryReconstruction)
                else None
            ),
            termination_domain=(
                termination_domain
                if isinstance(termination_domain, TerminationDomain)
                else None
            ),
            translation_domain=(
                translation_domain
                if isinstance(translation_domain, CartesianTranslationDomain)
                else None
            ),
            retain_warnings=retain_warnings,
        )
    try:
        initializer = TerminationInitializer(
            reconstruction=reconstruction,  # type: ignore[arg-type]
            feasibility_policy=feasibility_policy,  # type: ignore[arg-type]
            termination_domain=termination_domain,  # type: ignore[arg-type]
            translation_domain=translation_domain,  # type: ignore[arg-type]
            retain_warnings=retain_warnings,
            feasibility_override=feasibility_override,
        )
        return initializer.generate_seeds(max_seeds=max_seeds)
    except (TerminationInitializationError, TypeError, ValueError) as exc:
        return _invalid_result(
            str(exc),
            reconstruction=(
                reconstruction
                if isinstance(reconstruction, ExactBoundaryReconstruction)
                else None
            ),
            termination_domain=(
                termination_domain
                if isinstance(termination_domain, TerminationDomain)
                else None
            ),
            translation_domain=(
                translation_domain
                if isinstance(translation_domain, CartesianTranslationDomain)
                else None
            ),
            retain_warnings=retain_warnings,
            feasibility_override=(
                feasibility_override
                if isinstance(feasibility_override, FeasibilityOverride)
                else None
            ),
        )


__all__ = [
    "TERMINATION_INITIALIZATION_SCHEMA_VERSION",
    "DecoratedPopulationCheck",
    "ExactBoundaryReconstruction",
    "TerminationAttempt",
    "TerminationCandidate",
    "TerminationDisposition",
    "TerminationDomain",
    "TerminationInitializationError",
    "TerminationInitializationStatus",
    "TerminationInitializer",
    "TerminationSearchResult",
    "TerminationSeed",
    "TerminationSeedKind",
    "check_decorated_population",
    "generate_termination_seeds",
]
