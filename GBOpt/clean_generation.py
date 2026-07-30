# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Immutable configuration for deterministic clean-boundary campaign generation.

This module contains only typed policy/domain configuration and canonical parsing.  It
is independent of campaign process orchestration and target-property optimization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

from GBOpt.geometry_validation import (
    ContactPolicy,
    FeasibilityOverride,
    FeasibilityPolicy,
    SlabPolicy,
    SpeciesPairThresholds,
    VoidPolicy,
)
from GBOpt.interface_initialization import CartesianTranslationDomain
from GBOpt.termination import GrainTermination


CLEAN_GENERATION_CONFIG_SCHEMA_VERSION = 1


class CleanGenerationConfigError(ValueError):
    """Raised when clean-generation configuration is malformed or inconsistent."""


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


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CleanGenerationConfigError(f"{name} must be a mapping.")
    if any(not isinstance(key, str) for key in value):
        raise CleanGenerationConfigError(f"{name} keys must be strings.")
    return value  # type: ignore[return-value]


def _only_keys(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise CleanGenerationConfigError(
            f"{name} contains unsupported keys: {', '.join(unknown)}"
        )


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise CleanGenerationConfigError(f"{name} must be a bool.")
    return value


def _nonnegative_float(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CleanGenerationConfigError(
            f"{name} must be a finite nonnegative float."
        ) from exc
    if result < 0.0 or result in {float("inf"), float("-inf")} or result != result:
        raise CleanGenerationConfigError(
            f"{name} must be a finite nonnegative float."
        )
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CleanGenerationConfigError(f"{name} must be a positive integer.")
    return value


def _fraction_descriptor(value: object, name: str) -> Fraction:
    try:
        if isinstance(value, str):
            phase = Fraction(value)
        elif isinstance(value, Mapping):
            item = _mapping(value, name)
            _only_keys(item, {"numerator", "denominator"}, name)
            phase = Fraction(item["numerator"], item["denominator"])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raw = tuple(value)
            if len(raw) != 2:
                raise CleanGenerationConfigError(
                    f"{name} sequence must contain numerator and denominator."
                )
            phase = Fraction(raw[0], raw[1])
        elif isinstance(value, int) and not isinstance(value, bool):
            phase = Fraction(value, 1)
        else:
            raise TypeError
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise CleanGenerationConfigError(
            f"{name} must be an exact rational phase such as '1/2'."
        ) from exc
    return phase % 1


@dataclass(frozen=True, slots=True, order=True)
class RationalPhase:
    """Canonical exact phase in ``[0, 1)`` used by termination configuration."""

    numerator: int = 0
    denominator: int = 1

    def __post_init__(self) -> None:
        try:
            phase = Fraction(self.numerator, self.denominator) % 1
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise CleanGenerationConfigError(
                "termination phases require integer numerator and positive denominator."
            ) from exc
        object.__setattr__(self, "numerator", phase.numerator)
        object.__setattr__(self, "denominator", phase.denominator)

    @classmethod
    def parse(cls, value: object, name: str) -> "RationalPhase":
        phase = _fraction_descriptor(value, name)
        return cls(phase.numerator, phase.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}


@dataclass(frozen=True, slots=True)
class TerminationDomainSelection:
    """Case-independent selection rule for each exact finite termination domain."""

    mode: str = "all"
    left: tuple[RationalPhase, ...] = ()
    right: tuple[RationalPhase, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"all", "default_only", "explicit"}:
            raise CleanGenerationConfigError(
                "termination_domain.mode must be 'all', 'default_only', or 'explicit'."
            )
        left = tuple(self.left)
        right = tuple(self.right)
        if any(not isinstance(item, RationalPhase) for item in (*left, *right)):
            raise CleanGenerationConfigError(
                "explicit termination phases must be RationalPhase values."
            )
        if self.mode == "explicit":
            if not left or not right:
                raise CleanGenerationConfigError(
                    "explicit termination domains require non-empty left and right phases."
                )
            for grain, values in (("left", left), ("right", right)):
                phases = [item.fraction for item in values]
                if len(phases) != len(set(phases)):
                    raise CleanGenerationConfigError(
                        f"{grain} termination domain contains duplicate equivalent phases."
                    )
                if Fraction(0, 1) not in phases:
                    raise CleanGenerationConfigError(
                        f"{grain} explicit termination domain must include zero phase."
                    )
                ordered = tuple(
                    sorted(
                        values, key=lambda item: (item.fraction != 0, item.fraction)
                    )
                )
                if grain == "left":
                    left = ordered
                else:
                    right = ordered
        elif left or right:
            raise CleanGenerationConfigError(
                "left/right phases are permitted only when termination_domain.mode is explicit."
            )
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)

    @classmethod
    def from_mapping(cls, value: object) -> "TerminationDomainSelection":
        item = _mapping(value, "termination_domain")
        _only_keys(item, {"mode", "left", "right"}, "termination_domain")
        mode = str(item.get("mode", "all"))

        def phases(grain: str) -> tuple[RationalPhase, ...]:
            raw = item.get(grain, ())
            if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raise CleanGenerationConfigError(
                    f"termination_domain.{grain} must be a sequence."
                )
            return tuple(
                RationalPhase.parse(entry, f"termination_domain.{grain}[{index}]")
                for index, entry in enumerate(raw)
            )

        return cls(mode=mode, left=phases("left"), right=phases("right"))

    def descriptors(self, grain: str) -> tuple[GrainTermination, ...]:
        values = self.left if grain == "left" else self.right
        return tuple(
            GrainTermination(grain, item.numerator, item.denominator)  # type: ignore[arg-type]
            for item in values
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "left": [item.to_dict() for item in self.left],
            "right": [item.to_dict() for item in self.right],
        }

    @property
    def selection_hash(self) -> str:
        return _sha256(self.to_dict())


def _pair_thresholds(value: object) -> tuple[SpeciesPairThresholds, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CleanGenerationConfigError(
            "feasibility_policy.contact.pair_thresholds must be a sequence."
        )
    result = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"pair_thresholds[{index}]")
        _only_keys(
            item,
            {
                "species",
                "duplicate_angstrom",
                "hard_minimum_angstrom",
                "warning_minimum_angstrom",
            },
            f"pair_thresholds[{index}]",
        )
        result.append(
            SpeciesPairThresholds(
                species=tuple(item["species"]),  # type: ignore[arg-type]
                duplicate_angstrom=item["duplicate_angstrom"],
                hard_minimum_angstrom=item["hard_minimum_angstrom"],
                warning_minimum_angstrom=item["warning_minimum_angstrom"],
            )
        )
    return tuple(result)


def feasibility_policy_from_mapping(value: object) -> FeasibilityPolicy:
    """Parse a complete deterministic feasibility policy mapping."""
    item = _mapping(value, "feasibility_policy")
    _only_keys(item, {"contact", "void", "slab"}, "feasibility_policy")

    contact_raw = _mapping(item.get("contact", {}), "feasibility_policy.contact")
    _only_keys(
        contact_raw,
        {
            "pair_thresholds",
            "duplicate_tolerance_angstrom",
            "hard_minimum_bulk_factor",
            "warning_minimum_bulk_factor",
        },
        "feasibility_policy.contact",
    )
    contact = ContactPolicy(
        pair_thresholds=_pair_thresholds(contact_raw.get("pair_thresholds", ())),
        duplicate_tolerance_angstrom=contact_raw.get(
            "duplicate_tolerance_angstrom", 1.0e-6
        ),
        hard_minimum_bulk_factor=contact_raw.get("hard_minimum_bulk_factor", 0.45),
        warning_minimum_bulk_factor=contact_raw.get(
            "warning_minimum_bulk_factor", 0.60
        ),
    )

    void_raw = _mapping(item.get("void", {}), "feasibility_policy.void")
    void_allowed = {
        "hard_max_empty_bin_fraction",
        "warning_max_empty_bin_fraction",
        "hard_max_gap_range_bulk_factor",
        "warning_max_gap_range_bulk_factor",
        "hard_max_p95_bulk_factor",
        "warning_max_p95_bulk_factor",
        "min_bins_per_axis",
        "max_bins_per_axis",
    }
    _only_keys(void_raw, void_allowed, "feasibility_policy.void")
    void_defaults = VoidPolicy()
    void = VoidPolicy(
        **{
            name: void_raw.get(name, getattr(void_defaults, name))
            for name in void_allowed
        }
    )

    slab_raw = _mapping(item.get("slab", {}), "feasibility_policy.slab")
    slab_allowed = {
        "minimum_vacuum_thickness_angstrom",
        "warning_vacuum_thickness_angstrom",
        "minimum_surface_clearance_angstrom",
        "warning_surface_clearance_angstrom",
        "minimum_fixed_region_thickness_angstrom",
        "warning_fixed_region_thickness_angstrom",
        "minimum_buffer_thickness_angstrom",
        "warning_buffer_thickness_angstrom",
        "minimum_fixed_region_atoms",
        "minimum_buffer_region_atoms",
        "descriptor_tolerance_angstrom",
    }
    _only_keys(slab_raw, slab_allowed, "feasibility_policy.slab")
    slab_defaults = SlabPolicy()
    slab = SlabPolicy(
        **{
            name: slab_raw.get(name, getattr(slab_defaults, name))
            for name in slab_allowed
        }
    )
    return FeasibilityPolicy(contact=contact, void=void, slab=slab)


def feasibility_override_from_mapping(value: object) -> FeasibilityOverride | None:
    if value is None:
        return None
    item = _mapping(value, "override")
    _only_keys(item, {"status", "reason"}, "override")
    try:
        return FeasibilityOverride(status=item["status"], reason=item["reason"])
    except KeyError as exc:
        raise CleanGenerationConfigError(
            "override requires status and reason."
        ) from exc


def translation_domain_from_mapping(value: object) -> CartesianTranslationDomain:
    item = _mapping(value, "translation_domain")
    _only_keys(
        item,
        {
            "in_plane_components",
            "normal_offsets",
            "normal_axis",
            "in_plane_axes",
            "schema_version",
        },
        "translation_domain",
    )
    try:
        components = item["in_plane_components"]
    except KeyError as exc:
        raise CleanGenerationConfigError(
            "translation_domain requires in_plane_components."
        ) from exc
    return CartesianTranslationDomain(
        in_plane_components=tuple(tuple(values) for values in components),  # type: ignore[arg-type]
        normal_offsets=tuple(item.get("normal_offsets", (0.0,))),
        normal_axis=item.get("normal_axis"),
        in_plane_axes=(
            None
            if item.get("in_plane_axes") is None
            else tuple(item["in_plane_axes"])
        ),
        schema_version=item.get("schema_version", 1),
    )


@dataclass(frozen=True, slots=True)
class CleanGenerationSettings:
    """Complete immutable effective clean-generation settings."""

    topology: str = "periodic_bicrystal"
    boundary_conditions: tuple[str, str, str] = ("periodic", "periodic", "periodic")
    vacuum_angstrom: float = 0.0
    fixed_region_thickness_angstrom: float = 0.0
    surface_buffer_thickness_angstrom: float = 0.0
    feasibility_policy: FeasibilityPolicy = FeasibilityPolicy()
    feasibility_override: FeasibilityOverride | None = None
    translation_domain: CartesianTranslationDomain = CartesianTranslationDomain(
        in_plane_components=((0.0,), (0.0,)), normal_offsets=(0.0,)
    )
    termination_domain: TerminationDomainSelection = TerminationDomainSelection()
    retain_warnings: bool = False
    max_seeds: int = 1
    initialization_enabled: bool = True
    schema_version: int = CLEAN_GENERATION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.topology not in {"periodic_bicrystal", "single_interface_slab"}:
            raise CleanGenerationConfigError(
                "topology must be periodic_bicrystal or single_interface_slab."
            )
        conditions = tuple(self.boundary_conditions)
        if len(conditions) != 3 or any(
            item not in {"periodic", "fixed"} for item in conditions
        ):
            raise CleanGenerationConfigError(
                "boundary_conditions must contain three periodic/fixed values."
            )
        expected_x = "periodic" if self.topology == "periodic_bicrystal" else "fixed"
        if conditions[0] != expected_x:
            raise CleanGenerationConfigError(
                f"{self.topology} requires x boundary condition {expected_x}."
            )
        vacuum = _nonnegative_float(self.vacuum_angstrom, "vacuum_angstrom")
        fixed = _nonnegative_float(
            self.fixed_region_thickness_angstrom,
            "fixed_region_thickness_angstrom",
        )
        buffer = _nonnegative_float(
            self.surface_buffer_thickness_angstrom,
            "surface_buffer_thickness_angstrom",
        )
        if self.topology == "periodic_bicrystal" and (vacuum or fixed or buffer):
            raise CleanGenerationConfigError(
                "periodic_bicrystal requires zero vacuum, fixed-region thickness, "
                "and surface-buffer thickness."
            )
        if not isinstance(self.feasibility_policy, FeasibilityPolicy):
            raise CleanGenerationConfigError(
                "feasibility_policy must be a FeasibilityPolicy."
            )
        if self.feasibility_override is not None and not isinstance(
            self.feasibility_override, FeasibilityOverride
        ):
            raise CleanGenerationConfigError(
                "feasibility_override must be a FeasibilityOverride or None."
            )
        if not isinstance(self.translation_domain, CartesianTranslationDomain):
            raise CleanGenerationConfigError(
                "translation_domain must be a CartesianTranslationDomain."
            )
        if not isinstance(self.termination_domain, TerminationDomainSelection):
            raise CleanGenerationConfigError(
                "termination_domain must be a TerminationDomainSelection."
            )
        retain = _bool(self.retain_warnings, "retain_warnings")
        enabled = _bool(self.initialization_enabled, "initialization_enabled")
        max_seeds = _positive_int(self.max_seeds, "max_seeds")
        if self.schema_version != CLEAN_GENERATION_CONFIG_SCHEMA_VERSION:
            raise CleanGenerationConfigError(
                f"Unsupported schema_version {self.schema_version!r}."
            )
        object.__setattr__(self, "boundary_conditions", conditions)
        object.__setattr__(self, "vacuum_angstrom", vacuum)
        object.__setattr__(self, "fixed_region_thickness_angstrom", fixed)
        object.__setattr__(self, "surface_buffer_thickness_angstrom", buffer)
        object.__setattr__(self, "retain_warnings", retain)
        object.__setattr__(self, "initialization_enabled", enabled)
        object.__setattr__(self, "max_seeds", max_seeds)

    @classmethod
    def from_mapping(cls, value: object) -> "CleanGenerationSettings":
        item = _mapping(value, "clean-generation configuration")
        allowed = {
            "schema_version",
            "topology",
            "boundary_conditions",
            "vacuum_angstrom",
            "fixed_region_thickness_angstrom",
            "surface_buffer_thickness_angstrom",
            "feasibility_policy",
            "override",
            "translation_domain",
            "termination_domain",
            "retain_warnings",
            "max_seeds",
            "initialization_enabled",
        }
        _only_keys(item, allowed, "clean-generation configuration")
        topology = str(item.get("topology", "periodic_bicrystal"))
        default_conditions = (
            ("periodic", "periodic", "periodic")
            if topology == "periodic_bicrystal"
            else ("fixed", "periodic", "periodic")
        )
        return cls(
            topology=topology,
            boundary_conditions=tuple(
                item.get("boundary_conditions", default_conditions)
            ),  # type: ignore[arg-type]
            vacuum_angstrom=item.get("vacuum_angstrom", 0.0),
            fixed_region_thickness_angstrom=item.get(
                "fixed_region_thickness_angstrom", 0.0
            ),
            surface_buffer_thickness_angstrom=item.get(
                "surface_buffer_thickness_angstrom", 0.0
            ),
            feasibility_policy=feasibility_policy_from_mapping(
                item.get("feasibility_policy", {})
            ),
            feasibility_override=feasibility_override_from_mapping(
                item.get("override")
            ),
            translation_domain=translation_domain_from_mapping(
                item.get(
                    "translation_domain",
                    {
                        "in_plane_components": [[0.0], [0.0]],
                        "normal_offsets": [0.0],
                    },
                )
            ),
            termination_domain=TerminationDomainSelection.from_mapping(
                item.get("termination_domain", {"mode": "all"})
            ),
            retain_warnings=item.get("retain_warnings", False),
            max_seeds=item.get("max_seeds", 1),
            initialization_enabled=item.get("initialization_enabled", True),
            schema_version=item.get(
                "schema_version", CLEAN_GENERATION_CONFIG_SCHEMA_VERSION
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "CleanGenerationSettings":
        source = Path(path)
        try:
            if source.suffix.lower() == ".json":
                payload = json.loads(source.read_text(encoding="utf-8"))
            elif source.suffix.lower() in {".toml", ".tml"}:
                if tomllib is None:  # pragma: no cover
                    raise CleanGenerationConfigError(
                        "TOML configuration requires tomllib or the tomli package."
                    )
                payload = tomllib.loads(source.read_text(encoding="utf-8"))
            else:
                raise CleanGenerationConfigError(
                    "clean configuration must use a .json or .toml suffix."
                )
        except (OSError, json.JSONDecodeError) as exc:
            raise CleanGenerationConfigError(
                f"Cannot read clean-generation configuration {source}: {exc}"
            ) from exc
        if isinstance(payload, Mapping) and "clean_generation" in payload:
            payload = payload["clean_generation"]
        return cls.from_mapping(payload)

    def with_overrides(
        self,
        *,
        topology: str | None = None,
        boundary_conditions: Sequence[str] | None = None,
        vacuum_angstrom: float | None = None,
        fixed_region_thickness_angstrom: float | None = None,
        surface_buffer_thickness_angstrom: float | None = None,
        retain_warnings: bool | None = None,
        max_seeds: int | None = None,
        initialization_enabled: bool | None = None,
        default_termination_only: bool = False,
    ) -> "CleanGenerationSettings":
        selected_topology = self.topology if topology is None else topology
        selected_conditions = (
            self.boundary_conditions
            if boundary_conditions is None
            else tuple(boundary_conditions)
        )
        if topology is not None and boundary_conditions is None:
            selected_conditions = (
                ("periodic", "periodic", "periodic")
                if selected_topology == "periodic_bicrystal"
                else ("fixed", "periodic", "periodic")
            )
        selected_termination = (
            TerminationDomainSelection(mode="default_only")
            if default_termination_only
            else self.termination_domain
        )
        return replace(
            self,
            topology=selected_topology,
            boundary_conditions=selected_conditions,
            vacuum_angstrom=(
                self.vacuum_angstrom if vacuum_angstrom is None else vacuum_angstrom
            ),
            fixed_region_thickness_angstrom=(
                self.fixed_region_thickness_angstrom
                if fixed_region_thickness_angstrom is None
                else fixed_region_thickness_angstrom
            ),
            surface_buffer_thickness_angstrom=(
                self.surface_buffer_thickness_angstrom
                if surface_buffer_thickness_angstrom is None
                else surface_buffer_thickness_angstrom
            ),
            retain_warnings=(
                self.retain_warnings if retain_warnings is None else retain_warnings
            ),
            max_seeds=self.max_seeds if max_seeds is None else max_seeds,
            initialization_enabled=(
                self.initialization_enabled
                if initialization_enabled is None
                else initialization_enabled
            ),
            termination_domain=selected_termination,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "topology": self.topology,
            "boundary_conditions": list(self.boundary_conditions),
            "vacuum_angstrom": self.vacuum_angstrom,
            "fixed_region_thickness_angstrom": self.fixed_region_thickness_angstrom,
            "surface_buffer_thickness_angstrom": self.surface_buffer_thickness_angstrom,
            "feasibility_policy": self.feasibility_policy.to_dict(),
            "feasibility_policy_hash": self.feasibility_policy.policy_hash,
            "override": (
                None
                if self.feasibility_override is None
                else {
                    "status": self.feasibility_override.status,
                    "reason": self.feasibility_override.reason,
                }
            ),
            "translation_domain": self.translation_domain.to_dict(),
            "translation_domain_hash": self.translation_domain.domain_hash,
            "termination_domain": self.termination_domain.to_dict(),
            "termination_selection_hash": self.termination_domain.selection_hash,
            "retain_warnings": self.retain_warnings,
            "max_seeds": self.max_seeds,
            "initialization_enabled": self.initialization_enabled,
        }

    @property
    def configuration_hash(self) -> str:
        return _sha256(self.to_dict())


__all__ = [
    "CLEAN_GENERATION_CONFIG_SCHEMA_VERSION",
    "CleanGenerationConfigError",
    "CleanGenerationSettings",
    "RationalPhase",
    "TerminationDomainSelection",
    "feasibility_override_from_mapping",
    "feasibility_policy_from_mapping",
    "translation_domain_from_mapping",
]
