# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact crystallographic termination descriptors for GBMaker construction.

A termination is represented as an exact rational phase along the first row of a
canonical integer grain supercell.  The phase is applied while exact decorated sites
are still represented in supercell coordinates, before conversion to Cartesian
coordinates.  The finite supported phase set is derived from decorated-site layer
positions: each phase places one crystallographically distinct decorated layer on the
half-open cut at supercell coordinate zero.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, TypeAlias

import numpy as np

from GBOpt.gbmaker_supercell import (
    SupercellSites,
    build_supercell_matrix,
    enumerate_supercell_sites,
)

GrainSide: TypeAlias = Literal["left", "right"]
CutConvention: TypeAlias = Literal["decorated_layer_at_half_open_cut"]

_TERMINATION_SCHEMA_VERSION = 1
_VALID_GRAINS = frozenset({"left", "right"})
_CUT_CONVENTION: CutConvention = "decorated_layer_at_half_open_cut"


class TerminationError(ValueError):
    """Raised when an exact termination descriptor is malformed or unsupported."""


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


def _integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TerminationError(f"{name} must be an integer; got {value!r}.")
    return int(value)


@dataclass(frozen=True, slots=True, order=True)
class GrainTermination:
    """One exact lattice-phase termination for a named grain.

    ``phase_numerator / phase_denominator`` is canonicalized modulo one.  It is
    measured in units of the grain's unrepeated canonical supercell-normal row.  The
    fixed cut convention means the phase is not an arbitrary real-valued translation:
    supported values are the finite phases that place a decorated crystallographic
    layer at the lower half-open supercell cut.
    """

    grain: GrainSide
    phase_numerator: int = 0
    phase_denominator: int = 1
    cut_convention: CutConvention = _CUT_CONVENTION
    schema_version: int = _TERMINATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.grain not in _VALID_GRAINS:
            raise TerminationError(
                f"grain must be 'left' or 'right'; got {self.grain!r}."
            )
        numerator = _integer(self.phase_numerator, "phase_numerator")
        denominator = _integer(self.phase_denominator, "phase_denominator")
        if denominator <= 0:
            raise TerminationError(
                "phase_denominator must be a positive integer."
            )
        if self.cut_convention != _CUT_CONVENTION:
            raise TerminationError(
                f"Unsupported cut_convention {self.cut_convention!r}; expected "
                f"{_CUT_CONVENTION!r}."
            )
        if self.schema_version != _TERMINATION_SCHEMA_VERSION:
            raise TerminationError(
                f"Unsupported termination schema_version {self.schema_version!r}."
            )
        phase = Fraction(numerator, denominator) % 1
        object.__setattr__(self, "phase_numerator", phase.numerator)
        object.__setattr__(self, "phase_denominator", phase.denominator)

    @property
    def phase(self) -> Fraction:
        """Return the canonical exact phase in ``[0, 1)``."""
        return Fraction(self.phase_numerator, self.phase_denominator)

    @property
    def is_default(self) -> bool:
        """Return whether this is the historical zero-phase construction."""
        return self.phase_numerator == 0

    @property
    def interface_face(self) -> str:
        """Return the grain face adjacent to the central interface."""
        return "upper" if self.grain == "left" else "lower"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "grain": self.grain,
            "normal_coordinate": "canonical_supercell_axis_0",
            "interface_face": self.interface_face,
            "cut_convention": self.cut_convention,
            "phase": {
                "numerator": self.phase_numerator,
                "denominator": self.phase_denominator,
            },
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def descriptor_hash(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class TerminationPair:
    """Canonical left/right exact termination pair."""

    left: GrainTermination = GrainTermination("left")
    right: GrainTermination = GrainTermination("right")
    schema_version: int = _TERMINATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.left, GrainTermination) or self.left.grain != "left":
            raise TerminationError(
                "left must be a GrainTermination whose grain is 'left'."
            )
        if not isinstance(self.right, GrainTermination) or self.right.grain != "right":
            raise TerminationError(
                "right must be a GrainTermination whose grain is 'right'."
            )
        if self.schema_version != _TERMINATION_SCHEMA_VERSION:
            raise TerminationError(
                f"Unsupported termination-pair schema_version {self.schema_version!r}."
            )

    @property
    def is_default(self) -> bool:
        return self.left.is_default and self.right.is_default

    @property
    def canonical_key(self) -> tuple[Fraction, Fraction]:
        return (self.left.phase, self.right.phase)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def pair_hash(self) -> str:
        return _sha256(self.to_dict())


def enumerate_grain_terminations(
    grain: GrainSide,
    P_or_Q: np.ndarray,
    *,
    basis_numerators: np.ndarray,
    basis_denominator: int,
) -> tuple[GrainTermination, ...]:
    """Return the finite exact decorated-layer termination set for one grain.

    A one-repeat exact decorated supercell is enumerated.  If a decorated layer has
    supercell-normal coordinate ``u``, the canonical construction phase ``-u mod 1``
    places that layer at the half-open cut.  Equal phases from different basis rows or
    in-plane origins are deduplicated exactly.  Zero phase is always first; remaining
    phases are ordered by exact rational value.
    """
    if grain not in _VALID_GRAINS:
        raise TerminationError(
            f"grain must be 'left' or 'right'; got {grain!r}."
        )
    supercell = build_supercell_matrix(P_or_Q)
    sites = enumerate_supercell_sites(
        supercell,
        1,
        1,
        1,
        basis_numerators=basis_numerators,
        basis_denominator=basis_denominator,
    )
    denominator = sites.supercell_coordinate_denominator
    phases = {
        Fraction((-int(value)) % denominator, denominator)
        for value in sites.supercell_coordinate_numerators[:, 0]
    }
    phases.add(Fraction(0, 1))
    ordered = sorted(phases, key=lambda phase: (phase != 0, phase))
    return tuple(
        GrainTermination(
            grain=grain,
            phase_numerator=phase.numerator,
            phase_denominator=phase.denominator,
        )
        for phase in ordered
    )


def shifted_crystal_coordinates(
    sites: SupercellSites,
    supercell: np.ndarray,
    phase: Fraction,
    *,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> tuple[np.ndarray, int]:
    """Apply an exact supercell-normal phase and return crystal coordinates.

    The returned coordinates are exact numerators over a common positive denominator.
    Wrapping occurs in exact repeated-supercell coordinates.  Site count, basis-index
    mapping, and complete decorated populations are unchanged.
    """
    phase = Fraction(phase) % 1
    if phase == 0:
        return np.asarray(sites.crystal_numerators, dtype=object), sites.denominator

    site_denominator = sites.supercell_coordinate_denominator
    phase_denominator = phase.denominator
    common = math.lcm(site_denominator, phase_denominator)
    scale_sites = common // site_denominator
    scale_phase = common // phase_denominator

    u = np.asarray(sites.supercell_coordinate_numerators, dtype=object) * scale_sites
    u = np.array(u, dtype=object, copy=True)
    u[:, 0] += phase.numerator * scale_phase
    limits = np.asarray((repeat_x, repeat_y, repeat_z), dtype=object) * common
    u = np.mod(u, limits)

    crystal = u @ np.asarray(supercell, dtype=object)
    return np.asarray(crystal, dtype=object), common


__all__ = [
    "CutConvention",
    "GrainSide",
    "GrainTermination",
    "TerminationError",
    "TerminationPair",
    "enumerate_grain_terminations",
    "shifted_crystal_coordinates",
]
