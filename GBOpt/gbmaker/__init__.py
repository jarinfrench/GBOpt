# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Internal package for decomposing ``GBOpt.GBMaker`` construction.

The package-level surface exposes the construction-state contracts from ``types`` and
the pure normalization functions from ``config`` and ``material`` that GBMaker's legacy
constructor and ``from_boundary_spec`` factory now delegate to. GBMaker keeps its own
``__validate`` instance method and per-field static validators for property setters,
but those are thin wrappers over the pure functions here, so there is a single
implementation of each validation rule. No orientation, geometry, or grain-generation
logic has moved into this package yet; that extraction happens incrementally in later
roadmap issues (R05 through R09). Internal validation helpers prefixed with an
underscore are not promoted.
"""

from .config import (
    normalize_legacy_config,
    resolve_boundary_input,
    validate_boundary_mode,
    validate_exact_limit,
    validate_mismatch_max_cells,
    validate_mismatch_tol,
    validate_strain_grain,
)
from .material import resolve_material_state
from .types import (
    AxisAccommodation,
    BicrystalResult,
    BoundaryMode,
    DimensionPlan,
    GBBuildConfig,
    GBMakerConstructionError,
    GBMakerConstructionTypeError,
    GBMakerConstructionValueError,
    GrainBuildRequest,
    GrainBuildResult,
    GrainSide,
    MaterialState,
    OrientationState,
    ResolvedBoundaryInput,
    StrainGrainPolicy,
)

__all__ = [
    # Exceptions
    "GBMakerConstructionError",
    "GBMakerConstructionValueError",
    "GBMakerConstructionTypeError",
    # Type aliases
    "BoundaryMode",
    "StrainGrainPolicy",
    "GrainSide",
    # Construction-state contracts
    "MaterialState",
    "GBBuildConfig",
    "ResolvedBoundaryInput",
    "OrientationState",
    "AxisAccommodation",
    "DimensionPlan",
    "GrainBuildRequest",
    "GrainBuildResult",
    "BicrystalResult",
    # Pure normalization functions
    "resolve_material_state",
    "normalize_legacy_config",
    "resolve_boundary_input",
    "validate_boundary_mode",
    "validate_exact_limit",
    "validate_mismatch_tol",
    "validate_mismatch_max_cells",
    "validate_strain_grain",
]
