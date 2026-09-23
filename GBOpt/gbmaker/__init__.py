# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Internal package for decomposing ``GBOpt.GBMaker`` construction.

The package-level surface exposes the construction-state contracts from ``types``, the
pure normalization functions from ``config`` and ``material`` that GBMaker's legacy
constructor and ``from_boundary_spec`` factory now delegate to, and ``resolve_orientation``
from ``orientation``, the pure construction stage that resolves per-grain rotations,
periodic Miller rows, and in-plane periodicity. GBMaker keeps its own ``__validate``
instance method and per-field static validators for property setters, but those are
thin wrappers over the pure functions here, so there is a single implementation of each
validation rule; the same is true of GBMaker's remaining orientation-related private
methods and ``orientation``'s underscore-prefixed helpers (row reduction, angular
error, integer-row approximation, misorientation decomposition), which are not promoted
here but are imported directly by ``GBOpt.GBMaker``, the same way it already imports
``config``'s ``_validate_scalar``. No geometry-kernel or grain-generation logic has
moved into this package yet; that extraction happens incrementally in later roadmap
issues (R06 through R09).
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
from .orientation import resolve_orientation
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
    # Pure construction stages
    "resolve_orientation",
]
