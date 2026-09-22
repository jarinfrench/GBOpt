# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Internal package for decomposing ``GBOpt.GBMaker`` construction.

The package-level surface currently exposes only the construction-state contracts from
``types``: the exception hierarchy and immutable dataclasses describing normalized build
configuration, resolved boundary input, per-grain material identity, orientation state,
per-axis strain accommodation, box-dimension planning, per-grain build requests and
results, and the final assembled bicrystal. No construction, orientation-resolution, or
supercell logic has moved into this package yet; that extraction happens incrementally
in later roadmap issues (R04 through R09), which is why no compatibility facade is
required here. Validation helpers in ``types`` are internal and are not promoted.
"""

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
]
