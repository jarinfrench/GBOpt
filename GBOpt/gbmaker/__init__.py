# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Internal package for decomposing ``GBOpt.GBMaker`` construction.

The package-level surface exposes the construction-state contracts from ``types``, the
pure normalization functions from ``config`` and ``material`` that GBMaker's legacy
constructor and ``from_boundary_spec`` factory now delegate to, ``resolve_orientation``
from ``orientation``, the pure construction stage that resolves per-grain rotations,
periodic Miller rows, and in-plane periodicity, and ``plan_periodic_spacing``/
``plan_dimensions`` from ``dimension``, the pure construction stages that plan periodic
spacing, boundary-normal x extents, per-axis commensurate-repeat strain accommodation,
minimum in-plane dimension enforcement, and final simulation-box dimensions. GBMaker
keeps its own ``__validate`` instance method and per-field static validators for
property setters, but those are thin wrappers over the pure functions here, so there is
a single implementation of each validation rule; the same is true of GBMaker's
remaining orientation-, dimension-, and geometry-related private methods and
``orientation``'s/``dimension``'s/``geometry``'s underscore-prefixed helpers (row
reduction, angular error, integer-row approximation, misorientation decomposition,
commensurate-pair search, per-axis accommodation planning, reduced-coordinate wrapping
and tolerance, periodic/selection basis construction, box-coordinate transforms,
complete-origin masking/filtering/clipping/deduplication), which are not promoted here
but are imported directly by ``GBOpt.GBMaker``, the same way it already imports
``config``'s ``_validate_scalar``. ``geometry`` is a leaf with respect to its
``gbmaker`` siblings (it depends only on ``types``); ``orientation`` and ``dimension``
now import their shared ``_miller_row_norm`` from it instead of each carrying a private
copy.

``exact_grain`` and ``approximate_grain`` (R08, issue #69) extract the exact
decorated-site and floating-point lattice-enumeration grain builders. Both consume a
``GrainBuildRequest`` and return a ``GrainBuildResult``; ``build_exact_grain`` and
``build_approximate_grain`` are promoted here as the package's builder contracts.
``approximate_grain``'s ``filter_grain_result_complete_origins`` and
``trim_grain_result_to_upper_x`` are post-build trimming helpers used only by
``GBOpt.GBMaker``'s vacuum-zero and periodic-gap-equalization orchestration; like the
underscore-prefixed geometry helpers, they are not promoted here but are imported
directly by ``GBOpt.GBMaker``. Both new modules are leaves with respect to each other
-- ``exact_grain`` and ``approximate_grain`` each depend only on ``geometry`` and
``types``, not on one another. Grain-boundary assembly (merging left/right grains,
periodic-gap equalization orchestration) and LAMMPS/file-writer code remain in
``GBOpt.GBMaker``, per issue #69's non-goals.
"""

from .approximate_grain import build_approximate_grain
from .config import (
    normalize_legacy_config,
    resolve_boundary_input,
    validate_boundary_mode,
    validate_exact_limit,
    validate_mismatch_max_cells,
    validate_mismatch_tol,
    validate_strain_grain,
)
from .dimension import plan_dimensions, plan_periodic_spacing
from .exact_grain import build_exact_grain
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
    "plan_periodic_spacing",
    "plan_dimensions",
    # Grain builders
    "build_exact_grain",
    "build_approximate_grain",
]
