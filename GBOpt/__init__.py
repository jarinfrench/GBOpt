# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

__version__ = "0.2.0"

from GBOpt.Atom import Atom
from GBOpt.BicrystalState import (
    LEFT_GRAIN_ID,
    RIGHT_GRAIN_ID,
    STATE_SCHEMA_VERSION,
    TRANSLATION_HISTORY_KEY,
    TRANSLATION_OPERATION_SCHEMA_VERSION,
    TRANSLATION_CONVENTION,
    BicrystalState,
    BicrystalStateError,
    BicrystalStateTypeError,
    BicrystalStateValueError,
    BicrystalTopology,
    BoundaryCondition,
    GrainSelector,
    InterfaceDescriptor,
    RegionDescriptor,
    SurfaceDescriptor,
    translate_grain,
)
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import GBManipulator
from GBOpt.Position import Position
from GBOpt.UnitCell import UnitCell

__citation__ = """
French, J. C. and Bhave, C. V. (2026). GBOpt: Grain Boundary Structure Optimization
Using Monte Carlo and Evolutionary Algorithms. SoftwareX, 35, 102763.
https://doi.org/10.1016/j.softx.2026.102763
"""
from GBOpt.geometry_audit import GeometryAuditError
from GBOpt.geometry_validation import (
    BicrystalFeasibilityReport,
    ContactPolicy,
    FeasibilityOverride,
    FeasibilityPolicy,
    FeasibilityStatus,
    GeometryValidationError,
    SlabPolicy,
    SpeciesPairThresholds,
    ValidationReason,
    VoidPolicy,
    validate_bicrystal_state,
)
from GBOpt.interface_initialization import (
    INITIALIZATION_SCHEMA_VERSION,
    AttemptDisposition,
    CandidateKind,
    CartesianTranslationDomain,
    InitializationStatus,
    InterfaceInitializationError,
    InterfaceInitializer,
    TranslationAttempt,
    TranslationCandidate,
    TranslationSearchResult,
    TranslationSeed,
    generate_translation_seeds,
)
from GBOpt.termination import (
    CutConvention,
    GrainSide,
    GrainTermination,
    TerminationError,
    TerminationPair,
    enumerate_grain_terminations,
)
from GBOpt.termination_initialization import (
    TERMINATION_INITIALIZATION_SCHEMA_VERSION,
    DecoratedPopulationCheck,
    ExactBoundaryReconstruction,
    TerminationAttempt,
    TerminationCandidate,
    TerminationDisposition,
    TerminationDomain,
    TerminationInitializationError,
    TerminationInitializationStatus,
    TerminationInitializer,
    TerminationSearchResult,
    TerminationSeed,
    TerminationSeedKind,
    generate_termination_seeds,
)
