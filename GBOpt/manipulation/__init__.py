# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the manipulation-operation contract API and built-in operations.

The package-level surface contains the operation protocol, its context/result value
types, the full exception hierarchy, the explicit registry, and this package's built-in
operation implementations (right-grain translation, grain-local termination cycling,
interface separation, atom insertion/removal, single-mode soft-phonon displacement, and
two-parent slice-and-merge crossover). Registration in ``default_registry`` is still
always explicit -- importing this module registers no built-in operation under a name;
it only makes the operation classes importable. ``GBManipulator``'s own legacy methods
(``translate_right_grain``, ``cycle_grain_terminations``, ``apply_interface_separation``,
``insert_atoms``, ``remove_atoms``, ``displace_along_soft_modes``, ``slice_and_merge``,
and their ``make_*_candidate`` counterparts) share these operations' pure computational
core and remain the supported entry points for that scripted usage.
"""

from .crossover import SliceAndMerge
from .density import AtomInsertion, AtomRemoval
from .registry import ManipulationRegistry, default_registry
from .separation import InterfaceSeparation
from .soft_mode import SoftModeDisplacement
from .termination import GrainTerminationCycle
from .translation import RightGrainTranslation
from .types import (
    Manipulation,
    ManipulationArityError,
    ManipulationCapabilityError,
    ManipulationCompatibilityError,
    ManipulationConfigurationError,
    ManipulationContext,
    ManipulationError,
    ManipulationExecutionError,
    ManipulationLookupError,
    ManipulationRegistrationError,
    ManipulationResult,
)

__all__ = [
    # Exceptions
    "ManipulationError",
    "ManipulationConfigurationError",
    "ManipulationArityError",
    "ManipulationCompatibilityError",
    "ManipulationCapabilityError",
    "ManipulationExecutionError",
    "ManipulationRegistrationError",
    "ManipulationLookupError",
    # Protocol and value types
    "Manipulation",
    "ManipulationContext",
    "ManipulationResult",
    # Registry
    "ManipulationRegistry",
    "default_registry",
    # Built-in operations
    "RightGrainTranslation",
    "GrainTerminationCycle",
    "InterfaceSeparation",
    "AtomInsertion",
    "AtomRemoval",
    "SoftModeDisplacement",
    "SliceAndMerge",
]
