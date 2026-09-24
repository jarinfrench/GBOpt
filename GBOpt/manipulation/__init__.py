# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the manipulation-operation contract API and built-in operations.

The package-level surface contains the operation protocol, its context/result value
types, the full exception hierarchy, the explicit registry, and this package's built-in
operation implementations (right-grain translation, grain-local termination cycling,
and interface separation). Registration in ``default_registry`` is still always
explicit -- importing this module registers no built-in operation under a name; it only
makes the operation classes importable. ``GBManipulator``'s own legacy methods
(``translate_right_grain``, ``cycle_grain_terminations``, ``apply_interface_separation``,
and their ``make_*_candidate`` counterparts) delegate to these operations and remain the
supported entry points for that scripted usage.
"""

from .registry import ManipulationRegistry, default_registry
from .separation import InterfaceSeparation
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
]
