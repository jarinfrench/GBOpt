# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the manipulation-operation contract API.

The package-level surface contains the operation protocol, its context/result value
types, the full exception hierarchy, and the explicit registry. It does not export any
built-in operation implementation: this package defines the contract and plumbing only,
not the scientific algorithms ``GBManipulator`` still implements directly.
"""

from .registry import ManipulationRegistry, default_registry
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
]
