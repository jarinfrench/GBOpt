# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Explicit registration of this package's built-in manipulation operations.

Importing this module, like importing the rest of ``GBOpt.manipulation``, registers
nothing on its own. ``register_builtin_operations`` is the one explicit call that adds
every built-in operation (right-grain translation, grain-termination cycling, interface
separation, atom insertion/removal, single-mode soft-phonon displacement, and two-parent
slice-and-merge crossover) to a registry under its own ``Manipulation.name``, so a caller
that discovers operations purely by registry lookup (e.g. ``GBOpt.optimization``'s
``OperationSpec`` compatibility adapters) sees built-in and third-party operations
through the same seam.
"""

from __future__ import annotations

from GBOpt.manipulation.crossover import SliceAndMerge
from GBOpt.manipulation.density import AtomInsertion, AtomRemoval
from GBOpt.manipulation.registry import ManipulationRegistry, default_registry
from GBOpt.manipulation.separation import InterfaceSeparation
from GBOpt.manipulation.soft_mode import SoftModeDisplacement
from GBOpt.manipulation.termination import GrainTerminationCycle
from GBOpt.manipulation.translation import RightGrainTranslation
from GBOpt.manipulation.types import (
    Manipulation,
    ManipulationLookupError,
    ManipulationRegistrationError,
)

_BUILTIN_OPERATION_TYPES: tuple[type[Manipulation], ...] = (
    RightGrainTranslation,
    GrainTerminationCycle,
    InterfaceSeparation,
    AtomInsertion,
    AtomRemoval,
    SoftModeDisplacement,
    SliceAndMerge,
)


def register_builtin_operations(
    registry: ManipulationRegistry = default_registry,
) -> None:
    """Register every built-in operation under its own name, idempotently.

    Calling this more than once against the same registry (e.g. because more than one
    caller imports a package that performs this registration) is safe: an operation
    already registered under its own name is left alone.

    :param registry: Registry to register into, defaults to
        ``GBOpt.manipulation.default_registry``.
    :raises ManipulationRegistrationError: If a built-in's name is already registered to
        a different operation than the built-in itself.
    """
    for operation_type in _BUILTIN_OPERATION_TYPES:
        operation = operation_type()
        try:
            existing = registry.get(operation.name)
        except ManipulationLookupError:
            registry.register(operation.name, operation)
            continue
        if type(existing) is not operation_type:
            raise ManipulationRegistrationError(
                f"a manipulation named {operation.name!r} is already registered to a "
                f"different operation ({type(existing).__name__!r})"
            )


__all__ = ["register_builtin_operations"]
