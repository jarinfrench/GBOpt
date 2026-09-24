# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define the manipulation-operation exception hierarchy and value types.

Contains the ``GBOpt.manipulation`` exception hierarchy, the ``ManipulationContext``/
``ManipulationResult`` immutable value types operations exchange with a caller, and the
``Manipulation`` protocol an operation implements. This is the lowest-level module in
the package: everything else in ``GBOpt.manipulation`` may import from here, and this
module imports from nothing else in the package. No registry, no built-in operation, and
no ``GBManipulator`` facade logic belongs here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from GBOpt.interface import InterfaceCandidate


class ManipulationError(Exception):
    """Base for exceptions raised by the ``GBOpt.manipulation`` package."""


class ManipulationConfigurationError(ManipulationError, ValueError):
    """Raised when a manipulation, context, or result is malformed."""


class ManipulationArityError(ManipulationError, ValueError):
    """Raised when the supplied parent count does not match an operation's arity."""


class ManipulationCompatibilityError(ManipulationError, ValueError):
    """Raised when supplied parents are mutually incompatible for an operation."""


class ManipulationCapabilityError(ManipulationError, ValueError):
    """Raised when an operation cannot support the parents or parameters it was given."""


class ManipulationExecutionError(ManipulationError, RuntimeError):
    """Raised when an operation fails while executing against a valid context."""


class ManipulationRegistrationError(ManipulationError, ValueError):
    """Raised when registering a manipulation name that is already registered."""


class ManipulationLookupError(ManipulationError, LookupError):
    """Raised when looking up a manipulation name that is not registered."""


@dataclass(frozen=True, slots=True, init=False)
class ManipulationContext:
    """Immutable input an operation executes against.

    :param parents: One or more immutable parent candidates, in parent order.
    :param rng: Random-number generator the operation must use for any randomness, so
        that a supplied ``np.random.Generator`` fully determines the operation's outcome.
    :param params: Read-only mapping of operation-specific parameters.
    """

    parents: tuple[InterfaceCandidate, ...]
    rng: np.random.Generator
    params: Mapping[str, object]

    def __init__(
        self,
        *,
        parents: tuple[InterfaceCandidate, ...] | list[InterfaceCandidate],
        rng: np.random.Generator,
        params: Mapping[str, object] | None = None,
    ) -> None:
        """Construct a validated, immutable manipulation context.

        :param parents: Keyword argument, required. One or more parent candidates.
        :param rng: Keyword argument, required. Random-number generator to expose to
            the operation.
        :param params: Keyword argument, optional, defaults to ``None``. Operation
            parameters.
        :raises ManipulationConfigurationError: If ``parents`` is empty or contains a
            non-``InterfaceCandidate`` value, or ``rng`` is not a ``np.random.Generator``.
        """
        parents_tuple = tuple(parents)
        if not parents_tuple or not all(
            isinstance(parent, InterfaceCandidate) for parent in parents_tuple
        ):
            raise ManipulationConfigurationError(
                "parents must be a nonempty sequence of InterfaceCandidate instances"
            )
        if not isinstance(rng, np.random.Generator):
            raise ManipulationConfigurationError(
                "rng must be a numpy.random.Generator instance"
            )
        object.__setattr__(self, "parents", parents_tuple)
        object.__setattr__(self, "rng", rng)
        object.__setattr__(self, "params", MappingProxyType(dict(params or {})))


@dataclass(frozen=True, slots=True, init=False)
class ManipulationResult:
    """Immutable output an operation returns from executing a context.

    :param children: One or more independently stored child candidates produced by the
        operation. Parent candidates are never included here: producing a child never
        mutates or aliases a parent's state.
    :param parameters: Read-only mapping of the concrete parameter values the operation
        actually used (including any it resolved or sampled from defaults).
    :param lineage: Read-only mapping of operation-provenance metadata (e.g. the
        operation name and parent count) for downstream bookkeeping.
    """

    children: tuple[InterfaceCandidate, ...]
    parameters: Mapping[str, object]
    lineage: Mapping[str, object]

    def __init__(
        self,
        *,
        children: tuple[InterfaceCandidate, ...] | list[InterfaceCandidate],
        parameters: Mapping[str, object] | None = None,
        lineage: Mapping[str, object] | None = None,
    ) -> None:
        """Construct a validated, immutable manipulation result.

        :param children: Keyword argument, required. One or more child candidates.
        :param parameters: Keyword argument, optional, defaults to ``None``. Concrete
            parameter values used by the operation.
        :param lineage: Keyword argument, optional, defaults to ``None``. Operation-
            provenance metadata.
        :raises ManipulationConfigurationError: If ``children`` is empty or contains a
            non-``InterfaceCandidate`` value.
        """
        children_tuple = tuple(children)
        if not children_tuple or not all(
            isinstance(child, InterfaceCandidate) for child in children_tuple
        ):
            raise ManipulationConfigurationError(
                "children must be a nonempty sequence of InterfaceCandidate instances"
            )
        object.__setattr__(self, "children", children_tuple)
        object.__setattr__(self, "parameters", MappingProxyType(dict(parameters or {})))
        object.__setattr__(self, "lineage", MappingProxyType(dict(lineage or {})))


@runtime_checkable
class Manipulation(Protocol):
    """Protocol satisfied by one manipulation operation implementation."""

    @property
    def name(self) -> str:
        """Stable operation name used for registry lookup and lineage metadata."""
        ...

    @property
    def arity(self) -> int:
        """Number of parent candidates this operation requires."""
        ...

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        """Run this operation against an already arity-checked context.

        :param context: Validated input, whose ``parents`` length equals ``arity``.
        :return: The produced children and their provenance metadata.
        """
        ...


__all__ = [
    "ManipulationError",
    "ManipulationConfigurationError",
    "ManipulationArityError",
    "ManipulationCompatibilityError",
    "ManipulationCapabilityError",
    "ManipulationExecutionError",
    "ManipulationRegistrationError",
    "ManipulationLookupError",
    "ManipulationContext",
    "ManipulationResult",
    "Manipulation",
]
