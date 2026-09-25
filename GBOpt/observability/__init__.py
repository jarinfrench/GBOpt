# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the versioned MC/GA lifecycle event vocabulary and its sinks.

The package-level surface contains the immutable ``RunContext``/``OptimizationEvent``
value types, the ``OptimizationAlgorithm``/``OptimizationEventType``/
``TerminationReason`` enums, the ``EVENT_SCHEMA_VERSION`` constant, the full exception
hierarchy, and the ``EventSink`` protocol with its ``NullEventSink``/
``LoggingEventSink``/``CompositeEventSink`` implementations. No MC/GA loop is wired
through these here -- ``MonteCarloMinimizer``/``GeneticAlgorithmMinimizer`` construct and
emit events using this package's types, defaulting to ``NullEventSink`` when a caller
supplies none.
"""

from .adapters import evaluation_event_fields
from .sinks import CompositeEventSink, EventSink, LoggingEventSink, NullEventSink
from .types import (
    EVENT_SCHEMA_VERSION,
    ObservabilityError,
    ObservabilityTypeError,
    ObservabilityValueError,
    OptimizationAlgorithm,
    OptimizationEvent,
    OptimizationEventType,
    RunContext,
    TerminationReason,
)

__all__ = [
    # Exceptions
    "ObservabilityError",
    "ObservabilityTypeError",
    "ObservabilityValueError",
    # Enums
    "OptimizationAlgorithm",
    "OptimizationEventType",
    "TerminationReason",
    # Value types
    "EVENT_SCHEMA_VERSION",
    "RunContext",
    "OptimizationEvent",
    # Sinks
    "EventSink",
    "NullEventSink",
    "LoggingEventSink",
    "CompositeEventSink",
    # Adapters
    "evaluation_event_fields",
]
