# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Expose the versioned MC/GA lifecycle event vocabulary, its sinks, and its journal.

The package-level surface contains the immutable ``RunContext``/``OptimizationEvent``/
``RunManifest`` value types, the ``OptimizationAlgorithm``/``OptimizationEventType``/
``TerminationReason`` enums, the ``EVENT_SCHEMA_VERSION``/``MANIFEST_SCHEMA_VERSION``
constants, the full exception hierarchy, the ``EventSink`` protocol with its
``NullEventSink``/``LoggingEventSink``/``CompositeEventSink``/``JsonlEventSink``
implementations, and the journal/manifest reader/writer functions. No MC/GA loop is
wired through these here -- ``MonteCarloMinimizer``/``GeneticAlgorithmMinimizer``
construct and emit events using this package's types, defaulting to ``NullEventSink``
when a caller supplies none.
"""

from .adapters import evaluation_event_fields
from .journal import (
    JournalWriteMode,
    JsonlEventSink,
    read_journal_events,
    read_run_manifest,
    write_run_manifest,
)
from .sinks import CompositeEventSink, EventSink, LoggingEventSink, NullEventSink
from .types import (
    EVENT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    ObservabilityError,
    ObservabilityTypeError,
    ObservabilityValueError,
    OptimizationAlgorithm,
    OptimizationEvent,
    OptimizationEventType,
    RunContext,
    RunManifest,
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
    "MANIFEST_SCHEMA_VERSION",
    "RunContext",
    "OptimizationEvent",
    "RunManifest",
    # Sinks
    "EventSink",
    "NullEventSink",
    "LoggingEventSink",
    "CompositeEventSink",
    "JsonlEventSink",
    # Adapters
    "evaluation_event_fields",
    # Journal / manifest
    "JournalWriteMode",
    "read_journal_events",
    "read_run_manifest",
    "write_run_manifest",
]
