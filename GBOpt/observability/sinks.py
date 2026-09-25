# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define where an ``OptimizationEvent`` goes once a minimizer emits it.

This module consumes one already-built, validated ``OptimizationEvent`` per call and
delivers it -- nowhere (``NullEventSink``), to the standard logging module
(``LoggingEventSink``), or to several other sinks in a fixed order
(``CompositeEventSink``). It does not decide when a lifecycle occurrence happens or what
an event's fields mean; that belongs to ``GBOpt.observability.types`` and to
``MonteCarloMinimizer``/``GeneticAlgorithmMinimizer``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from GBOpt.observability.types import (
    ObservabilityTypeError,
    ObservabilityValueError,
    OptimizationEvent,
    OptimizationEventType,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class EventSink(Protocol):
    """Protocol satisfied by anything that can receive an ``OptimizationEvent``."""

    def emit(self, event: OptimizationEvent) -> None:
        """Deliver one already-validated lifecycle event.

        :param event: Event to deliver.
        """
        ...


class NullEventSink:
    """Discard every event. The default sink for both minimizers.

    A minimizer configured with no explicit sink uses an instance of this class, so
    lifecycle event emission is silent unless a caller opts in to a real sink.
    """

    def emit(self, event: OptimizationEvent) -> None:
        """Discard ``event`` without side effects.

        :param event: Event to discard.
        """
        del event


_DEFAULT_LEVELS: dict[OptimizationEventType, int] = {
    OptimizationEventType.RUN_STARTED: logging.INFO,
    OptimizationEventType.INITIAL_EVALUATION: logging.INFO,
    OptimizationEventType.PROPOSAL_EVALUATED: logging.DEBUG,
    OptimizationEventType.CANDIDATE_ACCEPTED: logging.DEBUG,
    OptimizationEventType.CANDIDATE_REJECTED: logging.DEBUG,
    OptimizationEventType.BEST_UPDATED: logging.INFO,
    OptimizationEventType.GENERATION_BOUNDARY: logging.INFO,
    OptimizationEventType.POPULATION_RESEEDED: logging.WARNING,
    OptimizationEventType.RUN_TERMINATED: logging.INFO,
    OptimizationEventType.RUN_FAILED: logging.ERROR,
}
"""Default event-type -> logging level mapping used by :class:`LoggingEventSink`.

Per-candidate, high-frequency occurrences (a single proposal's evaluation, or one
accept/reject decision) log at ``DEBUG`` so they stay out of an ``INFO``-configured
run's output by default; lifecycle summaries (run start, initial evaluation, a new
best, a generation boundary, termination) log at ``INFO``; a reseed (recovering from a
generation with no valid survivors) logs at ``WARNING`` since it signals every candidate
in that generation failed; a run-ending failure logs at ``ERROR``.
"""


class LoggingEventSink:
    """Format one event as a single log record, at a level appropriate to its type.

    Only scalar fields already on the event are formatted -- ``operation_parameters``
    (the one field that can hold a nested mapping) is never included, so this sink never
    duplicates the large, structured data a dedicated event consumer would read from the
    event object directly.

    :param target_logger: Keyword argument, optional, defaults to ``None``. Logger to
        write to. ``None`` uses this module's own logger.
    """

    def __init__(self, target_logger: logging.Logger | None = None) -> None:
        """Configure one logging-backed event sink.

        :param target_logger: Keyword argument, optional, defaults to ``None``. Logger
            to write to. ``None`` uses this module's own logger.
        :raises ObservabilityTypeError: If ``target_logger`` is neither ``None`` nor a
            ``logging.Logger``.
        """
        if target_logger is not None and not isinstance(target_logger, logging.Logger):
            raise ObservabilityTypeError(
                "target_logger must be a logging.Logger or None"
            )
        self._logger = logger if target_logger is None else target_logger

    def emit(self, event: OptimizationEvent) -> None:
        """Log one event at its event type's configured level.

        :param event: Event to log.
        """
        level = _DEFAULT_LEVELS.get(event.event_type, logging.DEBUG)
        self._logger.log(level, self._format(event))

    @staticmethod
    def _format(event: OptimizationEvent) -> str:
        """Render one event as a single, scalar-only log line.

        :param event: Event to render.
        :return: Human-readable summary of ``event``'s scalar fields.
        """
        fields = [
            f"run={event.run.run_id}",
            f"algorithm={event.run.algorithm.value}",
            f"iteration={event.iteration}",
        ]
        if event.candidate_id is not None:
            fields.append(f"candidate={event.candidate_id}")
        if event.input_index is not None:
            fields.append(f"input_index={event.input_index}")
        if event.status is not None:
            fields.append(f"status={event.status.value}")
        if event.selection_energy is not None:
            fields.append(f"selection_energy={event.selection_energy:.6g}")
        if event.energy is not None:
            fields.append(f"energy={event.energy:.6g}")
        if event.operation_name is not None:
            fields.append(f"operation={event.operation_name}")
        if event.failure_stage is not None:
            fields.append(f"failure_stage={event.failure_stage.value}")
        if event.failure_code is not None:
            fields.append(f"failure_code={event.failure_code}")
        if event.failure_message is not None:
            fields.append(f"failure_message={event.failure_message}")
        if event.termination_reason is not None:
            fields.append(f"termination_reason={event.termination_reason.value}")
        return f"{event.event_type.value}: " + " ".join(fields)


class CompositeEventSink:
    """Fan one event out to several sinks, in one fixed, deterministic order.

    **Emission order**: ``emit()`` calls each configured sink in exactly the order
    given at construction, every time, regardless of any earlier sink's outcome.

    **Sink-failure policy**: if a sink's ``emit()`` raises, the exception is caught and
    logged (via this module's own logger, at ``ERROR``) rather than propagated -- the
    remaining sinks, in order, still receive the event. One broken or slow-to-fail sink
    can therefore never stop another sink from observing an event, and never reaches the
    optimizer's own control flow. A sink that must never silently lose an event should
    implement its own internal retry/durability policy; this class only guarantees that
    a failure there stays contained to that one sink.

    :param sinks: Sinks to fan out to, in emission order.
    :raises ObservabilityValueError: If ``sinks`` is empty.
    :raises ObservabilityTypeError: If any element does not satisfy ``EventSink``.
    """

    def __init__(self, sinks: Sequence[EventSink]) -> None:
        """Configure one fixed-order composite event sink.

        :param sinks: Keyword argument omitted; sinks to fan out to, in emission order.
        :raises ObservabilityValueError: If ``sinks`` is empty.
        :raises ObservabilityTypeError: If any element does not satisfy ``EventSink``.
        """
        sinks_tuple = tuple(sinks)
        if not sinks_tuple:
            raise ObservabilityValueError(
                "CompositeEventSink requires at least one sink"
            )
        for sink in sinks_tuple:
            if not isinstance(sink, EventSink):
                raise ObservabilityTypeError(
                    "every CompositeEventSink sink must satisfy the EventSink protocol"
                )
        self._sinks = sinks_tuple

    @property
    def sinks(self) -> tuple[EventSink, ...]:
        """Configured sinks, in emission order."""
        return self._sinks

    def emit(self, event: OptimizationEvent) -> None:
        """Deliver ``event`` to every configured sink, in order.

        A sink whose ``emit()`` raises is logged and skipped; every other configured
        sink still receives ``event``, in the same fixed order.

        :param event: Event to deliver.
        """
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                # Deliberate recovery boundary, matching this codebase's established
                # evaluator/reconstruction-boundary pattern: one sink's failure must
                # never stop another sink from observing this event, and must never
                # reach the optimizer loop that triggered emission.
                logger.exception(
                    "event sink %r failed to emit %s", sink, event.event_type.value
                )


__all__ = [
    "EventSink",
    "NullEventSink",
    "LoggingEventSink",
    "CompositeEventSink",
]
