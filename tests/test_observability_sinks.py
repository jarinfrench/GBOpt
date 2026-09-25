# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused tests for the ``EventSink`` implementations."""

import logging

import pytest

from GBOpt.evaluation import FailureStage
from GBOpt.observability.sinks import (
    CompositeEventSink,
    EventSink,
    LoggingEventSink,
    NullEventSink,
)
from GBOpt.observability.types import (
    ObservabilityTypeError,
    ObservabilityValueError,
    OptimizationAlgorithm,
    OptimizationEvent,
    OptimizationEventType,
    RunContext,
    TerminationReason,
)


def _event(**overrides):
    run = RunContext(
        run_id="run-1", seed=1, algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM
    )
    arguments = {
        "event_type": OptimizationEventType.RUN_STARTED,
        "run": run,
        "iteration": 0,
    }
    arguments.update(overrides)
    return OptimizationEvent(**arguments)


class _RecordingSink:
    def __init__(self):
        self.received: list[OptimizationEvent] = []

    def emit(self, event: OptimizationEvent) -> None:
        self.received.append(event)


class _ExplodingSink:
    def emit(self, event: OptimizationEvent) -> None:
        raise RuntimeError("sink is broken")


def test_null_event_sink_discards_without_error():
    sink = NullEventSink()

    sink.emit(_event())  # must not raise


def test_null_event_sink_satisfies_event_sink_protocol():
    assert isinstance(NullEventSink(), EventSink)


def test_logging_event_sink_uses_default_logger_when_none_given():
    sink = LoggingEventSink()

    assert sink._logger.name == "GBOpt.observability.sinks"


def test_logging_event_sink_rejects_non_logger_target():
    with pytest.raises(ObservabilityTypeError):
        LoggingEventSink(target_logger="not-a-logger")


@pytest.mark.parametrize(
    ("event_type", "expected_level", "extra"),
    [
        pytest.param(
            OptimizationEventType.RUN_STARTED, logging.INFO, {}, id="run-started-info"
        ),
        pytest.param(
            OptimizationEventType.PROPOSAL_EVALUATED,
            logging.DEBUG,
            {},
            id="proposal-evaluated-debug",
        ),
        pytest.param(
            OptimizationEventType.POPULATION_RESEEDED,
            logging.WARNING,
            {},
            id="reseeded-warning",
        ),
        pytest.param(
            OptimizationEventType.RUN_FAILED,
            logging.ERROR,
            {
                "failure_stage": FailureStage.EVALUATOR,
                "failure_message": "boom",
            },
            id="run-failed-error",
        ),
        pytest.param(
            OptimizationEventType.RUN_TERMINATED,
            logging.INFO,
            {"termination_reason": TerminationReason.MAX_STEPS},
            id="run-terminated-info",
        ),
    ],
)
def test_logging_event_sink_maps_event_type_to_level(
    caplog, event_type, expected_level, extra
):
    sink = LoggingEventSink()
    event = _event(event_type=event_type, **extra)

    with caplog.at_level(logging.DEBUG, logger="GBOpt.observability.sinks"):
        sink.emit(event)

    (record,) = caplog.records
    assert record.levelno == expected_level
    assert event_type.value in record.message


def test_logging_event_sink_never_includes_operation_parameters_verbatim(caplog):
    sink = LoggingEventSink()
    event = _event(
        event_type=OptimizationEventType.PROPOSAL_EVALUATED,
        operation_name="insert_atoms",
        operation_parameters={"marker_field_xyz": "should-not-appear"},
    )

    with caplog.at_level(logging.DEBUG, logger="GBOpt.observability.sinks"):
        sink.emit(event)

    (record,) = caplog.records
    assert "marker_field_xyz" not in record.message
    assert "operation=insert_atoms" in record.message


def test_composite_event_sink_requires_at_least_one_sink():
    with pytest.raises(ObservabilityValueError):
        CompositeEventSink([])


def test_composite_event_sink_rejects_non_sink_members():
    with pytest.raises(ObservabilityTypeError):
        CompositeEventSink([object()])


def test_composite_event_sink_emits_in_configured_order():
    order: list[str] = []

    class _NamedSink:
        def __init__(self, name):
            self.name = name

        def emit(self, event):
            order.append(self.name)

    composite = CompositeEventSink(
        [_NamedSink("first"), _NamedSink("second"), _NamedSink("third")]
    )
    composite.emit(_event())

    assert order == ["first", "second", "third"]


def test_composite_event_sink_continues_past_a_failing_sink(caplog):
    recording = _RecordingSink()
    composite = CompositeEventSink([_ExplodingSink(), recording])
    event = _event()

    with caplog.at_level(logging.ERROR, logger="GBOpt.observability.sinks"):
        composite.emit(event)

    assert recording.received == [event]
    assert any("sink" in record.message.lower() for record in caplog.records)


def test_composite_event_sink_exposes_configured_sinks_in_order():
    first = _RecordingSink()
    second = _RecordingSink()

    composite = CompositeEventSink([first, second])

    assert composite.sinks == (first, second)
