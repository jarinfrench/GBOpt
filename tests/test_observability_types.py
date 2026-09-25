# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused tests for the versioned MC/GA lifecycle event value types."""

import numpy as np
import pytest

from GBOpt.evaluation import EvaluationStatus, FailureStage
from GBOpt.observability.types import (
    EVENT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    ObservabilityTypeError,
    ObservabilityValueError,
    OptimizationAlgorithm,
    OptimizationEvent,
    OptimizationEventType,
    RunContext,
    RunManifest,
    TerminationReason,
)


def _run(**overrides):
    arguments = {
        "run_id": "run-1",
        "seed": 42,
        "algorithm": OptimizationAlgorithm.MONTE_CARLO,
    }
    arguments.update(overrides)
    return RunContext(**arguments)


def test_run_context_normalizes_seed_and_stores_optional_identity():
    run = _run(seed=np.int64(7), case_id="case-a", campaign_id="campaign-b")

    assert run.seed == 7
    assert type(run.seed) is int
    assert run.case_id == "case-a"
    assert run.campaign_id == "campaign-b"


def test_run_context_defaults_case_and_campaign_to_none():
    run = _run()

    assert run.case_id is None
    assert run.campaign_id is None


@pytest.mark.parametrize(
    ("overrides", "error", "match"),
    [
        pytest.param({"run_id": ""}, ObservabilityTypeError, "run_id", id="empty-run-id"),
        pytest.param({"run_id": None}, ObservabilityTypeError, "run_id", id="none-run-id"),
        pytest.param({"seed": True}, ObservabilityTypeError, "seed", id="boolean-seed"),
        pytest.param({"seed": 1.5}, ObservabilityTypeError, "seed", id="non-integral-seed"),
        pytest.param(
            {"algorithm": "monte_carlo"},
            ObservabilityTypeError,
            "algorithm",
            id="raw-string-algorithm",
        ),
        pytest.param({"case_id": ""}, ObservabilityTypeError, "case_id", id="empty-case-id"),
        pytest.param(
            {"campaign_id": ""}, ObservabilityTypeError, "campaign_id", id="empty-campaign-id"
        ),
    ],
)
def test_run_context_rejects_invalid_fields(overrides, error, match):
    with pytest.raises(error, match=match):
        _run(**overrides)


def test_run_context_is_immutable():
    run = _run()

    with pytest.raises((AttributeError, TypeError)):
        run.seed = 99


def test_run_manifest_stamps_schema_version_and_shares_run_identity():
    run = _run(run_id="run-shared")

    manifest = RunManifest(run=run, created_at="2026-01-01T00:00:00+00:00")

    assert manifest.schema_version == MANIFEST_SCHEMA_VERSION
    assert manifest.run is run
    assert manifest.run.run_id == "run-shared"
    assert manifest.created_at == "2026-01-01T00:00:00+00:00"


def test_run_manifest_defaults_created_at_to_a_non_empty_timestamp():
    manifest = RunManifest(run=_run())

    assert isinstance(manifest.created_at, str)
    assert manifest.created_at.strip()


def test_run_manifest_rejects_non_run_context():
    with pytest.raises(ObservabilityTypeError, match="run"):
        RunManifest(run="not-a-run-context")


@pytest.mark.parametrize(
    "created_at",
    [
        pytest.param("", id="empty-string"),
        pytest.param(123, id="non-string"),
    ],
)
def test_run_manifest_rejects_invalid_created_at(created_at):
    with pytest.raises(ObservabilityTypeError, match="created_at"):
        RunManifest(run=_run(), created_at=created_at)


def test_run_manifest_is_immutable():
    manifest = RunManifest(run=_run())

    with pytest.raises((AttributeError, TypeError)):
        manifest.created_at = "later"


def _event(**overrides):
    arguments = {
        "event_type": OptimizationEventType.RUN_STARTED,
        "run": _run(),
        "iteration": 0,
    }
    arguments.update(overrides)
    return OptimizationEvent(**arguments)


def test_event_always_stamps_current_schema_version():
    event = _event()

    assert event.schema_version == EVENT_SCHEMA_VERSION


def test_event_normalizes_iteration_and_input_index():
    event = _event(iteration=np.int64(3), input_index=np.int64(1))

    assert event.iteration == 3
    assert type(event.iteration) is int
    assert event.input_index == 1
    assert type(event.input_index) is int


def test_event_input_index_accepts_negative_sentinel():
    # Some authoritative evaluation sources (e.g. an owned-mode initial candidate)
    # use input_index=-1 for "not a submitted population member" -- this field
    # must accept whatever EvaluationResult.input_index itself already accepts.
    event = _event(input_index=-1)

    assert event.input_index == -1


def test_event_normalizes_operation_parameters_to_json_safe_mapping():
    event = _event(
        event_type=OptimizationEventType.PROPOSAL_EVALUATED,
        operation_name="insert_atoms",
        operation_parameters={
            "num_to_insert": np.int64(2),
            "site_indices": (np.int64(1), np.int64(2)),
            "fraction": np.float64(0.5),
        },
    )

    assert event.operation_parameters == {
        "num_to_insert": 2,
        "site_indices": (1, 2),
        "fraction": 0.5,
    }
    assert type(event.operation_parameters["num_to_insert"]) is int
    assert type(event.operation_parameters["site_indices"][0]) is int


def test_event_operation_parameters_is_read_only():
    event = _event(operation_parameters={"a": 1})

    with pytest.raises(TypeError):
        event.operation_parameters["a"] = 2


@pytest.mark.parametrize(
    "bad_value",
    [
        pytest.param(np.array([1, 2, 3]), id="numpy-array"),
        pytest.param(object(), id="arbitrary-object"),
        pytest.param(lambda: None, id="callable"),
        pytest.param(b"bytes", id="bytes"),
    ],
)
def test_event_rejects_non_json_safe_operation_parameters(bad_value):
    with pytest.raises(ObservabilityTypeError):
        _event(operation_parameters={"bad": bad_value})


def test_event_rejects_non_string_operation_parameter_keys():
    with pytest.raises(ObservabilityValueError):
        _event(operation_parameters={1: "a"})


def test_run_terminated_requires_termination_reason():
    with pytest.raises(ObservabilityValueError, match="termination_reason"):
        _event(event_type=OptimizationEventType.RUN_TERMINATED)


def test_termination_reason_rejected_outside_run_terminated():
    with pytest.raises(ObservabilityValueError, match="termination_reason"):
        _event(
            event_type=OptimizationEventType.RUN_STARTED,
            termination_reason=TerminationReason.MAX_STEPS,
        )


def test_run_terminated_accepts_termination_reason():
    event = _event(
        event_type=OptimizationEventType.RUN_TERMINATED,
        termination_reason=TerminationReason.ENERGY_TOLERANCE,
    )

    assert event.termination_reason is TerminationReason.ENERGY_TOLERANCE


def test_run_failed_requires_failure_stage_and_message():
    with pytest.raises(ObservabilityValueError, match="RUN_FAILED"):
        _event(event_type=OptimizationEventType.RUN_FAILED)


def test_run_failed_accepts_failure_context():
    event = _event(
        event_type=OptimizationEventType.RUN_FAILED,
        failure_stage=FailureStage.EVALUATOR,
        failure_message="boom",
    )

    assert event.failure_stage is FailureStage.EVALUATOR
    assert event.failure_message == "boom"


def test_success_status_rejects_failure_context():
    with pytest.raises(ObservabilityValueError, match="SUCCESS"):
        _event(
            event_type=OptimizationEventType.PROPOSAL_EVALUATED,
            status=EvaluationStatus.SUCCESS,
            failure_message="should not be here",
        )


def test_failed_status_requires_failure_stage_and_message():
    with pytest.raises(ObservabilityValueError, match="FAILED"):
        _event(
            event_type=OptimizationEventType.PROPOSAL_EVALUATED,
            status=EvaluationStatus.FAILED,
        )


def test_failed_status_accepts_failure_context():
    event = _event(
        event_type=OptimizationEventType.PROPOSAL_EVALUATED,
        status=EvaluationStatus.FAILED,
        failure_stage=FailureStage.ARTIFACT,
        failure_message="missing structure path",
    )

    assert event.status is EvaluationStatus.FAILED
    assert event.failure_stage is FailureStage.ARTIFACT


def test_event_is_immutable():
    event = _event()

    with pytest.raises((AttributeError, TypeError)):
        event.iteration = 5


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        pytest.param({"event_type": "run_started"}, ObservabilityTypeError, id="raw-string-type"),
        pytest.param({"run": object()}, ObservabilityTypeError, id="non-run-context"),
        pytest.param({"iteration": -1}, ObservabilityValueError, id="negative-iteration"),
        pytest.param({"iteration": True}, ObservabilityTypeError, id="boolean-iteration"),
        pytest.param({"candidate_id": ""}, ObservabilityTypeError, id="empty-candidate-id"),
        pytest.param({"selection_energy": float("nan")}, ObservabilityValueError, id="nan-selection-energy"),
        pytest.param({"status": "success"}, ObservabilityTypeError, id="raw-string-status"),
        pytest.param({"failure_stage": "evaluator"}, ObservabilityTypeError, id="raw-string-failure-stage"),
    ],
)
def test_event_rejects_invalid_fields(overrides, error):
    with pytest.raises(error):
        _event(**overrides)
