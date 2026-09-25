# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused tests for the ``EvaluationResult`` -> ``OptimizationEvent`` field adapter."""

from GBOpt.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    FailureStage,
    StructureArtifact,
)
from GBOpt.observability.adapters import evaluation_event_fields


def test_evaluation_event_fields_from_success():
    result = EvaluationResult(
        candidate_id="GA_1_g0_c0",
        input_index=0,
        status=EvaluationStatus.SUCCESS,
        selection_energy=1.25,
        energy=1.25,
        artifact=StructureArtifact(path="/tmp/candidate.data", format="lammps"),
    )

    fields = evaluation_event_fields(result)

    assert fields == {
        "candidate_id": "GA_1_g0_c0",
        "input_index": 0,
        "status": EvaluationStatus.SUCCESS,
        "selection_energy": 1.25,
        "energy": 1.25,
        "failure_stage": None,
        "failure_code": None,
        "failure_message": None,
    }


def test_evaluation_event_fields_from_failure():
    result = EvaluationResult(
        candidate_id="GA_1_g0_c1",
        input_index=1,
        status=EvaluationStatus.FAILED,
        selection_energy=1.0e30,
        failure_stage=FailureStage.EVALUATOR,
        failure_code="callback_exception",
        failure_message="RuntimeError: boom",
    )

    fields = evaluation_event_fields(result)

    assert fields == {
        "candidate_id": "GA_1_g0_c1",
        "input_index": 1,
        "status": EvaluationStatus.FAILED,
        "selection_energy": 1.0e30,
        "energy": None,
        "failure_stage": FailureStage.EVALUATOR,
        "failure_code": "callback_exception",
        "failure_message": "RuntimeError: boom",
    }


def test_evaluation_event_fields_output_is_a_valid_optimization_event_payload():
    from GBOpt.observability.types import (
        OptimizationAlgorithm,
        OptimizationEvent,
        OptimizationEventType,
        RunContext,
    )

    result = EvaluationResult(
        candidate_id="MC_run_s3",
        input_index=3,
        status=EvaluationStatus.SUCCESS,
        selection_energy=0.5,
        energy=0.5,
        artifact=StructureArtifact(path="/tmp/candidate.data", format="lammps"),
    )
    run = RunContext(run_id="run-1", seed=1, algorithm=OptimizationAlgorithm.MONTE_CARLO)

    event = OptimizationEvent(
        event_type=OptimizationEventType.PROPOSAL_EVALUATED,
        run=run,
        iteration=3,
        **evaluation_event_fields(result),
    )

    assert event.candidate_id == "MC_run_s3"
    assert event.status is EvaluationStatus.SUCCESS
