# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused tests for adapters into the canonical evaluation result type."""

import pytest

from GBOpt._explicit_ownership_evaluation import CandidateEvaluation
from GBOpt.evaluation.adapters import (
    from_batch_dict,
    from_candidate_evaluation,
    from_scalar_tuple,
)
from GBOpt.evaluation.types import (
    EvaluationStatus,
    EvaluationTypeError,
    EvaluationValueError,
    FailureStage,
)


def _successful_candidate_evaluation(**overrides):
    arguments = {
        "candidate_id": "GA_1_g0_c0",
        "input_index": 0,
        "objective": 1.25,
        "structure_path": "/tmp/candidate.data",
        "mapping": object(),
        "manipulator": object(),
        "success": True,
        "failure_reason": None,
    }
    arguments.update(overrides)
    return CandidateEvaluation(**arguments)


def _failed_candidate_evaluation(**overrides):
    arguments = {
        "candidate_id": "GA_1_g0_c0",
        "input_index": 0,
        "objective": 1.0e30,
        "structure_path": None,
        "mapping": None,
        "manipulator": None,
        "success": False,
        "failure_reason": "failed",
    }
    arguments.update(overrides)
    return CandidateEvaluation(**arguments)


def test_from_candidate_evaluation_rejects_non_candidate_evaluation():
    with pytest.raises(EvaluationTypeError, match="record must be a CandidateEvaluation"):
        from_candidate_evaluation(object())


def test_from_candidate_evaluation_preserves_successful_fields():
    manipulator = object()
    record = _successful_candidate_evaluation(objective=2.5, manipulator=manipulator)

    result = from_candidate_evaluation(record)

    assert result.status is EvaluationStatus.SUCCESS
    assert result.candidate_id == record.candidate_id
    assert result.input_index == record.input_index
    assert result.energy == pytest.approx(2.5)
    assert result.selection_energy == pytest.approx(2.5)
    assert result.artifact is not None
    assert result.artifact.path == "/tmp/candidate.data"
    assert result.artifact.format == "lammps"
    assert result.manipulator is manipulator


def test_from_candidate_evaluation_classifies_ownership_failure_from_missing_mapping():
    record = _failed_candidate_evaluation(
        mapping=None,
        structure_path=None,
        failure_reason="explicit-ownership mutation did not propagate grain labels",
    )

    result = from_candidate_evaluation(record)

    assert result.status is EvaluationStatus.FAILED
    assert result.failure_stage is FailureStage.OWNERSHIP
    assert result.selection_energy == pytest.approx(1.0e30)
    assert result.artifact is None
    assert result.failure_message == record.failure_reason


def test_from_candidate_evaluation_defaults_non_ownership_failure_to_evaluator_stage():
    record = _failed_candidate_evaluation(
        mapping=object(),
        structure_path="/tmp/candidate.data",
        failure_reason="RuntimeError: calculator crashed",
    )

    result = from_candidate_evaluation(record)

    assert result.failure_stage is FailureStage.EVALUATOR
    assert result.artifact is not None
    assert result.artifact.path == "/tmp/candidate.data"


def test_from_candidate_evaluation_falls_back_to_generic_message_when_reason_is_empty():
    record = _failed_candidate_evaluation(failure_reason="failed")
    object.__setattr__(record, "failure_reason", None)

    result = from_candidate_evaluation(record)

    assert result.failure_message == "unknown evaluation failure"


@pytest.mark.parametrize(
    "result",
    [None, (1.0,), (1.0, "/tmp/candidate.data", "extra"), "not-a-tuple"],
)
def test_from_scalar_tuple_rejects_malformed_result(result):
    with pytest.raises(EvaluationTypeError, match="scalar evaluator result must be a 2-tuple"):
        from_scalar_tuple("GA_1_g0_c0", 0, result, penalty=1.0e30)


def test_from_scalar_tuple_rejects_nonfinite_penalty():
    with pytest.raises(EvaluationValueError, match="penalty must be a finite real scalar"):
        from_scalar_tuple("GA_1_g0_c0", 0, (1.0, "/tmp/candidate.data"), penalty=float("nan"))


def test_from_scalar_tuple_builds_successful_result():
    result = from_scalar_tuple("GA_1_g0_c0", 0, (1.5, "/tmp/candidate.data"), penalty=1.0e30)

    assert result.status is EvaluationStatus.SUCCESS
    assert result.energy == pytest.approx(1.5)
    assert result.artifact.path == "/tmp/candidate.data"
    assert result.artifact.format == "lammps"


@pytest.mark.parametrize(
    ("energy", "structure_path", "expected_stage"),
    [
        pytest.param(float("nan"), "/tmp/candidate.data", FailureStage.VALIDATION, id="nonfinite-energy"),
        pytest.param(None, "/tmp/candidate.data", FailureStage.VALIDATION, id="missing-energy"),
        pytest.param(1.5, None, FailureStage.ARTIFACT, id="missing-path"),
        pytest.param(1.5, "", FailureStage.ARTIFACT, id="empty-path"),
    ],
)
def test_from_scalar_tuple_treats_incomplete_result_as_failure_not_exception(
    energy, structure_path, expected_stage
):
    result = from_scalar_tuple(
        "GA_1_g0_c0", 0, (energy, structure_path), penalty=1.0e30
    )

    assert result.status is EvaluationStatus.FAILED
    assert result.selection_energy == pytest.approx(1.0e30)
    assert result.failure_stage is expected_stage


def test_from_batch_dict_rejects_non_dict_result():
    with pytest.raises(EvaluationTypeError, match="batch evaluator result must be a dictionary"):
        from_batch_dict("GA_1_g0_c0", 0, ["not", "a", "dict"], penalty=1.0e30)


def test_from_batch_dict_builds_successful_result():
    result = from_batch_dict(
        "GA_1_g0_c0",
        0,
        {"energy": 1.5, "final_dump": "/tmp/candidate.data"},
        penalty=1.0e30,
    )

    assert result.status is EvaluationStatus.SUCCESS
    assert result.energy == pytest.approx(1.5)
    assert result.artifact.path == "/tmp/candidate.data"


def test_from_batch_dict_treats_missing_keys_as_failure_not_exception():
    result = from_batch_dict("GA_1_g0_c0", 0, {}, penalty=1.0e30)

    assert result.status is EvaluationStatus.FAILED
    assert result.selection_energy == pytest.approx(1.0e30)
    assert result.failure_stage is FailureStage.VALIDATION


@pytest.mark.parametrize(
    ("entry_overrides", "match"),
    [
        pytest.param({"candidate_id": "GA_1_g0_c1"}, "candidate_id", id="candidate-id-mismatch"),
        pytest.param({"input_index": 1}, "input_index", id="input-index-mismatch"),
    ],
)
def test_from_batch_dict_rejects_self_described_identity_mismatch(entry_overrides, match):
    entry = {"energy": 1.5, "final_dump": "/tmp/candidate.data"}
    entry.update(entry_overrides)

    with pytest.raises(EvaluationValueError, match=match):
        from_batch_dict("GA_1_g0_c0", 0, entry, penalty=1.0e30)


def test_from_batch_dict_accepts_matching_self_described_identity():
    entry = {
        "energy": 1.5,
        "final_dump": "/tmp/candidate.data",
        "candidate_id": "GA_1_g0_c0",
        "input_index": 0,
    }

    result = from_batch_dict("GA_1_g0_c0", 0, entry, penalty=1.0e30)

    assert result.status is EvaluationStatus.SUCCESS


def test_from_batch_dict_rejects_nonintegral_self_described_input_index():
    entry = {
        "energy": 1.5,
        "final_dump": "/tmp/candidate.data",
        "input_index": "0",
    }

    with pytest.raises(EvaluationTypeError, match="input_index must be a non-Boolean integer"):
        from_batch_dict("GA_1_g0_c0", 0, entry, penalty=1.0e30)


def test_from_batch_dict_supports_custom_keys():
    result = from_batch_dict(
        "GA_1_g0_c0",
        0,
        {"gbe": 1.5, "dump_file_name": "/tmp/candidate.data"},
        penalty=1.0e30,
        energy_key="gbe",
        structure_path_key="dump_file_name",
    )

    assert result.status is EvaluationStatus.SUCCESS
    assert result.energy == pytest.approx(1.5)
