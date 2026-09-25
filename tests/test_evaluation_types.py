# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused tests for the canonical evaluation result value types."""

import numpy as np
import pytest

from GBOpt.evaluation.types import (
    EvaluationResult,
    EvaluationStatus,
    EvaluationTypeError,
    EvaluationValueError,
    FailureStage,
    StructureArtifact,
)


def _success_kwargs(**overrides):
    arguments = {
        "candidate_id": "GA_1_g0_c0",
        "input_index": 0,
        "status": EvaluationStatus.SUCCESS,
        "selection_energy": 1.25,
        "energy": 1.25,
        "artifact": StructureArtifact(path="/tmp/candidate.data", format="lammps"),
        "manipulator": object(),
    }
    arguments.update(overrides)
    return arguments


def _failed_kwargs(**overrides):
    arguments = {
        "candidate_id": "GA_1_g0_c0",
        "input_index": 0,
        "status": EvaluationStatus.FAILED,
        "selection_energy": 1.0e30,
        "failure_stage": FailureStage.EVALUATOR,
        "failure_message": "evaluator callback raised",
    }
    arguments.update(overrides)
    return arguments


def test_structure_artifact_normalizes_path_type():
    from pathlib import Path

    artifact = StructureArtifact(path=Path("/tmp/candidate.data"), format="lammps")

    assert artifact.path == "/tmp/candidate.data"
    assert type(artifact.path) is str


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param({"path": ""}, EvaluationValueError, "path must not be empty", id="empty-path"),
        pytest.param({"path": None}, EvaluationTypeError, "path must be a path-like string", id="none-path"),
        pytest.param({"format": ""}, EvaluationValueError, "format must not be empty", id="empty-format"),
        pytest.param({"format": None}, EvaluationTypeError, "format must be a string", id="none-format"),
        pytest.param(
            {"digest": ""},
            EvaluationTypeError,
            "digest must be a non-empty string or None",
            id="empty-digest",
        ),
    ],
)
def test_structure_artifact_rejects_invalid_fields(kwargs, error, match):
    arguments = {"path": "/tmp/candidate.data", "format": "lammps"}
    arguments.update(kwargs)

    with pytest.raises(error, match=match):
        StructureArtifact(**arguments)


def test_structure_artifact_from_path_computes_digest(tmp_path):
    structure_file = tmp_path / "candidate.data"
    structure_file.write_bytes(b"atoms")

    artifact = StructureArtifact.from_path(structure_file, "lammps", compute_digest=True)

    assert artifact.digest is not None
    assert len(artifact.digest) == 64


def test_structure_artifact_from_path_defaults_to_no_digest(tmp_path):
    structure_file = tmp_path / "candidate.data"
    structure_file.write_bytes(b"atoms")

    artifact = StructureArtifact.from_path(structure_file, "lammps")

    assert artifact.digest is None


def test_evaluation_result_normalizes_python_scalar_fields():
    result = EvaluationResult(
        **_success_kwargs(input_index=np.int64(2), selection_energy=np.float64(2.5), energy=np.float64(2.5))
    )

    assert result.input_index == 2
    assert type(result.input_index) is int
    assert result.selection_energy == pytest.approx(2.5)
    assert type(result.selection_energy) is float
    assert result.energy == pytest.approx(2.5)
    assert type(result.energy) is float


def test_successful_evaluation_result_keeps_energy_and_selection_energy_distinct():
    result = EvaluationResult(**_success_kwargs(selection_energy=1.25, energy=1.25))

    assert result.status is EvaluationStatus.SUCCESS
    assert result.energy == pytest.approx(1.25)
    assert result.selection_energy == pytest.approx(1.25)
    assert result.failure_stage is None
    assert result.failure_code is None
    assert result.failure_message is None


def test_failed_evaluation_result_carries_penalty_as_selection_energy_only():
    result = EvaluationResult(**_failed_kwargs(selection_energy=1.0e30))

    assert result.status is EvaluationStatus.FAILED
    assert result.energy is None
    assert result.selection_energy == pytest.approx(1.0e30)
    assert result.manipulator is None


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param({"candidate_id": ""}, EvaluationTypeError, "candidate_id must be a non-empty string", id="empty-id"),
        pytest.param({"input_index": True}, EvaluationTypeError, "input_index must be a non-Boolean integer", id="bool-index"),
        pytest.param({"status": "success"}, EvaluationTypeError, "status must be an EvaluationStatus", id="raw-status"),
        pytest.param({"selection_energy": np.nan}, EvaluationValueError, "selection_energy must be finite", id="nonfinite-selection"),
        pytest.param({"artifact": object()}, EvaluationTypeError, "artifact must be a StructureArtifact or None", id="bad-artifact"),
        pytest.param({"energy": None}, EvaluationValueError, "successful evaluation requires a physical energy", id="missing-energy"),
        pytest.param({"energy": np.nan}, EvaluationValueError, "energy must be finite", id="nonfinite-energy"),
        pytest.param({"artifact": None}, EvaluationValueError, "successful evaluation requires a structure artifact", id="missing-artifact"),
        pytest.param(
            {"failure_message": "unexpected"},
            EvaluationValueError,
            "successful evaluation must not include failure context",
            id="unexpected-failure-message",
        ),
    ],
)
def test_successful_evaluation_result_rejects_incoherent_state(kwargs, error, match):
    arguments = _success_kwargs()
    arguments.update(kwargs)

    with pytest.raises(error, match=match):
        EvaluationResult(**arguments)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        pytest.param({"energy": 1.0}, EvaluationValueError, "failed evaluation must not include an energy", id="unexpected-energy"),
        pytest.param({"manipulator": object()}, EvaluationValueError, "failed evaluation must not include a manipulator", id="unexpected-manipulator"),
        pytest.param({"failure_stage": None}, EvaluationValueError, "failed evaluation requires a failure_stage", id="missing-stage"),
        pytest.param({"failure_stage": "evaluator"}, EvaluationValueError, "failed evaluation requires a failure_stage", id="raw-stage"),
        pytest.param({"failure_message": None}, EvaluationValueError, "failed evaluation requires a failure_message", id="missing-message"),
        pytest.param({"failure_message": ""}, EvaluationValueError, "failed evaluation requires a failure_message", id="empty-message"),
        pytest.param({"failure_code": ""}, EvaluationTypeError, "failure_code must be a non-empty string or None", id="empty-code"),
    ],
)
def test_failed_evaluation_result_rejects_incoherent_state(kwargs, error, match):
    arguments = _failed_kwargs()
    arguments.update(kwargs)

    with pytest.raises(error, match=match):
        EvaluationResult(**arguments)


def test_failed_evaluation_result_accepts_optional_artifact_and_code():
    result = EvaluationResult(
        **_failed_kwargs(
            artifact=StructureArtifact(path="/tmp/candidate.data", format="lammps"),
            failure_code="GrainOwnershipError",
        )
    )

    assert result.artifact is not None
    assert result.failure_code == "GrainOwnershipError"
