# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Adapt existing evaluator result shapes into ``EvaluationResult``.

This module consumes an already-produced ``CandidateEvaluation``, a legacy scalar
``(energy, structure_path)`` tuple, or a legacy batch result dictionary. It returns one
``EvaluationResult`` per candidate, without changing any evaluator callback's external
signature. Callback invocation, artifact reconstruction, and MC/GA loop control do not
belong here.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np

from GBOpt._explicit_ownership_evaluation import CandidateEvaluation
from GBOpt.evaluation.types import (
    EvaluationResult,
    EvaluationStatus,
    EvaluationTypeError,
    EvaluationValueError,
    FailureStage,
    StructureArtifact,
)

_DEFAULT_FORMAT = "lammps"


def _finite_energy(value: object) -> float | None:
    """Return a finite Python float, or ``None`` if ``value`` is not one.

    :param value: Candidate energy value to normalize.
    :return: Finite Python float, or ``None`` when ``value`` is missing, Boolean,
        non-real, or non-finite.
    """
    if value is None or isinstance(value, (bool, np.bool_)) or not isinstance(
        value, Real
    ):
        return None
    normalized = float(value)
    if not np.isfinite(normalized):
        return None
    return normalized


def _normalize_structure_path(value: object) -> str | None:
    """Return a non-empty structure path string, or ``None`` if ``value`` is not one.

    :param value: Candidate structure path value to normalize.
    :return: Non-empty Python string, or ``None`` when ``value`` is missing, not a
        string, or blank.
    """
    if isinstance(value, str) and value.strip():
        return value
    return None


def _check_identity(
    expected_candidate_id: str,
    expected_input_index: int,
    provided_candidate_id: object,
    provided_input_index: object,
) -> None:
    """Cross-check a self-describing result entry's identity against expectations.

    :param expected_candidate_id: Candidate identity the caller expects at this
        position.
    :param expected_input_index: Candidate position the caller expects this entry to
        fill.
    :param provided_candidate_id: Candidate identity embedded in the raw entry, if any.
    :param provided_input_index: Candidate position embedded in the raw entry, if any.
    :raises EvaluationValueError: If a provided identity conflicts with what was
        expected.
    """
    if provided_candidate_id is not None and provided_candidate_id != expected_candidate_id:
        raise EvaluationValueError(
            f"result candidate_id {provided_candidate_id!r} does not match expected "
            f"{expected_candidate_id!r}"
        )
    if provided_input_index is not None:
        if isinstance(provided_input_index, (bool, np.bool_)) or not isinstance(
            provided_input_index, Integral
        ):
            raise EvaluationTypeError("result input_index must be a non-Boolean integer")
        if int(provided_input_index) != expected_input_index:
            raise EvaluationValueError(
                f"result input_index {int(provided_input_index)!r} does not match "
                f"expected {expected_input_index!r}"
            )


def from_candidate_evaluation(record: CandidateEvaluation) -> EvaluationResult:
    """Adapt one ownership-aware evaluation into the canonical result type.

    The legacy ``failure_reason`` is a single opaque string that already collapses
    several distinct failure origins (evaluator callback, artifact reload, ownership
    reconstruction, objective validation). Only the ownership-construction failure is
    unambiguously identifiable from ``record`` alone (``mapping is None`` happens
    exactly when candidate/file mapping construction itself failed, before any
    evaluator callback runs); every other failure is attributed to ``EVALUATOR`` as a
    disclosed, best-effort default. Full-fidelity stage attribution requires the
    evaluator itself to classify failures at the point they occur.

    :param record: Explicit-ownership evaluation to adapt.
    :return: Equivalent canonical evaluation result.
    :raises EvaluationTypeError: If ``record`` is not a ``CandidateEvaluation``.
    :raises EvaluationValueError: If a successful ``record`` has no ``structure_path``
        (unreachable through ``CandidateEvaluation``'s own construction invariants).
    """
    if not isinstance(record, CandidateEvaluation):
        raise EvaluationTypeError("record must be a CandidateEvaluation")

    if record.success:
        if record.structure_path is None:
            raise EvaluationValueError(
                "successful CandidateEvaluation must include a structure_path"
            )
        return EvaluationResult(
            candidate_id=record.candidate_id,
            input_index=record.input_index,
            status=EvaluationStatus.SUCCESS,
            selection_energy=record.objective,
            energy=record.objective,
            artifact=StructureArtifact(path=record.structure_path, format=_DEFAULT_FORMAT),
            manipulator=record.manipulator,
        )

    artifact = (
        StructureArtifact(path=record.structure_path, format=_DEFAULT_FORMAT)
        if record.structure_path is not None
        else None
    )
    stage = FailureStage.OWNERSHIP if record.mapping is None else FailureStage.EVALUATOR
    return EvaluationResult(
        candidate_id=record.candidate_id,
        input_index=record.input_index,
        status=EvaluationStatus.FAILED,
        selection_energy=record.objective,
        artifact=artifact,
        failure_stage=stage,
        failure_message=record.failure_reason or "unknown evaluation failure",
    )


def from_scalar_tuple(
    candidate_id: str,
    input_index: int,
    result: object,
    *,
    penalty: float,
    format: str = _DEFAULT_FORMAT,
) -> EvaluationResult:
    """Adapt one legacy scalar ``(energy, structure_path)`` evaluator result.

    A missing or non-finite energy, or a missing structure path, is a normal evaluation
    failure (the calculation did not produce a usable result) and is represented as a
    ``FAILED`` result carrying ``penalty``, matching the existing evaluator's own
    missing-result handling -- it is not raised. A ``result`` that is not a 2-tuple at
    all is a malformed entry and is raised explicitly.

    :param candidate_id: Stable logical candidate identity.
    :param input_index: Candidate position in the submitted population.
    :param result: Raw ``(energy, structure_path)`` tuple from a scalar evaluator
        callback.
    :param penalty: Keyword argument, required. Optimizer-supplied objective value for
        a failed calculation.
    :param format: Keyword argument, optional, defaults to ``"lammps"``. Structure file
        format identifier for a successful result's artifact.
    :return: Canonical evaluation result.
    :raises EvaluationTypeError: If ``result`` is not a 2-tuple.
    :raises EvaluationValueError: If ``penalty`` is non-finite.
    """
    if not isinstance(result, tuple) or len(result) != 2:
        raise EvaluationTypeError(
            "scalar evaluator result must be a 2-tuple of (energy, structure_path)"
        )
    energy_value, raw_structure_path = result
    numeric_penalty = _finite_energy(penalty)
    if numeric_penalty is None:
        raise EvaluationValueError("penalty must be a finite real scalar")

    numeric_energy = _finite_energy(energy_value)
    structure_path = _normalize_structure_path(raw_structure_path)
    if numeric_energy is None or structure_path is None:
        return EvaluationResult(
            candidate_id=candidate_id,
            input_index=input_index,
            status=EvaluationStatus.FAILED,
            selection_energy=numeric_penalty,
            artifact=(
                StructureArtifact(path=structure_path, format=format)
                if structure_path is not None
                else None
            ),
            failure_stage=(
                FailureStage.VALIDATION if numeric_energy is None else FailureStage.ARTIFACT
            ),
            failure_message=(
                "scalar evaluator result did not include a finite energy"
                if numeric_energy is None
                else "scalar evaluator result did not include a structure path"
            ),
        )

    return EvaluationResult(
        candidate_id=candidate_id,
        input_index=input_index,
        status=EvaluationStatus.SUCCESS,
        selection_energy=numeric_energy,
        energy=numeric_energy,
        artifact=StructureArtifact(path=structure_path, format=format),
    )


def from_batch_dict(
    candidate_id: str,
    input_index: int,
    result: object,
    *,
    penalty: float,
    format: str = _DEFAULT_FORMAT,
    energy_key: str = "energy",
    structure_path_key: str = "final_dump",
) -> EvaluationResult:
    """Adapt one legacy batch evaluator result dictionary.

    A dictionary missing ``energy_key``/``structure_path_key``, or carrying a
    non-finite energy, is a normal evaluation failure and is represented as a
    ``FAILED`` result carrying ``penalty``, matching the existing batch evaluator's own
    missing-result handling -- it is not raised. A ``result`` that is not a dictionary
    at all is a malformed entry and is raised explicitly, as is a self-described
    ``candidate_id``/``input_index`` entry that conflicts with the position it was
    returned at.

    :param candidate_id: Stable logical candidate identity expected at this position.
    :param input_index: Candidate position expected in the submitted batch.
    :param result: Raw per-candidate dictionary from a batch evaluator callback.
    :param penalty: Keyword argument, required. Optimizer-supplied objective value for
        a failed calculation.
    :param format: Keyword argument, optional, defaults to ``"lammps"``. Structure file
        format identifier for a successful result's artifact.
    :param energy_key: Keyword argument, optional, defaults to ``"energy"``. Dictionary
        key carrying the candidate's energy.
    :param structure_path_key: Keyword argument, optional, defaults to
        ``"final_dump"``. Dictionary key carrying the candidate's structure path.
    :return: Canonical evaluation result.
    :raises EvaluationTypeError: If ``result`` is not a dictionary, or a self-described
        ``input_index`` is non-integral.
    :raises EvaluationValueError: If ``penalty`` is non-finite, or a self-described
        ``candidate_id``/``input_index`` conflicts with the expected position.
    """
    if not isinstance(result, dict):
        raise EvaluationTypeError("batch evaluator result must be a dictionary")
    _check_identity(
        candidate_id,
        input_index,
        result.get("candidate_id"),
        result.get("input_index"),
    )
    numeric_penalty = _finite_energy(penalty)
    if numeric_penalty is None:
        raise EvaluationValueError("penalty must be a finite real scalar")

    numeric_energy = _finite_energy(result.get(energy_key))
    structure_path = _normalize_structure_path(result.get(structure_path_key))
    if numeric_energy is None or structure_path is None:
        return EvaluationResult(
            candidate_id=candidate_id,
            input_index=input_index,
            status=EvaluationStatus.FAILED,
            selection_energy=numeric_penalty,
            artifact=(
                StructureArtifact(path=structure_path, format=format)
                if structure_path is not None
                else None
            ),
            failure_stage=(
                FailureStage.VALIDATION if numeric_energy is None else FailureStage.ARTIFACT
            ),
            failure_message=(
                f"batch evaluator result did not include a finite {energy_key!r}"
                if numeric_energy is None
                else f"batch evaluator result did not include a {structure_path_key!r}"
            ),
        )

    return EvaluationResult(
        candidate_id=candidate_id,
        input_index=input_index,
        status=EvaluationStatus.SUCCESS,
        selection_energy=numeric_energy,
        energy=numeric_energy,
        artifact=StructureArtifact(path=structure_path, format=format),
    )
