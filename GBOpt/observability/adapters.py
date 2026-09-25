# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Adapt an authoritative ``EvaluationResult`` into ``OptimizationEvent`` fields.

This module consumes one already-classified ``EvaluationResult`` (from
``GBOpt.evaluation``) and returns the subset of ``OptimizationEvent`` keyword arguments
it authoritatively determines -- candidate identity, position, status, energies, and
failure context. It does not decide which lifecycle occurrence is being reported (that
is the event's own ``event_type``, chosen by the caller), run any evaluation, or emit
anything; those belong to ``MonteCarloMinimizer``/``GeneticAlgorithmMinimizer`` and to
``GBOpt.observability.sinks``, respectively. Sharing this one function is what lets both
minimizers report a candidate's outcome from the same authoritative source rather than
each re-deriving event fields from the result's already-adapted, lossier scalar/dict
outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from GBOpt.evaluation import EvaluationResult


def evaluation_event_fields(result: EvaluationResult) -> dict[str, object]:
    """Return ``OptimizationEvent`` keyword arguments sourced from one evaluation.

    :param result: Authoritative, already-classified candidate evaluation outcome.
    :return: Keyword arguments for ``OptimizationEvent``'s ``candidate_id``/
        ``input_index``/``status``/``selection_energy``/``energy``/``failure_stage``/
        ``failure_code``/``failure_message`` fields.
    """
    return {
        "candidate_id": result.candidate_id,
        "input_index": result.input_index,
        "status": result.status,
        "selection_energy": result.selection_energy,
        "energy": result.energy,
        "failure_stage": result.failure_stage,
        "failure_code": result.failure_code,
        "failure_message": result.failure_message,
    }


__all__ = ["evaluation_event_fields"]
