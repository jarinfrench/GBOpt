# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define the versioned MC/GA lifecycle event schema and its value types.

This module consumes an optimizer run's identity and already-classified per-candidate
outcome (``EvaluationStatus``/``FailureStage``, from ``GBOpt.evaluation``) or operation
provenance (``ManipulationResult.parameters``/``.lineage``, from ``GBOpt.manipulation``)
and returns one immutable, JSON-safe ``OptimizationEvent`` per lifecycle occurrence. It
does not decide when a lifecycle occurrence happens, run any manipulation or evaluation,
or deliver an event anywhere -- those belong to ``MonteCarloMinimizer``/
``GeneticAlgorithmMinimizer`` and to ``GBOpt.observability.sinks``, respectively.

``OptimizationEvent`` is a stable, versioned scientific record, not a substitute for a
checkpoint: it carries no full atom array, callback, logger, or other live object, and a
consumer must never treat a stream of these events as restart state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from GBOpt.evaluation import EvaluationStatus, FailureStage


class ObservabilityError(Exception):
    """Base class for observability-subsystem errors."""


class ObservabilityTypeError(ObservabilityError, TypeError):
    """Raised when observability-domain state has an invalid type."""


class ObservabilityValueError(ObservabilityError, ValueError):
    """Raised when observability-domain state has an invalid value."""


EVENT_SCHEMA_VERSION: int = 1
"""Version of the :class:`OptimizationEvent` field contract.

Every :class:`OptimizationEvent` stamps this value itself, so a consumer holding a
serialized event always knows which shape produced it, independent of which GBOpt
version is running. Bump this constant, and document the change, whenever a field is
added, removed, or given new meaning.
"""


class OptimizationAlgorithm(str, Enum):
    """Optimizer family that produced one run's events."""

    MONTE_CARLO = "monte_carlo"
    GENETIC_ALGORITHM = "genetic_algorithm"


class OptimizationEventType(str, Enum):
    """Stable lifecycle occurrence an :class:`OptimizationEvent` reports.

    ``CANDIDATE_ACCEPTED``/``CANDIDATE_REJECTED`` mean "survives into the optimizer's
    next state" for both algorithms, but the two algorithms reach that decision
    differently: Monte Carlo emits one accept/reject decision per proposed step
    (its Metropolis criterion), while the genetic algorithm emits one decision per
    evaluated population member, for whether that candidate's structure carries over
    or breeds into the next generation (its keep/intermediate selection), not a
    single-current-state choice.
    """

    RUN_STARTED = "run_started"
    INITIAL_EVALUATION = "initial_evaluation"
    PROPOSAL_EVALUATED = "proposal_evaluated"
    CANDIDATE_ACCEPTED = "candidate_accepted"
    CANDIDATE_REJECTED = "candidate_rejected"
    BEST_UPDATED = "best_updated"
    GENERATION_BOUNDARY = "generation_boundary"
    POPULATION_RESEEDED = "population_reseeded"
    RUN_TERMINATED = "run_terminated"
    RUN_FAILED = "run_failed"


class TerminationReason(str, Enum):
    """Stable, versioned reason a run's main loop stopped."""

    ENERGY_TOLERANCE = "energy_tolerance"
    MAX_REJECTIONS = "max_rejections"
    MAX_STEPS = "max_steps"
    MAX_GENERATIONS = "max_generations"


def _normalize_run_id(run_id: object) -> str:
    """Validate one stable run identity.

    :param run_id: Run identity to validate.
    :return: Validated non-empty identity.
    :raises ObservabilityTypeError: If ``run_id`` is not a non-empty string.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ObservabilityTypeError("run_id must be a non-empty string")
    return run_id


def _normalize_seed(seed: object) -> int:
    """Validate one resolved RNG seed.

    :param seed: Seed value to validate.
    :return: Python integer seed.
    :raises ObservabilityTypeError: If ``seed`` is Boolean or non-integral.
    """
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ObservabilityTypeError("seed must be a non-Boolean integer")
    return int(seed)


def _normalize_optional_identity(value: object, *, name: str) -> str | None:
    """Validate one optional non-empty identity string.

    :param value: Identity value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Validated identity, or ``None``.
    :raises ObservabilityTypeError: If ``value`` is neither ``None`` nor a non-empty
        string.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ObservabilityTypeError(f"{name} must be a non-empty string or None")
    return value


@dataclass(frozen=True, slots=True, init=False)
class RunContext:
    """Immutable identity shared by every event one optimizer run emits.

    :param run_id: Stable run identity (the same value MC/GA already resolve as their
        ``unique_id``).
    :param seed: Resolved RNG seed actually passed to ``numpy.random.default_rng`` --
        never the un-resolved ``None`` a caller may have supplied.
    :param algorithm: Optimizer family that produced this run.
    :param case_id: Keyword argument, optional, defaults to ``None``. Caller-supplied
        scientific case identity.
    :param campaign_id: Keyword argument, optional, defaults to ``None``. Caller-supplied
        campaign identity grouping several runs.
    :raises ObservabilityTypeError: If any field has an invalid type.
    :raises ObservabilityValueError: If ``run_id``, ``case_id``, or ``campaign_id`` is
        empty.
    """

    run_id: str
    seed: int
    algorithm: OptimizationAlgorithm
    case_id: str | None
    campaign_id: str | None

    def __init__(
        self,
        *,
        run_id: str,
        seed: int,
        algorithm: OptimizationAlgorithm,
        case_id: str | None = None,
        campaign_id: str | None = None,
    ) -> None:
        """Construct a validated, immutable run identity.

        :param run_id: Keyword argument, required. Stable run identity.
        :param seed: Keyword argument, required. Resolved RNG seed.
        :param algorithm: Keyword argument, required. Optimizer family.
        :param case_id: Keyword argument, optional, defaults to ``None``. Scientific
            case identity.
        :param campaign_id: Keyword argument, optional, defaults to ``None``. Campaign
            identity.
        :raises ObservabilityTypeError: If any field has an invalid type.
        :raises ObservabilityValueError: If ``run_id``, ``case_id``, or ``campaign_id``
            is empty.
        """
        run_id = _normalize_run_id(run_id)
        seed = _normalize_seed(seed)
        if not isinstance(algorithm, OptimizationAlgorithm):
            raise ObservabilityTypeError("algorithm must be an OptimizationAlgorithm")
        case_id = _normalize_optional_identity(case_id, name="case_id")
        campaign_id = _normalize_optional_identity(campaign_id, name="campaign_id")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "campaign_id", campaign_id)


_JSON_SAFE_SCALAR_TYPES = (str, bool, type(None))


def _normalize_json_safe(value: object, *, path: str) -> object:
    """Validate and normalize one JSON-safe scalar, mapping, or sequence.

    :param value: Value to validate.
    :param path: Keyword argument, required. Diagnostic path to ``value``.
    :return: Plain-Python equivalent (``dict``/``tuple``/``str``/``int``/``float``/
        ``bool``/``None``) with numeric types normalized.
    :raises ObservabilityTypeError: If ``value`` (or a nested value) is not a JSON-safe
        scalar, mapping, or sequence -- in particular, never a NumPy array, callable,
        logger, or other live object.
    :raises ObservabilityValueError: If a nested mapping has a non-string key.
    """
    if isinstance(value, _JSON_SAFE_SCALAR_TYPES):
        return value
    if isinstance(value, Mapping):
        return _normalize_json_safe_mapping(value, path=path)
    if isinstance(value, (Sequence, tuple)) and not isinstance(value, (str, bytes)):
        return tuple(
            _normalize_json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized_real = float(value)
        if not np.isfinite(normalized_real):
            raise ObservabilityValueError(f"{path} must be finite")
        return normalized_real
    raise ObservabilityTypeError(
        f"{path} must be a JSON-safe scalar, mapping, or sequence; got "
        f"{type(value).__name__}"
    )


def _normalize_json_safe_mapping(
    value: Mapping[Any, Any], *, path: str
) -> dict[str, object]:
    """Validate and normalize one JSON-safe mapping's keys and values.

    :param value: Mapping to validate.
    :param path: Keyword argument, required. Diagnostic path to ``value``.
    :return: Plain ``dict[str, object]`` with every value JSON-safe-normalized.
    :raises ObservabilityTypeError: If any value is not JSON-safe.
    :raises ObservabilityValueError: If any key is not a string.
    """
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ObservabilityValueError(f"{path} keys must be strings")
        normalized[key] = _normalize_json_safe(item, path=f"{path}.{key}")
    return normalized


def _normalize_operation_parameters(
    value: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    """Validate and freeze one optional operation-parameters mapping.

    :param value: Mapping to validate, typically ``ManipulationResult.parameters`` or
        ``.lineage``.
    :return: Read-only, JSON-safe equivalent, or ``None``.
    :raises ObservabilityTypeError: If ``value`` is neither ``None`` nor a mapping, or
        contains a non-JSON-safe value.
    :raises ObservabilityValueError: If a nested mapping has a non-string key.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ObservabilityTypeError("operation_parameters must be a mapping or None")
    normalized = _normalize_json_safe_mapping(value, path="operation_parameters")
    return MappingProxyType(normalized)


def _normalize_iteration(value: object, *, name: str) -> int:
    """Validate one non-negative iteration-like index.

    :param value: Index value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Python non-negative integer.
    :raises ObservabilityTypeError: If ``value`` is Boolean or non-integral.
    :raises ObservabilityValueError: If ``value`` is negative.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ObservabilityTypeError(f"{name} must be a non-Boolean integer")
    normalized = int(value)
    if normalized < 0:
        raise ObservabilityValueError(f"{name} must be non-negative")
    return normalized


def _validate_termination_reason_coherence(
    event_type: OptimizationEventType, termination_reason: TerminationReason | None
) -> None:
    """Enforce that ``termination_reason`` is present for, and only for, termination.

    :param event_type: Lifecycle occurrence being validated.
    :param termination_reason: Reason value to check against ``event_type``.
    :raises ObservabilityValueError: If ``termination_reason`` is missing on a
        ``RUN_TERMINATED`` event, or present on any other event type.
    """
    if event_type is OptimizationEventType.RUN_TERMINATED:
        if termination_reason is None:
            raise ObservabilityValueError(
                "RUN_TERMINATED requires a termination_reason"
            )
    elif termination_reason is not None:
        raise ObservabilityValueError(
            "termination_reason is only valid for RUN_TERMINATED"
        )


def _validate_failure_context_coherence(
    event_type: OptimizationEventType,
    status: EvaluationStatus | None,
    failure_stage: FailureStage | None,
    failure_code: str | None,
    failure_message: str | None,
) -> None:
    """Enforce that failure context matches ``event_type`` and ``status``.

    :param event_type: Lifecycle occurrence being validated.
    :param status: Success/failure outcome being validated, when applicable.
    :param failure_stage: Failure-stage value to check.
    :param failure_code: Failure-code value to check.
    :param failure_message: Failure-message value to check.
    :raises ObservabilityValueError: If ``RUN_FAILED`` lacks failure context, a
        ``SUCCESS`` status carries failure context, or a ``FAILED`` status (outside
        ``RUN_FAILED``) lacks failure context.
    """
    from GBOpt.evaluation import EvaluationStatus

    if event_type is OptimizationEventType.RUN_FAILED and (
        failure_stage is None or failure_message is None
    ):
        raise ObservabilityValueError(
            "RUN_FAILED requires failure_stage and failure_message"
        )
    if status is EvaluationStatus.SUCCESS and (
        failure_stage is not None
        or failure_code is not None
        or failure_message is not None
    ):
        raise ObservabilityValueError(
            "a SUCCESS status must not include failure context"
        )
    if (
        status is EvaluationStatus.FAILED
        and event_type is not OptimizationEventType.RUN_FAILED
        and (failure_stage is None or failure_message is None)
    ):
        raise ObservabilityValueError(
            "a FAILED status requires failure_stage and failure_message"
        )


def _normalize_optional_energy(value: object, *, name: str) -> float | None:
    """Validate one optional finite energy-like scalar.

    :param value: Energy-like value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Finite Python float, or ``None``.
    :raises ObservabilityTypeError: If ``value`` is neither ``None`` nor a non-Boolean
        real scalar.
    :raises ObservabilityValueError: If ``value`` is non-finite.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ObservabilityTypeError(f"{name} must be a non-Boolean real scalar or None")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ObservabilityValueError(f"{name} must be finite")
    return normalized


@dataclass(frozen=True, slots=True, init=False)
class OptimizationEvent:
    """One immutable, versioned MC/GA lifecycle occurrence.

    Every field is either a small scalar, a stable enum member, or a shallow, JSON-safe
    mapping -- never a full atom array, callback, logger, or other live object.
    ``status``/``selection_energy``/``energy``/``failure_stage``/``failure_code``/
    ``failure_message`` mirror the same-named fields of the authoritative
    ``EvaluationResult`` this event reports on, when applicable; ``operation_name``/
    ``operation_parameters`` mirror ``ManipulationResult.lineage``'s ``"operation"``
    entry and ``.parameters``, when an operation produced the reported candidate.

    :param event_type: Lifecycle occurrence this event reports.
    :param run: Identity of the run this event belongs to.
    :param iteration: Non-negative MC step or GA generation index.
    :param candidate_id: Keyword argument, optional, defaults to ``None``. Stable
        candidate identity, when this event reports on one candidate.
    :param input_index: Keyword argument, optional, defaults to ``None``. Candidate
        position within the submitted generation, when applicable.
    :param status: Keyword argument, optional, defaults to ``None``. Success/failure
        outcome, when this event reports on one evaluation.
    :param selection_energy: Keyword argument, optional, defaults to ``None``.
        Optimizer-facing selection energy, when applicable.
    :param energy: Keyword argument, optional, defaults to ``None``. Physical energy,
        populated only alongside a ``SUCCESS`` status.
    :param failure_stage: Keyword argument, optional, defaults to ``None``. Pipeline
        stage a failure originated from.
    :param failure_code: Keyword argument, optional, defaults to ``None``. Stable
        machine-readable failure identifier.
    :param failure_message: Keyword argument, optional, defaults to ``None``.
        Human-readable failure context.
    :param operation_name: Keyword argument, optional, defaults to ``None``. Name of the
        manipulation operation that produced the reported candidate.
    :param operation_parameters: Keyword argument, optional, defaults to ``None``.
        Read-only, JSON-safe concrete parameter values the operation used.
    :param termination_reason: Keyword argument, optional, defaults to ``None``. Reason
        the run's main loop stopped; required for, and only for,
        ``RUN_TERMINATED``.
    :raises ObservabilityTypeError: If any field has an invalid type.
    :raises ObservabilityValueError: If a numeric field is non-finite/negative, or
        ``status``/``termination_reason`` presence is inconsistent with ``event_type``
        or the failure fields.
    """

    schema_version: int
    event_type: OptimizationEventType
    run: RunContext
    iteration: int
    candidate_id: str | None
    input_index: int | None
    status: EvaluationStatus | None
    selection_energy: float | None
    energy: float | None
    failure_stage: FailureStage | None
    failure_code: str | None
    failure_message: str | None
    operation_name: str | None
    operation_parameters: Mapping[str, object] | None
    termination_reason: TerminationReason | None

    def __init__(
        self,
        *,
        event_type: OptimizationEventType,
        run: RunContext,
        iteration: int,
        candidate_id: str | None = None,
        input_index: int | None = None,
        status: EvaluationStatus | None = None,
        selection_energy: float | None = None,
        energy: float | None = None,
        failure_stage: FailureStage | None = None,
        failure_code: str | None = None,
        failure_message: str | None = None,
        operation_name: str | None = None,
        operation_parameters: Mapping[str, object] | None = None,
        termination_reason: TerminationReason | None = None,
    ) -> None:
        """Construct a validated, immutable lifecycle event.

        See the class docstring for parameter semantics and raised exceptions.
        """
        # Imported here, not at module scope, so this leaf-adjacent module never
        # requires GBOpt.evaluation to be importable merely to define the schema --
        # only to validate against its enums once an instance is actually built.
        from GBOpt.evaluation import EvaluationStatus, FailureStage

        if not isinstance(event_type, OptimizationEventType):
            raise ObservabilityTypeError("event_type must be an OptimizationEventType")
        if not isinstance(run, RunContext):
            raise ObservabilityTypeError("run must be a RunContext")
        iteration = _normalize_iteration(iteration, name="iteration")
        candidate_id = _normalize_optional_identity(candidate_id, name="candidate_id")
        input_index = (
            None
            if input_index is None
            else _normalize_iteration(input_index, name="input_index")
        )
        if status is not None and not isinstance(status, EvaluationStatus):
            raise ObservabilityTypeError("status must be an EvaluationStatus or None")
        selection_energy = _normalize_optional_energy(
            selection_energy, name="selection_energy"
        )
        energy = _normalize_optional_energy(energy, name="energy")
        if failure_stage is not None and not isinstance(failure_stage, FailureStage):
            raise ObservabilityTypeError("failure_stage must be a FailureStage or None")
        failure_code = _normalize_optional_identity(failure_code, name="failure_code")
        failure_message = _normalize_optional_identity(
            failure_message, name="failure_message"
        )
        operation_name = _normalize_optional_identity(
            operation_name, name="operation_name"
        )
        operation_parameters = _normalize_operation_parameters(operation_parameters)
        if termination_reason is not None and not isinstance(
            termination_reason, TerminationReason
        ):
            raise ObservabilityTypeError(
                "termination_reason must be a TerminationReason or None"
            )

        _validate_termination_reason_coherence(event_type, termination_reason)
        _validate_failure_context_coherence(
            event_type, status, failure_stage, failure_code, failure_message
        )

        object.__setattr__(self, "schema_version", EVENT_SCHEMA_VERSION)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "input_index", input_index)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "selection_energy", selection_energy)
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "failure_stage", failure_stage)
        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(self, "failure_message", failure_message)
        object.__setattr__(self, "operation_name", operation_name)
        object.__setattr__(self, "operation_parameters", operation_parameters)
        object.__setattr__(self, "termination_reason", termination_reason)


__all__ = [
    "ObservabilityError",
    "ObservabilityTypeError",
    "ObservabilityValueError",
    "EVENT_SCHEMA_VERSION",
    "OptimizationAlgorithm",
    "OptimizationEventType",
    "TerminationReason",
    "RunContext",
    "OptimizationEvent",
]
