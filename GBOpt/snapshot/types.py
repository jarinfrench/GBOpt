# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Define the immutable, versioned schema-v2 restart snapshot value types.

This module consumes already-classified run identity, RNG state, and per-candidate
evaluation/artifact data (``GBOpt.evaluation``'s ``StructureArtifact``/
``EvaluationStatus``/``FailureStage``, ``GBOpt.FileGrainOwnership``'s
``CandidateFileMapping``) and returns immutable, validated snapshot objects describing
all restart-critical Monte Carlo/genetic-algorithm state. It does not run an optimizer
loop, evaluate a candidate, read or write a checkpoint file, or migrate schema-v1
state -- those belong to ``MonteCarloMinimizer``/``GeneticAlgorithmMinimizer`` and to
``GBOpt.snapshot.migration``, respectively.

Every snapshot type is frozen and validates its own fields at construction time, the
same discipline ``GBOpt.evaluation.types``/``GBOpt.observability.types`` already
establish. No field accepts a callback, event sink, logger, open file, scheduler
client, or live manipulator/minimizer object; a value that is not one of this module's
own validated types, a small scalar, or a JSON-safe mapping/sequence is rejected at
construction, never silently accepted or coerced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np

from GBOpt.evaluation import EvaluationStatus, FailureStage, StructureArtifact
from GBOpt.FileGrainOwnership import CandidateFileMapping

if TYPE_CHECKING:
    from GBOpt.observability import OptimizationAlgorithm, RunContext


class SnapshotError(Exception):
    """Base class for restart-snapshot errors."""


class SnapshotTypeError(SnapshotError, TypeError):
    """Raised when restart-snapshot state has an invalid type."""


class SnapshotValueError(SnapshotError, ValueError):
    """Raised when restart-snapshot state has an invalid value."""


SNAPSHOT_SCHEMA_VERSION: int = 2
"""Version of the typed restart-snapshot field contract.

Independent of ``GBOpt.Checkpoint.CHECKPOINT_SCHEMA_VERSION`` (schema v1, the raw
dictionary envelope every minimizer still reads and writes). Every snapshot stamps
this value itself, so a consumer holding a serialized snapshot always knows which
shape produced it. Bump this constant, and document the change, whenever a field is
added, removed, or given new meaning.
"""


def _normalize_identity(value: object, *, name: str) -> str:
    """Validate one required non-empty identity string.

    :param value: Identity value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Validated non-empty identity.
    :raises SnapshotTypeError: If ``value`` is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise SnapshotTypeError(f"{name} must be a non-empty string")
    return value


def _normalize_optional_identity(value: object, *, name: str) -> str | None:
    """Validate one optional non-empty identity string.

    :param value: Identity value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Validated identity, or ``None``.
    :raises SnapshotTypeError: If ``value`` is neither ``None`` nor a non-empty string.
    """
    if value is None:
        return None
    return _normalize_identity(value, name=name)


def _normalize_seed(value: object) -> int:
    """Validate one resolved RNG seed.

    :param value: Seed value to validate.
    :return: Python integer seed.
    :raises SnapshotTypeError: If ``value`` is Boolean or non-integral.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise SnapshotTypeError("seed must be a non-Boolean integer")
    return int(value)


def _normalize_index(value: object, *, name: str) -> int:
    """Validate one non-negative integer index.

    :param value: Index value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Python non-negative integer.
    :raises SnapshotTypeError: If ``value`` is Boolean or non-integral.
    :raises SnapshotValueError: If ``value`` is negative.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise SnapshotTypeError(f"{name} must be a non-Boolean integer")
    normalized = int(value)
    if normalized < 0:
        raise SnapshotValueError(f"{name} must be non-negative")
    return normalized


def _normalize_signed_index(value: object, *, name: str) -> int:
    """Validate one integer index that may be a negative sentinel.

    Matches ``EvaluationResult.input_index``'s own contract: some authoritative
    candidates (e.g. an owned-mode initial candidate) use ``-1`` for "not a submitted
    population member," not a stricter non-negative range.

    :param value: Index value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Python integer.
    :raises SnapshotTypeError: If ``value`` is Boolean or non-integral.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise SnapshotTypeError(f"{name} must be a non-Boolean integer")
    return int(value)


def _normalize_energy(value: object, *, name: str) -> float:
    """Validate one finite energy-like scalar.

    :param value: Energy-like value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Finite Python float.
    :raises SnapshotTypeError: If ``value`` is Boolean or non-real.
    :raises SnapshotValueError: If ``value`` is non-finite.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise SnapshotTypeError(f"{name} must be a non-Boolean real scalar")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise SnapshotValueError(f"{name} must be finite")
    return normalized


def _normalize_optional_energy(value: object, *, name: str) -> float | None:
    """Validate one optional finite energy-like scalar.

    :param value: Energy-like value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Finite Python float, or ``None``.
    :raises SnapshotTypeError: If ``value`` is neither ``None`` nor a non-Boolean real
        scalar.
    :raises SnapshotValueError: If ``value`` is non-finite.
    """
    if value is None:
        return None
    return _normalize_energy(value, name=name)


def _reject_live_objects(value: object, *, path: str) -> object:
    """Recursively validate that *value* holds only JSON-safe scalars/mappings/sequences.

    This is the enforcement point for "arbitrary live Python objects are rejected from
    the typed snapshot codec": it is applied to every opaque passthrough payload this
    module accepts (RNG bit-generator state, an already-typed subsystem's own
    serialized state), so a callback, event sink, logger, open file, scheduler client,
    or live manipulator/minimizer object can never reach a constructed snapshot, even
    indirectly through one of these mappings.

    :param value: Value to validate.
    :param path: Keyword argument, required. Diagnostic path to ``value``.
    :return: Plain-Python equivalent (``dict``/``tuple``/``str``/``int``/``float``/
        ``bool``/``None``) with NumPy scalar types normalized.
    :raises SnapshotTypeError: If ``value`` (or a nested value) is not a JSON-safe
        scalar, mapping, or sequence.
    :raises SnapshotValueError: If a nested mapping has a non-string key, or a real
        number is non-finite.
    """
    if isinstance(value, (str, bool, type(None))):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SnapshotValueError(f"{path} keys must be strings")
            normalized[key] = _reject_live_objects(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (Sequence, tuple)) and not isinstance(value, (str, bytes)):
        return tuple(
            _reject_live_objects(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, np.ndarray):
        return tuple(
            _reject_live_objects(item, path=f"{path}[{index}]")
            for index, item in enumerate(value.tolist())
        )
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        normalized_real = float(value)
        if not np.isfinite(normalized_real):
            raise SnapshotValueError(f"{path} must be finite")
        return normalized_real
    raise SnapshotTypeError(
        f"{path} must be a JSON-safe scalar, mapping, or sequence; got "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True, slots=True, init=False)
class RngStateSnapshot:
    """Immutable, exact snapshot of one ``numpy.random.Generator``'s bit generator.

    Captures the bit generator's stable type name and its full internal state as
    returned by ``Generator.bit_generator.state`` -- not a reseed, and not a summary.
    :meth:`to_generator` reconstructs a ``Generator`` whose subsequent draws are
    bit-for-bit identical to the source generator's.

    :param bit_generator: Bit generator class name (e.g. ``"PCG64"``), matching
        ``numpy.random.Generator.bit_generator.state["bit_generator"]``.
    :param state: Full internal bit-generator state mapping, JSON-safe-normalized.
    :raises SnapshotTypeError: If ``bit_generator`` is not a non-empty string, ``state``
        is not a mapping, or ``state`` is not JSON-safe.
    :raises SnapshotValueError: If ``state`` does not itself declare a matching
        ``"bit_generator"`` entry.
    """

    bit_generator: str
    state: Mapping[str, object]

    def __init__(self, *, bit_generator: str, state: Mapping[str, object]) -> None:
        """Construct a validated, immutable RNG state snapshot.

        :param bit_generator: Keyword argument, required. Bit generator class name.
        :param state: Keyword argument, required. Full internal bit-generator state.
        :raises SnapshotTypeError: If ``bit_generator`` is not a non-empty string,
            ``state`` is not a mapping, or ``state`` is not JSON-safe.
        :raises SnapshotValueError: If ``state`` does not itself declare a matching
            ``"bit_generator"`` entry.
        """
        bit_generator = _normalize_identity(bit_generator, name="bit_generator")
        if not isinstance(state, Mapping):
            raise SnapshotTypeError("state must be a mapping")
        normalized_state = _reject_live_objects(state, path="state")
        if normalized_state.get("bit_generator") != bit_generator:
            raise SnapshotValueError(
                "state['bit_generator'] must match the bit_generator argument"
            )
        object.__setattr__(self, "bit_generator", bit_generator)
        object.__setattr__(self, "state", MappingProxyType(normalized_state))

    @classmethod
    def from_generator(cls, rng: np.random.Generator) -> RngStateSnapshot:
        """Capture one live generator's exact bit-generator state.

        :param rng: Generator to snapshot.
        :return: Validated, immutable RNG state snapshot.
        :raises SnapshotTypeError: If ``rng`` is not a ``numpy.random.Generator``.
        """
        if not isinstance(rng, np.random.Generator):
            raise SnapshotTypeError("rng must be a numpy.random.Generator")
        raw_state = rng.bit_generator.state
        return cls(bit_generator=raw_state["bit_generator"], state=raw_state)

    def to_generator(self) -> np.random.Generator:
        """Reconstruct a ``Generator`` whose future draws match the captured source's.

        :return: A new generator, seeded via exact bit-generator state restoration.
        :raises SnapshotValueError: If :attr:`bit_generator` does not name a bit
            generator class ``numpy.random`` provides.
        """
        bit_generator_cls = getattr(np.random, self.bit_generator, None)
        if not (
            isinstance(bit_generator_cls, type)
            and issubclass(bit_generator_cls, np.random.BitGenerator)
        ):
            raise SnapshotValueError(
                f"{self.bit_generator!r} is not a numpy.random bit generator"
            )
        bit_generator = bit_generator_cls()
        bit_generator.state = dict(self.state)
        return np.random.Generator(bit_generator)


@dataclass(frozen=True, slots=True, init=False)
class RunIdentitySnapshot:
    """Immutable run identity, field-compatible with ``GBOpt.observability.RunContext``.

    Deliberately does not import or depend on ``GBOpt.observability`` at construction
    time -- a snapshot's own validity never depends on the event system having run, or
    even being importable. :meth:`to_run_context` is an optional, one-way conversion
    for a caller that already has an ``OptimizationAlgorithm`` in hand and wants to
    correlate a restored snapshot with that run's event stream.

    :param run_id: Stable run identity (the same value MC/GA already resolve as their
        ``unique_id``).
    :param seed: Resolved RNG seed originally requested for this run. Snapshot restart
        correctness never depends on this value alone -- :class:`RngStateSnapshot`
        carries the actual bit-generator state -- but it is preserved as run
        provenance.
    :param case_id: Keyword argument, optional, defaults to ``None``. Caller-supplied
        scientific case identity.
    :param campaign_id: Keyword argument, optional, defaults to ``None``. Caller-supplied
        campaign identity grouping several runs.
    :raises SnapshotTypeError: If any field has an invalid type.
    :raises SnapshotValueError: If ``run_id``, ``case_id``, or ``campaign_id`` is empty.
    """

    run_id: str
    seed: int
    case_id: str | None
    campaign_id: str | None

    def __init__(
        self,
        *,
        run_id: str,
        seed: int,
        case_id: str | None = None,
        campaign_id: str | None = None,
    ) -> None:
        """Construct a validated, immutable run identity snapshot.

        See the class docstring for parameter semantics and raised exceptions.
        """
        run_id = _normalize_identity(run_id, name="run_id")
        seed = _normalize_seed(seed)
        case_id = _normalize_optional_identity(case_id, name="case_id")
        campaign_id = _normalize_optional_identity(campaign_id, name="campaign_id")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "campaign_id", campaign_id)

    def to_run_context(self, *, algorithm: OptimizationAlgorithm) -> RunContext:
        """Build a ``RunContext`` sharing this identity, for event correlation only.

        :param algorithm: Keyword argument, required. Optimizer family this run
            belongs to -- not itself part of a restart snapshot's own identity, since
            each snapshot type is already specific to one algorithm.
        :return: A ``RunContext`` carrying this snapshot's identity fields.
        """
        from GBOpt.observability import RunContext

        return RunContext(
            run_id=self.run_id,
            seed=self.seed,
            algorithm=algorithm,
            case_id=self.case_id,
            campaign_id=self.campaign_id,
        )


@dataclass(frozen=True, slots=True)
class LineageStepSnapshot:
    """One structured operation/parent-provenance step, never a prose description.

    :param operation_name: Name of the manipulation operation (or fixed lineage marker,
        e.g. ``"START"``) that produced this candidate from its parent.
    :param parent_reference: Stable reference to the parent this operation applied to
        -- a structure artifact path or a candidate identity, depending on which
        identity was authoritative when this step was recorded.
    :raises SnapshotTypeError: If either field is not a non-empty string.
    """

    operation_name: str
    parent_reference: str

    def __post_init__(self) -> None:
        """Validate both fields are non-empty strings.

        :raises SnapshotTypeError: If either field is not a non-empty string.
        """
        object.__setattr__(
            self,
            "operation_name",
            _normalize_identity(self.operation_name, name="operation_name"),
        )
        object.__setattr__(
            self,
            "parent_reference",
            _normalize_identity(self.parent_reference, name="parent_reference"),
        )


@dataclass(frozen=True, slots=True)
class GenerationHistoryEntrySnapshot:
    """One historical population member's structured lineage and recorded energy.

    :param lineage: Structured operation/parent-provenance step for this entry.
    :param energy: Recorded finite energy for this entry.
    :raises SnapshotTypeError: If ``lineage`` is not a :class:`LineageStepSnapshot`, or
        ``energy`` is Boolean or non-real.
    :raises SnapshotValueError: If ``energy`` is non-finite.
    """

    lineage: LineageStepSnapshot
    energy: float

    def __post_init__(self) -> None:
        """Validate lineage type and normalize the recorded energy.

        :raises SnapshotTypeError: If ``lineage`` is not a :class:`LineageStepSnapshot`,
            or ``energy`` is Boolean or non-real.
        :raises SnapshotValueError: If ``energy`` is non-finite.
        """
        if not isinstance(self.lineage, LineageStepSnapshot):
            raise SnapshotTypeError("lineage must be a LineageStepSnapshot")
        object.__setattr__(
            self, "energy", _normalize_energy(self.energy, name="energy")
        )


@dataclass(frozen=True, slots=True)
class FailureDiagnosticSnapshot:
    """Bounded, structured record of one failed evaluation's diagnostic source.

    :param candidate_id: Stable logical candidate identity.
    :param generation: GA generation where the evaluation failed.
    :param input_index: Candidate position within the submitted generation.
    :param failure_reason: Durable evaluator/reconstruction failure context.
    :param source_path: Evaluator-returned diagnostic source path, when available.
    :raises SnapshotTypeError: If a scalar field has an invalid type.
    :raises SnapshotValueError: If ``generation``/``input_index`` is negative or
        ``failure_reason`` is empty.
    """

    candidate_id: str
    generation: int
    input_index: int
    failure_reason: str
    source_path: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize every field.

        :raises SnapshotTypeError: If a scalar field has an invalid type.
        :raises SnapshotValueError: If ``generation``/``input_index`` is negative or
            ``failure_reason`` is empty.
        """
        object.__setattr__(
            self,
            "candidate_id",
            _normalize_identity(self.candidate_id, name="candidate_id"),
        )
        object.__setattr__(
            self, "generation", _normalize_index(self.generation, name="generation")
        )
        object.__setattr__(
            self,
            "input_index",
            _normalize_index(self.input_index, name="input_index"),
        )
        object.__setattr__(
            self,
            "failure_reason",
            _normalize_identity(self.failure_reason, name="failure_reason"),
        )
        object.__setattr__(
            self,
            "source_path",
            _normalize_optional_identity(self.source_path, name="source_path"),
        )


@dataclass(frozen=True, slots=True)
class MonteCarloStepRecordSnapshot:
    """One structured MC step's applied operation and Metropolis outcome.

    :param operation_name: Name of the manipulation operation proposed at this step.
    :param accepted: Whether the Metropolis criterion accepted this step's proposal.
    :raises SnapshotTypeError: If ``operation_name`` is not a non-empty string, or
        ``accepted`` is not exactly a Python ``bool``.
    """

    operation_name: str
    accepted: bool

    def __post_init__(self) -> None:
        """Validate both fields.

        :raises SnapshotTypeError: If ``operation_name`` is not a non-empty string, or
            ``accepted`` is not exactly a Python ``bool``.
        """
        object.__setattr__(
            self,
            "operation_name",
            _normalize_identity(self.operation_name, name="operation_name"),
        )
        if type(self.accepted) is not bool:
            raise SnapshotTypeError("accepted must be a bool")


@dataclass(frozen=True, slots=True, init=False)
class CandidateEvaluationSnapshot:
    """Canonical, algorithm-neutral restart record for one candidate evaluation.

    Mirrors ``GBOpt.evaluation.EvaluationResult``'s field contract, minus its live
    ``manipulator`` reference (never serialized) and with ``artifact`` optional even on
    success: this type also stands in for the artifact-independent historical summary
    ``GBOpt.evaluation``'s owned-mode adapter otherwise represents with
    ``CandidateEvaluationSummary`` (R22), so a successful record that never had -- or no
    longer needs -- a structure reference remains constructible.

    :param candidate_id: Stable logical candidate identity independent of artifact
        paths.
    :param input_index: Candidate position in the submitted population; accepts a
        negative sentinel (e.g. ``-1``) for a non-submitted authoritative candidate.
    :param status: Stable success/failure outcome.
    :param selection_energy: Optimizer-facing energy used for acceptance/selection.
    :param energy: Keyword argument, optional, defaults to ``None``. Physical energy,
        populated only when ``status`` is ``SUCCESS``.
    :param artifact: Keyword argument, optional, defaults to ``None``. Reconstructed
        structure artifact reference, when available.
    :param failure_stage: Keyword argument, optional, defaults to ``None``. Pipeline
        stage the failure originated from, required when ``status`` is ``FAILED``.
    :param failure_code: Keyword argument, optional, defaults to ``None``. Stable
        machine-readable failure identifier.
    :param failure_message: Keyword argument, optional, defaults to ``None``.
        Human-readable failure context, required when ``status`` is ``FAILED``.
    :raises SnapshotTypeError: If scalar, status, or reference fields have invalid
        types.
    :raises SnapshotValueError: If energies are non-finite or success/failure fields
        are internally inconsistent.
    """

    candidate_id: str
    input_index: int
    status: EvaluationStatus
    selection_energy: float
    energy: float | None
    artifact: StructureArtifact | None
    failure_stage: FailureStage | None
    failure_code: str | None
    failure_message: str | None

    def __init__(
        self,
        *,
        candidate_id: str,
        input_index: int,
        status: EvaluationStatus,
        selection_energy: float,
        energy: float | None = None,
        artifact: StructureArtifact | None = None,
        failure_stage: FailureStage | None = None,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> None:
        """Construct a validated, immutable candidate evaluation snapshot.

        See the class docstring for parameter semantics and raised exceptions.
        """
        candidate_id = _normalize_identity(candidate_id, name="candidate_id")
        input_index = _normalize_signed_index(input_index, name="input_index")
        if not isinstance(status, EvaluationStatus):
            raise SnapshotTypeError("status must be an EvaluationStatus")
        selection_energy = _normalize_energy(
            selection_energy, name="selection_energy"
        )
        if artifact is not None and not isinstance(artifact, StructureArtifact):
            raise SnapshotTypeError("artifact must be a StructureArtifact or None")
        if failure_stage is not None and not isinstance(failure_stage, FailureStage):
            raise SnapshotTypeError("failure_stage must be a FailureStage or None")
        failure_code = _normalize_optional_identity(failure_code, name="failure_code")

        if status is EvaluationStatus.SUCCESS:
            if energy is None:
                raise SnapshotValueError(
                    "successful evaluation requires a physical energy"
                )
            energy = _normalize_energy(energy, name="energy")
            if failure_stage is not None or failure_code is not None or (
                failure_message is not None
            ):
                raise SnapshotValueError(
                    "successful evaluation must not include failure context"
                )
        else:
            if energy is not None:
                raise SnapshotValueError(
                    "failed evaluation must not include an energy"
                )
            if failure_stage is None:
                raise SnapshotValueError("failed evaluation requires a failure_stage")
            if not isinstance(failure_message, str) or not failure_message:
                raise SnapshotValueError(
                    "failed evaluation requires a failure_message"
                )

        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "input_index", input_index)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "selection_energy", selection_energy)
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "failure_stage", failure_stage)
        object.__setattr__(self, "failure_code", failure_code)
        object.__setattr__(self, "failure_message", failure_message)

    @classmethod
    def from_evaluation_result(cls, result) -> CandidateEvaluationSnapshot:
        """Build a snapshot from one authoritative evaluation result.

        Deliberately never reads ``result.manipulator`` -- a live reconstructed
        candidate is exactly the kind of object this codec must reject.

        :param result: Authoritative evaluation result to snapshot.
        :return: Validated, immutable candidate evaluation snapshot.
        :raises SnapshotTypeError: If ``result`` is not an ``EvaluationResult``.
        """
        from GBOpt.evaluation import EvaluationResult

        if not isinstance(result, EvaluationResult):
            raise SnapshotTypeError("result must be an EvaluationResult")
        return cls(
            candidate_id=result.candidate_id,
            input_index=result.input_index,
            status=result.status,
            selection_energy=result.selection_energy,
            energy=result.energy,
            artifact=result.artifact,
            failure_stage=result.failure_stage,
            failure_code=result.failure_code,
            failure_message=result.failure_message,
        )


@dataclass(frozen=True, slots=True, init=False)
class PopulationCandidateSnapshot:
    """One GA population member's restart-critical state.

    Candidate state is always a validated structure artifact plus structured lineage.
    When the run carries persistent explicit grain ownership, :attr:`mapping` is also
    populated with the already-validated ``CandidateFileMapping`` that lets the
    candidate be reloaded without re-inferring grain membership; a run without
    persistent ownership (this schema's compatibility path for a pre-ownership legacy
    checkpoint) leaves it ``None`` and relies on the structure artifact alone, matching
    that mode's own historic reconstruction contract.

    :param artifact: Reconstructible structure artifact reference for this candidate.
    :param lineage: Structured operation/parent-provenance step that produced this
        candidate.
    :param mapping: Keyword argument, optional, defaults to ``None``. Persistent
        candidate-to-file ownership mapping, when the run tracks explicit ownership.
    :param candidate_id: Keyword argument, optional, defaults to ``None``. Stable
        logical candidate identity, when the run tracks one independent of population
        position.
    :raises SnapshotTypeError: If any field has an invalid type.
    """

    artifact: StructureArtifact
    lineage: LineageStepSnapshot
    mapping: CandidateFileMapping | None
    candidate_id: str | None

    def __init__(
        self,
        *,
        artifact: StructureArtifact,
        lineage: LineageStepSnapshot,
        mapping: CandidateFileMapping | None = None,
        candidate_id: str | None = None,
    ) -> None:
        """Construct a validated, immutable population candidate snapshot.

        See the class docstring for parameter semantics and raised exceptions.
        """
        if not isinstance(artifact, StructureArtifact):
            raise SnapshotTypeError("artifact must be a StructureArtifact")
        if not isinstance(lineage, LineageStepSnapshot):
            raise SnapshotTypeError("lineage must be a LineageStepSnapshot")
        if mapping is not None and not isinstance(mapping, CandidateFileMapping):
            raise SnapshotTypeError("mapping must be a CandidateFileMapping or None")
        candidate_id = _normalize_optional_identity(candidate_id, name="candidate_id")
        object.__setattr__(self, "artifact", artifact)
        object.__setattr__(self, "lineage", lineage)
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "candidate_id", candidate_id)


def _validate_optional_retention_state(value: object) -> Mapping[str, object] | None:
    """Validate one opaque, already-typed artifact-retention state payload.

    ``GBOpt.artifacts.store.ArtifactStore`` already owns its own typed
    ``to_state``/``from_state`` contract; this module treats its serialized form as an
    explicitly reviewed, already-typed representation (per this schema's own candidate-
    state contract) rather than re-typing it, and only enforces that it carries no live
    object.

    :param value: Retention state payload to validate.
    :return: JSON-safe-normalized payload, or ``None``.
    :raises SnapshotTypeError: If ``value`` is neither ``None`` nor a JSON-safe mapping.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SnapshotTypeError("retention_state must be a mapping or None")
    return MappingProxyType(_reject_live_objects(value, path="retention_state"))


@dataclass(frozen=True, slots=True, init=False)
class MonteCarloSnapshot:
    """Immutable restart-critical state for a Monte Carlo run, after a completed step.

    :param run: Run identity for this snapshot.
    :param rng: Exact RNG bit-generator state at the completed step boundary.
    :param completed_step: Non-negative index of the last fully completed MC step.
    :param temperature: Current MC temperature.
    :param rejection_count: Consecutive rejected trials since the last accepted step.
    :param previous_energy: Energy of the previously accepted/current structure.
    :param best_energy: Best (minimum) energy observed so far.
    :param current_artifact: Structure artifact for the current (accepted) structure.
    :param best_artifact: Keyword argument, optional, defaults to ``None``. Structure
        artifact for the best structure observed so far, when a durable archive exists.
    :param energy_history: Keyword argument, optional, defaults to ``()``. Energies
        recorded at every completed step, in order.
    :param accepted_steps: Keyword argument, optional, defaults to ``()``. Indices of
        every accepted step, in order.
    :param step_history: Keyword argument, optional, defaults to ``()``. Structured
        per-step operation/acceptance record, in order.
    :param retention_state: Keyword argument, optional, defaults to ``None``. Opaque,
        already-typed ``ArtifactStore`` state, when artifact retention is configured.
    :raises SnapshotTypeError: If any field has an invalid type.
    :raises SnapshotValueError: If a numeric field is non-finite/negative.
    """

    schema_version: int
    run: RunIdentitySnapshot
    rng: RngStateSnapshot
    completed_step: int
    temperature: float
    rejection_count: int
    previous_energy: float
    best_energy: float
    current_artifact: StructureArtifact
    best_artifact: StructureArtifact | None
    energy_history: tuple[float, ...]
    accepted_steps: tuple[int, ...]
    step_history: tuple[MonteCarloStepRecordSnapshot, ...]
    retention_state: Mapping[str, object] | None

    def __init__(
        self,
        *,
        run: RunIdentitySnapshot,
        rng: RngStateSnapshot,
        completed_step: int,
        temperature: float,
        rejection_count: int,
        previous_energy: float,
        best_energy: float,
        current_artifact: StructureArtifact,
        best_artifact: StructureArtifact | None = None,
        energy_history: Sequence[float] = (),
        accepted_steps: Sequence[int] = (),
        step_history: Sequence[MonteCarloStepRecordSnapshot] = (),
        retention_state: Mapping[str, object] | None = None,
    ) -> None:
        """Construct a validated, immutable Monte Carlo restart snapshot.

        See the class docstring for parameter semantics and raised exceptions.
        """
        if not isinstance(run, RunIdentitySnapshot):
            raise SnapshotTypeError("run must be a RunIdentitySnapshot")
        if not isinstance(rng, RngStateSnapshot):
            raise SnapshotTypeError("rng must be an RngStateSnapshot")
        completed_step = _normalize_index(completed_step, name="completed_step")
        temperature = _normalize_energy(temperature, name="temperature")
        rejection_count = _normalize_index(rejection_count, name="rejection_count")
        previous_energy = _normalize_energy(previous_energy, name="previous_energy")
        best_energy = _normalize_energy(best_energy, name="best_energy")
        if not isinstance(current_artifact, StructureArtifact):
            raise SnapshotTypeError("current_artifact must be a StructureArtifact")
        if best_artifact is not None and not isinstance(
            best_artifact, StructureArtifact
        ):
            raise SnapshotTypeError("best_artifact must be a StructureArtifact or None")
        energy_history_tuple = tuple(
            _normalize_energy(value, name="energy_history entry")
            for value in energy_history
        )
        accepted_steps_tuple = tuple(
            _normalize_index(value, name="accepted_steps entry")
            for value in accepted_steps
        )
        if not all(
            isinstance(entry, MonteCarloStepRecordSnapshot) for entry in step_history
        ):
            raise SnapshotTypeError(
                "step_history entries must be MonteCarloStepRecordSnapshot"
            )
        retention_state = _validate_optional_retention_state(retention_state)

        object.__setattr__(self, "schema_version", SNAPSHOT_SCHEMA_VERSION)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "rng", rng)
        object.__setattr__(self, "completed_step", completed_step)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "rejection_count", rejection_count)
        object.__setattr__(self, "previous_energy", previous_energy)
        object.__setattr__(self, "best_energy", best_energy)
        object.__setattr__(self, "current_artifact", current_artifact)
        object.__setattr__(self, "best_artifact", best_artifact)
        object.__setattr__(self, "energy_history", energy_history_tuple)
        object.__setattr__(self, "accepted_steps", accepted_steps_tuple)
        object.__setattr__(self, "step_history", tuple(step_history))
        object.__setattr__(self, "retention_state", retention_state)


@dataclass(frozen=True, slots=True, init=False)
class GeneticAlgorithmSnapshot:
    """Immutable restart-critical state for a GA run, after a completed generation.

    :param run: Run identity for this snapshot.
    :param rng: Exact RNG bit-generator state at the completed generation boundary.
    :param completed_generation: Non-negative index of the last fully completed GA
        generation.
    :param best: Best candidate evaluation observed so far; must be a successful
        result.
    :param population: Current pending population, in fixed order.
    :param population_cache: Keyword argument, optional, defaults to ``()``. Reusable
        carryover evaluation per population slot, aligned by position with
        :attr:`population`; an entry is ``None`` for a slot with no reusable cached
        result. Empty when the run tracks no carryover cache.
    :param energy_history: Keyword argument, optional, defaults to ``()``. Per-generation
        energy lists, in order.
    :param generation_history: Keyword argument, optional, defaults to ``()``. Structured
        per-generation lineage/energy records, in order.
    :param retention_lineages: Keyword argument, optional, defaults to ``None``. Stable
        parent candidate identities per population slot, aligned with
        :attr:`population`, when the run tracks retention lineage; ``None`` otherwise.
    :param last_generation_evaluations: Keyword argument, optional, defaults to
        ``None``. Most recent generation's per-slot evaluation outcome, aligned with
        :attr:`population`, when the run tracks it; ``None`` otherwise.
    :param failure_diagnostics: Keyword argument, optional, defaults to ``()``. Bounded,
        structured failed-evaluation diagnostic history.
    :param claimed_paths: Keyword argument, optional, defaults to ``()``. Canonical
        evaluator artifact paths already claimed by this run, for path-reuse detection.
    :param retention_state: Keyword argument, optional, defaults to ``None``. Opaque,
        already-typed ``ArtifactStore`` state, when artifact retention is configured.
    :param retention_archive_mappings: Keyword argument, optional, defaults to ``()``.
        Persistent ownership mapping per archived candidate identity, when retention
        tracks explicit ownership.
    :raises SnapshotTypeError: If any field has an invalid type.
    :raises SnapshotValueError: If ``population`` is empty, ``best`` is not a successful
        evaluation, or an aligned sequence's length does not match ``population``.
    """

    schema_version: int
    run: RunIdentitySnapshot
    rng: RngStateSnapshot
    completed_generation: int
    best: CandidateEvaluationSnapshot
    population: tuple[PopulationCandidateSnapshot, ...]
    population_cache: tuple[CandidateEvaluationSnapshot | None, ...]
    energy_history: tuple[tuple[float, ...], ...]
    generation_history: tuple[tuple[GenerationHistoryEntrySnapshot, ...], ...]
    retention_lineages: tuple[tuple[str, ...], ...] | None
    last_generation_evaluations: tuple[CandidateEvaluationSnapshot, ...] | None
    failure_diagnostics: tuple[FailureDiagnosticSnapshot, ...]
    claimed_paths: tuple[str, ...]
    retention_state: Mapping[str, object] | None
    retention_archive_mappings: Mapping[str, CandidateFileMapping]

    def __init__(
        self,
        *,
        run: RunIdentitySnapshot,
        rng: RngStateSnapshot,
        completed_generation: int,
        best: CandidateEvaluationSnapshot,
        population: Sequence[PopulationCandidateSnapshot],
        population_cache: Sequence[CandidateEvaluationSnapshot | None] = (),
        energy_history: Sequence[Sequence[float]] = (),
        generation_history: Sequence[
            Sequence[GenerationHistoryEntrySnapshot]
        ] = (),
        retention_lineages: Sequence[Sequence[str]] | None = None,
        last_generation_evaluations: (
            Sequence[CandidateEvaluationSnapshot] | None
        ) = None,
        failure_diagnostics: Sequence[FailureDiagnosticSnapshot] = (),
        claimed_paths: Sequence[str] = (),
        retention_state: Mapping[str, object] | None = None,
        retention_archive_mappings: Mapping[str, CandidateFileMapping] = (),
    ) -> None:
        """Construct a validated, immutable genetic-algorithm restart snapshot.

        See the class docstring for parameter semantics and raised exceptions.
        """
        if not isinstance(run, RunIdentitySnapshot):
            raise SnapshotTypeError("run must be a RunIdentitySnapshot")
        if not isinstance(rng, RngStateSnapshot):
            raise SnapshotTypeError("rng must be an RngStateSnapshot")
        completed_generation = _normalize_index(
            completed_generation, name="completed_generation"
        )
        if not isinstance(best, CandidateEvaluationSnapshot):
            raise SnapshotTypeError("best must be a CandidateEvaluationSnapshot")
        if best.status is not EvaluationStatus.SUCCESS:
            raise SnapshotValueError("best must be a successful evaluation")

        population_tuple = tuple(population)
        if not population_tuple:
            raise SnapshotValueError("population must not be empty")
        if not all(
            isinstance(entry, PopulationCandidateSnapshot)
            for entry in population_tuple
        ):
            raise SnapshotTypeError(
                "population entries must be PopulationCandidateSnapshot"
            )
        population_size = len(population_tuple)

        population_cache_tuple = tuple(population_cache)
        if population_cache_tuple and len(population_cache_tuple) != population_size:
            raise SnapshotValueError(
                "population_cache must be empty or aligned with population"
            )
        if not all(
            entry is None or isinstance(entry, CandidateEvaluationSnapshot)
            for entry in population_cache_tuple
        ):
            raise SnapshotTypeError(
                "population_cache entries must be CandidateEvaluationSnapshot or None"
            )

        energy_history_tuple = tuple(
            tuple(_normalize_energy(value, name="energy_history entry") for value in gen)
            for gen in energy_history
        )
        generation_history_tuple = tuple(
            tuple(_require_generation_history_entry(entry) for entry in gen)
            for gen in generation_history
        )

        retention_lineages_tuple: tuple[tuple[str, ...], ...] | None
        if retention_lineages is None:
            retention_lineages_tuple = None
        else:
            retention_lineages_tuple = tuple(
                tuple(
                    _normalize_identity(value, name="retention_lineages entry")
                    for value in lineage
                )
                for lineage in retention_lineages
            )
            if len(retention_lineages_tuple) != population_size:
                raise SnapshotValueError(
                    "retention_lineages must be None or aligned with population"
                )

        last_generation_evaluations_tuple: (
            tuple[CandidateEvaluationSnapshot, ...] | None
        )
        if last_generation_evaluations is None:
            last_generation_evaluations_tuple = None
        else:
            last_generation_evaluations_tuple = tuple(last_generation_evaluations)
            if len(last_generation_evaluations_tuple) != population_size:
                raise SnapshotValueError(
                    "last_generation_evaluations must be None or aligned with "
                    "population"
                )
            if not all(
                isinstance(entry, CandidateEvaluationSnapshot)
                for entry in last_generation_evaluations_tuple
            ):
                raise SnapshotTypeError(
                    "last_generation_evaluations entries must be "
                    "CandidateEvaluationSnapshot"
                )

        if not all(
            isinstance(entry, FailureDiagnosticSnapshot)
            for entry in failure_diagnostics
        ):
            raise SnapshotTypeError(
                "failure_diagnostics entries must be FailureDiagnosticSnapshot"
            )
        claimed_paths_tuple = tuple(
            _normalize_identity(value, name="claimed_paths entry")
            for value in claimed_paths
        )
        retention_state = _validate_optional_retention_state(retention_state)
        retention_archive_mappings_dict = dict(retention_archive_mappings)
        for candidate_id, mapping in retention_archive_mappings_dict.items():
            if not isinstance(candidate_id, str) or not candidate_id.strip():
                raise SnapshotTypeError(
                    "retention_archive_mappings keys must be non-empty strings"
                )
            if not isinstance(mapping, CandidateFileMapping):
                raise SnapshotTypeError(
                    "retention_archive_mappings values must be CandidateFileMapping"
                )

        object.__setattr__(self, "schema_version", SNAPSHOT_SCHEMA_VERSION)
        object.__setattr__(self, "run", run)
        object.__setattr__(self, "rng", rng)
        object.__setattr__(self, "completed_generation", completed_generation)
        object.__setattr__(self, "best", best)
        object.__setattr__(self, "population", population_tuple)
        object.__setattr__(self, "population_cache", population_cache_tuple)
        object.__setattr__(self, "energy_history", energy_history_tuple)
        object.__setattr__(self, "generation_history", generation_history_tuple)
        object.__setattr__(self, "retention_lineages", retention_lineages_tuple)
        object.__setattr__(
            self,
            "last_generation_evaluations",
            last_generation_evaluations_tuple,
        )
        object.__setattr__(
            self, "failure_diagnostics", tuple(failure_diagnostics)
        )
        object.__setattr__(self, "claimed_paths", claimed_paths_tuple)
        object.__setattr__(self, "retention_state", retention_state)
        object.__setattr__(
            self,
            "retention_archive_mappings",
            MappingProxyType(retention_archive_mappings_dict),
        )


def _require_generation_history_entry(
    entry: object,
) -> GenerationHistoryEntrySnapshot:
    """Validate one generation-history entry's type.

    :param entry: Entry to validate.
    :return: The validated entry, unchanged.
    :raises SnapshotTypeError: If ``entry`` is not a ``GenerationHistoryEntrySnapshot``.
    """
    if not isinstance(entry, GenerationHistoryEntrySnapshot):
        raise SnapshotTypeError(
            "generation_history entries must be GenerationHistoryEntrySnapshot"
        )
    return entry


__all__ = [
    "SnapshotError",
    "SnapshotTypeError",
    "SnapshotValueError",
    "SNAPSHOT_SCHEMA_VERSION",
    "RngStateSnapshot",
    "RunIdentitySnapshot",
    "LineageStepSnapshot",
    "GenerationHistoryEntrySnapshot",
    "FailureDiagnosticSnapshot",
    "MonteCarloStepRecordSnapshot",
    "CandidateEvaluationSnapshot",
    "PopulationCandidateSnapshot",
    "MonteCarloSnapshot",
    "GeneticAlgorithmSnapshot",
]
