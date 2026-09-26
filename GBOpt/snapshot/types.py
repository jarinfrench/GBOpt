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

from GBOpt.evaluation import (
    EvaluationError,
    EvaluationStatus,
    FailureStage,
    StructureArtifact,
)
from GBOpt.FileGrainOwnership import CandidateFileMapping, GrainOwnershipError

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


def _artifact_to_state(artifact: StructureArtifact) -> dict[str, object]:
    """Serialize one structure artifact reference into a JSON-safe mapping.

    :param artifact: Artifact reference to serialize.
    :return: JSON-safe mapping, restorable via :func:`_artifact_from_state`.
    """
    return {
        "path": artifact.path,
        "format": artifact.format,
        "digest": artifact.digest,
    }


def _artifact_from_state(value: object, *, name: str) -> StructureArtifact:
    """Restore one structure artifact reference from :func:`_artifact_to_state`'s output.

    :param value: Serialized mapping to restore.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Validated artifact reference.
    :raises SnapshotTypeError: If ``value`` is not a mapping or is missing a required
        field.
    :raises SnapshotValueError: If the restored artifact reference is invalid.
    """
    if not isinstance(value, Mapping):
        raise SnapshotTypeError(f"{name} must be a mapping")
    try:
        return StructureArtifact(
            path=value["path"], format=value["format"], digest=value.get("digest")
        )
    except KeyError as exc:
        raise SnapshotTypeError(f"{name} is missing required field: {exc}") from exc
    except EvaluationError as exc:
        raise SnapshotValueError(f"{name} is invalid: {exc}") from exc


def _status_from_state(value: object) -> EvaluationStatus:
    """Restore one ``EvaluationStatus`` from its serialized ``.value`` string.

    :param value: Serialized status value.
    :return: Validated status member.
    :raises SnapshotValueError: If ``value`` is not a known ``EvaluationStatus`` value.
    """
    try:
        return EvaluationStatus(value)
    except ValueError as exc:
        raise SnapshotValueError(f"invalid EvaluationStatus value: {value!r}") from exc


def _failure_stage_from_state(value: object) -> FailureStage:
    """Restore one ``FailureStage`` from its serialized ``.value`` string.

    :param value: Serialized failure-stage value.
    :return: Validated failure-stage member.
    :raises SnapshotValueError: If ``value`` is not a known ``FailureStage`` value.
    """
    try:
        return FailureStage(value)
    except ValueError as exc:
        raise SnapshotValueError(f"invalid FailureStage value: {value!r}") from exc


def _mapping_from_state(value: object, *, name: str) -> CandidateFileMapping:
    """Restore one candidate/file ownership mapping, translating its own error type.

    Reuses ``GBOpt.optimization.types``'s existing schema-v1 mapping
    (de)serialization helper rather than duplicating its field-by-field
    reconstruction.

    :param value: Serialized mapping state.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Validated candidate/file ownership mapping.
    :raises SnapshotValueError: If ``value`` is not a valid serialized mapping.
    """
    # Local import: GBOpt.optimization.types is only reachable through
    # GBOpt.optimization's own package __init__, which eagerly imports
    # GBOpt.optimization.genetic -- and genetic.py imports this subpackage at module
    # scope. A module-scope import here would form an import cycle; deferring it to
    # call time (after both packages have finished loading) breaks it.
    from GBOpt.optimization.types import _candidate_mapping_from_state

    try:
        return _candidate_mapping_from_state(value)
    except GrainOwnershipError as exc:
        raise SnapshotValueError(f"{name} is invalid: {exc}") from exc


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
        return _reject_live_objects_mapping(value, path=path)
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


def _reject_live_objects_mapping(
    value: Mapping[str, object], *, path: str
) -> dict[str, object]:
    """Recursively validate one mapping's keys/values are all JSON-safe.

    :param value: Mapping to validate.
    :param path: Keyword argument, required. Diagnostic path to ``value``.
    :return: Plain ``dict[str, object]`` with every value JSON-safe-normalized.
    :raises SnapshotTypeError: If any value is not JSON-safe.
    :raises SnapshotValueError: If any key is not a string.
    """
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise SnapshotValueError(f"{path} keys must be strings")
        normalized[key] = _reject_live_objects(item, path=f"{path}.{key}")
    return normalized


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
        normalized_state = _reject_live_objects_mapping(state, path="state")
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

    def to_state(self) -> dict[str, object]:
        """Serialize this snapshot into a JSON-safe mapping.

        :return: JSON-safe mapping, restorable via :meth:`from_state`.
        """
        return {"bit_generator": self.bit_generator, "state": dict(self.state)}

    @classmethod
    def from_state(cls, state: object) -> RngStateSnapshot:
        """Restore a validated RNG state snapshot from :meth:`to_state`'s own output.

        :param state: Serialized mapping, as returned by :meth:`to_state`.
        :return: Validated, immutable RNG state snapshot.
        :raises SnapshotTypeError: If ``state`` is not a mapping or is missing a
            required field.
        :raises SnapshotValueError: If ``state`` is internally inconsistent.
        """
        if not isinstance(state, Mapping):
            raise SnapshotTypeError("RngStateSnapshot state must be a mapping")
        try:
            return cls(bit_generator=state["bit_generator"], state=state["state"])
        except KeyError as exc:
            raise SnapshotTypeError(
                f"RngStateSnapshot state is missing required field: {exc}"
            ) from exc

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

    def to_state(self) -> dict[str, object]:
        """Serialize this identity into a JSON-safe mapping.

        :return: JSON-safe mapping, restorable via :meth:`from_state`.
        """
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "case_id": self.case_id,
            "campaign_id": self.campaign_id,
        }

    @classmethod
    def from_state(cls, state: object) -> RunIdentitySnapshot:
        """Restore a validated run identity from :meth:`to_state`'s own output.

        :param state: Serialized mapping, as returned by :meth:`to_state`.
        :return: Validated, immutable run identity snapshot.
        :raises SnapshotTypeError: If ``state`` is not a mapping or is missing a
            required field.
        :raises SnapshotValueError: If a restored identity field is invalid.
        """
        if not isinstance(state, Mapping):
            raise SnapshotTypeError("RunIdentitySnapshot state must be a mapping")
        try:
            return cls(
                run_id=state["run_id"],
                seed=state["seed"],
                case_id=state.get("case_id"),
                campaign_id=state.get("campaign_id"),
            )
        except KeyError as exc:
            raise SnapshotTypeError(
                f"RunIdentitySnapshot state is missing required field: {exc}"
            ) from exc

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


@dataclass(frozen=True, slots=True, init=False)
class LineageStepSnapshot:
    """One structured operation/parent-provenance step, never a prose description.

    An operation may draw from one parent (mutation, carryover, the fixed ``"START"``
    marker) or two (crossover), so :attr:`parent_references` holds however many a real
    step actually used rather than assuming a fixed arity. :attr:`diagnostic_note`
    carries free-text context schema-v1 itself already recorded alongside some lineage
    entries (e.g. crossover provenance parameters, a fallback attempt count) -- it is
    explicitly non-authoritative diagnostic text, never a substitute for
    :attr:`operation_name`/:attr:`parent_references`' own structured identity.

    :param operation_name: Name of the manipulation operation (or fixed lineage marker,
        e.g. ``"START"``) that produced this candidate from its parent(s).
    :param parent_references: Stable references to the parent(s) this operation applied
        to -- structure artifact paths or candidate identities, depending on which
        identity was authoritative when this step was recorded.
    :param diagnostic_note: Keyword argument, optional, defaults to ``None``.
        Non-authoritative free-text context recorded alongside this step.
    :raises SnapshotTypeError: If ``operation_name``/``diagnostic_note`` is not a
        string, or ``parent_references`` is not a sequence of non-empty strings.
    :raises SnapshotValueError: If ``parent_references`` is empty.
    """

    operation_name: str
    parent_references: tuple[str, ...]
    diagnostic_note: str | None

    def __init__(
        self,
        *,
        operation_name: str,
        parent_references: Sequence[str],
        diagnostic_note: str | None = None,
    ) -> None:
        """Construct a validated, immutable lineage step.

        See the class docstring for parameter semantics and raised exceptions.
        """
        operation_name = _normalize_identity(operation_name, name="operation_name")
        parent_references_tuple = tuple(
            _normalize_identity(value, name="parent_references entry")
            for value in parent_references
        )
        if not parent_references_tuple:
            raise SnapshotValueError("parent_references must not be empty")
        diagnostic_note = _normalize_optional_identity(
            diagnostic_note, name="diagnostic_note"
        )
        object.__setattr__(self, "operation_name", operation_name)
        object.__setattr__(self, "parent_references", parent_references_tuple)
        object.__setattr__(self, "diagnostic_note", diagnostic_note)

    def to_state(self) -> dict[str, object]:
        """Serialize this lineage step into a JSON-safe mapping.

        :return: JSON-safe mapping, restorable via :meth:`from_state`.
        """
        return {
            "operation_name": self.operation_name,
            "parent_references": list(self.parent_references),
            "diagnostic_note": self.diagnostic_note,
        }

    @classmethod
    def from_state(cls, state: object) -> LineageStepSnapshot:
        """Restore a validated lineage step from :meth:`to_state`'s own output.

        :param state: Serialized mapping, as returned by :meth:`to_state`.
        :return: Validated, immutable lineage step.
        :raises SnapshotTypeError: If ``state`` is not a mapping or is missing a
            required field.
        :raises SnapshotValueError: If ``state`` is internally inconsistent.
        """
        if not isinstance(state, Mapping):
            raise SnapshotTypeError("LineageStepSnapshot state must be a mapping")
        try:
            return cls(
                operation_name=state["operation_name"],
                parent_references=state["parent_references"],
                diagnostic_note=state.get("diagnostic_note"),
            )
        except KeyError as exc:
            raise SnapshotTypeError(
                f"LineageStepSnapshot state is missing required field: {exc}"
            ) from exc


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

    def to_state(self) -> dict[str, object]:
        """Serialize this history entry into a JSON-safe mapping.

        :return: JSON-safe mapping, restorable via :meth:`from_state`.
        """
        return {"lineage": self.lineage.to_state(), "energy": self.energy}

    @classmethod
    def from_state(cls, state: object) -> GenerationHistoryEntrySnapshot:
        """Restore a validated history entry from :meth:`to_state`'s own output.

        :param state: Serialized mapping, as returned by :meth:`to_state`.
        :return: Validated, immutable generation history entry.
        :raises SnapshotTypeError: If ``state`` is not a mapping or is missing a
            required field.
        :raises SnapshotValueError: If ``state`` is internally inconsistent.
        """
        if not isinstance(state, Mapping):
            raise SnapshotTypeError(
                "GenerationHistoryEntrySnapshot state must be a mapping"
            )
        try:
            return cls(
                lineage=LineageStepSnapshot.from_state(state["lineage"]),
                energy=state["energy"],
            )
        except KeyError as exc:
            raise SnapshotTypeError(
                f"GenerationHistoryEntrySnapshot state is missing required field: {exc}"
            ) from exc


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

    def to_state(self) -> dict[str, object]:
        """Serialize this failure diagnostic into a JSON-safe mapping.

        :return: JSON-safe mapping, restorable via :meth:`from_state`.
        """
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "input_index": self.input_index,
            "failure_reason": self.failure_reason,
            "source_path": self.source_path,
        }

    @classmethod
    def from_state(cls, state: object) -> FailureDiagnosticSnapshot:
        """Restore a validated failure diagnostic from :meth:`to_state`'s own output.

        :param state: Serialized mapping, as returned by :meth:`to_state`.
        :return: Validated, immutable failure diagnostic.
        :raises SnapshotTypeError: If ``state`` is not a mapping or is missing a
            required field.
        :raises SnapshotValueError: If ``state`` is internally inconsistent.
        """
        if not isinstance(state, Mapping):
            raise SnapshotTypeError("FailureDiagnosticSnapshot state must be a mapping")
        try:
            return cls(
                candidate_id=state["candidate_id"],
                generation=state["generation"],
                input_index=state["input_index"],
                failure_reason=state["failure_reason"],
                source_path=state.get("source_path"),
            )
        except KeyError as exc:
            raise SnapshotTypeError(
                f"FailureDiagnosticSnapshot state is missing required field: {exc}"
            ) from exc


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

    def to_state(self) -> dict[str, object]:
        """Serialize this candidate evaluation into a JSON-safe mapping.

        :return: JSON-safe mapping, restorable via :meth:`from_state`.
        """
        return {
            "candidate_id": self.candidate_id,
            "input_index": self.input_index,
            "status": self.status.value,
            "selection_energy": self.selection_energy,
            "energy": self.energy,
            "artifact": (
                None if self.artifact is None else _artifact_to_state(self.artifact)
            ),
            "failure_stage": (
                None if self.failure_stage is None else self.failure_stage.value
            ),
            "failure_code": self.failure_code,
            "failure_message": self.failure_message,
        }

    @classmethod
    def from_state(cls, state: object) -> CandidateEvaluationSnapshot:
        """Restore a validated candidate evaluation from :meth:`to_state`'s own output.

        :param state: Serialized mapping, as returned by :meth:`to_state`.
        :return: Validated, immutable candidate evaluation snapshot.
        :raises SnapshotTypeError: If ``state`` is not a mapping or is missing a
            required field.
        :raises SnapshotValueError: If ``state`` is internally inconsistent.
        """
        if not isinstance(state, Mapping):
            raise SnapshotTypeError(
                "CandidateEvaluationSnapshot state must be a mapping"
            )
        try:
            artifact_state = state.get("artifact")
            failure_stage_state = state.get("failure_stage")
            return cls(
                candidate_id=state["candidate_id"],
                input_index=state["input_index"],
                status=_status_from_state(state["status"]),
                selection_energy=state["selection_energy"],
                energy=state.get("energy"),
                artifact=(
                    None
                    if artifact_state is None
                    else _artifact_from_state(artifact_state, name="artifact")
                ),
                failure_stage=(
                    None
                    if failure_stage_state is None
                    else _failure_stage_from_state(failure_stage_state)
                ),
                failure_code=state.get("failure_code"),
                failure_message=state.get("failure_message"),
            )
        except KeyError as exc:
            raise SnapshotTypeError(
                f"CandidateEvaluationSnapshot state is missing required field: {exc}"
            ) from exc

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

    def to_state(self) -> dict[str, object]:
        """Serialize this population candidate into a JSON-safe mapping.

        :return: JSON-safe mapping, restorable via :meth:`from_state`.
        """
        # Local import: see _mapping_from_state's own comment on why this cannot be a
        # module-scope import.
        from GBOpt.optimization.types import _candidate_mapping_to_state

        return {
            "artifact": _artifact_to_state(self.artifact),
            "lineage": self.lineage.to_state(),
            "mapping": (
                None
                if self.mapping is None
                else _candidate_mapping_to_state(self.mapping)
            ),
            "candidate_id": self.candidate_id,
        }

    @classmethod
    def from_state(cls, state: object) -> PopulationCandidateSnapshot:
        """Restore a validated population candidate from :meth:`to_state`'s own output.

        :param state: Serialized mapping, as returned by :meth:`to_state`.
        :return: Validated, immutable population candidate snapshot.
        :raises SnapshotTypeError: If ``state`` is not a mapping or is missing a
            required field.
        :raises SnapshotValueError: If ``state`` is internally inconsistent.
        """
        if not isinstance(state, Mapping):
            raise SnapshotTypeError(
                "PopulationCandidateSnapshot state must be a mapping"
            )
        try:
            mapping_state = state.get("mapping")
            return cls(
                artifact=_artifact_from_state(state["artifact"], name="artifact"),
                lineage=LineageStepSnapshot.from_state(state["lineage"]),
                mapping=(
                    None
                    if mapping_state is None
                    else _mapping_from_state(mapping_state, name="mapping")
                ),
                candidate_id=state.get("candidate_id"),
            )
        except KeyError as exc:
            raise SnapshotTypeError(
                f"PopulationCandidateSnapshot state is missing required field: {exc}"
            ) from exc


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
    return MappingProxyType(
        _reject_live_objects_mapping(value, path="retention_state")
    )


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


def _normalize_optional_index(value: object, *, name: str) -> int | None:
    """Validate one optional non-negative integer.

    :param value: Value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Python non-negative integer, or ``None``.
    :raises SnapshotTypeError: If ``value`` is neither ``None`` nor a non-Boolean
        integer.
    :raises SnapshotValueError: If ``value`` is negative.
    """
    if value is None:
        return None
    return _normalize_index(value, name=name)


def _normalize_optional_bool(value: object, *, name: str) -> bool | None:
    """Validate one optional exact Python ``bool``.

    :param value: Value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: The Boolean, unchanged, or ``None``.
    :raises SnapshotTypeError: If ``value`` is neither ``None`` nor exactly a ``bool``.
    """
    if value is None:
        return None
    if type(value) is not bool:
        raise SnapshotTypeError(f"{name} must be a bool or None")
    return value


def _normalize_optional_energy_bare(value: object, *, name: str) -> float | None:
    """Validate one optional finite real scalar with no domain-specific naming.

    Alias for :func:`_normalize_optional_energy` used for non-energy real-valued
    configuration fields (e.g. a tilt angle), kept distinct for readability at call
    sites.

    :param value: Value to validate.
    :param name: Keyword argument, required. Field name used in diagnostics.
    :return: Finite Python float, or ``None``.
    """
    return _normalize_optional_energy(value, name=name)


@dataclass(frozen=True, slots=True, init=False)
class GeneticAlgorithmConfigurationSnapshot:
    """Immutable deterministic GA configuration, checked against a resuming run.

    Schema-v1 stored a different subset of this configuration depending on GA mode:
    the legacy (non-owned) path only ever validated :attr:`slice_and_merge_pct`/
    :attr:`reuse_carryover_evaluations` on resume, while the explicit-ownership path
    validated every field here. Every field beyond those two is therefore optional,
    ``None`` when the checkpoint that produced this snapshot never tracked it (always
    true for a legacy-mode snapshot; never true for an explicit-ownership one).

    :param slice_and_merge_pct: Percentage of non-carryover offspring generated by
        slice-and-merge crossover.
    :param reuse_carryover_evaluations: Whether unchanged carryover candidates reuse
        their prior evaluation instead of being re-evaluated.
    :param population_size: Keyword argument, optional, defaults to ``None``. Number of
        candidates per generation.
    :param keep_top_pct: Keyword argument, optional, defaults to ``None``. Percentage of
        lowest-energy structures carried over unchanged.
    :param intermediate_pct: Keyword argument, optional, defaults to ``None``.
        Percentage of structures eligible for crossover/mutation selection.
    :param allow_variable_cell: Keyword argument, optional, defaults to ``None``.
        Whether orthogonal box dimensions may evolve between generations.
    :param choices: Keyword argument, optional, defaults to ``None``. Configured
        mutation operation names.
    :param crossover_surface: Keyword argument, optional, defaults to ``None``.
        Formula-preserving crossover surface mode.
    :param crossover_max_tilt_degrees: Keyword argument, optional, defaults to ``None``.
        Maximum combined local periodic-wave tilt in degrees.
    :param crossover_attempts: Keyword argument, optional, defaults to ``None``.
        Maximum parent-pair attempts before one crossover slot falls back to mutation.
    :param failure_diagnostic_count: Keyword argument, optional, defaults to ``None``.
        Maximum number of most-recent failed evaluator sources preserved for
        diagnostics.
    :param composition_policy: Keyword argument, optional, defaults to ``None``.
        Fixed formula ratio the run enforces for every candidate.
    :raises SnapshotTypeError: If any field has an invalid type.
    :raises SnapshotValueError: If a numeric field is non-finite/out of range.
    """

    slice_and_merge_pct: float
    reuse_carryover_evaluations: bool
    population_size: int | None
    keep_top_pct: int | None
    intermediate_pct: int | None
    allow_variable_cell: bool | None
    choices: tuple[str, ...] | None
    crossover_surface: str | None
    crossover_max_tilt_degrees: float | None
    crossover_attempts: int | None
    failure_diagnostic_count: int | None
    composition_policy: tuple[tuple[str, int], ...] | None

    def __init__(
        self,
        *,
        slice_and_merge_pct: float,
        reuse_carryover_evaluations: bool,
        population_size: int | None = None,
        keep_top_pct: int | None = None,
        intermediate_pct: int | None = None,
        allow_variable_cell: bool | None = None,
        choices: Sequence[str] | None = None,
        crossover_surface: str | None = None,
        crossover_max_tilt_degrees: float | None = None,
        crossover_attempts: int | None = None,
        failure_diagnostic_count: int | None = None,
        composition_policy: Sequence[Sequence[object]] | None = None,
    ) -> None:
        """Construct a validated, immutable GA configuration snapshot.

        See the class docstring for parameter semantics and raised exceptions.
        """
        slice_and_merge_pct = _normalize_energy(
            slice_and_merge_pct, name="slice_and_merge_pct"
        )
        normalized_reuse_carryover_evaluations = _normalize_optional_bool(
            reuse_carryover_evaluations, name="reuse_carryover_evaluations"
        )
        if normalized_reuse_carryover_evaluations is None:
            raise SnapshotTypeError("reuse_carryover_evaluations must be a bool")
        reuse_carryover_evaluations = normalized_reuse_carryover_evaluations
        population_size = _normalize_optional_index(
            population_size, name="population_size"
        )
        keep_top_pct = _normalize_optional_index(keep_top_pct, name="keep_top_pct")
        intermediate_pct = _normalize_optional_index(
            intermediate_pct, name="intermediate_pct"
        )
        allow_variable_cell = _normalize_optional_bool(
            allow_variable_cell, name="allow_variable_cell"
        )
        choices_tuple = (
            None
            if choices is None
            else tuple(
                _normalize_identity(value, name="choices entry") for value in choices
            )
        )
        if crossover_surface is not None:
            crossover_surface = _normalize_identity(
                crossover_surface, name="crossover_surface"
            )
        crossover_max_tilt_degrees = _normalize_optional_energy_bare(
            crossover_max_tilt_degrees, name="crossover_max_tilt_degrees"
        )
        crossover_attempts = _normalize_optional_index(
            crossover_attempts, name="crossover_attempts"
        )
        failure_diagnostic_count = _normalize_optional_index(
            failure_diagnostic_count, name="failure_diagnostic_count"
        )
        composition_policy_tuple = (
            None
            if composition_policy is None
            else tuple(
                (
                    _normalize_identity(species, name="composition_policy species"),
                    _normalize_index(coefficient, name="composition_policy coefficient"),
                )
                for species, coefficient in composition_policy
            )
        )

        object.__setattr__(self, "slice_and_merge_pct", slice_and_merge_pct)
        object.__setattr__(
            self, "reuse_carryover_evaluations", reuse_carryover_evaluations
        )
        object.__setattr__(self, "population_size", population_size)
        object.__setattr__(self, "keep_top_pct", keep_top_pct)
        object.__setattr__(self, "intermediate_pct", intermediate_pct)
        object.__setattr__(self, "allow_variable_cell", allow_variable_cell)
        object.__setattr__(self, "choices", choices_tuple)
        object.__setattr__(self, "crossover_surface", crossover_surface)
        object.__setattr__(
            self, "crossover_max_tilt_degrees", crossover_max_tilt_degrees
        )
        object.__setattr__(self, "crossover_attempts", crossover_attempts)
        object.__setattr__(
            self, "failure_diagnostic_count", failure_diagnostic_count
        )
        object.__setattr__(self, "composition_policy", composition_policy_tuple)

    def to_state(self) -> dict[str, object]:
        """Serialize this configuration into a JSON-safe mapping.

        :return: JSON-safe mapping, restorable via :meth:`from_state`.
        """
        return {
            "slice_and_merge_pct": self.slice_and_merge_pct,
            "reuse_carryover_evaluations": self.reuse_carryover_evaluations,
            "population_size": self.population_size,
            "keep_top_pct": self.keep_top_pct,
            "intermediate_pct": self.intermediate_pct,
            "allow_variable_cell": self.allow_variable_cell,
            "choices": None if self.choices is None else list(self.choices),
            "crossover_surface": self.crossover_surface,
            "crossover_max_tilt_degrees": self.crossover_max_tilt_degrees,
            "crossover_attempts": self.crossover_attempts,
            "failure_diagnostic_count": self.failure_diagnostic_count,
            "composition_policy": (
                None
                if self.composition_policy is None
                else [list(entry) for entry in self.composition_policy]
            ),
        }

    @classmethod
    def from_state(cls, state: object) -> GeneticAlgorithmConfigurationSnapshot:
        """Restore a validated GA configuration from :meth:`to_state`'s own output.

        :param state: Serialized mapping, as returned by :meth:`to_state`.
        :return: Validated, immutable GA configuration snapshot.
        :raises SnapshotTypeError: If ``state`` is not a mapping or is missing a
            required field.
        :raises SnapshotValueError: If ``state`` is internally inconsistent.
        """
        if not isinstance(state, Mapping):
            raise SnapshotTypeError(
                "GeneticAlgorithmConfigurationSnapshot state must be a mapping"
            )
        try:
            return cls(
                slice_and_merge_pct=state["slice_and_merge_pct"],
                reuse_carryover_evaluations=state["reuse_carryover_evaluations"],
                population_size=state.get("population_size"),
                keep_top_pct=state.get("keep_top_pct"),
                intermediate_pct=state.get("intermediate_pct"),
                allow_variable_cell=state.get("allow_variable_cell"),
                choices=state.get("choices"),
                crossover_surface=state.get("crossover_surface"),
                crossover_max_tilt_degrees=state.get("crossover_max_tilt_degrees"),
                crossover_attempts=state.get("crossover_attempts"),
                failure_diagnostic_count=state.get("failure_diagnostic_count"),
                composition_policy=state.get("composition_policy"),
            )
        except KeyError as exc:
            raise SnapshotTypeError(
                "GeneticAlgorithmConfigurationSnapshot state is missing required "
                f"field: {exc}"
            ) from exc


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
    :param configuration: Keyword argument, required. Deterministic GA configuration
        checked against a resuming run's own construction arguments.
    :param best_mapping: Keyword argument, optional, defaults to ``None``. Persistent
        explicit-ownership reconstruction mapping for :attr:`best`, when the run tracks
        explicit ownership -- ``CandidateEvaluationSnapshot`` itself carries no mapping
        field, since it is also used by contexts (a legacy-mode candidate, an MC
        candidate) that never have one.
    :param population_cache_mappings: Keyword argument, optional, defaults to ``()``.
        Persistent explicit-ownership reconstruction mapping per :attr:`population_cache`
        entry, aligned by position; empty when the run tracks no carryover cache or no
        cache entry needs one.
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
    configuration: GeneticAlgorithmConfigurationSnapshot
    best_mapping: CandidateFileMapping | None
    population_cache_mappings: tuple[CandidateFileMapping | None, ...]

    def __init__(
        self,
        *,
        run: RunIdentitySnapshot,
        rng: RngStateSnapshot,
        completed_generation: int,
        best: CandidateEvaluationSnapshot,
        population: Sequence[PopulationCandidateSnapshot],
        configuration: GeneticAlgorithmConfigurationSnapshot,
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
        retention_archive_mappings: Mapping[str, CandidateFileMapping] = MappingProxyType(
            {}
        ),
        best_mapping: CandidateFileMapping | None = None,
        population_cache_mappings: Sequence[CandidateFileMapping | None] = (),
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
        if not isinstance(configuration, GeneticAlgorithmConfigurationSnapshot):
            raise SnapshotTypeError(
                "configuration must be a GeneticAlgorithmConfigurationSnapshot"
            )

        population_tuple = _validate_ga_population(population)
        population_size = len(population_tuple)
        population_cache_tuple = _validate_ga_population_cache(
            population_cache, population_size=population_size
        )
        energy_history_tuple = tuple(
            tuple(_normalize_energy(value, name="energy_history entry") for value in gen)
            for gen in energy_history
        )
        generation_history_tuple = tuple(
            tuple(_require_generation_history_entry(entry) for entry in gen)
            for gen in generation_history
        )
        retention_lineages_tuple = _validate_ga_retention_lineages(
            retention_lineages, population_size=population_size
        )
        last_generation_evaluations_tuple = _validate_ga_last_generation_evaluations(
            last_generation_evaluations, population_size=population_size
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
        retention_archive_mappings_dict = _validate_ga_retention_archive_mappings(
            retention_archive_mappings
        )
        if best_mapping is not None and not isinstance(
            best_mapping, CandidateFileMapping
        ):
            raise SnapshotTypeError("best_mapping must be a CandidateFileMapping or None")
        population_cache_mappings_tuple = tuple(population_cache_mappings)
        if population_cache_mappings_tuple and (
            len(population_cache_mappings_tuple) != population_size
        ):
            raise SnapshotValueError(
                "population_cache_mappings must be empty or aligned with population"
            )
        if not all(
            entry is None or isinstance(entry, CandidateFileMapping)
            for entry in population_cache_mappings_tuple
        ):
            raise SnapshotTypeError(
                "population_cache_mappings entries must be CandidateFileMapping or None"
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
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "best_mapping", best_mapping)
        object.__setattr__(
            self, "population_cache_mappings", population_cache_mappings_tuple
        )

    def to_state(self) -> dict[str, object]:
        """Serialize this snapshot into a JSON-safe mapping.

        :return: JSON-safe mapping, restorable via :meth:`from_state`.
        """
        # Local import: see _mapping_from_state's own comment on why this cannot be a
        # module-scope import.
        from GBOpt.optimization.types import _candidate_mapping_to_state

        return {
            "schema_version": self.schema_version,
            "run": self.run.to_state(),
            "rng": self.rng.to_state(),
            "completed_generation": self.completed_generation,
            "best": self.best.to_state(),
            "population": [entry.to_state() for entry in self.population],
            "population_cache": [
                None if entry is None else entry.to_state()
                for entry in self.population_cache
            ],
            "energy_history": [list(generation) for generation in self.energy_history],
            "generation_history": [
                [entry.to_state() for entry in generation]
                for generation in self.generation_history
            ],
            "retention_lineages": (
                None
                if self.retention_lineages is None
                else [list(lineage) for lineage in self.retention_lineages]
            ),
            "last_generation_evaluations": (
                None
                if self.last_generation_evaluations is None
                else [entry.to_state() for entry in self.last_generation_evaluations]
            ),
            "failure_diagnostics": [
                entry.to_state() for entry in self.failure_diagnostics
            ],
            "claimed_paths": list(self.claimed_paths),
            "retention_state": (
                None if self.retention_state is None else dict(self.retention_state)
            ),
            "retention_archive_mappings": {
                candidate_id: _candidate_mapping_to_state(mapping)
                for candidate_id, mapping in self.retention_archive_mappings.items()
            },
            "configuration": self.configuration.to_state(),
            "best_mapping": (
                None
                if self.best_mapping is None
                else _candidate_mapping_to_state(self.best_mapping)
            ),
            "population_cache_mappings": [
                None if mapping is None else _candidate_mapping_to_state(mapping)
                for mapping in self.population_cache_mappings
            ],
        }

    @classmethod
    def from_state(cls, state: object) -> GeneticAlgorithmSnapshot:
        """Restore a validated GA snapshot from :meth:`to_state`'s own output.

        :param state: Serialized mapping, as returned by :meth:`to_state`.
        :return: Validated, immutable genetic-algorithm restart snapshot.
        :raises SnapshotTypeError: If ``state`` is not a mapping or is missing a
            required field.
        :raises SnapshotValueError: If ``state`` declares an unsupported schema
            version, or a restored value is semantically invalid.
        """
        if not isinstance(state, Mapping):
            raise SnapshotTypeError("GeneticAlgorithmSnapshot state must be a mapping")
        if state.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise SnapshotValueError(
                "unsupported GeneticAlgorithmSnapshot schema version "
                f"{state.get('schema_version')!r}; expected "
                f"{SNAPSHOT_SCHEMA_VERSION!r}"
            )
        try:
            retention_lineages_state = state.get("retention_lineages")
            last_generation_evaluations_state = state.get(
                "last_generation_evaluations"
            )
            raw_archive_mappings = state.get("retention_archive_mappings", {})
            if not isinstance(raw_archive_mappings, Mapping):
                raise SnapshotTypeError(
                    "retention_archive_mappings state must be a mapping"
                )
            return cls(
                run=RunIdentitySnapshot.from_state(state["run"]),
                rng=RngStateSnapshot.from_state(state["rng"]),
                completed_generation=state["completed_generation"],
                best=CandidateEvaluationSnapshot.from_state(state["best"]),
                population=[
                    PopulationCandidateSnapshot.from_state(entry)
                    for entry in state["population"]
                ],
                configuration=GeneticAlgorithmConfigurationSnapshot.from_state(
                    state["configuration"]
                ),
                population_cache=[
                    None if entry is None else CandidateEvaluationSnapshot.from_state(
                        entry
                    )
                    for entry in state.get("population_cache", ())
                ],
                energy_history=state.get("energy_history", ()),
                generation_history=[
                    [
                        GenerationHistoryEntrySnapshot.from_state(entry)
                        for entry in generation
                    ]
                    for generation in state.get("generation_history", ())
                ],
                retention_lineages=(
                    None
                    if retention_lineages_state is None
                    else [list(lineage) for lineage in retention_lineages_state]
                ),
                last_generation_evaluations=(
                    None
                    if last_generation_evaluations_state is None
                    else [
                        CandidateEvaluationSnapshot.from_state(entry)
                        for entry in last_generation_evaluations_state
                    ]
                ),
                failure_diagnostics=[
                    FailureDiagnosticSnapshot.from_state(entry)
                    for entry in state.get("failure_diagnostics", ())
                ],
                claimed_paths=state.get("claimed_paths", ()),
                retention_state=state.get("retention_state"),
                retention_archive_mappings={
                    candidate_id: _mapping_from_state(
                        mapping_state, name="retention_archive_mappings entry"
                    )
                    for candidate_id, mapping_state in raw_archive_mappings.items()
                },
                best_mapping=(
                    None
                    if state.get("best_mapping") is None
                    else _mapping_from_state(
                        state["best_mapping"], name="best_mapping"
                    )
                ),
                population_cache_mappings=[
                    None
                    if entry is None
                    else _mapping_from_state(
                        entry, name="population_cache_mappings entry"
                    )
                    for entry in state.get("population_cache_mappings", ())
                ],
            )
        except KeyError as exc:
            raise SnapshotTypeError(
                f"GeneticAlgorithmSnapshot state is missing required field: {exc}"
            ) from exc


def _validate_ga_population(
    population: Sequence[PopulationCandidateSnapshot],
) -> tuple[PopulationCandidateSnapshot, ...]:
    """Validate a non-empty, well-typed GA population.

    :param population: Candidate sequence to validate.
    :return: Validated population tuple.
    :raises SnapshotTypeError: If any entry is not a ``PopulationCandidateSnapshot``.
    :raises SnapshotValueError: If ``population`` is empty.
    """
    population_tuple = tuple(population)
    if not population_tuple:
        raise SnapshotValueError("population must not be empty")
    if not all(
        isinstance(entry, PopulationCandidateSnapshot) for entry in population_tuple
    ):
        raise SnapshotTypeError(
            "population entries must be PopulationCandidateSnapshot"
        )
    return population_tuple


def _validate_ga_population_cache(
    population_cache: Sequence[CandidateEvaluationSnapshot | None],
    *,
    population_size: int,
) -> tuple[CandidateEvaluationSnapshot | None, ...]:
    """Validate an optional, population-aligned carryover cache.

    :param population_cache: Cache sequence to validate.
    :param population_size: Keyword argument, required. Population size to align
        against, when ``population_cache`` is non-empty.
    :return: Validated cache tuple.
    :raises SnapshotTypeError: If any entry is neither ``None`` nor a
        ``CandidateEvaluationSnapshot``.
    :raises SnapshotValueError: If ``population_cache`` is non-empty and misaligned.
    """
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
    return population_cache_tuple


def _validate_ga_retention_lineages(
    retention_lineages: Sequence[Sequence[str]] | None,
    *,
    population_size: int,
) -> tuple[tuple[str, ...], ...] | None:
    """Validate an optional, population-aligned sequence of retention lineages.

    :param retention_lineages: Lineage sequence to validate, or ``None``.
    :param population_size: Keyword argument, required. Population size to align
        against.
    :return: Validated lineage tuple, or ``None``.
    :raises SnapshotTypeError: If any lineage entry is not a non-empty string.
    :raises SnapshotValueError: If ``retention_lineages`` is given and misaligned.
    """
    if retention_lineages is None:
        return None
    retention_lineages_tuple = tuple(
        tuple(_normalize_identity(value, name="retention_lineages entry") for value in lineage)
        for lineage in retention_lineages
    )
    if len(retention_lineages_tuple) != population_size:
        raise SnapshotValueError(
            "retention_lineages must be None or aligned with population"
        )
    return retention_lineages_tuple


def _validate_ga_last_generation_evaluations(
    last_generation_evaluations: Sequence[CandidateEvaluationSnapshot] | None,
    *,
    population_size: int,
) -> tuple[CandidateEvaluationSnapshot, ...] | None:
    """Validate an optional, population-aligned sequence of prior-generation results.

    :param last_generation_evaluations: Evaluation sequence to validate, or ``None``.
    :param population_size: Keyword argument, required. Population size to align
        against.
    :return: Validated evaluation tuple, or ``None``.
    :raises SnapshotTypeError: If any entry is not a ``CandidateEvaluationSnapshot``.
    :raises SnapshotValueError: If given and misaligned with ``population_size``.
    """
    if last_generation_evaluations is None:
        return None
    last_generation_evaluations_tuple = tuple(last_generation_evaluations)
    if len(last_generation_evaluations_tuple) != population_size:
        raise SnapshotValueError(
            "last_generation_evaluations must be None or aligned with population"
        )
    if not all(
        isinstance(entry, CandidateEvaluationSnapshot)
        for entry in last_generation_evaluations_tuple
    ):
        raise SnapshotTypeError(
            "last_generation_evaluations entries must be CandidateEvaluationSnapshot"
        )
    return last_generation_evaluations_tuple


def _validate_ga_retention_archive_mappings(
    retention_archive_mappings: Mapping[str, CandidateFileMapping],
) -> dict[str, CandidateFileMapping]:
    """Validate a candidate-identity-keyed mapping of persistent ownership state.

    :param retention_archive_mappings: Mapping to validate.
    :return: Validated, detached ``dict`` copy.
    :raises SnapshotTypeError: If a key is not a non-empty string, or a value is not a
        ``CandidateFileMapping``.
    """
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
    return retention_archive_mappings_dict


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
    "GeneticAlgorithmConfigurationSnapshot",
    "GeneticAlgorithmSnapshot",
]
