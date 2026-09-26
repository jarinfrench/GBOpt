# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Migrate an on-disk schema-v1 minimizer checkpoint into a validated schema-v2 snapshot.

This module consumes a schema-v1 checkpoint envelope (as written by
``MonteCarloMinimizer.run_MC``/``GeneticAlgorithmMinimizer.run_GA``, either its legacy
or explicit-ownership form) and returns one validated :class:`~GBOpt.snapshot.types.
MonteCarloSnapshot`/:class:`~GBOpt.snapshot.types.GeneticAlgorithmSnapshot`. It does not
read from or write to a live minimizer, run an optimizer loop, or change how either
minimizer reads/writes its own schema-v1 checkpoints -- this is a standalone, strict,
read-only reader over an existing on-disk file.

Every field access below is wrapped so a missing required field, an unsupported schema
version, a wrong-algorithm checkpoint, or a semantically invalid reference fails
explicitly as :class:`SnapshotMigrationError`, never as an unrelated ``KeyError``/
``TypeError`` leaking out of this module, and never by silently guessing a default for
state schema-v1 never optionally omits.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from GBOpt.Checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointStore,
    validate_checkpoint_envelope,
)
from GBOpt.evaluation import EvaluationStatus, FailureStage, StructureArtifact
from GBOpt.FileGrainOwnership import GrainOwnershipError
from GBOpt.optimization.types import _candidate_mapping_from_state
from GBOpt.snapshot.types import (
    CandidateEvaluationSnapshot,
    FailureDiagnosticSnapshot,
    GenerationHistoryEntrySnapshot,
    GeneticAlgorithmSnapshot,
    LineageStepSnapshot,
    MonteCarloSnapshot,
    MonteCarloStepRecordSnapshot,
    PopulationCandidateSnapshot,
    RngStateSnapshot,
    RunIdentitySnapshot,
    SnapshotError,
)

_STRUCTURE_FORMAT = "lammps"
"""Structure file format every schema-v1 checkpoint path implicitly used.

Schema-v1 never recorded a format identifier alongside a structure path -- every
minimizer's evaluator writes/reads LAMMPS data files exclusively. This constant makes
that implicit assumption explicit at the one place it is relied on.
"""


class SnapshotMigrationError(SnapshotError):
    """Raised when a schema-v1 checkpoint cannot be migrated to a schema-v2 snapshot."""


def _fail(message: str) -> NoReturn:
    """Raise a uniform migration failure.

    :param message: Human-readable failure context.
    :raises SnapshotMigrationError: Always.
    """
    raise SnapshotMigrationError(message)


def _load_v1_state(path: str | Path, fmt: str) -> dict:
    """Load and minimally shape-check one schema-v1 checkpoint envelope from disk.

    :param path: Checkpoint file path.
    :param fmt: Serialization format -- ``"json"`` or ``"pickle"``.
    :return: Deserialized checkpoint envelope.
    :raises SnapshotMigrationError: If the file cannot be found or parsed, or does not
        contain a schema-v1 dictionary envelope.
    """
    try:
        store = CheckpointStore.from_optional(path, fmt=fmt)
    except CheckpointError as exc:
        raise SnapshotMigrationError(str(exc)) from exc
    try:
        state = store.load()
    except CheckpointError as exc:
        raise SnapshotMigrationError(
            f"could not load checkpoint {path}: {exc}"
        ) from exc
    if state is None:
        raise SnapshotMigrationError(f"no checkpoint file found at {path}")
    if not isinstance(state, dict):
        raise SnapshotMigrationError("checkpoint envelope must be a dictionary")
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise SnapshotMigrationError(
            f"unsupported checkpoint schema version {state.get('schema_version')!r}; "
            f"only schema v{CHECKPOINT_SCHEMA_VERSION} can be migrated to schema v2"
        )
    return state


def _rng_snapshot_from_v1(state: dict) -> RngStateSnapshot:
    """Build an RNG snapshot from one schema-v1 envelope's ``rng_state`` field.

    :param state: Schema-v1 checkpoint envelope.
    :return: Validated RNG state snapshot.
    :raises SnapshotMigrationError: If ``rng_state`` is missing or malformed.
    """
    try:
        rng_state = state["rng_state"]
        bit_generator = rng_state["bit_generator"]
    except (KeyError, TypeError) as exc:
        raise SnapshotMigrationError(
            f"checkpoint is missing required field: {exc}"
        ) from exc
    try:
        return RngStateSnapshot(bit_generator=bit_generator, state=rng_state)
    except SnapshotError as exc:
        raise SnapshotMigrationError(f"checkpoint rng_state is invalid: {exc}") from exc


def _artifact_or_none(path: object) -> StructureArtifact | None:
    """Build an optional structure artifact reference from one raw path value.

    :param path: Raw path value, or a falsy placeholder for "no artifact".
    :return: Validated artifact reference, or ``None``.
    :raises SnapshotMigrationError: If ``path`` is truthy but not a valid path string.
    """
    if not path:
        return None
    try:
        return StructureArtifact(path=str(path), format=_STRUCTURE_FORMAT)
    except SnapshotError as exc:
        raise SnapshotMigrationError(f"checkpoint structure path is invalid: {exc}") from exc


def _lineage_step_from_v1(raw: object) -> LineageStepSnapshot:
    """Build one structured lineage step from a raw schema-v1 lineage entry.

    Schema-v1 lineage entries are not uniformly ``[operation_label, parent]`` pairs --
    a one-parent operation (mutation, carryover, the fixed ``"START"`` marker) writes a
    2-element list; a crossover fallback appends a 3-element list carrying a trailing
    diagnostic note instead of a second parent; a two-parent crossover appends a
    4-element list carrying both parents plus a trailing provenance-repr diagnostic
    note. This dispatches on that already-established shape rather than assuming a
    fixed arity.

    :param raw: Raw lineage entry, as found in a schema-v1 checkpoint.
    :return: Validated lineage step.
    :raises SnapshotMigrationError: If ``raw`` is not one of the three well-formed
        schema-v1 lineage entry shapes.
    """
    if not isinstance(raw, (list, tuple)):
        raise SnapshotMigrationError(f"checkpoint lineage entry is malformed: {raw!r}")
    raw_entry = list(raw)
    if len(raw_entry) == 2:
        operation_name, parents, diagnostic_note = raw_entry[0], raw_entry[1:], None
    elif len(raw_entry) == 3:
        operation_name, parents, diagnostic_note = (
            raw_entry[0], raw_entry[1:2], raw_entry[2],
        )
    elif len(raw_entry) == 4:
        operation_name, parents, diagnostic_note = (
            raw_entry[0], raw_entry[1:3], raw_entry[3],
        )
    else:
        raise SnapshotMigrationError(f"checkpoint lineage entry is malformed: {raw!r}")
    try:
        return LineageStepSnapshot(
            operation_name=operation_name,
            parent_references=parents,
            diagnostic_note=diagnostic_note,
        )
    except SnapshotError as exc:
        raise SnapshotMigrationError(
            f"checkpoint lineage entry is invalid: {exc}"
        ) from exc


def _monte_carlo_snapshot_from_v1(state: dict) -> MonteCarloSnapshot:
    """Build a validated Monte Carlo snapshot from one schema-v1 envelope.

    :param state: Schema-v1 checkpoint envelope already confirmed to be
        ``minimizer="MonteCarloMinimizer"``.
    :return: Validated Monte Carlo restart snapshot.
    :raises SnapshotMigrationError: If a required field is missing or any value is
        semantically invalid.
    """
    try:
        run_params = state["run_params"]
        run = RunIdentitySnapshot(
            run_id=str(run_params["unique_id"]), seed=run_params["seed"]
        )
    except KeyError as exc:
        raise SnapshotMigrationError(
            f"checkpoint is missing required run_params field: {exc}"
        ) from exc
    except SnapshotError as exc:
        raise SnapshotMigrationError(f"checkpoint run identity is invalid: {exc}") from exc

    rng = _rng_snapshot_from_v1(state)

    try:
        mc_state = state["state"]
        current_artifact = _artifact_or_none(mc_state["current_structure_dump"])
        if current_artifact is None:
            _fail("checkpoint current_structure_dump is missing")
        operation_list = mc_state["operation_list"]
        step_history = tuple(
            MonteCarloStepRecordSnapshot(
                operation_name=entry[0], accepted=bool(entry[1])
            )
            for entry in operation_list
        )
        snapshot = MonteCarloSnapshot(
            run=run,
            rng=rng,
            completed_step=state["progress_index"],
            temperature=mc_state["T"],
            rejection_count=mc_state["rejection_count"],
            previous_energy=mc_state["prev_gbe"],
            best_energy=state["best_energy"],
            current_artifact=current_artifact,
            best_artifact=_artifact_or_none(state.get("best_dump")),
            energy_history=mc_state["GBE_vals"],
            accepted_steps=mc_state["accepted_idx"],
            step_history=step_history,
            retention_state=mc_state.get("artifact_store"),
        )
    except KeyError as exc:
        raise SnapshotMigrationError(
            f"checkpoint is missing required state field: {exc}"
        ) from exc
    except (TypeError, ValueError, IndexError) as exc:
        raise SnapshotMigrationError(
            f"checkpoint state is malformed: {exc}"
        ) from exc
    except SnapshotError as exc:
        raise SnapshotMigrationError(f"checkpoint state is invalid: {exc}") from exc
    return snapshot


def _ga_snapshot_from_v1_legacy(state: dict) -> GeneticAlgorithmSnapshot:
    """Build a validated GA snapshot from one legacy (non-owned) schema-v1 envelope.

    :param state: Schema-v1 checkpoint envelope already confirmed to be
        ``minimizer="GeneticAlgorithmMinimizer"`` in its legacy (non-explicit-ownership)
        form.
    :return: Validated genetic-algorithm restart snapshot.
    :raises SnapshotMigrationError: If a required field is missing or any value is
        semantically invalid.
    """
    try:
        run_params = state["run_params"]
        run = RunIdentitySnapshot(
            run_id=str(run_params["unique_id"]), seed=run_params["seed"]
        )
    except KeyError as exc:
        raise SnapshotMigrationError(
            "checkpoint is missing required run_params field (a legacy checkpoint "
            f"written before seed persistence cannot be migrated): {exc}"
        ) from exc
    except SnapshotError as exc:
        raise SnapshotMigrationError(f"checkpoint run identity is invalid: {exc}") from exc

    rng = _rng_snapshot_from_v1(state)

    try:
        ga_state = state["state"]
        population_lineages = ga_state["population_lineages"]
        population_paths = ga_state.get(
            "population_checkpoint_paths",
            [lineage[1] for lineage in population_lineages],
        )
        if len(population_paths) != len(population_lineages):
            _fail("checkpoint population paths are not aligned with lineages")
        population = tuple(
            _legacy_population_candidate_to_snapshot(lineage, path)
            for lineage, path in zip(population_lineages, population_paths, strict=True)
        )

        cached_states = ga_state.get(
            "population_cached_evaluations", [None] * len(population)
        )
        if len(cached_states) != len(population):
            _fail("checkpoint cached evaluations are not aligned with population")
        population_cache = tuple(
            None if cached is None else _legacy_cache_to_snapshot(index, cached)
            for index, cached in enumerate(cached_states)
        )

        energy_history, generation_history = _energy_and_generation_history_from_v1(
            ga_state
        )

        best_dump = state["best_dump"]
        best_artifact = _artifact_or_none(best_dump)
        if best_artifact is None:
            _fail("checkpoint best_dump is missing")
        best = CandidateEvaluationSnapshot(
            candidate_id=f"{run.run_id}-best",
            input_index=-1,
            status=EvaluationStatus.SUCCESS,
            selection_energy=state["best_energy"],
            energy=state["best_energy"],
            artifact=best_artifact,
        )

        snapshot = GeneticAlgorithmSnapshot(
            run=run,
            rng=rng,
            completed_generation=state["progress_index"],
            best=best,
            population=population,
            population_cache=population_cache,
            energy_history=energy_history,
            generation_history=generation_history,
            retention_state=ga_state.get("artifact_store"),
        )
    except KeyError as exc:
        raise SnapshotMigrationError(
            f"checkpoint is missing required state field: {exc}"
        ) from exc
    except (TypeError, ValueError, IndexError) as exc:
        raise SnapshotMigrationError(
            f"checkpoint state is malformed: {exc}"
        ) from exc
    except SnapshotError as exc:
        raise SnapshotMigrationError(f"checkpoint state is invalid: {exc}") from exc
    return snapshot


def _legacy_population_candidate_to_snapshot(
    lineage: object, path: object
) -> PopulationCandidateSnapshot:
    """Build one legacy-mode population candidate snapshot.

    :param lineage: Raw ``[operation_label, parent]`` lineage entry.
    :param path: Raw structure file path.
    :return: Validated population candidate snapshot.
    :raises SnapshotMigrationError: If either argument is malformed.
    """
    artifact = _artifact_or_none(path)
    if artifact is None:
        raise SnapshotMigrationError(
            "checkpoint population candidate lacks a structure path"
        )
    try:
        return PopulationCandidateSnapshot(
            artifact=artifact, lineage=_lineage_step_from_v1(lineage)
        )
    except SnapshotError as exc:
        raise SnapshotMigrationError(
            f"checkpoint population candidate is invalid: {exc}"
        ) from exc


def _legacy_cache_to_snapshot(
    index: int, cached: object
) -> CandidateEvaluationSnapshot:
    """Build a candidate evaluation snapshot from one legacy carryover cache entry.

    :param index: Population slot index this cache entry belongs to.
    :param cached: Raw ``{"energy": ..., "structure_path": ...}`` cache entry.
    :return: Validated candidate evaluation snapshot.
    :raises SnapshotMigrationError: If ``cached`` is malformed.
    """
    if not isinstance(cached, dict):
        raise SnapshotMigrationError("checkpoint cached evaluation must be a mapping")
    try:
        energy = cached["energy"]
        structure_path = cached["structure_path"]
    except KeyError as exc:
        raise SnapshotMigrationError(
            f"checkpoint cached evaluation is malformed: {exc}"
        ) from exc
    artifact = _artifact_or_none(structure_path)
    if artifact is None:
        raise SnapshotMigrationError(
            "checkpoint cached evaluation lacks a structure path"
        )
    try:
        return CandidateEvaluationSnapshot(
            candidate_id=f"legacy-carryover-{index}",
            input_index=index,
            status=EvaluationStatus.SUCCESS,
            selection_energy=energy,
            energy=energy,
            artifact=artifact,
        )
    except SnapshotError as exc:
        raise SnapshotMigrationError(
            f"checkpoint cached evaluation is invalid: {exc}"
        ) from exc


def _energy_and_generation_history_from_v1(
    ga_state: dict,
) -> tuple[
    tuple[tuple[float, ...], ...], tuple[tuple[GenerationHistoryEntrySnapshot, ...], ...]
]:
    """Build the structured energy/generation history shared by both GA checkpoint shapes.

    :param ga_state: Schema-v1 GA checkpoint envelope's ``state`` field.
    :return: Validated ``(energy_history, generation_history)`` pair.
    """
    energy_history = tuple(tuple(generation) for generation in ga_state["GBE_vals"])
    generation_history = tuple(
        tuple(
            GenerationHistoryEntrySnapshot(
                lineage=_lineage_step_from_v1(lineage), energy=energy
            )
            for lineage, energy in generation
        )
        for generation in ga_state["history"]
    )
    return energy_history, generation_history


def _owned_evaluation_to_snapshot(raw: object) -> CandidateEvaluationSnapshot:
    """Build a candidate evaluation snapshot from one owned-mode evaluation state.

    :param raw: Raw ``_owned_evaluation_to_state``-shaped mapping.
    :return: Validated candidate evaluation snapshot.
    :raises SnapshotMigrationError: If ``raw`` is malformed.
    """
    if not isinstance(raw, dict):
        raise SnapshotMigrationError("checkpoint owned evaluation must be a mapping")
    try:
        candidate_id = raw["candidate_id"]
        input_index = raw["input_index"]
        energy = raw["energy"]
        structure_path = raw["structure_path"]
        success = raw["success"]
        failure_reason = raw.get("failure_reason")
    except KeyError as exc:
        raise SnapshotMigrationError(
            f"checkpoint owned evaluation is malformed: {exc}"
        ) from exc
    try:
        if success:
            artifact = _artifact_or_none(structure_path)
            return CandidateEvaluationSnapshot(
                candidate_id=candidate_id,
                input_index=input_index,
                status=EvaluationStatus.SUCCESS,
                selection_energy=energy,
                energy=energy,
                artifact=artifact,
            )
        return CandidateEvaluationSnapshot(
            candidate_id=candidate_id,
            input_index=input_index,
            status=EvaluationStatus.FAILED,
            selection_energy=energy,
            failure_stage=FailureStage.EVALUATOR,
            failure_message=failure_reason or "unknown evaluation failure",
        )
    except SnapshotError as exc:
        raise SnapshotMigrationError(
            f"checkpoint owned evaluation is invalid: {exc}"
        ) from exc


def _owned_evaluation_summary_to_snapshot(raw: object) -> CandidateEvaluationSnapshot:
    """Build a candidate evaluation snapshot from one ``CandidateEvaluationSummary`` state.

    Unlike :func:`_owned_evaluation_to_snapshot`, this historical record never carried a
    structure artifact -- :attr:`~GBOpt.snapshot.types.CandidateEvaluationSnapshot.
    artifact` is deliberately left ``None`` even on success, matching the artifact-
    independence ``CandidateEvaluationSummary`` (R22) was designed for.

    :param raw: Raw ``CandidateEvaluationSummary.to_state()``-shaped mapping.
    :return: Validated candidate evaluation snapshot.
    :raises SnapshotMigrationError: If ``raw`` is malformed.
    """
    if not isinstance(raw, dict):
        raise SnapshotMigrationError(
            "checkpoint generation evaluation must be a mapping"
        )
    try:
        candidate_id = raw["candidate_id"]
        input_index = raw["input_index"]
        objective = raw["objective"]
        success = raw["success"]
        failure_reason = raw.get("failure_reason")
    except KeyError as exc:
        raise SnapshotMigrationError(
            f"checkpoint generation evaluation is malformed: {exc}"
        ) from exc
    try:
        if success:
            return CandidateEvaluationSnapshot(
                candidate_id=candidate_id,
                input_index=input_index,
                status=EvaluationStatus.SUCCESS,
                selection_energy=objective,
                energy=objective,
            )
        return CandidateEvaluationSnapshot(
            candidate_id=candidate_id,
            input_index=input_index,
            status=EvaluationStatus.FAILED,
            selection_energy=objective,
            failure_stage=FailureStage.EVALUATOR,
            failure_message=failure_reason or "unknown evaluation failure",
        )
    except SnapshotError as exc:
        raise SnapshotMigrationError(
            f"checkpoint generation evaluation is invalid: {exc}"
        ) from exc


def _owned_population_from_v1(ga_state: dict) -> tuple[PopulationCandidateSnapshot, ...]:
    """Build the owned-mode population from one GA checkpoint's ``state`` field.

    :param ga_state: Schema-v1 GA checkpoint envelope's ``state`` field.
    :return: Validated population tuple.
    :raises SnapshotMigrationError: If population candidates/lineages are misaligned.
    """
    population_candidates = ga_state["population_candidates"]
    population_lineages = ga_state["population_lineages"]
    if len(population_candidates) != len(population_lineages):
        _fail("checkpoint population candidates are not aligned with lineages")
    return tuple(
        _owned_population_candidate_to_snapshot(candidate, lineage)
        for candidate, lineage in zip(
            population_candidates, population_lineages, strict=True
        )
    )


def _owned_population_cache_from_v1(
    ga_state: dict, *, population_size: int
) -> tuple[CandidateEvaluationSnapshot | None, ...]:
    """Build the owned-mode carryover cache from one GA checkpoint's ``state`` field.

    :param ga_state: Schema-v1 GA checkpoint envelope's ``state`` field.
    :param population_size: Keyword argument, required. Population size to align
        against.
    :return: Validated cache tuple.
    :raises SnapshotMigrationError: If the cache is not population-aligned.
    """
    cached_states = ga_state.get(
        "population_cached_evaluations", [None] * population_size
    )
    if len(cached_states) != population_size:
        _fail("checkpoint cached evaluations are not aligned with population")
    return tuple(
        None if cached is None else _owned_evaluation_to_snapshot(cached)
        for cached in cached_states
    )


def _owned_retention_lineages_from_v1(ga_state: dict) -> list[tuple[str, ...]] | None:
    """Build the owned-mode retention lineages from one GA checkpoint's ``state`` field.

    :param ga_state: Schema-v1 GA checkpoint envelope's ``state`` field.
    :return: Retention lineages, or ``None`` when the checkpoint carries none.
    """
    raw = ga_state.get("population_retention_lineages")
    if raw is None:
        return None
    return [tuple(lineage) for lineage in raw]


def _owned_last_generation_evaluations_from_v1(
    ga_state: dict,
) -> tuple[CandidateEvaluationSnapshot, ...] | None:
    """Build the owned-mode prior-generation evaluations from one GA ``state`` field.

    :param ga_state: Schema-v1 GA checkpoint envelope's ``state`` field.
    :return: Prior-generation evaluations, or ``None`` when the checkpoint carries none.
    """
    raw = ga_state.get("last_generation_evaluations")
    if raw is None:
        return None
    return tuple(_owned_evaluation_summary_to_snapshot(entry) for entry in raw)


def _owned_failure_diagnostics_from_v1(
    ga_state: dict,
) -> tuple[FailureDiagnosticSnapshot, ...]:
    """Build the bounded failure diagnostics from one GA checkpoint's ``state`` field.

    :param ga_state: Schema-v1 GA checkpoint envelope's ``state`` field.
    :return: Validated failure diagnostics tuple.
    :raises SnapshotMigrationError: If any diagnostic entry is malformed or invalid.
    """
    try:
        return tuple(
            FailureDiagnosticSnapshot(
                candidate_id=entry["candidate_id"],
                generation=entry["generation"],
                input_index=entry["input_index"],
                failure_reason=entry["failure_reason"],
                source_path=entry.get("source_path"),
            )
            for entry in ga_state.get("failure_diagnostics", [])
        )
    except (KeyError, TypeError) as exc:
        raise SnapshotMigrationError(
            f"checkpoint failure diagnostic is malformed: {exc}"
        ) from exc
    except SnapshotError as exc:
        raise SnapshotMigrationError(
            f"checkpoint failure diagnostic is invalid: {exc}"
        ) from exc


def _owned_retention_archive_mappings_from_v1(ga_state: dict) -> dict:
    """Build the owned-mode retention archive mappings from one GA ``state`` field.

    :param ga_state: Schema-v1 GA checkpoint envelope's ``state`` field.
    :return: Candidate-identity-keyed ``CandidateFileMapping`` dictionary.
    :raises SnapshotMigrationError: If any mapping entry is invalid.
    """
    raw_archive_mappings = ga_state.get("retention_archive_mappings", {})
    return {
        candidate_id: _mapping_from_v1_state(candidate_id, mapping_state)
        for candidate_id, mapping_state in raw_archive_mappings.items()
    }


def _ga_snapshot_from_v1_owned(state: dict) -> GeneticAlgorithmSnapshot:
    """Build a validated GA snapshot from one explicit-ownership schema-v1 envelope.

    :param state: Schema-v1 checkpoint envelope already confirmed to be
        ``minimizer="GeneticAlgorithmMinimizer"`` in its explicit-ownership form.
    :return: Validated genetic-algorithm restart snapshot.
    :raises SnapshotMigrationError: If a required field is missing or any value is
        semantically invalid.
    """
    try:
        run_params = state["run_params"]
        run = RunIdentitySnapshot(
            run_id=str(run_params["unique_id"]), seed=run_params["seed"]
        )
    except KeyError as exc:
        raise SnapshotMigrationError(
            f"checkpoint is missing required run_params field: {exc}"
        ) from exc
    except SnapshotError as exc:
        raise SnapshotMigrationError(f"checkpoint run identity is invalid: {exc}") from exc

    rng = _rng_snapshot_from_v1(state)

    try:
        ga_state = state["state"]
        best = _owned_evaluation_to_snapshot(ga_state["best_evaluation"])
        population = _owned_population_from_v1(ga_state)
        population_cache = _owned_population_cache_from_v1(
            ga_state, population_size=len(population)
        )
        retention_lineages = _owned_retention_lineages_from_v1(ga_state)
        last_generation_evaluations = _owned_last_generation_evaluations_from_v1(
            ga_state
        )
        failure_diagnostics = _owned_failure_diagnostics_from_v1(ga_state)
        claimed_paths = tuple(ga_state.get("claimed_paths", []))
        retention_archive_mappings = _owned_retention_archive_mappings_from_v1(
            ga_state
        )
        energy_history, generation_history = _energy_and_generation_history_from_v1(
            ga_state
        )

        snapshot = GeneticAlgorithmSnapshot(
            run=run,
            rng=rng,
            completed_generation=state["progress_index"],
            best=best,
            population=population,
            population_cache=population_cache,
            energy_history=energy_history,
            generation_history=generation_history,
            retention_lineages=retention_lineages,
            last_generation_evaluations=last_generation_evaluations,
            failure_diagnostics=failure_diagnostics,
            claimed_paths=claimed_paths,
            retention_state=ga_state.get("artifact_store"),
            retention_archive_mappings=retention_archive_mappings,
        )
    except KeyError as exc:
        raise SnapshotMigrationError(
            f"checkpoint is missing required state field: {exc}"
        ) from exc
    except (TypeError, ValueError, IndexError) as exc:
        raise SnapshotMigrationError(
            f"checkpoint state is malformed: {exc}"
        ) from exc
    except SnapshotError as exc:
        raise SnapshotMigrationError(f"checkpoint state is invalid: {exc}") from exc
    return snapshot


def _owned_population_candidate_to_snapshot(
    candidate: object, lineage: object
) -> PopulationCandidateSnapshot:
    """Build one owned-mode population candidate snapshot.

    :param candidate: Raw ``{"structure_path": ..., "mapping": ...}`` snapshot entry.
    :param lineage: Raw ``[operation_label, parent]`` lineage entry.
    :return: Validated population candidate snapshot.
    :raises SnapshotMigrationError: If either argument is malformed.
    """
    if not isinstance(candidate, dict):
        raise SnapshotMigrationError(
            "checkpoint population candidate must be a mapping"
        )
    try:
        structure_path = candidate["structure_path"]
        mapping_state = candidate.get("mapping")
    except KeyError as exc:
        raise SnapshotMigrationError(
            f"checkpoint population candidate is malformed: {exc}"
        ) from exc
    artifact = _artifact_or_none(structure_path)
    if artifact is None:
        raise SnapshotMigrationError(
            "checkpoint population candidate lacks a structure path"
        )
    mapping = (
        None
        if mapping_state is None
        else _mapping_from_v1_state(structure_path, mapping_state)
    )
    try:
        return PopulationCandidateSnapshot(
            artifact=artifact,
            lineage=_lineage_step_from_v1(lineage),
            mapping=mapping,
        )
    except SnapshotError as exc:
        raise SnapshotMigrationError(
            f"checkpoint population candidate is invalid: {exc}"
        ) from exc


def _mapping_from_v1_state(identity: str, mapping_state: object):
    """Reconstruct and validate one checkpointed candidate/file ownership mapping.

    Reuses ``GBOpt.optimization.types``'s existing schema-v1 mapping (de)serialization
    helper rather than duplicating its field-by-field reconstruction.

    :param identity: Candidate identity or path, used only in diagnostics.
    :param mapping_state: Raw serialized mapping state.
    :return: Validated ``CandidateFileMapping``.
    :raises SnapshotMigrationError: If ``mapping_state`` is malformed.
    """
    try:
        return _candidate_mapping_from_state(mapping_state)
    except GrainOwnershipError as exc:
        raise SnapshotMigrationError(
            f"checkpoint ownership mapping for {identity!r} is invalid: {exc}"
        ) from exc


def migrate_monte_carlo_checkpoint(
    path: str | Path, fmt: str = "json"
) -> MonteCarloSnapshot:
    """Migrate one on-disk schema-v1 Monte Carlo checkpoint into a schema-v2 snapshot.

    :param path: Checkpoint file path.
    :param fmt: Optional, defaults to ``"json"``. Serialization format -- ``"json"`` or
        ``"pickle"``.
    :return: Validated Monte Carlo restart snapshot.
    :raises SnapshotMigrationError: If the file cannot be found or parsed, is not a
        schema-v1 ``MonteCarloMinimizer`` checkpoint, or is missing a required field or
        carries a semantically invalid reference.
    """
    state = _load_v1_state(path, fmt)
    try:
        validate_checkpoint_envelope(
            state, minimizer="MonteCarloMinimizer", progress_unit="step"
        )
    except CheckpointCompatibilityError as exc:
        raise SnapshotMigrationError(str(exc)) from exc
    return _monte_carlo_snapshot_from_v1(state)


def migrate_genetic_algorithm_checkpoint(
    path: str | Path, fmt: str = "json"
) -> GeneticAlgorithmSnapshot:
    """Migrate one on-disk schema-v1 genetic-algorithm checkpoint into a v2 snapshot.

    Handles both the legacy (non-owned) and explicit-ownership schema-v1 checkpoint
    shapes, dispatching on the same ``state["state"]["ga_mode"]`` marker
    ``GeneticAlgorithmMinimizer.run_GA`` itself already checks.

    :param path: Checkpoint file path.
    :param fmt: Optional, defaults to ``"json"``. Serialization format -- ``"json"`` or
        ``"pickle"``.
    :return: Validated genetic-algorithm restart snapshot.
    :raises SnapshotMigrationError: If the file cannot be found or parsed, is not a
        schema-v1 ``GeneticAlgorithmMinimizer`` checkpoint, or is missing a required
        field or carries a semantically invalid reference.
    """
    state = _load_v1_state(path, fmt)
    try:
        validate_checkpoint_envelope(
            state, minimizer="GeneticAlgorithmMinimizer", progress_unit="generation"
        )
    except CheckpointCompatibilityError as exc:
        raise SnapshotMigrationError(str(exc)) from exc
    ga_state = state.get("state")
    if isinstance(ga_state, dict) and ga_state.get("ga_mode") == "explicit_ownership":
        return _ga_snapshot_from_v1_owned(state)
    return _ga_snapshot_from_v1_legacy(state)


__all__ = [
    "SnapshotMigrationError",
    "migrate_monte_carlo_checkpoint",
    "migrate_genetic_algorithm_checkpoint",
]
