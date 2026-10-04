# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

import copy as copy_module
import inspect
import logging
import math
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from GBOpt import GBMaker, GBManipulator
from GBOpt._candidate_admissibility import (
    CandidateAdmissibilityError,
    validate_formula_composition,
)
from GBOpt._explicit_ownership_evaluation import (
    CandidateEvaluation,
    CandidateEvaluationSummary,
    ExplicitOwnershipEvaluator,
)
from GBOpt.artifacts.cleanup import (
    ArtifactCleanupError,
    ArtifactCleanupRequest,
    _ArtifactCleaner,
)
from GBOpt.artifacts.policy import ArtifactPolicyError, ArtifactRetentionPolicy
from GBOpt.artifacts.provenance import (
    ArtifactProvenanceError,
    _ArtifactProvenance,
)
from GBOpt.artifacts.store import ArtifactStore, ArtifactStoreError
from GBOpt.artifacts.types import (
    ArtifactPin,
    ArtifactValueError,
    CandidatePropertyContext,
)
from GBOpt.Checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CandidateCheckpoint,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointStore,
    _wrap_batch_func_with_checkpoint,
    validate_checkpoint_envelope,
)
from GBOpt.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    FailureStage,
    StructureArtifact,
    from_batch_dict,
    from_candidate_evaluation,
    from_scalar_tuple,
)
from GBOpt.FileGrainOwnership import (
    CandidateFileMapping,
    GrainOwnership,
    GrainOwnershipError,
    LammpsDataError,
)
from GBOpt.GBMaker import GBMakerError
from GBOpt.GBManipulator import (
    GBManipulatorError,
    ParentError,
)
from GBOpt.manipulation import (
    Manipulation,
    ManipulationContext,
    ManipulationLookupError,
    ManipulationRegistry,
    SliceAndMerge,
    default_registry,
)
from GBOpt.observability import (
    EventSink,
    NullEventSink,
    OptimizationAlgorithm,
    OptimizationEvent,
    OptimizationEventType,
    RunContext,
    TerminationReason,
    evaluation_event_fields,
)
from GBOpt.optimization.checkpointing import (
    _artifact_archive_root,
    _cleanup_committed_artifacts,
    _configure_artifact_runtime,
    _materialize_archive_file,
    _normalize_calculation_context_config,
    _normalize_failure_diagnostic_count,
    _prepare_archive_state,
    _register_retention_candidate,
    _run_artifact_provenance,
    _tuples_to_lists,
    _write_artifact_manifest,
)
from GBOpt.optimization.dispatch import run_legacy_compat_binary_operation
from GBOpt.optimization.mutation import Mutator
from GBOpt.optimization.types import (
    GBMinimizerError,
    GBMinimizerTypeError,
    GBMinimizerValueError,
    OperationSpec,
    _CachedEvaluation,
    _candidate_mapping_from_state,
    _candidate_mapping_to_state,
    _FailureDiagnostic,
    resolve_rng_seed,
)
from GBOpt.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    CandidateEvaluationSnapshot,
    FailureDiagnosticSnapshot,
    GenerationHistoryEntrySnapshot,
    GeneticAlgorithmConfigurationSnapshot,
    GeneticAlgorithmSnapshot,
    PopulationCandidateSnapshot,
    RngStateSnapshot,
    RunIdentitySnapshot,
    SnapshotError,
)
from GBOpt.snapshot.migration import (
    _ga_snapshot_from_v1_legacy,
    _ga_snapshot_from_v1_owned,
    _lineage_step_from_v1,
    _owned_evaluation_to_snapshot,
)

ENERGY_PENALTY: float = 1.0e30
"""Optimizer policy for ranking failed candidate evaluations."""

_OWNED_GA_CHECKPOINT_VERSION = 4

_GA_MINIMIZER_NAME = "GeneticAlgorithmMinimizer"
_GA_PROGRESS_UNIT = "generation"
_STRUCTURE_FORMAT = "lammps"

logger = logging.getLogger(__name__)


def _lineage_entry_from_snapshot(step) -> list:
    """Rebuild one raw GA lineage entry list from its typed snapshot form.

    Exact inverse of ``GBOpt.snapshot.migration._lineage_step_from_v1``: a one-parent
    step (mutation, carryover, the fixed ``"START"`` marker) becomes a 2-element list;
    a step carrying a diagnostic note becomes a 3- or 4-element list with the note
    trailing, matching whichever of GA's own established lineage-entry shapes produced
    it.

    :param step: Structured lineage step to rebuild.
    :return: Raw ``[operation_name, *parent_references, diagnostic_note?]`` list, in
        the exact shape GA's own runtime population/history bookkeeping expects.
    """
    if step.diagnostic_note is None:
        return [step.operation_name, *step.parent_references]
    return [step.operation_name, *step.parent_references, step.diagnostic_note]


def _ga_snapshot_from_checkpoint_state(state: object, *, owned: bool) -> GeneticAlgorithmSnapshot:
    """Build a validated GA snapshot from one loaded checkpoint envelope.

    Dispatches on the envelope's own ``schema_version``: a schema-v2 envelope's typed
    snapshot payload is restored directly via
    :meth:`~GBOpt.snapshot.GeneticAlgorithmSnapshot.from_state`; a schema-v1 envelope is
    migrated through the established, mode-appropriate
    :func:`GBOpt.snapshot.migration._ga_snapshot_from_v1_owned`/
    :func:`GBOpt.snapshot.migration._ga_snapshot_from_v1_legacy` reader. No other schema
    version is accepted.

    :param state: Deserialized checkpoint envelope, as returned by
        :meth:`~GBOpt.Checkpoint.CheckpointStore.load`.
    :param owned: Keyword argument, required. Whether this run is executing in
        explicit-ownership mode.
    :return: Validated genetic-algorithm restart snapshot.
    :raises GBMinimizerError: If ``state`` is not a dictionary, declares an unsupported
        schema version, or its recorded mode does not match ``owned``.
    :raises CheckpointCompatibilityError: If a schema-v1 envelope fails structural
        validation.
    :raises SnapshotError: If the envelope's typed or migrated snapshot payload is
        semantically invalid.
    """
    if not isinstance(state, dict):
        raise GBMinimizerError("checkpoint envelope must be a dictionary")
    schema_version = state.get("schema_version")
    if schema_version == SNAPSHOT_SCHEMA_VERSION:
        if state.get("minimizer") != _GA_MINIMIZER_NAME:
            raise GBMinimizerError(
                f"checkpoint was written by {state.get('minimizer')!r}, expected "
                f"{_GA_MINIMIZER_NAME!r}"
            )
        if state.get("progress_unit") != _GA_PROGRESS_UNIT:
            raise GBMinimizerError(
                f"checkpoint progress_unit {state.get('progress_unit')!r} does not "
                f"match expected {_GA_PROGRESS_UNIT!r}"
            )
        return GeneticAlgorithmSnapshot.from_state(state.get("snapshot"))
    if schema_version == CHECKPOINT_SCHEMA_VERSION:
        validate_checkpoint_envelope(
            state, minimizer=_GA_MINIMIZER_NAME, progress_unit=_GA_PROGRESS_UNIT
        )
        v1_state = state.get("state")
        is_owned_v1 = (
            isinstance(v1_state, dict)
            and v1_state.get("ga_mode") == "explicit_ownership"
        )
        if is_owned_v1 != owned:
            raise GBMinimizerError(
                "checkpoint explicit-ownership mode does not match how this "
                "minimizer is being run"
            )
        return (
            _ga_snapshot_from_v1_owned(state)
            if is_owned_v1
            else _ga_snapshot_from_v1_legacy(state)
        )
    raise GBMinimizerError(
        f"unsupported GeneticAlgorithmMinimizer checkpoint schema version "
        f"{schema_version!r}"
    )


class GeneticAlgorithmMinimizer:
    """
    Minimizer class for finding the lowest energy configuration of a grain boundary
    using a simple genetic algorithm (GA). Mirrors the interface of MonteCarloMinimizer
    while using GA operations to explore the configuration space.
    """

    def __init__(
        self,
        GB: GBMaker,
        gb_energy_func: Callable,
        choices: list,
        seed=None,
        *,
        initial_structure: GBMaker | str | Path | None = None,
        initial_ownership: GrainOwnership | None = None,
        allow_variable_cell: bool = False,
        population_size: int = 20,
        generations: int = 50,
        keep_top_pct: int = 10,
        intermediate_pct: int = 60,
        slice_and_merge_pct: float = 50.0,
        reuse_carryover_evaluations: bool = False,
        gb_batch_energy_func: Callable | None = None,
        crossover_surface: str = "periodic_wave",
        crossover_max_tilt_degrees: float = 5.0,
        crossover_attempts: int = 8,
        binary_operations: Sequence[str] = (),
        registry: ManipulationRegistry | None = None,
        retention_policy: ArtifactRetentionPolicy | None = None,
        calculation_context: Mapping[str, object] | None = None,
        failure_diagnostic_count: int = 3,
        managed_artifact_root: str | Path | None = None,
        cleanup_candidate: Callable[[ArtifactCleanupRequest], None] | None = None,
        event_sink: EventSink | None = None,
        case_id: str | None = None,
        campaign_id: str | None = None,
    ):
        """Configure one genetic-algorithm grain-boundary minimizer.

        :param GB: GBMaker object to perform minimization on.
        :param gb_energy_func: Function that returns the energy of a GB structure. It
            must be callable with (GBMaker, GBManipulator, atom_positions, unique_id).
        :param choices: List of strings corresponding to GBManipulator operations. Used
            to configure the Mutator. Any name that is not one of the three legacy
            operation names is resolved by lookup in ``registry``.
        :param seed: Seed for numpy.random.default_rng. Keyword argument, optional,
            defaults to ``None``; ``None`` seeds from the current time.
        :ivar seed: The resolved seed actually passed to ``numpy.random.default_rng``
            (the current time when the constructor's ``seed`` argument is ``None``).
        :param initial_structure: Keyword argument, optional, defaults to ``None``.
            GBMaker or file-backed initial structure.
        :param initial_ownership: Keyword argument, optional, defaults to ``None``.
            Explicit ownership aligned to atom IDs in a file-backed initial structure.
        :param allow_variable_cell: Keyword argument, optional, defaults to ``False``.
            Allow orthogonal box dimensions returned by explicit-ownership evaluators to
            evolve between GA generations. Requires ``initial_ownership``.
        :param population_size: Number of candidates per generation. Keyword argument,
            optional, defaults to 20.
        :param generations: Number of generations to iterate. Keyword argument,
            optional, defaults to 50.
        :param keep_top_pct: Percentage of lowest-energy structures carried over
            unchanged. Keyword argument, optional, defaults to 10.
        :param intermediate_pct: Percentage of structures eligible for
            crossover/mutation selection. Keyword argument, optional, defaults to 60.
        :param slice_and_merge_pct: Percentage of non-carryover offspring generated by
            slice-and-merge crossover. The remaining offspring are generated by
            mutation. Keyword argument, optional, defaults to 50.0.
        :param reuse_carryover_evaluations: Reuse the validated energy and relaxed
            artifact of unchanged successful carryover candidates instead of invoking
            the evaluator again. Keyword argument, optional, defaults to ``False``.
        :param gb_batch_energy_func: Keyword argument, optional, defaults to ``None``.
            Batch-evaluation function for processing a population in one call. It should
            accept (GBMaker, manipulators, atom_positions_list, lineages, unique_ids)
            and return a list of dictionaries containing at least ``"energy"`` and
            ``"final_dump"`` keys. If not provided, fall back to calling
            ``gb_energy_func`` per candidate. If the function does not declare a
            ``checkpoint`` keyword argument it is automatically wrapped so that
            checkpointing still occurs at batch-return granularity; a ``UserWarning`` is
            emitted in that case. Declare a ``checkpoint=None`` parameter and call
            ``checkpoint.record(unique_id, energy, dump)`` per job to get per-job
            recovery granularity.
        :param crossover_surface: Keyword argument, optional, defaults to
            ``"periodic_wave"``. Formula-preserving crossover surface mode,
            ``"normal_plane"`` or ``"periodic_wave"``.
        :param crossover_max_tilt_degrees: Keyword argument, optional, defaults to
            ``5.0``. Maximum combined local periodic-wave tilt in degrees.
        :param crossover_attempts: Keyword argument, optional, defaults to ``8``.
            Maximum parent-pair attempts before one crossover slot falls back to
            mutation.
        :param binary_operations: Keyword argument, optional, defaults to ``()``.
            Additional two-parent operation names, resolved by lookup in ``registry``,
            added to the crossover pool alongside the always-present
            ``slice_and_merge``. Empty by default, matching pre-OperationSpec behavior
            exactly.
        :param registry: Keyword argument, optional, defaults to ``None``. Registry used
            to resolve any ``choices``/``binary_operations`` name that is not one of the
            three legacy unary names; ``None`` uses
            ``GBOpt.manipulation.default_registry``.
        :param retention_policy: Keyword argument, optional, defaults to ``None``.
            Scientific artifact-retention policy for explicit-ownership GA execution.
            ``None`` preserves keep-all artifact behavior.
        :param calculation_context: Keyword argument, optional, defaults to ``None``.
            JSON-safe run-level calculator/campaign provenance. A non-empty mapping is
            required when pruning is enabled.
        :param failure_diagnostic_count: Keyword argument, optional, defaults to ``3``.
            Maximum number of most-recent failed evaluator sources preserved for
            diagnostics when pruning is enabled.
        :param managed_artifact_root: Keyword argument, optional, defaults to ``None``.
            Root beneath which GBOpt may remove evaluator-returned source paths after a
            durable checkpoint commit. Mutually exclusive with ``cleanup_candidate``.
        :param cleanup_candidate: Keyword argument, optional, defaults to ``None``.
            Backend-owned callback invoked after a durable checkpoint commit for each
            evaluator source that has become transient. Mutually exclusive with
            ``managed_artifact_root``.
        :param event_sink: Keyword argument, optional, defaults to ``None``. Destination
            for this run's versioned lifecycle events. ``None`` uses ``NullEventSink``,
            so a run is silent unless a caller opts in. A sink whose ``emit()`` raises is
            logged and otherwise ignored -- event emission never aborts or otherwise
            alters the run.
        :param case_id: Keyword argument, optional, defaults to ``None``. Caller-supplied
            scientific case identity stamped on every emitted event's ``RunContext``.
        :param campaign_id: Keyword argument, optional, defaults to ``None``.
            Caller-supplied campaign identity grouping several runs, stamped on every
            emitted event's ``RunContext``.
        :raises TypeError: If ``seed`` is neither ``None`` nor an ``int``,
            ``initial_ownership`` is not GrainOwnership, accompanies a non-file initial
            structure, ``allow_variable_cell`` is not Boolean, a crossover/cleanup
            policy argument has an invalid type, ``retention_policy`` is not an
            ``ArtifactRetentionPolicy``, or ``event_sink`` is neither ``None`` nor an
            ``EventSink``.
        :raises ValueError: If ownership is supplied without an initial structure,
            variable-cell execution is requested without explicit ownership, cleanup
            ownership is ambiguous, or pruning lacks an explicit cleanup owner.
        """
        if event_sink is not None and not isinstance(event_sink, EventSink):
            raise GBMinimizerTypeError("event_sink must be an EventSink or None")
        self._event_sink: EventSink = (
            NullEventSink() if event_sink is None else event_sink
        )
        self.case_id = case_id
        self.campaign_id = campaign_id
        if not isinstance(allow_variable_cell, (bool, np.bool_)):
            raise TypeError("allow_variable_cell must be a Boolean")
        allow_variable_cell = bool(allow_variable_cell)
        if initial_ownership is not None:
            if not isinstance(initial_ownership, GrainOwnership):
                raise TypeError("initial_ownership must be a GrainOwnership instance")
            if initial_structure is None:
                raise ValueError("initial_ownership requires an initial_structure")
            if not isinstance(initial_structure, (str, Path)):
                raise TypeError(
                    "initial_ownership requires a str or Path initial_structure"
                )
        elif allow_variable_cell:
            raise ValueError("allow_variable_cell requires initial_ownership")
        artifact_cleaner, artifact_store = _configure_artifact_runtime(
            retention_policy,
            managed_artifact_root,
            cleanup_candidate,
        )
        calculation_context = _normalize_calculation_context_config(
            calculation_context,
            retention_policy=retention_policy,
        )
        failure_diagnostic_count = _normalize_failure_diagnostic_count(
            failure_diagnostic_count
        )
        if retention_policy is not None and initial_ownership is None:
            raise GBMinimizerValueError(
                "retention_policy currently requires explicit ownership"
            )
        if (
            isinstance(slice_and_merge_pct, (bool, np.bool_))
            or not isinstance(slice_and_merge_pct, Real)
        ):
            raise GBMinimizerTypeError(
                "slice_and_merge_pct must be a real number"
            )
        slice_and_merge_pct = float(slice_and_merge_pct)
        if not math.isfinite(slice_and_merge_pct) or not (
            0.0 <= slice_and_merge_pct <= 100.0
        ):
            raise GBMinimizerValueError(
                "slice_and_merge_pct must be finite and between 0 and 100"
            )
        if not isinstance(reuse_carryover_evaluations, (bool, np.bool_)):
            raise GBMinimizerTypeError(
                "reuse_carryover_evaluations must be a Boolean"
            )
        if not isinstance(crossover_surface, str):
            raise GBMinimizerTypeError("crossover_surface must be a string")
        if crossover_surface not in {"normal_plane", "periodic_wave"}:
            raise GBMinimizerValueError(
                "crossover_surface must be 'normal_plane' or 'periodic_wave'"
            )
        if (
            isinstance(crossover_max_tilt_degrees, (bool, np.bool_))
            or not isinstance(crossover_max_tilt_degrees, Real)
        ):
            raise GBMinimizerTypeError(
                "crossover_max_tilt_degrees must be a non-Boolean real scalar"
            )
        if (
            not np.isfinite(crossover_max_tilt_degrees)
            or float(crossover_max_tilt_degrees) < 0.0
            or float(crossover_max_tilt_degrees) >= 90.0
        ):
            raise GBMinimizerValueError(
                "crossover_max_tilt_degrees must be finite and satisfy 0 <= value < 90"
            )
        if (
            isinstance(crossover_attempts, (bool, np.bool_))
            or not isinstance(crossover_attempts, Integral)
        ):
            raise GBMinimizerTypeError(
                "crossover_attempts must be a non-Boolean integer"
            )
        if int(crossover_attempts) <= 0:
            raise GBMinimizerValueError(
                "crossover_attempts must be a positive integer"
            )
        self.GB: GBMaker = GB
        self.gb_energy_func: Callable = gb_energy_func
        if gb_batch_energy_func is not None:
            try:
                sig = inspect.signature(gb_batch_energy_func)
                if "checkpoint" not in sig.parameters:
                    warnings.warn(
                        "gb_batch_energy_func does not accept a 'checkpoint' kwarg. "
                        "It has been automatically wrapped so checkpointing occurs at "
                        "batch-return granularity. For per-job recovery, add "
                        "'checkpoint=None' to your batch function signature and call "
                        "checkpoint.record(unique_id, energy, dump) as each job completes.",
                        UserWarning,
                        stacklevel=2,
                    )
                    gb_batch_energy_func = _wrap_batch_func_with_checkpoint(
                        gb_batch_energy_func, penalty=ENERGY_PENALTY
                    )
            except ValueError:
                # C callables have no inspectable signature — wrap at batch-return granularity.
                warnings.warn(
                    "gb_batch_energy_func signature could not be inspected. "
                    "It has been automatically wrapped so checkpointing occurs at "
                    "batch-return granularity.",
                    UserWarning,
                    stacklevel=2,
                )
                gb_batch_energy_func = _wrap_batch_func_with_checkpoint(
                    gb_batch_energy_func, penalty=ENERGY_PENALTY
                )
            except TypeError as exc:
                raise GBMinimizerTypeError(
                    "gb_batch_energy_func must be callable."
                ) from exc
        self.gb_batch_energy_func: Callable | None = gb_batch_energy_func
        self.history: list = []
        self.initial_structure: GBMaker | str | Path | None = initial_structure
        self.initial_ownership: GrainOwnership | None = initial_ownership
        self.allow_variable_cell: bool = allow_variable_cell
        self.retention_policy: ArtifactRetentionPolicy | None = retention_policy
        self.calculation_context: dict[str, object] | None = calculation_context
        self.failure_diagnostic_count: int = failure_diagnostic_count
        self._failure_diagnostics: list[_FailureDiagnostic] = []
        self._artifact_cleaner: _ArtifactCleaner = artifact_cleaner
        self.artifact_store: ArtifactStore | None = artifact_store
        self._retention_archive_mappings: dict[str, dict] = {}
        self._artifact_provenance: _ArtifactProvenance | None = None
        self.seed: int = resolve_rng_seed(seed)
        self.local_random: np.random.Generator = np.random.default_rng(self.seed)
        self._owned_evaluator: ExplicitOwnershipEvaluator | None = (
            ExplicitOwnershipEvaluator(
                GB=GB,
                scalar_energy_func=gb_energy_func,
                batch_energy_func=gb_batch_energy_func,
                local_random=self.local_random,
                penalty=ENERGY_PENALTY,
                allow_variable_cell=allow_variable_cell,
            )
            if initial_ownership is not None
            else None
        )
        self.manipulator: GBManipulator = self._make_initial_manipulator()
        initial_parent = self.manipulator.parents[0]
        try:
            validate_formula_composition(
                initial_parent.whole_system,
                initial_parent.unit_cell,
            )
        except CandidateAdmissibilityError as exc:
            raise GBMinimizerValueError(
                f"initial candidate composition is inadmissible: {exc}"
            ) from exc
        self.composition_policy: tuple[tuple[str, int], ...] = tuple(
            initial_parent.unit_cell.formula_ratio
        )
        self._registry: ManipulationRegistry = (
            default_registry if registry is None else registry
        )
        self.mutator: Mutator = Mutator(choices, self.manipulator, registry=self._registry)
        self.manipulator.rng = self.local_random
        self.population_size: int = population_size
        self.generations: int = generations
        self.keep_top_pct: int = keep_top_pct
        self.intermediate_pct: int = intermediate_pct
        self.slice_and_merge_pct: float = slice_and_merge_pct
        self.reuse_carryover_evaluations: bool = bool(reuse_carryover_evaluations)
        self.crossover_surface: str = crossover_surface
        self.crossover_max_tilt_degrees: float = float(crossover_max_tilt_degrees)
        self.crossover_attempts: int = int(crossover_attempts)
        self.binary_operations: tuple[str, ...] = tuple(dict.fromkeys(binary_operations))
        for name in self.binary_operations:
            try:
                operation = self._registry.get(name)
            except ManipulationLookupError as exc:
                raise GBMinimizerValueError(
                    f"Unknown binary_operations entry: {name!r}"
                ) from exc
            if operation.arity != 2:
                raise GBMinimizerValueError(
                    f"binary_operations entry {name!r} resolves to an operation with "
                    f"arity {operation.arity}; only two-parent (arity 2) operations "
                    "are usable as a binary_operations entry"
                )
        self.GBE_vals: list[list[float]] = []

    def _make_initial_manipulator(self) -> GBManipulator:
        seed = self.initial_structure
        if seed is None:
            manip = GBManipulator(self.GB)
        elif isinstance(seed, GBMaker):
            manip = GBManipulator(seed)
        else:
            manip = GBManipulator(
                str(seed),
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
                grain_ownership=self.initial_ownership,
            )

        manip.rng = self.local_random

        return manip

    def _make_manipulator_from_file(self, filename: str) -> GBManipulator:
        if self.initial_ownership is not None:
            raise RuntimeError(
                "explicit-ownership file reloads must use reload_explicit_manipulator"
            )
        manipulator = GBManipulator(
            filename,
            unit_cell=self.GB.unit_cell,
            gb_thickness=self.GB.gb_thickness,
        )
        manipulator.rng = self.local_random
        return manipulator

    def _clone_owned_record(self, record: CandidateEvaluation) -> GBManipulator:
        """Clone a successfully reconstructed owned candidate.

        :param record: Successful explicit-ownership candidate evaluation.
        :return: Independent manipulator carrying the validated candidate state.
        :raises ValueError: If the evaluation did not produce a reusable candidate.
        """
        if (
            not record.success
            or record.manipulator is None
            or record.structure_path is None
        ):
            raise ValueError("cannot clone a failed candidate evaluation")
        manipulator = copy_module.copy(record.manipulator)
        manipulator.rng = self.local_random
        return manipulator

    def _binary_operation_specs(self) -> list[OperationSpec]:
        """Return this run's two-parent operation pool, weighted for selection.

        ``slice_and_merge`` is always present, configurable through the legacy
        ``slice_and_merge_pct``/``crossover_surface``/``crossover_max_tilt_degrees``
        constructor arguments; any ``binary_operations`` entries are additional pool
        members, resolved by registry lookup. With no ``binary_operations`` (the
        default), this pool always has exactly one member, and the generic
        weighted-selection machinery (``GBOpt.optimization.dispatch``) still runs over
        it, which is a documented no-op on RNG state for a single-member pool.

        :return: The ``slice_and_merge`` spec, followed by any configured
            ``binary_operations`` specs.
        """
        specs = [
            OperationSpec(name="slice_and_merge", operation=SliceAndMerge(), weight=1.0)
        ]
        for name in self.binary_operations:
            specs.append(
                OperationSpec(name=name, operation=self._registry.get(name), weight=1.0)
            )
        return specs

    def _owned_slice_and_merge_invoker(self):
        """Return a legacy binary invoker producing an owned-mode manipulator."""

        def _invoke(parent1, parent2, rng):
            new_manipulator = GBManipulator._from_parents(parent1, parent2, rng=rng)
            new_structure = new_manipulator.slice_and_merge(
                surface_mode=self.crossover_surface,
                max_tilt_degrees=self.crossover_max_tilt_degrees,
            )
            crossover_parameters = {
                "surface_mode": self.crossover_surface,
                "max_tilt_degrees": self.crossover_max_tilt_degrees,
            }
            return "slice_and_merge", new_manipulator, new_structure, crossover_parameters

        return _invoke

    def _legacy_slice_and_merge_invoker(self):
        """Return a legacy binary invoker producing a file-backed manipulator."""

        def _invoke(parent1, parent2, rng):
            new_manipulator = GBManipulator(
                parent1,
                parent2,
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
            )
            new_manipulator.rng = rng
            new_structure = new_manipulator.slice_and_merge(
                surface_mode=self.crossover_surface,
                max_tilt_degrees=self.crossover_max_tilt_degrees,
            )
            crossover_parameters = {
                "surface_mode": self.crossover_surface,
                "max_tilt_degrees": self.crossover_max_tilt_degrees,
            }
            return "slice_and_merge", new_manipulator, new_structure, crossover_parameters

        return _invoke

    def _run_generic_binary_operation(
        self, operation: Manipulation, candidate1, candidate2, rng
    ) -> tuple[str, GBManipulator, np.ndarray, Mapping[str, object] | None]:
        """Run a registry-resolved two-parent operation through ``ManipulationContext``.

        Shared by the owned and legacy binary invokers below; only how each builds its
        two ``InterfaceCandidate`` parents differs. Has no legacy tolerance to preserve
        (this seam only ever runs a ``binary_operations`` entry, never
        ``slice_and_merge``), so the stricter, uniform ``ManipulationContext`` boundary
        is safe here, matching the unary registry-resolved case in
        ``GBOpt.optimization.mutation``.

        :param operation: Two-parent operation to run.
        :param candidate1: First parent candidate.
        :param candidate2: Second parent candidate.
        :param rng: Random-number generator to draw from.
        :return: The operation's name, a manipulator wrapping its single output child,
            that child's atom positions, and a JSON-safe mapping of the concrete
            parameter values used (or ``None``).
        :raises GBMinimizerValueError: If the operation produces other than one child.
        """
        context = ManipulationContext(
            parents=(candidate1, candidate2), rng=rng, params={}
        )
        result = operation.execute(context)
        if len(result.children) != 1:
            raise GBMinimizerValueError(
                f"operation {operation.name!r} produced {len(result.children)} "
                "children; GA dispatch requires exactly one"
            )
        (child,) = result.children
        new_manipulator = GBManipulator._from_interface_candidate(
            child,
            unit_cell=self.GB.unit_cell,
            gb_thickness=self.GB.gb_thickness,
            rng=rng,
        )
        return (
            operation.name,
            new_manipulator,
            np.array(child.atoms, copy=True),
            dict(result.parameters) if result.parameters else None,
        )

    def _owned_generic_binary_invoker(self, operation: Manipulation):
        """Return a binary invoker running ``operation`` against two ``Parent``s."""

        def _invoke(parent1, parent2, rng):
            candidate1 = GBManipulator._from_parents(parent1, rng=rng).make_parent_candidate()
            candidate2 = GBManipulator._from_parents(parent2, rng=rng).make_parent_candidate()
            return self._run_generic_binary_operation(
                operation, candidate1, candidate2, rng
            )

        return _invoke

    def _legacy_generic_binary_invoker(self, operation: Manipulation):
        """Return a binary invoker running ``operation`` against two parent files."""

        def _invoke(parent1, parent2, rng):
            candidate1 = GBManipulator(
                parent1, unit_cell=self.GB.unit_cell, gb_thickness=self.GB.gb_thickness
            ).make_parent_candidate()
            candidate2 = GBManipulator(
                parent2, unit_cell=self.GB.unit_cell, gb_thickness=self.GB.gb_thickness
            ).make_parent_candidate()
            return self._run_generic_binary_operation(
                operation, candidate1, candidate2, rng
            )

        return _invoke

    def _run_artifact_provenance(self, action: Callable[[], None]) -> None:
        """Run one non-authoritative provenance write with warning-only failure policy.

        Provenance is observability rather than restart state. A write failure therefore
        must not invalidate an otherwise valid optimizer transition or checkpoint.

        :param action: Zero-argument provenance operation to execute.
        """
        _run_artifact_provenance(self._artifact_provenance, action)

    def _register_owned_retention_candidate(
        self,
        record: CandidateEvaluation,
        *,
        generation: int,
        lineage: tuple[str, ...],
    ) -> None:
        """Register one newly evaluated relaxed candidate with the artifact subsystem.

        Property acquisition runs only after explicit-ownership reconstruction succeeds,
        so callbacks receive validated relaxed physical state rather than submitted
        input. Provenance records the successful evaluation, normalized properties, and
        any scientific-retention membership deltas caused by the new candidate.

        :param record: Successful newly evaluated candidate.
        :param generation: Keyword argument, required. Generation where evaluation
            occurred.
        :param lineage: Keyword argument, required. Stable logical parent identities.
        :raises GBMinimizerError: If candidate physical state or retention policy
            evaluation is invalid.
        """
        if self.artifact_store is None or self.retention_policy is None:
            return
        if record.candidate_id in self.artifact_store:
            return
        if (
            not record.success
            or record.manipulator is None
            or record.structure_path is None
        ):
            raise GBMinimizerError(
                "only successful explicit-ownership evaluations may enter retention"
            )
        parent = record.manipulator.parents[0]
        try:
            context = CandidatePropertyContext(
                candidate_id=record.candidate_id,
                generation=generation,
                objective=record.objective,
                atoms=parent.whole_system,
                box_dims=parent.box_dims,
                grain_labels=parent.grain_labels,
                gb_plane_x=parent.gb_plane_x,
            )
            _register_retention_candidate(
                artifact_store=self.artifact_store,
                retention_policy=self.retention_policy,
                context=context,
                source_path=record.structure_path,
                lineage=lineage,
                provenance=self._artifact_provenance,
            )
        except (ArtifactPolicyError, ArtifactStoreError, ArtifactValueError) as exc:
            raise GBMinimizerError(
                f"artifact retention failed for candidate {record.candidate_id!r}: "
                f"{exc}"
            ) from exc

    @staticmethod
    def _owned_archive_root(checkpoint_file: Path | None, unique_id: str) -> Path:
        """Return the canonical archive root for one GA run.

        :param checkpoint_file: Run checkpoint path, or ``None`` when checkpointing is
            disabled.
        :param unique_id: Stable run identifier.
        :return: Run-specific artifact archive root.
        """
        return _artifact_archive_root(
            checkpoint_file,
            fallback_stem=f"GA_{unique_id}",
        )

    def _materialize_owned_archive(
        self,
        record: CandidateEvaluation,
        archive_root: Path,
    ) -> str:
        """Create one canonical retained structure without changing candidate identity.

        The source representation is already validated by the explicit-ownership
        evaluator. A hard link is preferred and an ordinary copy is used when linking is
        unavailable. Explicit reconstruction metadata remains checkpoint state rather
        than being inferred from the archived coordinates.

        :param record: Successful candidate whose structure must be retained.
        :param archive_root: Run-owned archive root.
        :return: Canonical retained structure path.
        :raises GBMinimizerError: If the candidate or filesystem state cannot be
            archived safely.
        """
        if self.artifact_store is None or self._owned_evaluator is None:
            raise GBMinimizerError(
                "owned archive materialization requires artifact state")
        if (
            not record.success
            or record.structure_path is None
            or record.mapping is None
            or record.manipulator is None
        ):
            raise GBMinimizerError("cannot archive an incomplete owned evaluation")
        candidate_id = record.candidate_id
        if Path(candidate_id).name != candidate_id or any(
            separator in candidate_id for separator in ("/", "\\")
        ):
            raise GBMinimizerError("candidate identity is unsafe for archive naming")
        source = Path(record.structure_path)
        destination = archive_root / "structures" / f"{candidate_id}.data"
        try:
            _materialize_archive_file(source, destination)
            self._owned_evaluator._reload_mapping(str(destination), record.mapping)
        except (OSError, LammpsDataError, GrainOwnershipError) as exc:
            raise GBMinimizerError(
                f"could not materialize retained candidate {candidate_id!r}"
            ) from exc
        self.artifact_store.set_archive_path(candidate_id, destination)
        self._retention_archive_mappings[candidate_id] = _candidate_mapping_to_state(
            record.mapping
        )
        _run_artifact_provenance(
            self._artifact_provenance,
            lambda: self._artifact_provenance.record_archive_created(
                candidate_id, destination
            ),
        )
        return str(destination)

    @staticmethod
    def _rebase_owned_evaluation(
        record: CandidateEvaluation,
        *,
        structure_path: str,
        mapping: CandidateFileMapping,
        manipulator: GBManipulator,
    ) -> CandidateEvaluation:
        """Return one successful evaluation rebased onto an equivalent durable artifact.

        :param record: Successful evaluation whose identity and objective are preserved.
        :param structure_path: Keyword argument, required. Equivalent durable structure.
        :param mapping: Keyword argument, required. Explicit reconstruction mapping for
            the durable structure.
        :param manipulator: Keyword argument, required. Aligned in-memory candidate.
        :return: Rebased successful evaluation.
        :raises TypeError: If the durable structure path has an invalid type.
        :raises ValueError: If ``record`` is not successful or durable reconstruction
            state is incomplete.
        """
        if not record.success:
            raise ValueError("cannot rebase a failed owned evaluation")
        return CandidateEvaluation(
            candidate_id=record.candidate_id,
            input_index=record.input_index,
            objective=record.objective,
            structure_path=structure_path,
            mapping=mapping,
            manipulator=manipulator,
            success=True,
        )

    def _rebase_owned_carryover_cache(
        self,
        cached_evaluations: list[CandidateEvaluation | None],
        snapshots: list[dict],
        manipulators: list[GBManipulator],
    ) -> list[CandidateEvaluation | None]:
        """Rebase reusable carryover evaluations onto next-population snapshots.

        :param cached_evaluations: Carryover cache aligned to the next population.
        :param snapshots: Newly written ``.owned.pending`` population state.
        :param manipulators: Next-population manipulators aligned to ``snapshots``.
        :return: Cache entries that no longer depend on evaluator source artifacts.
        :raises GBMinimizerError: If checkpoint population state is malformed.
        """
        if not (
            len(cached_evaluations) == len(snapshots) == len(manipulators)
        ):
            raise GBMinimizerError("owned carryover cache lost population alignment")
        rebased: list[CandidateEvaluation | None] = []
        for cached, snapshot, manipulator in zip(
            cached_evaluations, snapshots, manipulators, strict=True
        ):
            if cached is None:
                rebased.append(None)
                continue
            try:
                path = snapshot["structure_path"]
                mapping = _candidate_mapping_from_state(snapshot["mapping"])
                rebased_record = self._rebase_owned_evaluation(
                    cached,
                    structure_path=path,
                    mapping=mapping,
                    manipulator=manipulator,
                )
            except (KeyError, TypeError, ValueError, GrainOwnershipError) as exc:
                raise GBMinimizerError(
                    "owned carryover cache cannot be rebased onto checkpoint population"
                ) from exc
            rebased.append(rebased_record)
        return rebased

    def _prepare_owned_archive_state(
        self,
        records_by_id: dict[str, CandidateEvaluation],
        archive_root: Path,
    ) -> list[tuple[str, str]]:
        """Materialize required owned archives and detach eligible evictions.

        :param records_by_id: Successful live evaluations available for new archive
            copies.
        :param archive_root: Run-owned canonical archive root.
        :return: Candidate IDs and archive paths eligible for post-commit deletion.
        :raises GBMinimizerError: If a required candidate lacks materializable state.
        """
        def _materialize(candidate_id: str) -> None:
            """Materialize one required owned candidate archive.

            :param candidate_id: Stable logical candidate identity.
            :raises GBMinimizerError: If no live evaluation can materialize the archive.
            """
            record = records_by_id.get(candidate_id)
            if record is None:
                raise GBMinimizerError(
                    f"retained candidate {candidate_id!r} lacks materializable state"
                )
            self._materialize_owned_archive(record, archive_root)

        return _prepare_archive_state(
            self.artifact_store,
            _materialize,
            archive_detached=lambda candidate_id: (
                self._retention_archive_mappings.pop(candidate_id, None)
            ),
        )

    def _failure_diagnostic_states(self) -> tuple[dict[str, object], ...]:
        """Return current bounded failure diagnostics in deterministic order."""
        return tuple(
            diagnostic.to_state()
            for diagnostic in sorted(
                self._failure_diagnostics,
                key=lambda item: (
                    item.generation,
                    item.input_index,
                    item.candidate_id,
                ),
            )
        )

    def _record_failure_provenance(
        self,
        diagnostic: _FailureDiagnostic,
    ) -> bool:
        """Persist one failed-evaluation event without changing optimizer state.

        :param diagnostic: Failed-evaluation metadata to record.
        :return: Whether required failure provenance was persisted successfully.
        """
        return _run_artifact_provenance(
            self._artifact_provenance,
            lambda: self._artifact_provenance.record_evaluation_failed(
                diagnostic.candidate_id,
                diagnostic.generation,
                diagnostic.failure_reason,
                diagnostic_path=diagnostic.source_path,
                metadata={"input_index": diagnostic.input_index},
            ),
        )

    def _update_failure_diagnostics(
        self,
        pending: list[_FailureDiagnostic],
    ) -> list[_FailureDiagnostic]:
        """Apply the most-recent-N failure diagnostic bound before checkpointing.

        :param pending: Failed evaluator-source diagnostics accumulated since the last
            durable generation boundary.
        :return: Diagnostics detached from checkpoint state and eligible for post-commit
            cleanup if their required provenance is durable.
        """
        by_id = {
            diagnostic.candidate_id: diagnostic
            for diagnostic in (*self._failure_diagnostics, *pending)
        }
        ordered = sorted(
            by_id.values(),
            key=lambda item: (
                item.generation,
                item.input_index,
                item.candidate_id,
            ),
        )
        if self.failure_diagnostic_count == 0:
            retained: list[_FailureDiagnostic] = []
            evicted = ordered
        else:
            retained = ordered[-self.failure_diagnostic_count:]
            retained_ids = {item.candidate_id for item in retained}
            evicted = [
                item for item in ordered if item.candidate_id not in retained_ids
            ]
        self._failure_diagnostics = retained
        return evicted

    def _cleanup_failure_diagnostics(
        self,
        diagnostics: list[_FailureDiagnostic],
    ) -> None:
        """Best-effort cleanup of detached failed evaluator sources after commit.

        A failed source is removed only after its lightweight failure event is durable.
        Paths still referenced by successful artifact-store records or by the current
        bounded diagnostic set are protected even if an evaluator reused a path.

        :param diagnostics: Failed evaluator diagnostics detached before checkpoint
            commit.
        """
        protected_paths = {
            diagnostic.source_path
            for diagnostic in self._failure_diagnostics
            if diagnostic.source_path is not None
        }
        if self.artifact_store is not None:
            try:
                protected_paths.update(
                    artifact.source_path
                    for artifact in self.artifact_store.records()
                    if artifact.source_path is not None
                )
            except ArtifactStoreError as exc:
                warnings.warn(
                    (
                        "Failure diagnostic cleanup state could not be inspected: "
                        f"{exc}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                return

        for diagnostic in diagnostics:
            source_path = diagnostic.source_path
            if source_path is None or source_path in protected_paths:
                continue
            if not self._record_failure_provenance(diagnostic):
                warnings.warn(
                    (
                        "Artifact cleanup deferred because required failed-evaluation "
                        "provenance could not be persisted for "
                        f"{diagnostic.candidate_id!r}"
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            try:
                self._artifact_cleaner.cleanup_source(
                    ArtifactCleanupRequest(
                        candidate_id=diagnostic.candidate_id,
                        source_path=Path(source_path),
                    )
                )
                _run_artifact_provenance(
                    self._artifact_provenance,
                    lambda diagnostic=diagnostic: (
                        self._artifact_provenance.record_failure_diagnostic_pruned(
                            diagnostic.candidate_id,
                            diagnostic.source_path,
                        )
                    ),
                )
            except ArtifactCleanupError as exc:
                _run_artifact_provenance(
                    self._artifact_provenance,
                    lambda diagnostic=diagnostic, exc=exc: (
                        self._artifact_provenance.record_cleanup_failed(
                            "failure_diagnostic_prune",
                            diagnostic.source_path,
                            str(exc),
                            candidate_id=diagnostic.candidate_id,
                        )
                    ),
                )
                warnings.warn(
                    "Artifact cleanup failed for failed candidate "
                    f"{diagnostic.candidate_id!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _write_owned_artifact_manifest(self) -> bool:
        """Persist current owned-artifact state and required run provenance.

        :return: Whether the manifest was persisted successfully.
        """
        return _write_artifact_manifest(
            self.artifact_store,
            self._artifact_provenance,
            ownership_metadata=self._retention_archive_mappings,
            failure_diagnostics=self._failure_diagnostic_states(),
        )

    def _make_next_owned_generation(
        self,
        records: list[CandidateEvaluation],
        intermediate_indices: list[int],
        offspring_count: int,
    ) -> tuple[
        list[GBManipulator],
        list[np.ndarray],
        list[list[str]],
        list[Mapping[str, object] | None],
    ]:
        """Create exactly the requested number of ownership-aware offspring.

        :param records: Successful evaluations eligible for breeding.
        :param intermediate_indices: Indices eligible to become parents.
        :param offspring_count: Number of unfilled population slots.
        :return: Aligned manipulators, atom arrays, lineages, and each offspring's
            operation parameters (or ``None``) -- the last is never checkpointed and is
            only available for the generation that just produced it.
        :raises ValueError: If records are empty or ``offspring_count`` is invalid.
        """
        if not records:
            raise ValueError("no valid candidate records provided for breeding")
        if (
            isinstance(offspring_count, (bool, np.bool_))
            or not isinstance(offspring_count, Integral)
            or offspring_count < 0
        ):
            raise ValueError("offspring_count must be a nonnegative integer")
        offspring_count = int(offspring_count)
        if offspring_count == 0:
            return [], [], [], []
        if not intermediate_indices:
            intermediate_indices = list(range(len(records)))

        manipulators: list[GBManipulator] = []
        candidates: list[np.ndarray] = []
        lineages: list[list[str]] = []
        operation_parameters: list[Mapping[str, object] | None] = []
        n_slice = math.floor(
            offspring_count * self.slice_and_merge_pct / 100.0
        )
        n_mutate = offspring_count - n_slice

        binary_specs = self._binary_operation_specs()
        legacy_binary_invokers = {
            "slice_and_merge": self._owned_slice_and_merge_invoker(),
        }
        for name in self.binary_operations:
            legacy_binary_invokers[name] = self._owned_generic_binary_invoker(
                self._registry.get(name)
            )
        for _ in range(n_slice):
            inadmissible_attempts = 0
            record1 = records[intermediate_indices[0]]
            crossed = False
            for _attempt in range(self.crossover_attempts):
                replace = len(intermediate_indices) < 2
                idx_1, idx_2 = self.local_random.choice(
                    intermediate_indices,
                    size=2,
                    replace=replace,
                )
                record1 = records[int(idx_1)]
                record2 = records[int(idx_2)]
                parent1 = self._clone_owned_record(record1).parents[0]
                parent2 = self._clone_owned_record(record2).parents[0]
                outcome = run_legacy_compat_binary_operation(
                    binary_specs,
                    rng=self.local_random,
                    parent1=parent1,
                    parent2=parent2,
                    legacy_invokers=legacy_binary_invokers,
                )
                if outcome is None:
                    inadmissible_attempts += 1
                    continue
                label, new_manipulator, new_structure, crossover_parameters = outcome
                provenance = dict(new_manipulator.last_crossover_provenance or ())
                manipulators.append(new_manipulator)
                candidates.append(new_structure)
                lineages.append(
                    [
                        label,
                        str(record1.structure_path),
                        str(record2.structure_path),
                        repr(provenance),
                    ]
                )
                operation_parameters.append(crossover_parameters)
                crossed = True
                break
            if crossed:
                continue
            fallback = self._clone_owned_record(record1)
            mutation, new_structure, mutation_parameters = self.mutator.mutate(
                local_random=self.local_random,
                GB=self.GB,
                manipulator=fallback,
            )
            manipulators.append(fallback)
            candidates.append(new_structure)
            lineages.append(
                [
                    "crossover_fallback_" + mutation,
                    str(record1.structure_path),
                    f"{inadmissible_attempts} inadmissible crossover attempts",
                ]
            )
            operation_parameters.append(mutation_parameters)

        if n_mutate:
            selected = self.local_random.choice(
                intermediate_indices,
                size=n_mutate,
                replace=True,
            )
            for idx in selected:
                record = records[int(idx)]
                new_manipulator = self._clone_owned_record(record)
                mutation, new_structure, mutation_parameters = self.mutator.mutate(
                    local_random=self.local_random,
                    GB=self.GB,
                    manipulator=new_manipulator,
                )
                manipulators.append(new_manipulator)
                candidates.append(new_structure)
                lineages.append([mutation, str(record.structure_path)])
                operation_parameters.append(mutation_parameters)

        return manipulators, candidates, lineages, operation_parameters

    def _select_indices_by_energy(self, energies: list) -> tuple[list[int], list[int]]:
        idx_sorted = sorted(range(len(energies)), key=lambda i: energies[i])

        n_top = max(0, (len(energies) * self.keep_top_pct) // 100)
        n_inter = max(0, (len(energies) * self.intermediate_pct) // 100)

        lowest_top = idx_sorted[:n_top]
        intermediate = idx_sorted[:n_inter]
        return lowest_top, intermediate

    def _evaluate_generation(
        self,
        population_manipulators: list[GBManipulator],
        population_structures: list[np.ndarray],
        population_lineages: list[list[str]],
        gen: int,
        unique_id: int,
        gen_checkpoint: CandidateCheckpoint | None = None,
        cached_evaluations: list[_CachedEvaluation | None] | None = None,
    ) -> tuple[
        list[float], list[str | None], list[GBManipulator | None], list[EvaluationResult]
    ]:
        """Evaluate all candidates, optionally using a batch energy function.

        Each candidate's raw callback result is classified through the same
        ``EvaluationResult`` contract regardless of source (fresh callback, batch
        callback, checkpoint restore, or carryover cache), so a candidate always
        penalizes to ``ENERGY_PENALTY`` on a missing/non-finite energy or a missing
        structure path, not just on an outright callback exception. The returned
        ``EvaluationResult`` reflects a subsequent candidate-reconstruction failure too
        (reclassified to ``FailureStage.ARTIFACT``), not just the evaluator callback's
        own raw outcome, so it always matches the aligned energy/path this method
        actually returns.

        :param gen_checkpoint: If provided, already-evaluated candidates are skipped and
            new results are recorded after each evaluation.
        :param cached_evaluations: Successful results aligned to unchanged carryover
            candidates. ``None`` entries are evaluated normally.
        :return: Aligned energies, evaluator artifact paths, manipulators, and
            authoritative per-candidate evaluation results.
        :raises ValueError: If cached results are not population-aligned.
        :raises EvaluationTypeError: If a callback result is not the documented tuple
            or dictionary shape.
        """

        population_length = len(population_structures)
        if cached_evaluations is None:
            cached_evaluations = [None] * population_length
        elif len(cached_evaluations) != population_length:
            raise ValueError("cached evaluations must remain population-aligned")

        all_uids = [
            f"GA_{unique_id}_g{gen}_c{i}"
            for i in range(len(population_structures))
        ]

        if self.gb_batch_energy_func is not None:
            batch_results: list[dict[str, object] | None] = [
                None
            ] * population_length
            # Indices whose batch call itself raised: from_batch_dict cannot see a
            # raw dict for these, so their EvaluationResult is built directly with
            # FailureStage.EVALUATOR rather than reclassified from a penalty-shaped
            # placeholder dict (which would misattribute them to FailureStage.ARTIFACT).
            callback_exception_indices: dict[int, str] = {}
            pending = []
            for index, uid in enumerate(all_uids):
                cached = cached_evaluations[index]
                if cached is not None and self._is_valid_file(
                    cached.structure_path
                ):
                    batch_results[index] = {
                        "energy": cached.energy,
                        "final_dump": cached.structure_path,
                    }
                elif gen_checkpoint is None or not gen_checkpoint.is_done(uid):
                    pending.append((index, uid))

            if gen_checkpoint is not None:
                if pending:
                    pending_idxs, pending_uids = zip(*pending)
                    pending_idxs = list(pending_idxs)
                    pending_uids = list(pending_uids)
                    try:
                        new_results = self.gb_batch_energy_func(
                            self.GB,
                            [population_manipulators[i] for i in pending_idxs],
                            [population_structures[i] for i in pending_idxs],
                            [population_lineages[i] for i in pending_idxs],
                            pending_uids,
                            checkpoint=gen_checkpoint,
                        )
                    except Exception as exc:
                        # The external batch evaluator callback is a deliberate
                        # recovery boundary: any failure here penalizes only the
                        # pending candidates in this batch rather than aborting
                        # the generation.
                        logger.warning(
                            "gb_batch_energy_func failed for candidates %r: %s: %s",
                            pending_uids,
                            type(exc).__name__,
                            exc,
                        )
                        message = f"{type(exc).__name__}: {exc}"
                        for index, uid in zip(pending_idxs, pending_uids):
                            callback_exception_indices[index] = message
                            batch_results[index] = {
                                "energy": ENERGY_PENALTY,
                                "final_dump": None,
                            }
                            if not gen_checkpoint.is_done(uid):
                                gen_checkpoint.record(uid, ENERGY_PENALTY, None)
                    else:
                        # Record any results the batch func did not record itself
                        for uid, result in zip(pending_uids, new_results):
                            if not gen_checkpoint.is_done(uid):
                                gen_checkpoint.record(
                                    uid,
                                    float(result.get("energy", ENERGY_PENALTY)),
                                    result.get("final_dump", None),
                                )
                for index, uid in enumerate(all_uids):
                    if batch_results[index] is not None:
                        continue
                    energy, final_dump = gen_checkpoint.get_result(uid)
                    batch_results[index] = {
                        "energy": energy,
                        "final_dump": final_dump,
                    }
            else:
                if pending:
                    pending_idxs, pending_uids = zip(*pending)
                    try:
                        raw_results = self.gb_batch_energy_func(
                            self.GB,
                            [population_manipulators[i] for i in pending_idxs],
                            [population_structures[i] for i in pending_idxs],
                            [population_lineages[i] for i in pending_idxs],
                            list(pending_uids),
                        )
                    except Exception as exc:
                        # The external batch evaluator callback is a deliberate
                        # recovery boundary: any failure here penalizes only the
                        # pending candidates in this batch rather than aborting
                        # the generation.
                        logger.warning(
                            "gb_batch_energy_func failed for candidates %r: %s: %s",
                            pending_uids,
                            type(exc).__name__,
                            exc,
                        )
                        message = f"{type(exc).__name__}: {exc}"
                        for index in pending_idxs:
                            callback_exception_indices[index] = message
                            batch_results[index] = {
                                "energy": ENERGY_PENALTY,
                                "final_dump": None,
                            }
                    else:
                        for index, result in zip(
                            pending_idxs,
                            raw_results,
                            strict=True,
                        ):
                            batch_results[index] = result

            gen_energies = []
            gen_files = []
            evaluated_manipulators = []
            results: list[EvaluationResult] = []
            for index, raw_result in enumerate(batch_results):
                if raw_result is None:
                    raise RuntimeError("batch evaluation lost candidate alignment")
                uid = all_uids[index]
                if index in callback_exception_indices:
                    result = EvaluationResult(
                        candidate_id=uid,
                        input_index=index,
                        status=EvaluationStatus.FAILED,
                        selection_energy=ENERGY_PENALTY,
                        failure_stage=FailureStage.EVALUATOR,
                        failure_message=callback_exception_indices[index],
                    )
                else:
                    result = from_batch_dict(
                        uid, index, raw_result, penalty=ENERGY_PENALTY
                    )
                gen_energies.append(result.selection_energy)
                dump = result.artifact.path if result.artifact is not None else None

                if self._is_valid_file(dump):
                    gen_files.append(dump)
                    try:
                        evaluated_manipulators.append(
                            self._make_manipulator_from_file(dump)
                        )
                    except Exception as exc:
                        # Reconstructing one candidate's evaluator output is a
                        # deliberate recovery boundary: any failure here penalizes
                        # only this candidate rather than aborting the generation.
                        logger.warning(
                            "Candidate reconstruction failed for %r: %s: %s",
                            dump,
                            type(exc).__name__,
                            exc,
                        )
                        gen_files[-1] = None
                        gen_energies[-1] = ENERGY_PENALTY
                        evaluated_manipulators.append(None)
                        result = EvaluationResult(
                            candidate_id=uid,
                            input_index=index,
                            status=EvaluationStatus.FAILED,
                            selection_energy=ENERGY_PENALTY,
                            failure_stage=FailureStage.ARTIFACT,
                            failure_message=f"{type(exc).__name__}: {exc}",
                        )
                else:
                    gen_files.append(None)
                    gen_energies[-1] = ENERGY_PENALTY
                    evaluated_manipulators.append(None)
                results.append(result)

            return gen_energies, gen_files, evaluated_manipulators, results

        gen_energies: list[float] = []
        gen_files: list[str | None] = []
        evaluated_manipulators: list[GBManipulator | None] = []
        results: list[EvaluationResult] = []

        for idx, (manipulator, atom_positions) in enumerate(
                zip(population_manipulators, population_structures)):
            uid = all_uids[idx]
            cached = cached_evaluations[idx]
            if cached is not None and self._is_valid_file(cached.structure_path):
                result = from_scalar_tuple(
                    uid,
                    idx,
                    (cached.energy, cached.structure_path),
                    penalty=ENERGY_PENALTY,
                )
            elif gen_checkpoint is not None and gen_checkpoint.is_done(uid):
                gbe, dump_file_name = gen_checkpoint.get_result(uid)
                result = from_scalar_tuple(
                    uid, idx, (gbe, dump_file_name), penalty=ENERGY_PENALTY
                )
            else:
                try:
                    gbe, dump_file_name = self.gb_energy_func(
                        self.GB, manipulator, atom_positions, uid)
                except Exception as exc:
                    # The external evaluator callback is a deliberate recovery
                    # boundary: any failure here penalizes only this candidate
                    # rather than aborting the generation.
                    logger.warning(
                        "gb_energy_func failed for candidate %r: %s: %s",
                        uid,
                        type(exc).__name__,
                        exc,
                    )
                    gbe, dump_file_name = ENERGY_PENALTY, None
                    result = EvaluationResult(
                        candidate_id=uid,
                        input_index=idx,
                        status=EvaluationStatus.FAILED,
                        selection_energy=ENERGY_PENALTY,
                        failure_stage=FailureStage.EVALUATOR,
                        failure_message=f"{type(exc).__name__}: {exc}",
                    )
                else:
                    result = from_scalar_tuple(
                        uid, idx, (gbe, dump_file_name), penalty=ENERGY_PENALTY
                    )
                if gen_checkpoint is not None:
                    gen_checkpoint.record(uid, gbe, dump_file_name)

            gen_energies.append(result.selection_energy)
            dump_file_name = (
                result.artifact.path if result.artifact is not None else None
            )
            if self._is_valid_file(dump_file_name):
                gen_files.append(dump_file_name)
                try:
                    evaluated_manipulators.append(
                        self._make_manipulator_from_file(dump_file_name)
                    )
                except Exception as exc:
                    # Reconstructing one candidate's evaluator output is a
                    # deliberate recovery boundary: any failure here penalizes
                    # only this candidate rather than aborting the generation.
                    logger.warning(
                        "Candidate reconstruction failed for %r: %s: %s",
                        dump_file_name,
                        type(exc).__name__,
                        exc,
                    )
                    gen_files[-1] = None
                    gen_energies[-1] = ENERGY_PENALTY
                    evaluated_manipulators.append(None)
                    result = EvaluationResult(
                        candidate_id=uid,
                        input_index=idx,
                        status=EvaluationStatus.FAILED,
                        selection_energy=ENERGY_PENALTY,
                        failure_stage=FailureStage.ARTIFACT,
                        failure_message=f"{type(exc).__name__}: {exc}",
                    )
            else:
                gen_files.append(None)
                gen_energies[-1] = ENERGY_PENALTY
                evaluated_manipulators.append(None)
            results.append(result)

        return gen_energies, gen_files, evaluated_manipulators, results

    def _make_next_generation(
        self,
        files: list[str],
        intermediate_indices: list[int],
        offspring_count: int,
    ) -> tuple[
        list[GBManipulator],
        list[np.ndarray],
        list[list[str]],
        list[Mapping[str, object] | None],
    ]:
        """Create exactly the requested number of legacy-path offspring.

        :param files: Valid evaluated structure files eligible for breeding.
        :param intermediate_indices: Indices eligible to become parents.
        :param offspring_count: Number of unfilled population slots.
        :return: Aligned manipulators, atom arrays, lineages, and each offspring's
            operation parameters (or ``None``) -- the last is never checkpointed and is
            only available for the generation that just produced it.
        :raises ValueError: If no parent files are provided or ``offspring_count`` is
            invalid.
        """
        if not files:
            raise ValueError(
                "No valid parent files provided to _make_next_generation()."
            )
        if (
            isinstance(offspring_count, (bool, np.bool_))
            or not isinstance(offspring_count, Integral)
            or offspring_count < 0
        ):
            raise ValueError("offspring_count must be a nonnegative integer")
        offspring_count = int(offspring_count)
        if offspring_count == 0:
            return [], [], [], []

        if not intermediate_indices:
            intermediate_indices = list(range(len(files)))
        candidates: list[np.ndarray] = []
        manipulators: list[GBManipulator] = []
        lineages: list[list[str]] = []
        operation_parameters: list[Mapping[str, object] | None] = []

        N_slice = math.floor(
            offspring_count * self.slice_and_merge_pct / 100.0
        )
        N_mutate = offspring_count - N_slice

        # Slice & merge
        binary_specs = self._binary_operation_specs()
        legacy_binary_invokers = {
            "slice_and_merge": self._legacy_slice_and_merge_invoker(),
        }
        for name in self.binary_operations:
            legacy_binary_invokers[name] = self._legacy_generic_binary_invoker(
                self._registry.get(name)
            )
        for _ in range(N_slice):
            p1 = files[intermediate_indices[0]]
            crossed = False
            for _attempt in range(self.crossover_attempts):
                replace = len(intermediate_indices) < 2
                idx_1, idx_2 = self.local_random.choice(
                    intermediate_indices,
                    size=2,
                    replace=replace,
                )
                p1, p2 = files[int(idx_1)], files[int(idx_2)]
                outcome = run_legacy_compat_binary_operation(
                    binary_specs,
                    rng=self.local_random,
                    parent1=p1,
                    parent2=p2,
                    legacy_invokers=legacy_binary_invokers,
                )
                if outcome is None:
                    continue
                label, new_manip, new_struct, crossover_parameters = outcome
                candidates.append(new_struct)
                manipulators.append(new_manip)
                lineages.append(
                    [
                        label,
                        p1,
                        p2,
                        repr(dict(new_manip.last_crossover_provenance or ())),
                    ]
                )
                operation_parameters.append(crossover_parameters)
                crossed = True
                break
            if crossed:
                continue
            fallback = self._make_manipulator_from_file(p1)
            mutation, new_struct, mutation_parameters = self.mutator.mutate(
                local_random=self.local_random,
                GB=self.GB,
                manipulator=fallback,
            )
            candidates.append(new_struct)
            manipulators.append(fallback)
            lineages.append(["crossover_fallback_" + mutation, p1])
            operation_parameters.append(mutation_parameters)

        # Mutations
        if not intermediate_indices:
            intermediate_indices = list(range(len(files)))
        choices = self.local_random.choice(
            intermediate_indices, size=N_mutate, replace=True
        )
        for idx in choices:
            parent = files[idx]
            new_manip = GBManipulator(
                parent,
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
            )
            new_manip.rng = self.local_random
            mutation, new_struct, mutation_parameters = self.mutator.mutate(
                local_random=self.local_random,
                GB=self.GB,
                manipulator=new_manip,
            )

            candidates.append(new_struct)
            manipulators.append(new_manip)
            lineages.append([mutation, parent])
            operation_parameters.append(mutation_parameters)

        return manipulators, candidates, lineages, operation_parameters

    def _is_valid_file(self, p: str | None) -> bool:
        return bool(p) and Path(p).is_file()

    @staticmethod
    def _cached_evaluation_to_state(
        record: _CachedEvaluation | None,
    ) -> dict | None:
        """Serialize one optional legacy carryover cache entry.

        :param record: Reusable result or ``None`` for a cache miss.
        :return: JSON-safe cache state.
        """
        if record is None:
            return None
        return {
            "energy": record.energy,
            "structure_path": record.structure_path,
        }

    @staticmethod
    def _cached_evaluation_from_state(state: object) -> _CachedEvaluation | None:
        """Restore one optional legacy carryover cache entry.

        :param state: Deserialized optional cache state.
        :return: Validated reusable result or ``None``.
        :raises GBMinimizerError: If cache state is malformed.
        """
        if state is None:
            return None
        if not isinstance(state, dict):
            raise GBMinimizerError("cached evaluation state must be a dictionary")
        try:
            energy = float(state["energy"])
            structure_path = state["structure_path"]
        except (KeyError, TypeError, ValueError) as exc:
            raise GBMinimizerError("cached evaluation state is malformed") from exc
        if not math.isfinite(energy):
            raise GBMinimizerError("cached evaluation energy must be finite")
        if not isinstance(structure_path, str) or not structure_path:
            raise GBMinimizerError("cached evaluation structure_path is invalid")
        return _CachedEvaluation(energy=energy, structure_path=structure_path)

    @staticmethod
    def _owned_evaluation_to_state(record: CandidateEvaluation) -> dict:
        """Serialize one typed owned evaluation without its live manipulator.

        :param record: Explicit-ownership evaluation to persist.
        :return: JSON-safe evaluation state.
        """
        return {
            "candidate_id": record.candidate_id,
            "input_index": record.input_index,
            "energy": record.objective,
            "structure_path": record.structure_path,
            "mapping": (
                None
                if record.mapping is None
                else _candidate_mapping_to_state(record.mapping)
            ),
            "success": record.success,
            "failure_reason": record.failure_reason,
        }

    def _owned_evaluation_from_state(self, state: object) -> CandidateEvaluation:
        """Reconstruct one typed owned evaluation from checkpoint state.

        Successful artifacts are reloaded through the authoritative explicit-ownership
        path. Failed records remain non-reusable and do not require their diagnostic
        artifact to exist.

        :param state: Deserialized evaluation state.
        :return: Validated typed evaluation.
        :raises GBMinimizerError: If the state or a required successful artifact is
            invalid.
        """
        if self._owned_evaluator is None:
            raise GBMinimizerError(
                "owned evaluation restore requires an evaluator adapter"
            )
        if not isinstance(state, dict):
            raise GBMinimizerError("owned evaluation state must be a dictionary")
        try:
            candidate_id = state["candidate_id"]
            input_index = int(state["input_index"])
            energy = float(state["energy"])
            structure_path = state["structure_path"]
            success = state["success"]
            failure_reason = state.get("failure_reason")
            mapping_state = state["mapping"]
        except (KeyError, TypeError, ValueError) as exc:
            raise GBMinimizerError("owned evaluation state is malformed") from exc
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise GBMinimizerError("owned evaluation candidate_id is invalid")
        if isinstance(state.get("input_index"), (bool, np.bool_)) or input_index < -1:
            raise GBMinimizerError("owned evaluation input_index is invalid")
        if not np.isfinite(energy):
            raise GBMinimizerError("owned evaluation energy must be finite")
        if not isinstance(success, bool):
            raise GBMinimizerError("owned evaluation success must be Boolean")
        if structure_path is not None and not isinstance(structure_path, str):
            raise GBMinimizerError("owned evaluation structure_path is invalid")
        try:
            mapping = (
                None
                if mapping_state is None
                else _candidate_mapping_from_state(mapping_state)
            )
        except GrainOwnershipError as exc:
            raise GBMinimizerError(
                f"owned evaluation mapping is invalid: {exc}"
            ) from exc
        if not success:
            if not isinstance(failure_reason, str) or not failure_reason:
                raise GBMinimizerError(
                    "failed owned evaluation lacks failure context"
                )
            if energy != self._owned_evaluator.penalty:
                raise GBMinimizerError(
                    "failed owned evaluation does not carry the configured penalty"
                )
            # pyraisecontract: ignore=DOC115[TypeError]
            # pyraisecontract: ignore=DOC115[ValueError]
            #   All CandidateEvaluation scalar and failure-state invariants are
            #   explicitly validated above before reconstruction.
            return CandidateEvaluation(
                candidate_id=candidate_id,
                input_index=input_index,
                objective=energy,
                structure_path=structure_path,
                mapping=mapping,
                manipulator=None,
                success=False,
                failure_reason=failure_reason,
            )
        if mapping is None or structure_path is None:
            raise GBMinimizerError(
                "successful owned evaluation lacks reconstruction state"
            )
        try:
            manipulator = self._owned_evaluator._reload_mapping(
                structure_path,
                mapping,
            )
        except (
            OSError,
            LammpsDataError,
            GrainOwnershipError,
            ParentError,
            GBManipulatorError,
        ) as exc:
            raise GBMinimizerError(
                "Checkpoint owned evaluation artifact is missing, unreadable, or "
                f"inconsistent: {structure_path}"
            ) from exc
        # pyraisecontract: ignore=DOC115[TypeError]
        # pyraisecontract: ignore=DOC115[ValueError]
        #   Successful checkpoint scalar state is validated above, and the reload
        #   path proves that the required mapping/manipulator state is present.
        return CandidateEvaluation(
            candidate_id=candidate_id,
            input_index=input_index,
            objective=energy,
            structure_path=structure_path,
            mapping=mapping,
            manipulator=manipulator,
            success=True,
        )

    def _owned_record_from_snapshot(
        self,
        snapshot: CandidateEvaluationSnapshot,
        mapping: CandidateFileMapping | None,
    ) -> CandidateEvaluation:
        """Reconstruct one typed owned evaluation from its algorithm-neutral snapshot.

        ``CandidateEvaluationSnapshot`` carries no ``mapping`` field of its own -- it is
        also used by contexts (a legacy-mode candidate, an MC candidate) that never have
        one -- so the persistent explicit-ownership reconstruction mapping is supplied
        separately here, exactly as
        :attr:`~GBOpt.snapshot.GeneticAlgorithmSnapshot.best_mapping`/
        :attr:`~GBOpt.snapshot.GeneticAlgorithmSnapshot.population_cache_mappings`
        carry it alongside :attr:`~GBOpt.snapshot.GeneticAlgorithmSnapshot.best`/
        :attr:`~GBOpt.snapshot.GeneticAlgorithmSnapshot.population_cache`. Delegates to
        the established :meth:`_owned_evaluation_from_state` by rebuilding its exact
        expected raw-dictionary shape, rather than duplicating its reload/validation
        logic.

        :param snapshot: Canonical, algorithm-neutral candidate evaluation record.
        :param mapping: Persistent explicit-ownership reconstruction mapping, or
            ``None``.
        :return: Validated typed evaluation.
        :raises GBMinimizerError: If the snapshot or a required successful artifact is
            invalid.
        """
        v1_shape = {
            "candidate_id": snapshot.candidate_id,
            "input_index": snapshot.input_index,
            "energy": snapshot.selection_energy,
            "structure_path": (
                None if snapshot.artifact is None else snapshot.artifact.path
            ),
            "mapping": (
                None if mapping is None else _candidate_mapping_to_state(mapping)
            ),
            "success": snapshot.status is EvaluationStatus.SUCCESS,
            "failure_reason": snapshot.failure_message,
        }
        return self._owned_evaluation_from_state(v1_shape)

    def _write_owned_population_checkpoint(
        self,
        checkpoint_file: Path,
        unique_id: str,
        next_generation: int,
        manipulators: list[GBManipulator],
        structures: list[np.ndarray],
    ) -> list[dict]:
        """Write owned pending structures and their explicit reconstruction metadata.

        :param checkpoint_file: Run-level checkpoint path whose directory owns artifacts.
        :param unique_id: Stable run identifier.
        :param next_generation: Generation that will consume the pending population.
        :param manipulators: Candidate manipulators in population order.
        :param structures: Candidate atom rows in matching population order.
        :return: Ordered serialized population snapshots.
        :raises GBMinimizerError: If population alignment or ownership is invalid.
        """
        if self._owned_evaluator is None:
            raise GBMinimizerError(
                "owned population checkpoint requires an evaluator adapter"
            )
        if len(manipulators) != len(structures):
            raise GBMinimizerError(
                "owned checkpoint population lost manipulator/structure alignment"
            )
        snapshots = []
        for index, (manipulator, structure) in enumerate(
            zip(manipulators, structures, strict=True)
        ):
            try:
                mapping = self._owned_evaluator._candidate_file_mapping(
                    manipulator,
                    structure,
                )
            except GrainOwnershipError as exc:
                raise GBMinimizerError(
                    f"owned checkpoint candidate {index} has invalid ownership: {exc}"
                ) from exc
            pending_path = checkpoint_file.parent / (
                f"GA_{unique_id}_g{next_generation}_c{index}.owned.pending"
            )
            try:
                self.GB.write_lammps(
                    str(pending_path),
                    structure,
                    mapping.box_dims,
                    precision=15,
                )
            except (OSError, GBMakerError) as exc:
                raise GBMinimizerError(
                    f"could not persist owned checkpoint candidate {index}"
                ) from exc
            snapshots.append(
                {
                    "structure_path": str(pending_path),
                    "mapping": _candidate_mapping_to_state(mapping),
                }
            )
        return snapshots

    def _restore_owned_population(
        self,
        snapshots: object,
    ) -> tuple[list[GBManipulator], list[np.ndarray]]:
        """Restore an aligned pending owned population from checkpoint snapshots.

        :param snapshots: Ordered serialized structure/mapping snapshots.
        :return: Reconstructed manipulators and atom arrays.
        :raises GBMinimizerError: If state is malformed or any required artifact fails
            explicit reload validation.
        """
        if self._owned_evaluator is None:
            raise GBMinimizerError(
                "owned population restore requires an evaluator adapter"
            )
        if not isinstance(snapshots, list) or len(snapshots) != self.population_size:
            raise GBMinimizerError(
                "owned checkpoint population has an invalid candidate count"
            )
        manipulators = []
        structures = []
        for index, snapshot in enumerate(snapshots):
            if not isinstance(snapshot, dict):
                raise GBMinimizerError(
                    f"owned checkpoint candidate {index} is malformed"
                )
            path = snapshot.get("structure_path")
            if not isinstance(path, str):
                raise GBMinimizerError(
                    f"owned checkpoint candidate {index} lacks a structure path"
                )
            try:
                mapping = _candidate_mapping_from_state(snapshot.get("mapping"))
                manipulator = self._owned_evaluator._reload_mapping(path, mapping)
            except (
                OSError,
                LammpsDataError,
                GrainOwnershipError,
                ParentError,
                GBManipulatorError,
            ) as exc:
                raise GBMinimizerError(
                    f"Checkpoint owned population path {path} is missing, unreadable, "
                    "or inconsistent."
                ) from exc
            manipulators.append(manipulator)
            structures.append(
                np.array(manipulator.parents[0].whole_system, copy=True)
            )
        return manipulators, structures

    def _emit(
        self,
        event_type: OptimizationEventType,
        *,
        run_context: RunContext,
        iteration: int,
        **fields: object,
    ) -> None:
        """Build and deliver one lifecycle event, never letting a sink abort the run.

        :param event_type: Lifecycle occurrence to report.
        :param run_context: Keyword argument, required. Identity of the emitting run.
        :param iteration: Keyword argument, required. GA generation this event reports
            on.
        :param **fields: Additional ``OptimizationEvent`` keyword arguments.
        """
        try:
            event = OptimizationEvent(
                event_type=event_type,
                run=run_context,
                iteration=iteration,
                **fields,
            )
            self._event_sink.emit(event)
        except Exception:
            # Deliberate recovery boundary: a broken event -- or a sink that raises --
            # is an observability-side failure, never a reason to abort or otherwise
            # change the numerical outcome of the run that triggered it.
            logger.exception(
                "event emission failed for %s at GA generation %d; continuing the run",
                event_type.value,
                iteration,
            )

    def run_GA(
        self,
        unique_id: int | uuid.UUID | None = None,
        *,
        checkpoint_file: str | Path | None = None,
        checkpoint_format: str = "json",
        checkpoint_interval: int = 1
    ) -> tuple:
        """
        Runs a genetic algorithm loop on the grain boundary structure.

        Checkpointing is optional. Pass ``checkpoint_file`` to enable it; omit it (or
        pass ``None``) to run without any checkpoint file. When enabled, a per-candidate
        sidecar(``{stem}.iter{N}{ext}``) is also written so a mid-generation crash can
        be resumed without re-evaluating completed candidates. The checkpoint file is
        **not** deleted on normal completion — it can be used to continue the run later
        by calling ``run_GA`` again with the same ``checkpoint_file`` after increasing
        ``generations``. The checkpoint file and the sibling ``*.pending`` structure
        files in the same directory form a unit - both must be present to resume or
        extend a run. Do not delete or move the ``.pending`` files independently of the
        checkpoint file.

        :param unique_id: Argument, optional, defaults to ``None``. Label applied to all
            output files. Restored from the checkpoint on resume if not provided.
        :param checkpoint_file: Keyword argument, optional, defaults to ``None``. Path to
            the run-level checkpoint file. If the file exists the run resumes from it;
            otherwise a fresh run begins and the file is created.
        :param checkpoint_format: Keyword argument, optional, defaults to ``"json"``.
            Serialization format: ``"json"`` (human-readable) or ``"pickle"`` (binary,
            no NumPy conversion needed).
        :param checkpoint_interval: Keyword argument, optional, defaults to 1. Save a
            run-level checkpoint every N generations.
        :return: Tuple containing the minimum energy value observed and the associated
            dump filename.
        :raises GBMinimizerError: If a checkpoint is malformed or references a missing,
            unreadable, or ownership-inconsistent required structure artifact.
        :raises GBMinimizerValueError: If checkpoint configuration is invalid.
        """

        if self.initial_ownership is not None:
            return self._run_owned_GA(
                unique_id=unique_id,
                checkpoint_file=checkpoint_file,
                checkpoint_format=checkpoint_format,
                checkpoint_interval=checkpoint_interval,
            )

        try:
            if checkpoint_file is not None:
                checkpoint_file = Path(checkpoint_file)
                checkpoint = CheckpointStore.from_optional(
                    checkpoint_file, checkpoint_format, checkpoint_interval
                )
                try:
                    state = checkpoint.load()
                except CheckpointError as e:
                    raise GBMinimizerError(str(e)) from e
                if state is None:
                    unique_id = str(unique_id) if unique_id is not None else str(
                        uuid.uuid4())
            else:
                unique_id = str(unique_id) if unique_id is not None else str(
                    uuid.uuid4())
                checkpoint = CheckpointStore.disabled()
                state = None
        except CheckpointError as e:
            raise GBMinimizerValueError(str(e)) from e

        logger.debug(
            "GA run %s starting: generations=%d, population_size=%d, seed=%d, "
            "resuming=%s",
            unique_id,
            self.generations,
            self.population_size,
            self.seed,
            state is not None,
        )

        if state is not None:
            try:
                snapshot = _ga_snapshot_from_checkpoint_state(state, owned=False)
                configuration = snapshot.configuration
                if configuration.slice_and_merge_pct != self.slice_and_merge_pct:
                    raise GBMinimizerError(
                        "checkpoint slice_and_merge_pct does not match the "
                        "minimizer configuration"
                    )
                if (
                    configuration.reuse_carryover_evaluations
                    != self.reuse_carryover_evaluations
                ):
                    raise GBMinimizerError(
                        "checkpoint reuse_carryover_evaluations does not match "
                        "the minimizer configuration"
                    )
                unique_id = snapshot.run.run_id
                self.GBE_vals = [list(gen) for gen in snapshot.energy_history]
                self.history = [
                    [
                        [_lineage_entry_from_snapshot(entry.lineage), entry.energy]
                        for entry in gen
                    ]
                    for gen in snapshot.generation_history
                ]
                self.local_random = snapshot.rng.to_generator()
                self.seed = snapshot.run.seed
                _start_gen = snapshot.completed_generation + 1
                best_energy = snapshot.best.selection_energy
                best_dump = snapshot.best.artifact.path
                # Drop any stale iter checkpoint for the just-completed generation
                stale = CandidateCheckpoint._derive_path(
                    checkpoint_file, snapshot.completed_generation)
                if stale.exists():
                    stale.unlink()
                population_lineages = [
                    _lineage_entry_from_snapshot(candidate.lineage)
                    for candidate in snapshot.population
                ]
                # operation_parameters is in-memory-only (never checkpointed), so a
                # resumed population's candidates report no operation parameters until
                # the next generation produces new offspring.
                population_operation_parameters: list[Mapping[str, object] | None] = (
                    [None] * len(population_lineages)
                )
                population_cached_evaluations = [
                    None
                    if cache is None
                    else _CachedEvaluation(
                        energy=cache.energy, structure_path=cache.artifact.path
                    )
                    for cache in snapshot.population_cache
                ]
                population_checkpoint_paths = [
                    candidate.artifact.path for candidate in snapshot.population
                ]
                population_manipulators = []
                population_structures = []
                for cp_path in population_checkpoint_paths:
                    try:
                        manip = self._make_manipulator_from_file(cp_path)
                    except Exception as exc:
                        raise GBMinimizerError(
                            f"Checkpoint population path {cp_path} is "
                            "missing/unreadable."
                        ) from exc
                    population_manipulators.append(manip)
                    population_structures.append(
                        np.array(manip.parents[0].whole_system, copy=True)
                    )
                run_context = RunContext(
                    run_id=str(unique_id),
                    seed=self.seed,
                    algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
                    case_id=self.case_id,
                    campaign_id=self.campaign_id,
                )
                self._emit(
                    OptimizationEventType.RUN_STARTED,
                    run_context=run_context,
                    iteration=_start_gen - 1,
                )
            except GBMinimizerError:
                raise
            except (
                CheckpointCompatibilityError,
                SnapshotError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise GBMinimizerError(
                    f"Invalid GeneticAlgorithmMinimizer checkpoint envelope: {exc}"
                ) from exc
        else:
            run_context = RunContext(
                run_id=str(unique_id),
                seed=self.seed,
                algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
                case_id=self.case_id,
                campaign_id=self.campaign_id,
            )
            self._emit(
                OptimizationEventType.RUN_STARTED, run_context=run_context, iteration=0
            )
            # Evaluate the initial structure
            init_system = np.array(
                self.manipulator.parents[0].whole_system, copy=True)
            initial_candidate_id = "GA_initial" + str(unique_id)
            try:
                init_gbe, init_dump = self.gb_energy_func(
                    self.GB,
                    self.manipulator,
                    init_system,
                    initial_candidate_id,
                )
            except Exception as exc:
                # There is no sensible penalized starting point for a whole GA run,
                # so an initial-evaluation failure is fatal -- matching
                # MonteCarloMinimizer's initial evaluation, which raises the same way
                # rather than seeding a run from a penalty value.
                initial_result = EvaluationResult(
                    candidate_id=initial_candidate_id,
                    input_index=0,
                    status=EvaluationStatus.FAILED,
                    selection_energy=ENERGY_PENALTY,
                    failure_stage=FailureStage.EVALUATOR,
                    failure_message=f"{type(exc).__name__}: {exc}",
                )
                self._emit(
                    OptimizationEventType.INITIAL_EVALUATION,
                    run_context=run_context,
                    iteration=0,
                    **evaluation_event_fields(initial_result),
                )
                self._emit(
                    OptimizationEventType.RUN_FAILED,
                    run_context=run_context,
                    iteration=0,
                    failure_stage=initial_result.failure_stage,
                    failure_message=initial_result.failure_message,
                )
                raise GBMinimizerError(
                    f"initial evaluation failed: {initial_result.failure_message}"
                ) from exc
            self._emit(
                OptimizationEventType.INITIAL_EVALUATION,
                run_context=run_context,
                iteration=0,
                **evaluation_event_fields(
                    from_scalar_tuple(
                        initial_candidate_id,
                        0,
                        (init_gbe, init_dump),
                        penalty=ENERGY_PENALTY,
                    )
                ),
            )
            self.GBE_vals.append([init_gbe])
            self.history = []

            best_energy = init_gbe
            best_dump = init_dump
            logger.debug(
                "GA run %s initial evaluation: energy=%.6g", unique_id, init_gbe
            )

            base_parent = init_dump
            population_manipulators = []
            population_structures = []
            population_lineages = []
            population_operation_parameters: list[Mapping[str, object] | None] = []

            if self.initial_structure is not None:
                seed_manip = self._make_manipulator_from_file(base_parent)
                population_manipulators.append(seed_manip)
                population_structures.append(
                    np.array(seed_manip.parents[0].whole_system, copy=True)
                )
                population_lineages.append(["START", base_parent])
                population_operation_parameters.append(None)

            n_to_generate = self.population_size - len(population_manipulators)
            for _ in range(n_to_generate):
                candidate_manip = self._make_manipulator_from_file(base_parent)
                mutation, candidate_struct, mutation_parameters = self.mutator.mutate(
                    local_random=self.local_random,
                    GB=self.GB,
                    manipulator=candidate_manip,
                )
                population_manipulators.append(candidate_manip)
                population_structures.append(candidate_struct)
                population_lineages.append([mutation, base_parent])
                population_operation_parameters.append(mutation_parameters)

            population_checkpoint_paths = [lin[1] for lin in population_lineages]
            population_cached_evaluations = [None] * self.population_size
            _start_gen = 0

        def _build_ga_state(gen):
            """Return one callback-free, typed-snapshot checkpoint payload for ``gen``.

            :param gen: Completed GA generation represented by the checkpoint.
            :return: Serializable checkpoint payload.
            """
            population_snapshot = [
                PopulationCandidateSnapshot(
                    artifact=StructureArtifact(
                        path=str(path), format=_STRUCTURE_FORMAT
                    ),
                    lineage=_lineage_step_from_v1(lineage),
                )
                for lineage, path in zip(
                    population_lineages, population_checkpoint_paths, strict=True
                )
            ]
            population_cache_snapshot = [
                None
                if cache is None
                else CandidateEvaluationSnapshot(
                    candidate_id=f"legacy-carryover-{index}",
                    input_index=index,
                    status=EvaluationStatus.SUCCESS,
                    selection_energy=cache.energy,
                    energy=cache.energy,
                    artifact=StructureArtifact(
                        path=cache.structure_path, format=_STRUCTURE_FORMAT
                    ),
                )
                for index, cache in enumerate(population_cached_evaluations)
            ]
            generation_history_snapshot = [
                [
                    GenerationHistoryEntrySnapshot(
                        lineage=_lineage_step_from_v1(lineage), energy=energy
                    )
                    for lineage, energy in gen_history
                ]
                for gen_history in self.history
            ]
            snapshot = GeneticAlgorithmSnapshot(
                run=RunIdentitySnapshot(
                    run_id=str(unique_id),
                    seed=self.seed,
                    case_id=self.case_id,
                    campaign_id=self.campaign_id,
                ),
                rng=RngStateSnapshot.from_generator(self.local_random),
                completed_generation=gen,
                best=CandidateEvaluationSnapshot(
                    candidate_id=f"{unique_id}-best",
                    input_index=-1,
                    status=EvaluationStatus.SUCCESS,
                    selection_energy=best_energy,
                    energy=best_energy,
                    artifact=StructureArtifact(
                        path=str(best_dump), format=_STRUCTURE_FORMAT
                    ),
                ),
                population=population_snapshot,
                configuration=GeneticAlgorithmConfigurationSnapshot(
                    slice_and_merge_pct=self.slice_and_merge_pct,
                    reuse_carryover_evaluations=self.reuse_carryover_evaluations,
                ),
                population_cache=population_cache_snapshot,
                energy_history=[list(gen_vals) for gen_vals in self.GBE_vals],
                generation_history=generation_history_snapshot,
            )
            return {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "minimizer": _GA_MINIMIZER_NAME,
                "progress_unit": _GA_PROGRESS_UNIT,
                "snapshot": snapshot.to_state(),
            }

        _current_pending = []
        _last_completed_gen = -1
        # Main GA loop
        for gen in range(_start_gen, self.generations):
            if checkpoint.enabled:
                _current_pending = [
                    p for p in population_checkpoint_paths
                    if str(p).endswith(".pending")
                ]
            all_uids = [
                f"GA_{unique_id}_g{gen}_c{i}"
                for i in range(len(population_manipulators))
            ]
            gen_checkpoint = (
                CandidateCheckpoint.new_or_resume(
                    checkpoint_file, checkpoint_format, gen, all_uids)
                if checkpoint.enabled else None
            )

            gen_energies, gen_files, evaluated_manipulators, gen_results = (
                self._evaluate_generation(
                    population_manipulators,
                    population_structures,
                    population_lineages,
                    gen,
                    unique_id,
                    gen_checkpoint=gen_checkpoint,
                    cached_evaluations=population_cached_evaluations,
                )
            )

            for i, result in enumerate(gen_results):
                self._emit(
                    OptimizationEventType.PROPOSAL_EVALUATED,
                    run_context=run_context,
                    iteration=gen,
                    operation_name=population_lineages[i][0],
                    operation_parameters=population_operation_parameters[i],
                    **evaluation_event_fields(result),
                )

            valid_old_idxs = [
                i for i, f in enumerate(gen_files) if self._is_valid_file(f)
            ]

            self.GBE_vals.append(gen_energies)
            self.history.append(list(zip(population_lineages, gen_energies)))

            if not valid_old_idxs:
                # If nothing valid survived evaluation, re-seed from best.
                for i, result in enumerate(gen_results):
                    self._emit(
                        OptimizationEventType.CANDIDATE_REJECTED,
                        run_context=run_context,
                        iteration=gen,
                        operation_name=population_lineages[i][0],
                        operation_parameters=population_operation_parameters[i],
                        **evaluation_event_fields(result),
                    )
                self._emit(
                    OptimizationEventType.POPULATION_RESEEDED,
                    run_context=run_context,
                    iteration=gen,
                    candidate_id=None,
                    selection_energy=best_energy,
                )
                next_manipulators = []
                next_structures = []
                next_lineages = []
                next_operation_parameters: list[Mapping[str, object] | None] = []
                next_cached_evaluations: list[_CachedEvaluation | None] = []

                for _ in range(self.population_size):
                    candidate_manip = self._make_manipulator_from_file(
                        best_dump
                    )
                    mutation, candidate_struct, mutation_parameters = self.mutator.mutate(
                        local_random=self.local_random,
                        GB=self.GB,
                        manipulator=candidate_manip,
                    )
                    next_manipulators.append(candidate_manip)
                    next_structures.append(candidate_struct)
                    next_lineages.append([mutation, best_dump])
                    next_operation_parameters.append(mutation_parameters)
                    next_cached_evaluations.append(None)

                population_manipulators = next_manipulators
                population_structures = next_structures
                population_lineages = next_lineages
                population_operation_parameters = next_operation_parameters
                population_cached_evaluations = next_cached_evaluations
            else:
                for i in valid_old_idxs:
                    gbe = gen_energies[i]
                    dump_file_name = gen_files[i]
                    if gbe < best_energy:
                        logger.debug(
                            "GA run %s new best at generation %d: energy %.6g "
                            "(was %.6g)",
                            unique_id,
                            gen,
                            gbe,
                            best_energy,
                        )
                        best_energy = gbe
                        best_dump = dump_file_name
                        self._emit(
                            OptimizationEventType.BEST_UPDATED,
                            run_context=run_context,
                            iteration=gen,
                            **evaluation_event_fields(gen_results[i]),
                        )

                # Build compressed arrays of only valid candidates for selection and breeding.
                valid_energies = [gen_energies[i] for i in valid_old_idxs]
                valid_files = [gen_files[i] for i in valid_old_idxs]

                lowest_valid_idxs, inter_valid_idxs = self._select_indices_by_energy(
                    valid_energies
                )

                selected_old_idxs = {valid_old_idxs[j] for j in lowest_valid_idxs} | {
                    valid_old_idxs[j] for j in inter_valid_idxs
                }
                for i, result in enumerate(gen_results):
                    self._emit(
                        (
                            OptimizationEventType.CANDIDATE_ACCEPTED
                            if i in selected_old_idxs
                            else OptimizationEventType.CANDIDATE_REJECTED
                        ),
                        run_context=run_context,
                        iteration=gen,
                        operation_name=population_lineages[i][0],
                        operation_parameters=population_operation_parameters[i],
                        **evaluation_event_fields(result),
                    )

                # Carry over lowest energies.
                next_manipulators = []
                next_structures = []
                next_lineages = []
                next_operation_parameters = []
                next_cached_evaluations = []
                for j in lowest_valid_idxs:
                    old_idx = valid_old_idxs[j]
                    manip = evaluated_manipulators[old_idx]
                    dump = gen_files[old_idx]
                    if manip is None or dump is None:
                        continue
                    next_manipulators.append(manip)
                    next_structures.append(manip.parents[0].whole_system)
                    next_lineages.append(["carryover", dump])
                    next_operation_parameters.append(None)
                    next_cached_evaluations.append(
                        _CachedEvaluation(gen_energies[old_idx], dump)
                        if self.reuse_carryover_evaluations
                        else None
                    )

                valid_files_str = [f for f in valid_files if f is not None]
                offspring_count = self.population_size - len(next_manipulators)
                new_manips, new_structs, new_lineages, new_operation_parameters = (
                    self._make_next_generation(
                        valid_files_str,
                        inter_valid_idxs,
                        offspring_count,
                    )
                )

                next_manipulators.extend(new_manips)
                next_structures.extend(new_structs)
                next_lineages.extend(new_lineages)
                next_operation_parameters.extend(new_operation_parameters)
                next_cached_evaluations.extend([None] * len(new_lineages))

                population_manipulators = next_manipulators
                population_structures = next_structures
                population_lineages = next_lineages
                population_operation_parameters = next_operation_parameters
                population_cached_evaluations = next_cached_evaluations

            logger.debug(
                "GA run %s generation %d complete: %d/%d candidates valid, best "
                "energy %.6g",
                unique_id,
                gen,
                len(valid_old_idxs),
                len(gen_energies),
                best_energy,
            )

            _last_completed_gen = gen
            is_final_gen = (gen == self.generations - 1)
            if checkpoint.enabled and (checkpoint.is_due(gen + 1) or is_final_gen):
                new_pending = []
                for i, (manip, struct) in enumerate(
                    zip(population_manipulators, population_structures)
                ):
                    pending_path = str(
                        checkpoint_file.parent
                        / f"GA_{unique_id}_g{gen + 1}_c{i}.pending"
                    )
                    self.GB.write_lammps(
                        pending_path, struct, manip.parents[0].box_dims
                    )
                    new_pending.append(pending_path)
                population_checkpoint_paths = new_pending
                checkpoint.save_final(_build_ga_state(gen))
                for p in _current_pending:
                    Path(p).unlink(missing_ok=True)
                _current_pending = new_pending

            # Iter checkpoint is transient; main checkpoint covers this boundary
            if gen_checkpoint is not None:
                gen_checkpoint.delete()

            self._emit(
                OptimizationEventType.GENERATION_BOUNDARY,
                run_context=run_context,
                iteration=gen,
                selection_energy=best_energy,
            )

        if _last_completed_gen >= 0:
            self._emit(
                OptimizationEventType.RUN_TERMINATED,
                run_context=run_context,
                iteration=_last_completed_gen,
                termination_reason=TerminationReason.MAX_GENERATIONS,
            )

        return (best_energy, best_dump)

    def _run_owned_GA(
        self,
        unique_id: int | uuid.UUID | None = None,
        *,
        checkpoint_file: str | Path | None = None,
        checkpoint_format: str = "json",
        checkpoint_interval: int = 1,
    ) -> tuple[float, str]:
        """Run the GA while preserving explicit ownership through every reload.

        :param unique_id: Argument, optional, defaults to ``None``. Run identifier.
        :param checkpoint_file: Keyword argument, optional, defaults to ``None``.
            Run-level checkpoint path used for generation-boundary and candidate-sidecar
            recovery.
        :param checkpoint_format: Keyword argument, optional, defaults to ``"json"``.
            Checkpoint serialization format, either ``"json"`` or ``"pickle"``.
        :param checkpoint_interval: Keyword argument, optional, defaults to 1. Save the
            run-level checkpoint every N completed generations.
        :return: Minimum energy and validated structure path.
        :raises GBMinimizerError: If evaluation fails initially, aligned population
            state cannot be maintained, or checkpoint state cannot be reconstructed
            safely.
        :raises GBMinimizerValueError: If checkpoint configuration is invalid.
        """
        if self._owned_evaluator is None:
            raise GBMinimizerError(
                "explicit-ownership execution requires an evaluator adapter"
            )

        early_snapshot: GeneticAlgorithmSnapshot | None = None
        try:
            if checkpoint_file is None:
                checkpoint = CheckpointStore.disabled()
                state = None
                unique_id = str(unique_id) if unique_id is not None else str(
                    uuid.uuid4())
            else:
                checkpoint_file = Path(checkpoint_file)
                checkpoint = CheckpointStore.from_optional(
                    checkpoint_file,
                    checkpoint_format,
                    checkpoint_interval,
                )
                state = checkpoint.load()
                if state is None:
                    unique_id = (
                        str(unique_id) if unique_id is not None else str(uuid.uuid4())
                    )
                else:
                    if state.get("schema_version") == CHECKPOINT_SCHEMA_VERSION:
                        v1_owned_state = state.get("state")
                        if not isinstance(v1_owned_state, dict) or (
                            v1_owned_state.get("ga_mode") != "explicit_ownership"
                            or v1_owned_state.get("owned_checkpoint_version")
                            != _OWNED_GA_CHECKPOINT_VERSION
                        ):
                            raise GBMinimizerError(
                                "checkpoint does not contain supported "
                                "explicit-ownership state"
                            )
                    early_snapshot = _ga_snapshot_from_checkpoint_state(
                        state, owned=True
                    )
                    unique_id = early_snapshot.run.run_id
        except CheckpointError as exc:
            raise GBMinimizerValueError(str(exc)) from exc
        except GBMinimizerError:
            raise
        except (CheckpointCompatibilityError, SnapshotError, KeyError, TypeError) as exc:
            raise GBMinimizerError(
                "Invalid explicit-ownership GA checkpoint envelope."
            ) from exc

        logger.debug(
            "GA run %s starting (owned mode): generations=%d, population_size=%d, "
            "seed=%d, resuming=%s",
            unique_id,
            self.generations,
            self.population_size,
            self.seed,
            state is not None,
        )

        self._owned_evaluator.begin_run()
        if (
            self.retention_policy is not None
            and self.retention_policy.prune
            and not checkpoint.enabled
        ):
            raise GBMinimizerValueError(
                "retention_policy prune=True requires checkpoint_file for durable "
                "cleanup"
            )

        self._artifact_provenance = None
        if self.artifact_store is not None:
            archive_root = self._owned_archive_root(checkpoint_file, str(unique_id))
            try:
                self._artifact_provenance = _ArtifactProvenance(
                    archive_root,
                    calculation_context=self.calculation_context,
                )
            except ArtifactProvenanceError as exc:
                warnings.warn(
                    f"Artifact provenance initialization failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        population_snapshots: list[dict] = []
        materializable_records: dict[str, CandidateEvaluation] = {}
        if state is not None:
            try:
                # early_snapshot is always built above whenever state is not None.
                if early_snapshot is None:
                    raise GBMinimizerError(
                        "Invalid explicit-ownership GA checkpoint envelope."
                    )
                snapshot = early_snapshot
                configuration = snapshot.configuration
                expected_params = {
                    "population_size": self.population_size,
                    "keep_top_pct": self.keep_top_pct,
                    "intermediate_pct": self.intermediate_pct,
                    "slice_and_merge_pct": self.slice_and_merge_pct,
                    "reuse_carryover_evaluations": (
                        self.reuse_carryover_evaluations
                    ),
                    "allow_variable_cell": self.allow_variable_cell,
                    "choices": tuple(self.mutator.choices_keys),
                    "crossover_surface": self.crossover_surface,
                    "crossover_max_tilt_degrees": (
                        self.crossover_max_tilt_degrees
                    ),
                    "crossover_attempts": self.crossover_attempts,
                    "failure_diagnostic_count": self.failure_diagnostic_count,
                    "composition_policy": tuple(
                        tuple(entry) for entry in self.composition_policy
                    ),
                }
                actual_params = {
                    "population_size": configuration.population_size,
                    "keep_top_pct": configuration.keep_top_pct,
                    "intermediate_pct": configuration.intermediate_pct,
                    "slice_and_merge_pct": configuration.slice_and_merge_pct,
                    "reuse_carryover_evaluations": (
                        configuration.reuse_carryover_evaluations
                    ),
                    "allow_variable_cell": configuration.allow_variable_cell,
                    "choices": (
                        None
                        if configuration.choices is None
                        else tuple(configuration.choices)
                    ),
                    "crossover_surface": configuration.crossover_surface,
                    "crossover_max_tilt_degrees": (
                        configuration.crossover_max_tilt_degrees
                    ),
                    "crossover_attempts": configuration.crossover_attempts,
                    "failure_diagnostic_count": configuration.failure_diagnostic_count,
                    "composition_policy": configuration.composition_policy,
                }
                for name, expected in expected_params.items():
                    if actual_params[name] != expected:
                        raise GBMinimizerError(
                            f"owned checkpoint run parameter {name!r} does not match "
                            "the minimizer configuration"
                        )

                self.GBE_vals = [list(gen) for gen in snapshot.energy_history]
                self.history = [
                    [
                        [_lineage_entry_from_snapshot(entry.lineage), entry.energy]
                        for entry in gen
                    ]
                    for gen in snapshot.generation_history
                ]
                if (
                    len(self.GBE_vals) != snapshot.completed_generation + 2
                    or len(self.history) != snapshot.completed_generation + 1
                ):
                    raise GBMinimizerError(
                        "owned checkpoint energy/history progress is inconsistent"
                    )
                self.local_random = snapshot.rng.to_generator()
                self.seed = snapshot.run.seed
                retention_state = snapshot.retention_state
                if retention_state is None:
                    if self.retention_policy is not None:
                        raise GBMinimizerError(
                            "checkpoint retention policy does not match the minimizer "
                            "configuration"
                        )
                    self.artifact_store = None
                    self._retention_archive_mappings = {}
                else:
                    try:
                        self.artifact_store = ArtifactStore.from_state(
                            _tuples_to_lists(dict(retention_state)),
                            policy=self.retention_policy,
                        )
                    except ArtifactStoreError as exc:
                        raise GBMinimizerError(str(exc)) from exc
                    self._retention_archive_mappings = {
                        candidate_id: _candidate_mapping_to_state(mapping)
                        for candidate_id, mapping in sorted(
                            snapshot.retention_archive_mappings.items()
                        )
                    }
                    for artifact in self.artifact_store.records():
                        if artifact.archive_path is None:
                            continue
                        if (
                            artifact.candidate_id
                            not in self._retention_archive_mappings
                        ):
                            raise GBMinimizerError(
                                "checkpoint retained candidate "
                                f"{artifact.candidate_id!r} lacks ownership metadata"
                            )
                        if not Path(artifact.archive_path).is_file():
                            raise GBMinimizerError(
                                f"retained archive path {artifact.archive_path} is "
                                "missing"
                            )
                self._failure_diagnostics = [
                    _FailureDiagnostic(
                        candidate_id=diagnostic.candidate_id,
                        generation=diagnostic.generation,
                        input_index=diagnostic.input_index,
                        failure_reason=diagnostic.failure_reason,
                        source_path=diagnostic.source_path,
                    )
                    for diagnostic in snapshot.failure_diagnostics
                ]
                if len(self._failure_diagnostics) > self.failure_diagnostic_count:
                    raise GBMinimizerError(
                        "checkpoint failure diagnostics exceed the configured bound"
                    )
                _start_gen = snapshot.completed_generation + 1
                best_record = self._owned_record_from_snapshot(
                    snapshot.best, snapshot.best_mapping
                )
                if not best_record.success:
                    raise GBMinimizerError(
                        "owned checkpoint best evaluation is not reusable"
                    )
                if snapshot.retention_lineages is None:
                    raise GBMinimizerError(
                        "owned checkpoint retention lineages are invalid"
                    )
                if len(snapshot.retention_lineages) != self.population_size:
                    raise GBMinimizerError(
                        "owned checkpoint retention lineages are invalid"
                    )
                population_retention_lineages = [
                    tuple(lineage) for lineage in snapshot.retention_lineages
                ]
                population_lineages = [
                    _lineage_entry_from_snapshot(candidate.lineage)
                    for candidate in snapshot.population
                ]
                if len(population_lineages) != self.population_size:
                    raise GBMinimizerError(
                        "owned checkpoint population lineages are invalid"
                    )
                # operation_parameters is in-memory-only (never checkpointed), so a
                # resumed population's candidates report no operation parameters until
                # the next generation produces new offspring.
                population_operation_parameters: list[Mapping[str, object] | None] = (
                    [None] * len(population_lineages)
                )
                population_snapshots = [
                    {
                        "structure_path": candidate.artifact.path,
                        "mapping": (
                            None
                            if candidate.mapping is None
                            else _candidate_mapping_to_state(candidate.mapping)
                        ),
                    }
                    for candidate in snapshot.population
                ]
                population_manipulators, population_structures = (
                    self._restore_owned_population(population_snapshots)
                )
                cache_entries = snapshot.population_cache or (
                    (None,) * len(population_manipulators)
                )
                cache_mappings = snapshot.population_cache_mappings or (
                    (None,) * len(population_manipulators)
                )
                if len(cache_entries) != len(population_manipulators) or len(
                    cache_mappings
                ) != len(population_manipulators):
                    raise GBMinimizerError(
                        "owned checkpoint cached evaluations are not "
                        "population-aligned"
                    )
                population_cached_evaluations = [
                    None
                    if cache is None
                    else self._owned_record_from_snapshot(cache, mapping)
                    for cache, mapping in zip(
                        cache_entries, cache_mappings, strict=True
                    )
                ]
                if (
                    snapshot.last_generation_evaluations is None
                    or len(snapshot.last_generation_evaluations)
                    != self.population_size
                ):
                    raise GBMinimizerError(
                        "owned checkpoint generation evaluations are invalid"
                    )
                self.last_generation_evaluations = [
                    CandidateEvaluationSummary(
                        candidate_id=entry.candidate_id,
                        input_index=entry.input_index,
                        objective=entry.selection_energy,
                        success=entry.status is EvaluationStatus.SUCCESS,
                        failure_reason=entry.failure_message,
                    )
                    for entry in snapshot.last_generation_evaluations
                ]
                self._owned_evaluator.restore_claimed_paths(
                    list(snapshot.claimed_paths)
                )
                self.best_evaluation = best_record
                stale = CandidateCheckpoint._derive_path(
                    checkpoint_file,
                    snapshot.completed_generation,
                )
                if stale.exists():
                    stale.unlink()
                run_context = RunContext(
                    run_id=str(unique_id),
                    seed=self.seed,
                    algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
                    case_id=self.case_id,
                    campaign_id=self.campaign_id,
                )
                self._emit(
                    OptimizationEventType.RUN_STARTED,
                    run_context=run_context,
                    iteration=snapshot.completed_generation,
                )
            except GBMinimizerError:
                raise
            except (
                CheckpointCompatibilityError, SnapshotError, KeyError, TypeError,
                ValueError,
            ) as exc:
                raise GBMinimizerError(
                    f"Invalid explicit-ownership GA checkpoint state: {exc}"
                ) from exc
        else:
            run_context = RunContext(
                run_id=str(unique_id),
                seed=self.seed,
                algorithm=OptimizationAlgorithm.GENETIC_ALGORITHM,
                case_id=self.case_id,
                campaign_id=self.campaign_id,
            )
            self._emit(
                OptimizationEventType.RUN_STARTED, run_context=run_context, iteration=0
            )
            self.GBE_vals = []
            self.history = []
            self.last_generation_evaluations = []
            initial_atoms = np.array(
                self.manipulator.parents[0].whole_system,
                copy=True,
            )
            # No mutation has occurred yet, so initial labels are the persistent labels
            # carried by the owned parent.
            initial_record = self._owned_evaluator.evaluate_candidate(
                self.manipulator,
                initial_atoms,
                f"GA_initial{unique_id}",
                -1,
            )
            self._emit(
                OptimizationEventType.INITIAL_EVALUATION,
                run_context=run_context,
                iteration=0,
                **evaluation_event_fields(from_candidate_evaluation(initial_record)),
            )
            if not initial_record.success or initial_record.structure_path is None:
                self._emit(
                    OptimizationEventType.RUN_FAILED,
                    run_context=run_context,
                    iteration=0,
                    failure_stage=(
                        initial_record.failure_stage or FailureStage.EVALUATOR
                    ),
                    failure_message=(
                        initial_record.failure_reason or "initial evaluation failed"
                    ),
                )
                raise GBMinimizerError(
                    "initial explicit-ownership evaluation failed: "
                    f"{initial_record.failure_reason}"
                )
            self.GBE_vals.append(
                [from_candidate_evaluation(initial_record).selection_energy]
            )
            best_record = initial_record
            self.best_evaluation = best_record
            logger.debug(
                "GA run %s initial evaluation (owned mode): energy=%.6g",
                unique_id,
                initial_record.objective,
            )
            if self.artifact_store is not None:
                self._register_owned_retention_candidate(
                    initial_record, generation=0, lineage=()
                )
                self.artifact_store.replace_pin(
                    ArtifactPin.BEST_RESULT, initial_record.candidate_id
                )
                if checkpoint.enabled:
                    materializable_records[initial_record.candidate_id] = initial_record

            population_manipulators = []
            population_structures = []
            population_lineages = []
            population_operation_parameters: list[Mapping[str, object] | None] = []
            population_retention_lineages: list[tuple[str, ...]] = []
            population_cached_evaluations: list[
                CandidateEvaluation | None
            ] = []
            seed_manipulator = self._clone_owned_record(initial_record)
            population_manipulators.append(seed_manipulator)
            population_structures.append(
                np.array(seed_manipulator.parents[0].whole_system, copy=True)
            )
            population_lineages.append(["START", initial_record.structure_path])
            population_operation_parameters.append(None)
            population_retention_lineages.append((initial_record.candidate_id,))
            population_cached_evaluations.append(None)

            for _ in range(self.population_size - 1):
                candidate_manipulator = self._clone_owned_record(initial_record)
                mutation, candidate_structure, mutation_parameters = self.mutator.mutate(
                    local_random=self.local_random,
                    GB=self.GB,
                    manipulator=candidate_manipulator,
                )
                population_manipulators.append(candidate_manipulator)
                population_structures.append(candidate_structure)
                population_lineages.append([mutation, initial_record.structure_path])
                population_operation_parameters.append(mutation_parameters)
                population_retention_lineages.append((initial_record.candidate_id,))
                population_cached_evaluations.append(None)
            _start_gen = 0

        def _build_owned_state(gen: int) -> dict:
            """Return one callback-free, typed-snapshot checkpoint payload for ``gen``.

            :param gen: Completed GA generation represented by the checkpoint.
            :return: Serializable checkpoint payload.
            """
            best_snapshot = _owned_evaluation_to_snapshot(
                self._owned_evaluation_to_state(best_record)
            )
            population_snapshot = [
                PopulationCandidateSnapshot(
                    artifact=StructureArtifact(
                        path=snap["structure_path"], format=_STRUCTURE_FORMAT
                    ),
                    lineage=_lineage_step_from_v1(lineage),
                    mapping=(
                        None
                        if snap["mapping"] is None
                        else _candidate_mapping_from_state(snap["mapping"])
                    ),
                )
                for snap, lineage in zip(
                    population_snapshots, population_lineages, strict=True
                )
            ]
            population_cache_snapshot = [
                None
                if record is None
                else _owned_evaluation_to_snapshot(
                    self._owned_evaluation_to_state(record)
                )
                for record in population_cached_evaluations
            ]
            population_cache_mapping_objects = [
                None if record is None else record.mapping
                for record in population_cached_evaluations
            ]
            generation_history_snapshot = [
                [
                    GenerationHistoryEntrySnapshot(
                        lineage=_lineage_step_from_v1(lineage), energy=energy
                    )
                    for lineage, energy in gen_history
                ]
                for gen_history in self.history
            ]
            last_generation_snapshot = [
                CandidateEvaluationSnapshot(
                    candidate_id=record.candidate_id,
                    input_index=record.input_index,
                    status=(
                        EvaluationStatus.SUCCESS
                        if record.success
                        else EvaluationStatus.FAILED
                    ),
                    selection_energy=record.objective,
                    energy=record.objective if record.success else None,
                    failure_stage=(
                        None if record.success else FailureStage.EVALUATOR
                    ),
                    failure_message=(
                        None
                        if record.success
                        else (record.failure_reason or "unknown evaluation failure")
                    ),
                )
                for record in self.last_generation_evaluations
            ]
            failure_diagnostics_snapshot = [
                FailureDiagnosticSnapshot(
                    candidate_id=diagnostic.candidate_id,
                    generation=diagnostic.generation,
                    input_index=diagnostic.input_index,
                    failure_reason=diagnostic.failure_reason,
                    source_path=diagnostic.source_path,
                )
                for diagnostic in self._failure_diagnostics
            ]
            retention_archive_mapping_objects = {
                candidate_id: _candidate_mapping_from_state(
                    self._retention_archive_mappings[candidate_id]
                )
                for candidate_id in sorted(self._retention_archive_mappings)
            }
            snapshot = GeneticAlgorithmSnapshot(
                run=RunIdentitySnapshot(
                    run_id=str(unique_id),
                    seed=self.seed,
                    case_id=self.case_id,
                    campaign_id=self.campaign_id,
                ),
                rng=RngStateSnapshot.from_generator(self.local_random),
                completed_generation=gen,
                best=best_snapshot,
                population=population_snapshot,
                configuration=GeneticAlgorithmConfigurationSnapshot(
                    slice_and_merge_pct=self.slice_and_merge_pct,
                    reuse_carryover_evaluations=self.reuse_carryover_evaluations,
                    population_size=self.population_size,
                    keep_top_pct=self.keep_top_pct,
                    intermediate_pct=self.intermediate_pct,
                    allow_variable_cell=self.allow_variable_cell,
                    choices=self.mutator.choices_keys,
                    crossover_surface=self.crossover_surface,
                    crossover_max_tilt_degrees=self.crossover_max_tilt_degrees,
                    crossover_attempts=self.crossover_attempts,
                    failure_diagnostic_count=self.failure_diagnostic_count,
                    composition_policy=self.composition_policy,
                ),
                population_cache=population_cache_snapshot,
                energy_history=[list(gen_vals) for gen_vals in self.GBE_vals],
                generation_history=generation_history_snapshot,
                retention_lineages=[
                    list(lineage) for lineage in population_retention_lineages
                ],
                last_generation_evaluations=last_generation_snapshot,
                failure_diagnostics=failure_diagnostics_snapshot,
                claimed_paths=self._owned_evaluator.claimed_paths_state(),
                retention_state=(
                    None
                    if self.artifact_store is None
                    else self.artifact_store.to_state()
                ),
                retention_archive_mappings=retention_archive_mapping_objects,
                best_mapping=best_record.mapping,
                population_cache_mappings=population_cache_mapping_objects,
            )
            return {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "minimizer": _GA_MINIMIZER_NAME,
                "progress_unit": _GA_PROGRESS_UNIT,
                "snapshot": snapshot.to_state(),
            }

        pending_failure_diagnostics: list[_FailureDiagnostic] = []
        _last_completed_gen = -1

        for gen in range(_start_gen, self.generations):
            current_pending = [
                snapshot["structure_path"]
                for snapshot in population_snapshots
                if str(snapshot.get("structure_path", "")).endswith(
                    ".owned.pending"
                )
            ]
            all_uids = [
                f"GA_{unique_id}_g{gen}_c{index}"
                for index in range(len(population_structures))
            ]
            try:
                gen_checkpoint = (
                    CandidateCheckpoint.new_or_resume(
                        checkpoint_file,
                        checkpoint_format,
                        gen,
                        all_uids,
                    )
                    if checkpoint.enabled
                    else None
                )
                records = self._owned_evaluator.evaluate_generation(
                    population_manipulators,
                    population_structures,
                    population_lineages,
                    gen,
                    unique_id,
                    gen_checkpoint=gen_checkpoint,
                    cached_evaluations=population_cached_evaluations,
                )
            except CheckpointError as exc:
                raise GBMinimizerError(str(exc)) from exc
            for record in records:
                self._emit(
                    OptimizationEventType.PROPOSAL_EVALUATED,
                    run_context=run_context,
                    iteration=gen,
                    operation_name=population_lineages[record.input_index][0],
                    operation_parameters=population_operation_parameters[
                        record.input_index
                    ],
                    **evaluation_event_fields(from_candidate_evaluation(record)),
                )
            self.last_generation_evaluations = records
            if self.artifact_store is not None:
                for record in records:
                    if record.success:
                        continue
                    diagnostic = _FailureDiagnostic.from_evaluation(
                        record, generation=gen
                    )
                    self._record_failure_provenance(diagnostic)
                    if (
                        self.artifact_store.pruning_enabled
                        and diagnostic.source_path is not None
                    ):
                        pending_failure_diagnostics.append(diagnostic)

            if self.artifact_store is not None:
                for record, lineage in zip(
                    records, population_retention_lineages, strict=True
                ):
                    if record.success:
                        self._register_owned_retention_candidate(
                            record, generation=gen, lineage=lineage
                        )
                        if checkpoint.enabled:
                            materializable_records[record.candidate_id] = record
                        if gen_checkpoint is not None and record.candidate_id == (
                            f"GA_{unique_id}_g{gen}_c{record.input_index}"
                        ):
                            self.artifact_store.pin(
                                record.candidate_id, ArtifactPin.CANDIDATE_CHECKPOINT
                            )
            # GA selection consumes selection_energy (the optimizer-facing value:
            # physical energy on success, penalty on failure), never overwriting the
            # physical energy CandidateEvaluation.objective itself already carries
            # unambiguously via its own success flag -- this adapts on read rather
            # than migrating the surrounding checkpoint/lineage bookkeeping, which
            # already keys off CandidateEvaluation's existing, correct shape.
            generation_energies = [
                from_candidate_evaluation(record).selection_energy
                for record in records
            ]
            self.GBE_vals.append(generation_energies)
            self.history.append(list(zip(population_lineages, generation_energies)))
            valid_records = [record for record in records if record.success]

            if not valid_records:
                for record in records:
                    self._emit(
                        OptimizationEventType.CANDIDATE_REJECTED,
                        run_context=run_context,
                        iteration=gen,
                        operation_name=population_lineages[record.input_index][0],
                        operation_parameters=population_operation_parameters[
                            record.input_index
                        ],
                        **evaluation_event_fields(from_candidate_evaluation(record)),
                    )
                self._emit(
                    OptimizationEventType.POPULATION_RESEEDED,
                    run_context=run_context,
                    iteration=gen,
                    candidate_id=None,
                    selection_energy=from_candidate_evaluation(
                        best_record
                    ).selection_energy,
                )
                next_manipulators: list[GBManipulator] = []
                next_structures: list[np.ndarray] = []
                next_lineages: list[list[str]] = []
                next_operation_parameters: list[Mapping[str, object] | None] = []
                next_retention_lineages: list[tuple[str, ...]] = []
                next_cached_evaluations: list[
                    CandidateEvaluation | None
                ] = []
                for _ in range(self.population_size):
                    candidate_manipulator = self._clone_owned_record(best_record)
                    mutation, candidate_structure, mutation_parameters = self.mutator.mutate(
                        local_random=self.local_random,
                        GB=self.GB,
                        manipulator=candidate_manipulator,
                    )
                    next_manipulators.append(candidate_manipulator)
                    next_structures.append(candidate_structure)
                    next_lineages.append([mutation, best_record.structure_path])
                    next_operation_parameters.append(mutation_parameters)
                    next_retention_lineages.append((best_record.candidate_id,))
                    next_cached_evaluations.append(None)
                population_manipulators = next_manipulators
                population_structures = next_structures
                population_lineages = next_lineages
                population_operation_parameters = next_operation_parameters
                population_cached_evaluations = next_cached_evaluations
            else:
                best_selection_energy = from_candidate_evaluation(
                    best_record
                ).selection_energy
                for record in valid_records:
                    record_selection_energy = from_candidate_evaluation(
                        record
                    ).selection_energy
                    if record_selection_energy < best_selection_energy:
                        logger.debug(
                            "GA run %s new best at generation %d (owned mode): "
                            "energy %.6g (was %.6g)",
                            unique_id,
                            gen,
                            record_selection_energy,
                            best_selection_energy,
                        )
                        best_record = record
                        best_selection_energy = record_selection_energy
                        self.best_evaluation = record
                        self._emit(
                            OptimizationEventType.BEST_UPDATED,
                            run_context=run_context,
                            iteration=gen,
                            **evaluation_event_fields(
                                from_candidate_evaluation(record)
                            ),
                        )
                        if self.artifact_store is not None:
                            self.artifact_store.replace_pin(
                                ArtifactPin.BEST_RESULT, record.candidate_id
                            )

                valid_energies = [
                    from_candidate_evaluation(record).selection_energy
                    for record in valid_records
                ]
                lowest_indices, intermediate_indices = self._select_indices_by_energy(
                    valid_energies
                )
                selected_candidate_ids = {
                    valid_records[index].candidate_id for index in lowest_indices
                } | {
                    valid_records[index].candidate_id for index in intermediate_indices
                }
                for record in records:
                    self._emit(
                        (
                            OptimizationEventType.CANDIDATE_ACCEPTED
                            if record.candidate_id in selected_candidate_ids
                            else OptimizationEventType.CANDIDATE_REJECTED
                        ),
                        run_context=run_context,
                        iteration=gen,
                        operation_name=population_lineages[record.input_index][0],
                        operation_parameters=population_operation_parameters[
                            record.input_index
                        ],
                        **evaluation_event_fields(from_candidate_evaluation(record)),
                    )
                next_manipulators = []
                next_structures = []
                next_lineages = []
                next_operation_parameters = []
                next_retention_lineages = []
                next_cached_evaluations = []
                for index in lowest_indices:
                    record = valid_records[index]
                    carryover = self._clone_owned_record(record)
                    next_manipulators.append(carryover)
                    next_structures.append(
                        np.array(carryover.parents[0].whole_system, copy=True)
                    )
                    next_lineages.append(["carryover", record.structure_path])
                    next_operation_parameters.append(None)
                    next_retention_lineages.append((record.candidate_id,))
                    next_cached_evaluations.append(
                        record if self.reuse_carryover_evaluations else None
                    )

                offspring_count = self.population_size - len(next_manipulators)
                new_manipulators, new_structures, new_lineages, new_operation_parameters = (
                    self._make_next_owned_generation(
                        valid_records,
                        intermediate_indices,
                        offspring_count,
                    )
                )
                next_manipulators.extend(new_manipulators)
                next_structures.extend(new_structures)
                next_lineages.extend(new_lineages)
                next_operation_parameters.extend(new_operation_parameters)
                path_to_candidate_id = {
                    str(record.structure_path): record.candidate_id
                    for record in valid_records
                    if record.structure_path is not None
                }
                for lineage in new_lineages:
                    next_retention_lineages.append(
                        tuple(
                            path_to_candidate_id[value]
                            for value in lineage[1:]
                            if value in path_to_candidate_id
                        )
                    )
                next_cached_evaluations.extend([None] * len(new_lineages))
            if not (
                len(next_manipulators)
                == len(next_structures)
                == len(next_lineages)
                == len(next_retention_lineages)
                == self.population_size
            ):
                raise GBMinimizerError(
                    "owned GA failed to construct a complete aligned population"
                )
            population_manipulators = next_manipulators
            population_structures = next_structures
            population_lineages = next_lineages
            population_operation_parameters = next_operation_parameters
            population_retention_lineages = next_retention_lineages
            population_cached_evaluations = next_cached_evaluations

            is_final_gen = gen == self.generations - 1
            committed = checkpoint.enabled and (
                checkpoint.is_due(gen + 1) or is_final_gen
            )
            archive_evictions: list[tuple[str, str]] = []
            failure_evictions: list[_FailureDiagnostic] = []
            if committed:
                new_snapshots = self._write_owned_population_checkpoint(
                    checkpoint_file,
                    str(unique_id),
                    gen + 1,
                    population_manipulators,
                    population_structures,
                )
                population_snapshots = new_snapshots
                population_cached_evaluations = self._rebase_owned_carryover_cache(
                    population_cached_evaluations,
                    new_snapshots,
                    population_manipulators,
                )
                if self.artifact_store is not None:
                    for artifact in self.artifact_store.records():
                        self.artifact_store.release_pin(
                            artifact.candidate_id, ArtifactPin.CANDIDATE_CHECKPOINT
                        )
                    records_by_id = dict(materializable_records)
                    records_by_id[best_record.candidate_id] = best_record
                    archive_root = self._owned_archive_root(
                        checkpoint_file, str(unique_id)
                    )
                    archive_evictions = self._prepare_owned_archive_state(
                        records_by_id, archive_root
                    )
                    best_archive = self.artifact_store.archive_path(
                        best_record.candidate_id
                    )
                    if best_archive is None or best_record.mapping is None:
                        raise GBMinimizerError(
                            "current best candidate lacks a durable archive"
                        )
                    # pyraisecontract: ignore=DOC115[TypeError]
                    #   BEST_RESULT archives are normalized to string paths by the
                    #   artifact store, and best_record is a validated successful result.
                    best_record = self._rebase_owned_evaluation(
                        best_record,
                        structure_path=best_archive,
                        mapping=best_record.mapping,
                        manipulator=best_record.manipulator,
                    )
                    self.best_evaluation = best_record
                    if self.artifact_store.pruning_enabled:
                        failure_evictions = self._update_failure_diagnostics(
                            pending_failure_diagnostics
                        )
                try:
                    checkpoint.save_final(_build_owned_state(gen))
                except CheckpointError as exc:
                    raise GBMinimizerError(str(exc)) from exc
                materializable_records.clear()
                for path in current_pending:
                    try:
                        Path(path).unlink(missing_ok=True)
                    except OSError as exc:
                        warnings.warn(
                            f"Artifact cleanup failed for {path}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

            # Candidate sidecars are transient once the generation boundary is safely
            # represented by the main checkpoint (or checkpointing is disabled).
            if gen_checkpoint is not None:
                try:
                    gen_checkpoint.delete()
                except OSError as exc:
                    warnings.warn(
                        f"Candidate-sidecar cleanup failed: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if self.artifact_store is not None and not committed:
                    for record in records:
                        if (
                            record.success
                            and record.candidate_id in self.artifact_store
                        ):
                            self.artifact_store.release_pin(
                                record.candidate_id, ArtifactPin.CANDIDATE_CHECKPOINT
                            )
            if committed and self.artifact_store is not None:
                provenance_ready = self._write_owned_artifact_manifest()
                if provenance_ready:
                    _cleanup_committed_artifacts(
                        self.artifact_store,
                        self._artifact_cleaner,
                        self._artifact_provenance,
                        archive_evictions,
                        archive_root=archive_root,
                    )
                    self._cleanup_failure_diagnostics(failure_evictions)
                else:
                    warnings.warn(
                        (
                            "Artifact cleanup deferred because required calculation "
                            "provenance could not be persisted"
                        ),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                pending_failure_diagnostics.clear()
                self._write_owned_artifact_manifest()
            elif self.artifact_store is not None:
                self._write_owned_artifact_manifest()

            logger.debug(
                "GA run %s generation %d complete (owned mode): %d/%d candidates "
                "valid, best energy %.6g",
                unique_id,
                gen,
                len(valid_records),
                len(records),
                from_candidate_evaluation(best_record).selection_energy,
            )

            _last_completed_gen = gen
            self._emit(
                OptimizationEventType.GENERATION_BOUNDARY,
                run_context=run_context,
                iteration=gen,
                selection_energy=from_candidate_evaluation(
                    best_record
                ).selection_energy,
            )

        if _last_completed_gen >= 0:
            self._emit(
                OptimizationEventType.RUN_TERMINATED,
                run_context=run_context,
                iteration=_last_completed_gen,
                termination_reason=TerminationReason.MAX_GENERATIONS,
            )

        self.best_evaluation = best_record
        return best_record.objective, str(best_record.structure_path)
