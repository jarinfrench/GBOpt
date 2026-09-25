# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from __future__ import annotations

import logging
import math
import shutil
import uuid
import warnings
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from GBOpt import GBMaker, GBManipulator
from GBOpt.artifacts.cleanup import (
    ArtifactCleanupRequest,
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
    CheckpointError,
    CheckpointStore,
)
from GBOpt.evaluation import (
    EvaluationResult,
    EvaluationStatus,
    FailureStage,
    from_scalar_tuple,
)
from GBOpt.GBManipulator import (
    GBManipulatorError,
    ParentError,
)
from GBOpt.manipulation import ManipulationRegistry
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
    _prepare_archive_state,
    _register_retention_candidate,
    _run_artifact_provenance,
    _write_artifact_manifest,
)
from GBOpt.optimization.mutation import Mutator
from GBOpt.optimization.types import (
    GBMinimizerError,
    GBMinimizerTypeError,
    GBMinimizerValueError,
)

logger = logging.getLogger(__name__)

MC_ENERGY_PENALTY: float = 1.0e30


class MonteCarloMinimizer:
    """
    Minimizer class for finding the lowest energy configuration of a grain boundary.
    Runs a Monte-Carlo minimization approach on the provided GBMaker object, applying
    the provided manipulator options stochastically.
    :param GB: GBMaker object to perform minimization on.
    :param gb_energy_func: A function that returns the energy of test GB structure.
        Currently expects a function that can be called with the params
        (GBMaker,GBManipulator,atom_positions,unique_id) .
    :param choices: A list of strings corresponding to GBManipulator operations. Used in
        setting up the Mutator class.
    :param seed: The seed to initialize the numpy.random.default_rng with.
    :ivar seed: The resolved seed actually passed to ``numpy.random.default_rng``
        (the current time when the constructor's ``seed`` argument is ``None``).
    """

    def __init__(
        self,
        GB: GBMaker,
        gb_energy_func: Callable,
        choices: list,
        seed=None,
        *,
        initial_structure: Any = None,
        registry: ManipulationRegistry | None = None,
        retention_policy: ArtifactRetentionPolicy | None = None,
        calculation_context: Mapping[str, object] | None = None,
        managed_artifact_root: str | Path | None = None,
        cleanup_candidate: Callable[[ArtifactCleanupRequest], None] | None = None,
        event_sink: EventSink | None = None,
        case_id: str | None = None,
        campaign_id: str | None = None,
    ):
        """Configure one Monte Carlo grain-boundary minimizer.

        :param GB: GBMaker object to perform minimization on.
        :param gb_energy_func: Function called with GBMaker, GBManipulator, atom
            positions, and a run identifier; returns objective and relaxed dump path.
        :param choices: GBManipulator operation names available to the mutator. Any name
            that is not one of the three legacy operation names is resolved by lookup in
            ``registry``.
        :param seed: Random-number seed. Keyword argument, optional, defaults to
            ``None``; ``None`` seeds from the current time.
        :param initial_structure: Keyword argument, optional, defaults to ``None``.
            Optional GBMaker or file-backed initial structure accepted by GBManipulator.
        :param registry: Keyword argument, optional, defaults to ``None``. Registry used
            to resolve any non-legacy ``choices`` name; ``None`` uses
            ``GBOpt.manipulation.default_registry``.
        :param retention_policy: Keyword argument, optional, defaults to ``None``.
            Scientific artifact-retention policy. ``None`` preserves legacy keep-all
            artifact behavior.
        :param calculation_context: Keyword argument, optional, defaults to ``None``.
            JSON-safe run-level calculator/campaign provenance. A non-empty mapping is
            required when pruning is enabled.
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
        :raises GBMinimizerTypeError: If artifact retention/cleanup configuration has an
            invalid type, or ``event_sink`` is neither ``None`` nor an ``EventSink``.
        :raises GBMinimizerValueError: If cleanup ownership is ambiguous or inconsistent
            with pruning configuration.
        """
        if event_sink is not None and not isinstance(event_sink, EventSink):
            raise GBMinimizerTypeError("event_sink must be an EventSink or None")
        self._event_sink: EventSink = (
            NullEventSink() if event_sink is None else event_sink
        )
        self.case_id = case_id
        self.campaign_id = campaign_id
        self.GB = GB
        self.gb_energy_func = gb_energy_func
        self.initial_structure = initial_structure
        self.retention_policy = retention_policy
        self._artifact_cleaner, self.artifact_store = _configure_artifact_runtime(
            retention_policy,
            managed_artifact_root,
            cleanup_candidate,
        )
        self.calculation_context = _normalize_calculation_context_config(
            calculation_context,
            retention_policy=retention_policy,
        )
        self._artifact_provenance: _ArtifactProvenance | None = None
        self.manipulator = self._make_initial_manipulator()
        self.mutator = Mutator(choices, self.manipulator, registry=registry)
        self.accepted_idx = [0]  # Initial guess is accepted by definition
        self.operation_list = [["START", True]]
        self.seed: int = int(time()) if seed is None else seed
        self.local_random = np.random.default_rng(self.seed)
        self.manipulator.rng = self.local_random
        self.GBE_vals: list[float] = []

    def _make_initial_manipulator(self) -> GBManipulator:
        """Build the starting GBManipulator from configured seed state.

        - gbmaker (self.GB) remains the authoritative reference for
          unit_cell/gb_thickness.
        - initial structure may be:
          * None -> Use GBManipulator(self.GB)
          * GBMaker -> generate starting structure from that maker
          * anything else -> pass to GBManipulator as a "structure spec" that it can
            read, while still injecting unit_cell/gb_thickness from self.GB.

        :return: Starting manipulator used by Monte Carlo mutation.
        """
        seed = self.initial_structure
        if seed is None:
            manip = GBManipulator(self.GB)
        elif isinstance(seed, GBMaker):
            manip = GBManipulator(seed)
        else:
            manip = GBManipulator(
                seed, unit_cell=self.GB.unit_cell, gb_thickness=self.GB.gb_thickness
            )

        return manip

    def _load_mc_relaxed_manipulator(
        self,
        structure_path: str | Path,
        *,
        type_dict: dict,
    ) -> GBManipulator:
        """Load one relaxed MC evaluator output into validated manipulator state.

        :param structure_path: Evaluator-returned relaxed structure path.
        :param type_dict: Keyword argument, required. LAMMPS type-to-element mapping.
        :return: File-backed manipulator aligned with the relaxed output.
        :raises GBMinimizerError: If the evaluator output cannot be reconstructed.
        """
        try:
            manipulator = GBManipulator(
                str(structure_path),
                unit_cell=self.GB.unit_cell,
                gb_thickness=self.GB.gb_thickness,
                type_dict=type_dict,
            )
        except (ParentError, GBManipulatorError) as exc:
            raise GBMinimizerError(
                f"could not reconstruct relaxed MC structure {structure_path!s}"
            ) from exc
        manipulator.rng = self.local_random
        return manipulator

    @staticmethod
    def _mc_candidate_id(unique_id: str, step: int) -> str:
        """Return a stable path-independent identity for one evaluated MC state.

        :param unique_id: Stable run identifier.
        :param step: Non-negative MC evaluation step; zero identifies the initial state.
        :return: Stable logical candidate identity.
        :raises GBMinimizerValueError: If identity components cannot produce a safe
            archive filename.
        """
        if not isinstance(unique_id, str) or not unique_id:
            raise GBMinimizerValueError("MC unique_id must be a non-empty string")
        if (
            isinstance(step, (bool, np.bool_))
            or not isinstance(step, Integral)
            or step < 0
        ):
            raise GBMinimizerValueError(
                "MC candidate step must be a non-negative integer")
        candidate_id = f"MC_{unique_id}_s{int(step)}"
        if Path(candidate_id).name != candidate_id or any(
            separator in candidate_id for separator in ("/", "\\")
        ):
            raise GBMinimizerValueError(
                "MC unique_id contains path separators that are unsafe for artifact "
                "identity"
            )
        return candidate_id

    def _register_mc_retention_candidate(
        self,
        *,
        candidate_id: str,
        step: int,
        objective: float,
        structure_path: str | Path,
        lineage: tuple[str, ...],
        type_dict: dict,
    ) -> GBManipulator:
        """Register one relaxed MC result and return its validated file-backed state.

        Property callbacks receive the relaxed evaluator output even when the trial is
        subsequently rejected by MC selection.

        :param candidate_id: Keyword argument, required. Stable logical candidate
            identity.
        :param step: Keyword argument, required. MC step where evaluation occurred.
        :param objective: Keyword argument, required. Evaluated grain-boundary energy.
        :param structure_path: Keyword argument, required. Relaxed evaluator output.
        :param lineage: Keyword argument, required. Currently accepted MC parent
            identity.
        :param type_dict: Keyword argument, required. LAMMPS type-to-element mapping.
        :return: Validated file-backed manipulator for the relaxed evaluator output.
        :raises GBMinimizerError: If retention is disabled or relaxed output/retention
            state is invalid.
        """
        if self.artifact_store is None or self.retention_policy is None:
            raise GBMinimizerError(
                "MC retention candidate registration requires artifact state"
            )
        try:
            candidate_manipulator = self._load_mc_relaxed_manipulator(
                structure_path,
                type_dict=type_dict,
            )
            parent = candidate_manipulator.parents[0]
            context = CandidatePropertyContext(
                candidate_id=candidate_id,
                generation=step,
                objective=objective,
                atoms=parent.whole_system,
                box_dims=parent.box_dims,
                grain_labels=parent.grain_labels,
                gb_plane_x=parent.gb_plane_x,
            )
            _register_retention_candidate(
                artifact_store=self.artifact_store,
                retention_policy=self.retention_policy,
                context=context,
                source_path=structure_path,
                lineage=lineage,
                provenance=self._artifact_provenance,
            )
        except (
            ArtifactPolicyError,
            ArtifactStoreError,
            ArtifactValueError,
        ) as exc:
            raise GBMinimizerError(
                f"artifact retention failed for candidate {candidate_id!r}: {exc}"
            ) from exc
        return candidate_manipulator

    @staticmethod
    def _mc_archive_root(checkpoint_file: Path | None, unique_id: str) -> Path:
        """Return the canonical artifact root for one MC run.

        :param checkpoint_file: Run checkpoint path, or ``None`` when disabled.
        :param unique_id: Stable run identifier.
        :return: Run-specific artifact archive root.
        """
        return _artifact_archive_root(
            checkpoint_file,
            fallback_stem=f"MC_{unique_id}",
        )

    def _materialize_mc_archive(self, candidate_id: str, archive_root: Path) -> str:
        """Create one canonical retained MC structure without changing identity.

        :param candidate_id: Registered logical candidate identity.
        :param archive_root: Run-owned archive root.
        :return: Canonical retained structure path.
        :raises GBMinimizerError: If source/store state or filesystem materialization is
            invalid.
        """
        if self.artifact_store is None:
            raise GBMinimizerError("MC archive materialization requires artifact state")
        try:
            record = self.artifact_store.record(candidate_id)
        except ArtifactStoreError as exc:
            raise GBMinimizerError(str(exc)) from exc
        if record.source_path is None:
            raise GBMinimizerError(
                f"retained candidate {candidate_id!r} lacks a source structure"
            )
        source = Path(record.source_path)
        if not source.is_file():
            raise GBMinimizerError(
                f"retained candidate source path {source} is missing"
            )
        destination = archive_root / "structures" / f"{candidate_id}.data"
        try:
            _materialize_archive_file(source, destination)
            self.artifact_store.set_archive_path(candidate_id, destination)
        except (OSError, ArtifactStoreError) as exc:
            raise GBMinimizerError(
                f"could not materialize retained candidate {candidate_id!r}"
            ) from exc
        _run_artifact_provenance(
            self._artifact_provenance,
            lambda: self._artifact_provenance.record_archive_created(
                candidate_id, destination
            ),
        )
        return str(destination)

    def _prepare_mc_archive_state(
        self,
        archive_root: Path,
    ) -> list[tuple[str, str]]:
        """Materialize required MC archives and detach eligible archive evictions.

        :param archive_root: Run-owned canonical archive root.
        :return: Candidate IDs and archive paths eligible for post-commit deletion.
        :raises GBMinimizerError: If required archive state cannot be materialized.
        """
        try:
            return _prepare_archive_state(
                self.artifact_store,
                lambda candidate_id: self._materialize_mc_archive(
                    candidate_id, archive_root
                ),
            )
        except ArtifactStoreError as exc:
            raise GBMinimizerError(str(exc)) from exc

    def _mc_pin_owner(self, pin: ArtifactPin) -> str:
        """Return the unique MC candidate carrying one singleton operational pin.

        :param pin: Operational pin expected to have exactly one owner.
        :return: Stable logical candidate identity.
        :raises GBMinimizerError: If artifact state is disabled or pin ownership is not
            unique.
        """
        if self.artifact_store is None:
            raise GBMinimizerError("MC artifact pin lookup requires artifact state")
        try:
            owners = [
                artifact.candidate_id
                for artifact in self.artifact_store.records()
                if pin in artifact.pins
            ]
        except ArtifactStoreError as exc:
            raise GBMinimizerError(str(exc)) from exc
        if len(owners) != 1:
            raise GBMinimizerError(
                f"MC artifact checkpoint requires exactly one {pin.value!r} pin owner"
            )
        return owners[0]

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
        :param iteration: Keyword argument, required. MC step this event reports on.
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
                "event emission failed for %s at MC step %d; continuing the run",
                event_type.value,
                iteration,
            )

    def run_MC(
        self,
        E_accept: float = 1e-1,
        min_steps: int = None,
        max_steps: int = 50,
        E_tol: float = 1e-4,
        max_rejections: int = 20,
        cooldown_rate: float = 1.0,
        unique_id: int | uuid.UUID | None = None,
        *,
        checkpoint_file: str | Path | None = None,
        checkpoint_format: str = "json",
        checkpoint_interval: int = 1,
        **kwargs,
    ) -> float:
        # TODO: Add options for changing from linear to logarithmic cooldown
        """Run Monte Carlo iterations until a configured convergence criterion is met.

        When artifact retention is configured, every successful relaxed evaluator result
        is classified before MC acceptance. Currently accepted and global-best
        structures receive independent operational pins. Pruning occurs only after a
        durable checkpoint commit.

        :param E_accept: Optional, defaults to ``1e-1``. Energy increase with a 50%
            acceptance probability at the initial MC temperature, in J/m^2.
        :param min_steps: Optional, defaults to ``None``. Minimum number of MC
            iterations before the energy-tolerance termination criterion may stop the
            run.
        :param max_steps: Optional, defaults to ``50``. Maximum MC iteration index.
        :param E_tol: Optional, defaults to ``1e-4``. Positive best-energy decrease at
            or below which the run may terminate, in J/m^2.
        :param max_rejections: Optional, defaults to ``20``. Maximum consecutive
            rejected trials before termination.
        :param cooldown_rate: Optional, defaults to ``1.0``. Finite factor in ``(0, 1]``
            applied to the MC temperature after each completed iteration.
        :param unique_id: Optional, defaults to ``None``. Output label for a fresh run;
            a UUID is generated when omitted and checkpoint resume restores the saved
            label.
        :param checkpoint_file: Keyword argument, optional, defaults to ``None``. Run
            checkpoint path. Resume restores current structure, RNG state, temperature,
            accepted history, stable run identity, retention state, ``min_steps``, and
            ``cooldown_rate``. ``max_steps`` may be increased on resume.
        :param checkpoint_format: Keyword argument, optional, defaults to ``"json"``.
            Checkpoint serialization format, ``"json"`` or ``"pickle"``.
        :param checkpoint_interval: Keyword argument, optional, defaults to ``1``. Save
            a periodic checkpoint every N completed steps; final state is always saved
            when checkpointing is enabled.
        :param **kwargs: Keyword arguments forwarded to ``gb_energy_func``.
        :return: Minimum grain-boundary energy encountered.
        :raises GBMinimizerTypeError: If ``cooldown_rate`` is not a non-Boolean real
            scalar.
        :raises GBMinimizerValueError: If ``cooldown_rate`` is non-finite/out of range,
            checkpoint configuration is invalid, or pruning is requested without a
            durable checkpoint.
        :raises GBMinimizerError: If checkpoint load/save, relaxed-result
            reconstruction, retention compatibility, archive materialization, or
            artifact state is invalid.
        """

        if isinstance(cooldown_rate, (bool, np.bool_)) or not isinstance(
            cooldown_rate, Real
        ):
            raise GBMinimizerTypeError(
                "cooldown_rate must be a non-Boolean real scalar"
            )
        cooldown_rate = float(cooldown_rate)
        if not math.isfinite(cooldown_rate) or not 0.0 < cooldown_rate <= 1.0:
            raise GBMinimizerValueError(
                "cooldown_rate must be finite and satisfy 0 < value <= 1"
            )
        checkpoint_path = None if checkpoint_file is None else Path(checkpoint_file)
        if (
            self.retention_policy is not None
            and self.retention_policy.prune
            and checkpoint_path is None
        ):
            raise GBMinimizerValueError(
                "retention_policy prune=True requires checkpoint_file for durable "
                "cleanup"
            )

        try:
            checkpoint = CheckpointStore.from_optional(
                checkpoint_file, checkpoint_format, checkpoint_interval
            )
        except CheckpointError as e:
            raise GBMinimizerValueError(str(e)) from e

        type_dict = {value: key for key,
                     value in self.GB.unit_cell.type_map.items()}

        try:
            state = checkpoint.load()
        except CheckpointError as e:
            raise GBMinimizerError(str(e)) from e

        current_candidate_id: str | None = None
        best_candidate_id: str | None = None
        if state is not None:
            self.GBE_vals = state["state"]["GBE_vals"]
            self.accepted_idx = state["state"]["accepted_idx"]
            self.operation_list = state["state"]["operation_list"]
            self.local_random.bit_generator.state = state["rng_state"]
            unique_id = str(state["run_params"]["unique_id"])
            min_steps = state["run_params"]["min_steps"]
            cooldown_rate = state["run_params"]["cooldown_rate"]
            self.seed = state["run_params"].get("seed", self.seed)
            _resume_step = state["progress_index"] + 1
            T = state["state"]["T"]
            rejection_count = state["state"]["rejection_count"]
            min_gbe = state["best_energy"]
            prev_gbe = state["state"]["prev_gbe"]
            best_dump = state["best_dump"]
            _current_dump = state["state"]["current_structure_dump"]
            self.manipulator = self._load_mc_relaxed_manipulator(
                _current_dump,
                type_dict=type_dict,
            )

            retention_state = state["state"].get("artifact_store")
            if retention_state is None:
                if self.retention_policy is not None:
                    raise GBMinimizerError(
                        "checkpoint retention policy does not match the minimizer "
                        "configuration"
                    )
                self.artifact_store = None
            else:
                try:
                    self.artifact_store = ArtifactStore.from_state(
                        retention_state,
                        policy=self.retention_policy,
                    )
                except ArtifactStoreError as exc:
                    raise GBMinimizerError(str(exc)) from exc
                try:
                    for artifact in self.artifact_store.records():
                        if artifact.archive_path is not None and not Path(
                            artifact.archive_path
                        ).is_file():
                            raise GBMinimizerError(
                                f"retained archive path {artifact.archive_path} is "
                                "missing"
                            )
                except ArtifactStoreError as exc:
                    raise GBMinimizerError(str(exc)) from exc
                current_candidate_id = self._mc_pin_owner(ArtifactPin.RUN_CHECKPOINT)
                best_candidate_id = self._mc_pin_owner(ArtifactPin.BEST_RESULT)
            run_context = RunContext(
                run_id=unique_id,
                seed=self.seed,
                algorithm=OptimizationAlgorithm.MONTE_CARLO,
                case_id=self.case_id,
                campaign_id=self.campaign_id,
            )
            self._emit(
                OptimizationEventType.RUN_STARTED,
                run_context=run_context,
                iteration=_resume_step - 1,
            )
        else:
            _resume_step = 1
            unique_id = str(uuid.uuid4()) if unique_id is None else str(unique_id)
            run_context = RunContext(
                run_id=unique_id,
                seed=self.seed,
                algorithm=OptimizationAlgorithm.MONTE_CARLO,
                case_id=self.case_id,
                campaign_id=self.campaign_id,
            )
            self._emit(
                OptimizationEventType.RUN_STARTED, run_context=run_context, iteration=0
            )
            init_system = np.array(
                self.manipulator.parents[0].whole_system, copy=True)
            initial_candidate_id = "initial" + str(unique_id)
            try:
                init_gbe, _current_dump = self.gb_energy_func(
                    self.GB,
                    self.manipulator,
                    init_system,
                    initial_candidate_id,
                    **kwargs,
                )
            except Exception as exc:
                # There is no sensible penalized starting point for a whole MC run,
                # so an initial-evaluation failure is fatal -- matching
                # GeneticAlgorithmMinimizer's owned-mode initial evaluation, which
                # raises the same way rather than seeding a run from a penalty value.
                initial_result = EvaluationResult(
                    candidate_id=initial_candidate_id,
                    input_index=0,
                    status=EvaluationStatus.FAILED,
                    selection_energy=MC_ENERGY_PENALTY,
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
            initial_result = from_scalar_tuple(
                initial_candidate_id, 0, (init_gbe, _current_dump), penalty=MC_ENERGY_PENALTY
            )
            self._emit(
                OptimizationEventType.INITIAL_EVALUATION,
                run_context=run_context,
                iteration=0,
                **evaluation_event_fields(initial_result),
            )
            if initial_result.status is not EvaluationStatus.SUCCESS:
                self._emit(
                    OptimizationEventType.RUN_FAILED,
                    run_context=run_context,
                    iteration=0,
                    failure_stage=initial_result.failure_stage,
                    failure_message=initial_result.failure_message,
                )
                raise GBMinimizerError(
                    f"initial evaluation failed: {initial_result.failure_message}"
                )
            init_gbe = initial_result.selection_energy
            _current_dump = initial_result.artifact.path
            self.GBE_vals.append(init_gbe)
            T = -1 * E_accept / math.log(0.5)
            rejection_count = 0
            min_gbe = min(self.GBE_vals)
            prev_gbe = init_gbe
            best_dump = None

        archive_root = self._mc_archive_root(checkpoint_path, str(unique_id))
        self._artifact_provenance = None
        if self.artifact_store is not None:
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

        if state is None and self.artifact_store is not None:
            current_candidate_id = self._mc_candidate_id(str(unique_id), 0)
            self._register_mc_retention_candidate(
                candidate_id=current_candidate_id,
                step=0,
                objective=init_gbe,
                structure_path=_current_dump,
                lineage=(),
                type_dict=type_dict,
            )
            try:
                self.artifact_store.replace_pin(
                    ArtifactPin.RUN_CHECKPOINT, current_candidate_id
                )
                self.artifact_store.replace_pin(
                    ArtifactPin.BEST_RESULT, current_candidate_id
                )
            except ArtifactStoreError as exc:
                raise GBMinimizerError(str(exc)) from exc
            best_candidate_id = current_candidate_id

        _write_artifact_manifest(self.artifact_store, self._artifact_provenance)

        def _build_state(step):
            """Return one callback-free checkpoint payload for ``step``.

            :param step: Completed MC step represented by the checkpoint.
            :return: Serializable checkpoint payload.
            :raises GBMinimizerError: If artifact-store state cannot be serialized.
            """
            # Note that E_tol, max_rejections, and E_accept can be changed on resume;
            # run_params reflects the latest resume call for adjustable controls.
            try:
                artifact_state = (
                    None
                    if self.artifact_store is None
                    else self.artifact_store.to_state()
                )
            except ArtifactStoreError as exc:
                raise GBMinimizerError(str(exc)) from exc
            return {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "minimizer": "MonteCarloMinimizer",
                "progress_unit": "step",
                "progress_index": step,
                "best_energy": min_gbe,
                "best_dump": str(best_dump) if best_dump else None,
                "rng_state": self.local_random.bit_generator.state,
                "run_params": {
                    "E_accept": E_accept,
                    "min_steps": min_steps,
                    "max_steps": max_steps,
                    "E_tol": E_tol,
                    "max_rejections": max_rejections,
                    "cooldown_rate": cooldown_rate,
                    "unique_id": str(unique_id),
                    "seed": self.seed,
                },
                "state": {
                    "T": T,
                    "rejection_count": rejection_count,
                    "prev_gbe": prev_gbe,
                    "current_structure_dump": str(_current_dump),
                    "GBE_vals": self.GBE_vals,
                    "accepted_idx": self.accepted_idx,
                    "operation_list": self.operation_list,
                    "artifact_store": artifact_state,
                },
            }

        def _commit_step(step: int, *, final: bool) -> None:
            """Persist one durable MC boundary and clean only after commit.

            :param step: Completed MC step to persist.
            :param final: Keyword argument, required. Bypass periodic interval gating
                when ``True``.
            :raises GBMinimizerError: If archive preparation, artifact state, or
                checkpoint persistence fails.
            """
            nonlocal best_dump
            if not checkpoint.enabled:
                return
            if not final and not checkpoint.is_due(step):
                return
            archive_evictions: list[tuple[str, str]] = []
            if self.artifact_store is not None:
                archive_evictions = self._prepare_mc_archive_state(archive_root)
                if best_candidate_id is None:
                    raise GBMinimizerError(
                        "MC artifact state is missing the current best identity"
                    )
                try:
                    best_archive = self.artifact_store.archive_path(best_candidate_id)
                except ArtifactStoreError as exc:
                    raise GBMinimizerError(str(exc)) from exc
                if best_archive is None:
                    raise GBMinimizerError(
                        "current MC best candidate lacks a durable archive"
                    )
                best_dump = best_archive
            try:
                if final:
                    checkpoint.save_final(_build_state(step))
                else:
                    checkpoint.save_if_due(step, lambda: _build_state(step))
            except CheckpointError as exc:
                raise GBMinimizerError(str(exc)) from exc
            if self.artifact_store is not None:
                provenance_ready = _write_artifact_manifest(
                    self.artifact_store,
                    self._artifact_provenance,
                )
                if provenance_ready:
                    _cleanup_committed_artifacts(
                        self.artifact_store,
                        self._artifact_cleaner,
                        self._artifact_provenance,
                        archive_evictions,
                        archive_root=archive_root,
                    )
                else:
                    warnings.warn(
                        "Artifact cleanup deferred because required calculation "
                        "provenance could not be persisted",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            _write_artifact_manifest(self.artifact_store, self._artifact_provenance)

        _last_completed_step = state["progress_index"] if state is not None else -1
        _early_exit = False
        for i in range(_resume_step, max_steps + 1):
            lineage = (
                ()
                if current_candidate_id is None
                else (current_candidate_id,)
            )
            mutation, new_system = self.mutator.mutate(
                self.local_random, self.GB, self.manipulator
            )

            step_candidate_id = str(unique_id)
            try:
                new_gbe, dump_file_name = self.gb_energy_func(
                    self.GB,
                    self.manipulator,
                    new_system,
                    step_candidate_id,
                    **kwargs,
                )
            except Exception as exc:
                # The external evaluator callback is a deliberate recovery boundary:
                # any failure here penalizes and deterministically rejects only this
                # step rather than aborting the run, matching GA's own established
                # per-candidate recovery-boundary pattern.
                logger.warning(
                    "gb_energy_func failed for MC step %d (candidate %r): %s: %s",
                    i,
                    step_candidate_id,
                    type(exc).__name__,
                    exc,
                )
                step_result = EvaluationResult(
                    candidate_id=step_candidate_id,
                    input_index=i,
                    status=EvaluationStatus.FAILED,
                    selection_energy=MC_ENERGY_PENALTY,
                    failure_stage=FailureStage.EVALUATOR,
                    failure_message=f"{type(exc).__name__}: {exc}",
                )
            else:
                step_result = from_scalar_tuple(
                    step_candidate_id,
                    i,
                    (new_gbe, dump_file_name),
                    penalty=MC_ENERGY_PENALTY,
                )

            self._emit(
                OptimizationEventType.PROPOSAL_EVALUATED,
                run_context=run_context,
                iteration=i,
                operation_name=mutation,
                **evaluation_event_fields(step_result),
            )

            new_gbe = step_result.selection_energy
            dump_file_name = (
                step_result.artifact.path if step_result.artifact is not None else None
            )
            self.GBE_vals.append(new_gbe)

            if step_result.status is not EvaluationStatus.SUCCESS:
                # A failed proposal is deterministically rejected -- it never becomes
                # the new current/best structure, and this rejection does not consume
                # the acceptance-draw RNG state.
                accepted = False
            else:
                accepted = new_gbe <= prev_gbe or self.local_random.uniform(
                    0, 1
                ) <= math.exp(-(new_gbe - prev_gbe) / T)

            trial_candidate_id = None
            trial_manipulator = None
            if (
                self.artifact_store is not None
                and step_result.status is EvaluationStatus.SUCCESS
            ):
                trial_candidate_id = self._mc_candidate_id(str(unique_id), i)
                trial_manipulator = self._register_mc_retention_candidate(
                    candidate_id=trial_candidate_id,
                    step=i,
                    objective=new_gbe,
                    structure_path=dump_file_name,
                    lineage=lineage,
                    type_dict=type_dict,
                )

            if accepted:
                self._emit(
                    OptimizationEventType.CANDIDATE_ACCEPTED,
                    run_context=run_context,
                    iteration=i,
                    operation_name=mutation,
                    **evaluation_event_fields(step_result),
                )
                self.operation_list.append([mutation, True])
                self.manipulator = (
                    trial_manipulator
                    if trial_manipulator is not None
                    else self._load_mc_relaxed_manipulator(
                        dump_file_name,
                        type_dict=type_dict,
                    )
                )
                self.manipulator.rng = self.local_random
                _current_dump = dump_file_name
                prev_gbe = new_gbe
                self.accepted_idx.append(i)
                rejection_count = 0
                if self.artifact_store is not None:
                    if trial_candidate_id is None:
                        raise GBMinimizerError(
                            "accepted MC artifact is missing a candidate identity"
                        )
                    try:
                        self.artifact_store.replace_pin(
                            ArtifactPin.RUN_CHECKPOINT, trial_candidate_id
                        )
                    except ArtifactStoreError as exc:
                        raise GBMinimizerError(str(exc)) from exc
                    current_candidate_id = trial_candidate_id

                if new_gbe <= min_gbe:
                    best_dump = Path(dump_file_name).with_name(
                        "min_" + Path(dump_file_name).name)
                    shutil.copyfile(dump_file_name, best_dump)
                    del_E = min_gbe - new_gbe
                    min_gbe = new_gbe
                    self._emit(
                        OptimizationEventType.BEST_UPDATED,
                        run_context=run_context,
                        iteration=i,
                        **evaluation_event_fields(step_result),
                    )
                    if self.artifact_store is not None:
                        if trial_candidate_id is None:
                            raise GBMinimizerError(
                                "best MC artifact is missing a candidate identity"
                            )
                        try:
                            self.artifact_store.replace_pin(
                                ArtifactPin.BEST_RESULT, trial_candidate_id
                            )
                        except ArtifactStoreError as exc:
                            raise GBMinimizerError(str(exc)) from exc
                        best_candidate_id = trial_candidate_id
                    if 0 < del_E <= E_tol and (min_steps is None or i >= min_steps):
                        logger.info(
                            "MC run %s met energy tolerance criterion at step %d "
                            "(best energy %.6g)",
                            unique_id,
                            i,
                            min_gbe,
                        )
                        _last_completed_step = i
                        _commit_step(i, final=True)
                        self._emit(
                            OptimizationEventType.RUN_TERMINATED,
                            run_context=run_context,
                            iteration=i,
                            termination_reason=TerminationReason.ENERGY_TOLERANCE,
                        )
                        _early_exit = True
                        break
            else:
                self._emit(
                    OptimizationEventType.CANDIDATE_REJECTED,
                    run_context=run_context,
                    iteration=i,
                    operation_name=mutation,
                    **evaluation_event_fields(step_result),
                )
                self.operation_list.append([mutation, False])
                rejection_count += 1
                if rejection_count > max_rejections:
                    logger.info(
                        "MC run %s terminated after %d consecutive rejections "
                        "at step %d",
                        unique_id,
                        rejection_count,
                        i,
                    )
                    T *= cooldown_rate
                    _last_completed_step = i
                    _commit_step(i, final=True)
                    self._emit(
                        OptimizationEventType.RUN_TERMINATED,
                        run_context=run_context,
                        iteration=i,
                        termination_reason=TerminationReason.MAX_REJECTIONS,
                    )
                    _early_exit = True
                    break

            T *= cooldown_rate

            _last_completed_step = i
            if i < max_steps:
                _commit_step(i, final=False)
            _write_artifact_manifest(self.artifact_store, self._artifact_provenance)
        if not _early_exit and _last_completed_step >= 0:
            _commit_step(_last_completed_step, final=True)
            self._emit(
                OptimizationEventType.RUN_TERMINATED,
                run_context=run_context,
                iteration=_last_completed_step,
                termination_reason=TerminationReason.MAX_STEPS,
            )

        return min_gbe

