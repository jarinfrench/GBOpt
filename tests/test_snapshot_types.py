# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import numpy as np
import pytest

from GBOpt.evaluation import EvaluationResult, EvaluationStatus, FailureStage, StructureArtifact
from GBOpt.FileGrainOwnership import BoundaryNormalTopology, CandidateFileMapping
from GBOpt.snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
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
    SnapshotTypeError,
    SnapshotValueError,
)


def _artifact(path="candidate.data"):
    return StructureArtifact(path=path, format="lammps")


def _mapping():
    return CandidateFileMapping(
        atom_ids=np.arange(1, 5),
        labels=np.array([0, 0, 1, 1], dtype=np.int8),
        species=np.array(["Ni", "Ni", "Ni", "Ni"], dtype=object),
        box_dims=np.array([[0.0, 10.0], [0.0, 5.0], [0.0, 5.0]]),
        gb_plane_x=5.0,
        inplane_periodic=(True, True),
        left_grain_x_bounds=(0.0, 5.0),
        right_grain_x_bounds=(5.0, 10.0),
        coordinate_tolerance=1e-6,
        normal_topology=BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
    )


def _run():
    return RunIdentitySnapshot(run_id="run-1", seed=7)


def _success_eval(candidate_id="c1", *, artifact=None):
    return CandidateEvaluationSnapshot(
        candidate_id=candidate_id,
        input_index=0,
        status=EvaluationStatus.SUCCESS,
        selection_energy=1.5,
        energy=1.5,
        artifact=_artifact() if artifact is None else artifact,
    )


def _failed_eval(candidate_id="c2"):
    return CandidateEvaluationSnapshot(
        candidate_id=candidate_id,
        input_index=1,
        status=EvaluationStatus.FAILED,
        selection_energy=1.0e30,
        failure_stage=FailureStage.EVALUATOR,
        failure_message="boom",
    )


def _lineage():
    return LineageStepSnapshot(
        operation_name="translate_right_grain", parent_references=("parent.data",)
    )


class TestRngStateSnapshot:
    def test_round_trips_exact_draws(self):
        rng = np.random.default_rng(12345)
        snapshot = RngStateSnapshot.from_generator(rng)
        restored = snapshot.to_generator()
        assert np.array_equal(rng.random(10), restored.random(10))

    def test_from_generator_rejects_non_generator(self):
        with pytest.raises(SnapshotTypeError):
            RngStateSnapshot.from_generator(object())

    def test_mismatched_bit_generator_name_rejected(self):
        raw = np.random.default_rng(1).bit_generator.state
        with pytest.raises(SnapshotValueError):
            RngStateSnapshot(bit_generator="MT19937", state=raw)

    def test_rejects_non_json_safe_state(self):
        raw = dict(np.random.default_rng(1).bit_generator.state)
        raw["callback"] = lambda: None
        with pytest.raises(SnapshotTypeError):
            RngStateSnapshot(bit_generator=raw["bit_generator"], state=raw)

    def test_unknown_bit_generator_name_rejected_on_reconstruction(self):
        snapshot = RngStateSnapshot(
            bit_generator="NotARealBitGenerator",
            state={"bit_generator": "NotARealBitGenerator"},
        )
        with pytest.raises(SnapshotValueError):
            snapshot.to_generator()


class TestRunIdentitySnapshot:
    def test_valid_construction(self):
        run = RunIdentitySnapshot(run_id="abc", seed=1, case_id="case", campaign_id="camp")
        assert run.run_id == "abc"
        assert run.seed == 1

    def test_empty_run_id_rejected(self):
        with pytest.raises(SnapshotTypeError):
            RunIdentitySnapshot(run_id="", seed=1)

    def test_boolean_seed_rejected(self):
        with pytest.raises(SnapshotTypeError):
            RunIdentitySnapshot(run_id="abc", seed=True)

    def test_to_run_context_shares_identity(self):
        from GBOpt.observability import OptimizationAlgorithm

        run = RunIdentitySnapshot(run_id="abc", seed=3, case_id="c", campaign_id="k")
        context = run.to_run_context(algorithm=OptimizationAlgorithm.MONTE_CARLO)
        assert context.run_id == "abc"
        assert context.seed == 3
        assert context.case_id == "c"
        assert context.campaign_id == "k"
        assert context.algorithm is OptimizationAlgorithm.MONTE_CARLO


class TestCandidateEvaluationSnapshot:
    def test_successful_requires_energy(self):
        with pytest.raises(SnapshotValueError):
            CandidateEvaluationSnapshot(
                candidate_id="c",
                input_index=0,
                status=EvaluationStatus.SUCCESS,
                selection_energy=1.0,
            )

    def test_successful_artifact_is_optional(self):
        record = CandidateEvaluationSnapshot(
            candidate_id="c",
            input_index=0,
            status=EvaluationStatus.SUCCESS,
            selection_energy=1.0,
            energy=1.0,
        )
        assert record.artifact is None

    def test_failed_requires_stage_and_message(self):
        with pytest.raises(SnapshotValueError):
            CandidateEvaluationSnapshot(
                candidate_id="c",
                input_index=0,
                status=EvaluationStatus.FAILED,
                selection_energy=1.0e30,
            )

    def test_failed_rejects_energy(self):
        with pytest.raises(SnapshotValueError):
            CandidateEvaluationSnapshot(
                candidate_id="c",
                input_index=0,
                status=EvaluationStatus.FAILED,
                selection_energy=1.0e30,
                energy=1.0,
                failure_stage=FailureStage.EVALUATOR,
                failure_message="boom",
            )

    def test_from_evaluation_result_drops_manipulator(self):
        result = EvaluationResult(
            candidate_id="c1",
            input_index=0,
            status=EvaluationStatus.SUCCESS,
            selection_energy=1.0,
            energy=1.0,
            artifact=_artifact(),
            manipulator=object(),
        )
        snapshot = CandidateEvaluationSnapshot.from_evaluation_result(result)
        assert snapshot.candidate_id == "c1"
        assert not hasattr(snapshot, "manipulator")

    def test_from_evaluation_result_rejects_wrong_type(self):
        with pytest.raises(SnapshotTypeError):
            CandidateEvaluationSnapshot.from_evaluation_result(object())

    def test_accepts_negative_input_index_sentinel(self):
        record = _success_eval()
        assert record.input_index == 0
        failed = CandidateEvaluationSnapshot(
            candidate_id="init",
            input_index=-1,
            status=EvaluationStatus.FAILED,
            selection_energy=1.0e30,
            failure_stage=FailureStage.EVALUATOR,
            failure_message="boom",
        )
        assert failed.input_index == -1


class TestLineageStepSnapshot:
    def test_single_parent(self):
        step = LineageStepSnapshot(
            operation_name="mutate", parent_references=("parent.data",)
        )
        assert step.parent_references == ("parent.data",)
        assert step.diagnostic_note is None

    def test_two_parents_with_diagnostic_note(self):
        step = LineageStepSnapshot(
            operation_name="slice_and_merge",
            parent_references=("p1.data", "p2.data"),
            diagnostic_note="{'surface_mode': 'periodic_wave'}",
        )
        assert step.parent_references == ("p1.data", "p2.data")
        assert step.diagnostic_note == "{'surface_mode': 'periodic_wave'}"

    def test_empty_parent_references_rejected(self):
        with pytest.raises(SnapshotValueError):
            LineageStepSnapshot(operation_name="mutate", parent_references=())

    def test_empty_operation_name_rejected(self):
        with pytest.raises(SnapshotTypeError):
            LineageStepSnapshot(operation_name="", parent_references=("p.data",))


class TestPopulationCandidateSnapshot:
    def test_valid_construction_without_mapping(self):
        candidate = PopulationCandidateSnapshot(artifact=_artifact(), lineage=_lineage())
        assert candidate.mapping is None

    def test_valid_construction_with_mapping(self):
        candidate = PopulationCandidateSnapshot(
            artifact=_artifact(), lineage=_lineage(), mapping=_mapping()
        )
        assert isinstance(candidate.mapping, CandidateFileMapping)

    def test_rejects_non_artifact(self):
        with pytest.raises(SnapshotTypeError):
            PopulationCandidateSnapshot(artifact=object(), lineage=_lineage())

    def test_rejects_live_object_as_mapping(self):
        with pytest.raises(SnapshotTypeError):
            PopulationCandidateSnapshot(
                artifact=_artifact(), lineage=_lineage(), mapping=object()
            )


class TestMonteCarloSnapshot:
    def _snapshot(self, **overrides):
        kwargs = dict(
            run=_run(),
            rng=RngStateSnapshot.from_generator(np.random.default_rng(1)),
            completed_step=3,
            temperature=0.5,
            rejection_count=1,
            previous_energy=2.0,
            best_energy=1.5,
            current_artifact=_artifact("current.data"),
            best_artifact=_artifact("best.data"),
            energy_history=(2.0, 1.8, 1.5),
            accepted_steps=(0, 2),
            step_history=(
                MonteCarloStepRecordSnapshot(operation_name="translate_right_grain", accepted=True),
            ),
        )
        kwargs.update(overrides)
        return MonteCarloSnapshot(**kwargs)

    def test_valid_construction_stamps_schema_version(self):
        snapshot = self._snapshot()
        assert snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION

    def test_best_artifact_optional(self):
        snapshot = self._snapshot(best_artifact=None)
        assert snapshot.best_artifact is None

    def test_negative_completed_step_rejected(self):
        with pytest.raises(SnapshotValueError):
            self._snapshot(completed_step=-1)

    def test_retention_state_rejects_live_object(self):
        with pytest.raises(SnapshotTypeError):
            self._snapshot(retention_state={"cleanup": lambda: None})

    def test_retention_state_accepts_json_safe_mapping(self):
        snapshot = self._snapshot(retention_state={"records": [1, 2, 3]})
        assert snapshot.retention_state["records"] == (1, 2, 3)


class TestGeneticAlgorithmSnapshot:
    def _snapshot(self, **overrides):
        population = (
            PopulationCandidateSnapshot(artifact=_artifact("p0.data"), lineage=_lineage()),
            PopulationCandidateSnapshot(artifact=_artifact("p1.data"), lineage=_lineage()),
        )
        kwargs = dict(
            run=_run(),
            rng=RngStateSnapshot.from_generator(np.random.default_rng(2)),
            completed_generation=1,
            best=_success_eval(),
            population=population,
            population_cache=(None, _success_eval("cache-1")),
            energy_history=((2.0, 1.5), (1.8, 1.4)),
            generation_history=(
                (GenerationHistoryEntrySnapshot(lineage=_lineage(), energy=2.0),),
            ),
            failure_diagnostics=(
                FailureDiagnosticSnapshot(
                    candidate_id="c-fail",
                    generation=0,
                    input_index=2,
                    failure_reason="evaluator crashed",
                ),
            ),
        )
        kwargs.update(overrides)
        return GeneticAlgorithmSnapshot(**kwargs)

    def test_valid_construction_stamps_schema_version(self):
        snapshot = self._snapshot()
        assert snapshot.schema_version == SNAPSHOT_SCHEMA_VERSION
        assert len(snapshot.population) == 2

    def test_empty_population_rejected(self):
        with pytest.raises(SnapshotValueError):
            self._snapshot(population=())

    def test_best_must_be_successful(self):
        with pytest.raises(SnapshotValueError):
            self._snapshot(best=_failed_eval())

    def test_population_cache_length_must_match_population(self):
        with pytest.raises(SnapshotValueError):
            self._snapshot(population_cache=(None,))

    def test_retention_lineages_length_must_match_population(self):
        with pytest.raises(SnapshotValueError):
            self._snapshot(retention_lineages=[("a",)])

    def test_retention_lineages_aligned_accepted(self):
        snapshot = self._snapshot(retention_lineages=[("a",), ("b", "c")])
        assert snapshot.retention_lineages == (("a",), ("b", "c"))

    def test_retention_archive_mappings_validated(self):
        snapshot = self._snapshot(
            retention_archive_mappings={"cand-1": _mapping()}
        )
        assert isinstance(snapshot.retention_archive_mappings["cand-1"], CandidateFileMapping)

    def test_retention_archive_mappings_rejects_live_object(self):
        with pytest.raises(SnapshotTypeError):
            self._snapshot(retention_archive_mappings={"cand-1": object()})

    def test_last_generation_evaluations_length_must_match_population(self):
        with pytest.raises(SnapshotValueError):
            self._snapshot(last_generation_evaluations=(_success_eval(),))
