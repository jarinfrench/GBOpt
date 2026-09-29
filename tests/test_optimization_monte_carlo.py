# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from GBOpt.artifacts import ArtifactRetentionPolicy, KeepBest
from GBOpt.evaluation import EvaluationStatus
from GBOpt.GBMaker import GBMaker
from GBOpt.manipulation import ManipulationRegistry, ManipulationResult
from GBOpt.observability import (
    OptimizationAlgorithm,
    OptimizationEvent,
    OptimizationEventType,
    TerminationReason,
)
from GBOpt.optimization.monte_carlo import MC_ENERGY_PENALTY, MonteCarloMinimizer
from GBOpt.optimization.types import (
    GBMinimizerError,
    GBMinimizerTypeError,
    GBMinimizerValueError,
)

_TEST_CALCULATION_CONTEXT = {"calculator": {"name": "test-evaluator"}}


pytestmark = pytest.mark.filterwarnings(
    "ignore:File-backed Parent initialization without explicit grain ownership is "
    "deprecated.*:DeprecationWarning"
)


@pytest.fixture
def gb():
    theta = math.radians(36.869898)
    misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
    return GBMaker(
        3.52,
        "fcc",
        10.0,
        misorientation,
        "Ni",
        repeat_factor=(2, 5),
        x_dim_min=30.0,
        vacuum=8.0,
        interaction_distance=8.0,
    )


@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _make_energy_func(gb, crash_after=None):
    """Return a deterministic evaluator that may fail after a fixed call count."""
    call_count = 0

    def energy_func(GB, manipulator, atom_positions, unique_id):
        nonlocal call_count
        call_count += 1
        if crash_after is not None and call_count > crash_after:
            raise RuntimeError(f"Simulated crash at call {call_count}")
        path = f"{unique_id}_{call_count}.data"
        GB.write_lammps(
            path,
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return 2.0 - call_count * 0.001, path

    return energy_func


def _make_minimizer(gb, energy_func):
    return MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
    )


def _install_mutate_crash(mc, crash_after):
    """Make ``mc``'s mutator raise after ``crash_after`` successful calls.

    Evaluator exceptions no longer abort an MC run (a failed proposal is now
    deterministically rejected instead), so a test that needs a genuine mid-run
    crash -- to inspect checkpoint state at a specific, otherwise-unreachable
    intermediate step -- raises from the mutator instead, which still propagates
    uncaught. Returns the original ``mutate`` bound method so a caller that resumes
    the same minimizer can restore non-crashing behavior first.
    """
    original_mutate = mc.mutator.mutate
    call_count = 0

    def crashing_mutate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > crash_after:
            raise RuntimeError(f"Simulated crash at mutate call {call_count}")
        return original_mutate(*args, **kwargs)

    mc.mutator.mutate = crashing_mutate
    return original_mutate


def _make_sequence_energy_func(energies, root, *, start_index=0):
    """Return an evaluator writing deterministic relaxed files beneath ``root``."""
    values = iter(energies)
    call_count = start_index
    root.mkdir(parents=True, exist_ok=True)

    def energy_func(GB, manipulator, atom_positions, unique_id):
        nonlocal call_count
        call_count += 1
        path = root / f"{unique_id}_{call_count}.data"
        GB.write_lammps(
            str(path),
            atom_positions,
            manipulator.parents[0].box_dims,
        )
        return next(values), str(path)

    return energy_func


def test_run_mc_no_checkpoint_no_file_created(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))

    mc.run_MC(max_steps=2, unique_id=1)

    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.pkl")) == []
    assert mc.artifact_store is None
    assert list(tmp_path.glob("*.artifacts")) == []


def test_mc_retention_prunes_superseded_accepted_source_after_commit(gb, tmp_path):
    managed_root = tmp_path / "managed"
    energy_func = _make_sequence_energy_func([2.0, 1.0], managed_root)
    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="objective_best",
                property="objective",
                direction="min",
                count=1,
            ),
        ),
        prune=True,
    )
    mc = MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=managed_root,
    )
    checkpoint = tmp_path / "mc_retention.json"

    mc.run_MC(max_steps=1, unique_id=41, checkpoint_file=checkpoint)

    initial_source = managed_root / "initial41_1.data"
    current_source = managed_root / "41_2.data"
    archive = tmp_path / "mc_retention.artifacts" / "structures" / "MC_41_s1.data"
    assert not initial_source.exists()
    assert current_source.exists()
    assert archive.is_file()

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    manifest = json.loads(
        (checkpoint.with_suffix(".artifacts") / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["calculation_context"] == _TEST_CALCULATION_CONTEXT
    assert state["snapshot"]["best_artifact"]["path"] == str(archive)
    records = state["snapshot"]["retention_state"]["records"]
    by_id = {record["candidate"]["candidate_id"]: record for record in records}
    assert by_id["MC_41_s1"]["pins"] == ["best_result", "run_checkpoint"]
    assert by_id["MC_41_s1"]["retention_reasons"] == ["rule:objective_best"]


def test_mc_retains_rejected_scientific_result_and_prunes_its_source(gb, tmp_path):
    managed_root = tmp_path / "managed_rejected"
    energy_func = _make_sequence_energy_func([1.0, 1000.0], managed_root)
    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="largest_objective",
                property="objective",
                direction="max",
                count=1,
            ),
        ),
        prune=True,
    )
    mc = MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=managed_root,
    )
    checkpoint = tmp_path / "mc_rejected.json"

    mc.run_MC(max_steps=1, unique_id=42, checkpoint_file=checkpoint)

    rejected_source = managed_root / "42_2.data"
    rejected_archive = (
        tmp_path / "mc_rejected.artifacts" / "structures" / "MC_42_s1.data"
    )
    assert not rejected_source.exists()
    assert rejected_archive.is_file()

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    records = state["snapshot"]["retention_state"]["records"]
    rejected = next(
        record
        for record in records
        if record["candidate"]["candidate_id"] == "MC_42_s1"
    )
    assert rejected["pins"] == []
    assert rejected["retention_reasons"] == ["rule:largest_objective"]
    assert (tmp_path / "mc_rejected.artifacts" / "manifest.json").is_file()
    assert (tmp_path / "mc_rejected.artifacts" / "history.jsonl").is_file()


def test_mc_resume_rejects_retention_policy_mismatch(gb, tmp_path):
    managed_root = tmp_path / "managed_mismatch"
    energy_func = _make_sequence_energy_func([2.0, 1.0], managed_root)
    first_policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="objective_best",
                property="objective",
                direction="min",
                count=1,
            ),
        ),
    )
    checkpoint = tmp_path / "mc_mismatch.json"
    MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
        retention_policy=first_policy,
    ).run_MC(max_steps=1, unique_id=43, checkpoint_file=checkpoint)

    changed_policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="objective_best",
                property="objective",
                direction="min",
                count=2,
            ),
        ),
    )
    resumed = MonteCarloMinimizer(
        gb,
        _make_energy_func(gb),
        ["translate_right_grain"],
        seed=0,
        retention_policy=changed_policy,
    )

    with pytest.raises(
        GBMinimizerError,
        match="artifact retention policy signature mismatch",
    ):
        resumed.run_MC(max_steps=2, checkpoint_file=checkpoint)


def test_mc_pruning_requires_calculation_context(gb, tmp_path):
    managed_root = tmp_path / "managed_context"
    policy = ArtifactRetentionPolicy(prune=True)

    with pytest.raises(
        GBMinimizerValueError,
        match="requires a non-empty calculation_context",
    ):
        MonteCarloMinimizer(
            gb,
            _make_energy_func(gb),
            ["translate_right_grain"],
            seed=0,
            retention_policy=policy,
            managed_artifact_root=managed_root,
        )


def test_mc_pruning_requires_checkpoint_file(gb, tmp_path):
    managed_root = tmp_path / "managed_no_checkpoint"
    policy = ArtifactRetentionPolicy(prune=True)
    mc = MonteCarloMinimizer(
        gb,
        _make_energy_func(gb),
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=managed_root,
    )

    with pytest.raises(GBMinimizerValueError, match="requires checkpoint_file"):
        mc.run_MC(max_steps=1, unique_id=44)


def test_mc_cleanup_failure_leaks_source_but_checkpoint_resumes(gb, tmp_path):
    managed_root = tmp_path / "managed_cleanup_failure"
    energy_func = _make_sequence_energy_func([1.0, 1000.0], managed_root)
    policy = ArtifactRetentionPolicy(prune=True)

    def failing_cleanup(_request):
        raise OSError("backend cleanup failed")

    checkpoint = tmp_path / "mc_cleanup_failure.json"
    mc = MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        cleanup_candidate=failing_cleanup,
    )

    with pytest.warns(RuntimeWarning, match="Artifact cleanup failed"):
        mc.run_MC(max_steps=1, unique_id=45, checkpoint_file=checkpoint)

    leaked_source = managed_root / "45_2.data"
    assert checkpoint.is_file()
    assert leaked_source.is_file()

    resumed = MonteCarloMinimizer(
        gb,
        _make_sequence_energy_func([1000.0], managed_root),
        ["translate_right_grain"],
        seed=0,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        cleanup_candidate=lambda _request: None,
    )
    resumed.run_MC(max_steps=2, checkpoint_file=checkpoint)

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["snapshot"]["completed_step"] == 2


def test_mc_prune_fresh_resume_matches_continuous_run(gb, tmp_path):
    continuous_root = tmp_path / "continuous"
    resumed_root = tmp_path / "resumed"
    policy = ArtifactRetentionPolicy(
        rules=(
            KeepBest(
                name="objective_best",
                property="objective",
                direction="min",
                count=1,
            ),
        ),
        prune=True,
    )

    continuous_checkpoint = tmp_path / "continuous.json"
    continuous = MonteCarloMinimizer(
        gb,
        _make_sequence_energy_func([4.0, 3.0, 2.0, 1.0], continuous_root),
        ["translate_right_grain"],
        seed=11,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=continuous_root,
    )
    continuous_energy = continuous.run_MC(
        max_steps=3,
        unique_id=46,
        checkpoint_file=continuous_checkpoint,
    )

    resumed_checkpoint = tmp_path / "resumed.json"
    partial = MonteCarloMinimizer(
        gb,
        _make_sequence_energy_func([4.0, 3.0], resumed_root),
        ["translate_right_grain"],
        seed=11,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=resumed_root,
    )
    partial.run_MC(max_steps=1, unique_id=46, checkpoint_file=resumed_checkpoint)

    partial_state = json.loads(resumed_checkpoint.read_text(encoding="utf-8"))
    for record in partial_state["snapshot"]["retention_state"]["records"]:
        if "run_checkpoint" in record["pins"]:
            assert Path(record["source_path"]).is_file()
        elif record["source_path"] is not None:
            assert not Path(record["source_path"]).exists()

    resumed = MonteCarloMinimizer(
        gb,
        _make_sequence_energy_func([2.0, 1.0], resumed_root, start_index=2),
        ["translate_right_grain"],
        seed=999,
        retention_policy=policy,
        calculation_context=_TEST_CALCULATION_CONTEXT,
        managed_artifact_root=resumed_root,
    )
    resumed_energy = resumed.run_MC(
        max_steps=3,
        checkpoint_file=resumed_checkpoint,
    )

    assert resumed_energy == pytest.approx(continuous_energy)
    assert resumed.GBE_vals == continuous.GBE_vals
    assert resumed.accepted_idx == continuous.accepted_idx
    assert resumed.operation_list == continuous.operation_list

    continuous_state = json.loads(
        continuous_checkpoint.read_text(encoding="utf-8")
    )["snapshot"]
    resumed_state = json.loads(resumed_checkpoint.read_text(encoding="utf-8"))[
        "snapshot"
    ]

    def normalized_records(state):
        return [
            {
                "candidate": record["candidate"],
                "pins": record["pins"],
                "retention_reasons": record["retention_reasons"],
                "has_archive": record["archive_path"] is not None,
            }
            for record in state["retention_state"]["records"]
        ]

    assert normalized_records(resumed_state) == normalized_records(continuous_state)


def test_run_mc_checkpoint_kept_on_completion(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))
    checkpoint = tmp_path / "mc.json"

    mc.run_MC(max_steps=3, unique_id=2, checkpoint_file=checkpoint)

    assert checkpoint.exists()


def test_run_mc_checkpoint_file_is_valid_json(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))
    _install_mutate_crash(mc, crash_after=2)
    checkpoint = tmp_path / "mc.json"

    with pytest.raises(RuntimeError):
        mc.run_MC(max_steps=10, unique_id=3, checkpoint_file=checkpoint)

    assert checkpoint.exists()
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert {"schema_version", "minimizer", "progress_unit", "snapshot"} <= state.keys()
    assert {
        "run",
        "rng",
        "completed_step",
        "temperature",
        "rejection_count",
        "previous_energy",
        "best_energy",
        "current_artifact",
        "energy_history",
        "accepted_steps",
        "step_history",
    } <= state["snapshot"].keys()
    assert state["minimizer"] == "MonteCarloMinimizer"
    assert state["progress_unit"] == "step"


def test_run_mc_checkpoint_format_pickle(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))
    _install_mutate_crash(mc, crash_after=2)
    checkpoint = tmp_path / "mc.pkl"

    with pytest.raises(RuntimeError):
        mc.run_MC(
            max_steps=10,
            unique_id=4,
            checkpoint_file=checkpoint,
            checkpoint_format="pickle",
        )

    assert checkpoint.exists()
    with checkpoint.open("rb") as stream:
        state = pickle.load(stream)
    assert "snapshot" in state
    assert "energy_history" in state["snapshot"]


def test_run_mc_resume_from_json(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))
    original_mutate = _install_mutate_crash(mc, crash_after=2)
    checkpoint = tmp_path / "mc_resume.json"

    with pytest.raises(RuntimeError):
        mc.run_MC(max_steps=10, unique_id=5, checkpoint_file=checkpoint)

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    resumed_from_step = saved["snapshot"]["completed_step"]
    gbe_count_before_resume = len(mc.GBE_vals)
    assert resumed_from_step > 0

    mc.mutator.mutate = original_mutate
    mc.run_MC(max_steps=10, unique_id=5, checkpoint_file=checkpoint)

    assert checkpoint.exists()
    assert len(mc.GBE_vals) > gbe_count_before_resume


def test_run_mc_resume_from_pickle(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))
    original_mutate = _install_mutate_crash(mc, crash_after=2)
    checkpoint = tmp_path / "mc_resume.pkl"

    with pytest.raises(RuntimeError):
        mc.run_MC(
            max_steps=10,
            unique_id=6,
            checkpoint_file=checkpoint,
            checkpoint_format="pickle",
        )

    assert checkpoint.exists()
    mc.mutator.mutate = original_mutate
    mc.run_MC(
        max_steps=10,
        unique_id=6,
        checkpoint_file=checkpoint,
        checkpoint_format="pickle",
    )
    assert checkpoint.exists()


def test_run_mc_corrupted_checkpoint_raises(gb, tmp_path):
    checkpoint = tmp_path / "corrupt.json"
    checkpoint.write_bytes(b"not valid json {{{")
    mc = _make_minimizer(gb, _make_energy_func(gb))

    with pytest.raises(GBMinimizerError):
        mc.run_MC(max_steps=5, unique_id=7, checkpoint_file=checkpoint)


def test_run_mc_invalid_format_raises(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))
    checkpoint = tmp_path / "mc.hdf5"

    with pytest.raises(GBMinimizerValueError):
        mc.run_MC(
            max_steps=5,
            unique_id=8,
            checkpoint_file=checkpoint,
            checkpoint_format="hdf5",
        )


def test_run_mc_checkpoint_interval_respected(gb, tmp_path):
    # Steps 1-4 succeed and step 5's mutate call raises. With interval=3, only
    # step 3 has been checkpointed at that point.
    mc = _make_minimizer(gb, _make_energy_func(gb))
    _install_mutate_crash(mc, crash_after=4)
    checkpoint = tmp_path / "mc_interval.json"

    with pytest.raises(RuntimeError):
        mc.run_MC(
            max_steps=10,
            unique_id=9,
            checkpoint_file=checkpoint,
            checkpoint_interval=3,
        )

    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert state["snapshot"]["completed_step"] == 3


def test_proposal_evaluator_exception_rejects_step_without_crashing(gb, tmp_path):
    # call 1 = initial (succeeds), call 2 = step 1 (succeeds); every call after that
    # (steps 2 and 3) raises. A proposal-step evaluator exception no longer aborts
    # the run: each failed step is deterministically rejected, exactly like a normal
    # Metropolis rejection, and the run reaches max_steps regardless.
    energy_func = _make_energy_func(gb, crash_after=2)
    mc = _make_minimizer(gb, energy_func)

    mc.run_MC(max_steps=3, unique_id=100)

    assert len(mc.GBE_vals) == 4
    assert mc.GBE_vals[1] != MC_ENERGY_PENALTY
    assert mc.GBE_vals[2] == MC_ENERGY_PENALTY
    assert mc.GBE_vals[3] == MC_ENERGY_PENALTY
    assert mc.operation_list[-2][1] is False
    assert mc.operation_list[-1][1] is False
    assert 2 not in mc.accepted_idx
    assert 3 not in mc.accepted_idx


def test_initial_evaluator_exception_raises(gb, tmp_path):
    # There is no sensible penalized starting point for a whole MC run, so an
    # initial-evaluation failure is fatal rather than gracefully rejected.
    energy_func = _make_energy_func(gb, crash_after=0)
    mc = _make_minimizer(gb, energy_func)

    with pytest.raises(GBMinimizerError, match="initial evaluation failed"):
        mc.run_MC(max_steps=3, unique_id=101)


def test_resume_without_unique_id_restores_original_label(gb, tmp_path):
    checkpoint = tmp_path / "mc_uid.json"
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=2,
        unique_id=7777,
        checkpoint_file=checkpoint,
    )

    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=4,
        checkpoint_file=checkpoint,
    )

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["snapshot"]["run"]["run_id"] == "7777"


def test_two_fresh_runs_without_unique_id_use_different_labels(gb, tmp_path):
    checkpoint_1 = tmp_path / "mc1.json"
    checkpoint_2 = tmp_path / "mc2.json"

    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=1,
        checkpoint_file=checkpoint_1,
    )
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=1,
        checkpoint_file=checkpoint_2,
    )

    uid_1 = json.loads(checkpoint_1.read_text(encoding="utf-8"))["snapshot"]["run"][
        "run_id"
    ]
    uid_2 = json.loads(checkpoint_2.read_text(encoding="utf-8"))["snapshot"]["run"][
        "run_id"
    ]
    assert uid_1 != uid_2


def test_resume_restores_cooldown_rate_from_checkpoint(gb, tmp_path):
    checkpoint = tmp_path / "mc_cr.json"
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=2,
        cooldown_rate=0.8,
        unique_id=1,
        checkpoint_file=checkpoint,
    )

    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=4,
        checkpoint_file=checkpoint,
    )

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["snapshot"]["cooldown_rate"] == pytest.approx(0.8)


def test_resume_restores_min_steps_from_checkpoint(gb, tmp_path):
    checkpoint = tmp_path / "mc_ms.json"
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=2,
        min_steps=5,
        unique_id=2,
        checkpoint_file=checkpoint,
    )

    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=10,
        checkpoint_file=checkpoint,
    )

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["snapshot"]["min_steps"] == 5


def test_resolved_seed_is_retained_on_the_minimizer(gb):
    minimizer = MonteCarloMinimizer(
        gb, _make_energy_func(gb), ["translate_right_grain"], seed=0
    )
    assert minimizer.seed == 0


def test_resolved_seed_from_none_is_retained_on_the_minimizer(gb):
    minimizer = MonteCarloMinimizer(
        gb, _make_energy_func(gb), ["translate_right_grain"], seed=None
    )
    assert isinstance(minimizer.seed, int)


def test_energy_tolerance_termination_logs_instead_of_printing(gb, tmp_path, capsys):
    root = tmp_path / "structures"
    energy_func = _make_sequence_energy_func([2.0, 1.99995], root)
    mc = MonteCarloMinimizer(gb, energy_func, ["translate_right_grain"], seed=0)

    mc.run_MC(max_steps=5, unique_id=99)

    captured = capsys.readouterr()
    assert "Meets energy tolerance criterion" not in captured.out
    assert captured.out == ""


def test_energy_tolerance_termination_emits_info_log(gb, tmp_path, caplog):
    import logging

    root = tmp_path / "structures"
    energy_func = _make_sequence_energy_func([2.0, 1.99995], root)
    mc = MonteCarloMinimizer(gb, energy_func, ["translate_right_grain"], seed=0)

    with caplog.at_level(logging.INFO, logger="GBOpt.optimization.monte_carlo"):
        mc.run_MC(max_steps=5, unique_id=99)

    assert any(
        "met energy tolerance criterion" in record.getMessage()
        for record in caplog.records
    )


def test_library_is_silent_by_default_without_handlers_configured(gb, tmp_path):
    import logging

    root = tmp_path / "structures"
    energy_func = _make_sequence_energy_func([2.0, 1.99995], root)
    mc_logger = logging.getLogger("GBOpt.optimization.monte_carlo")
    assert mc_logger.handlers == []

    mc = MonteCarloMinimizer(gb, energy_func, ["translate_right_grain"], seed=0)
    mc.run_MC(max_steps=5, unique_id=99)

    assert mc_logger.handlers == []


def test_resume_restores_seed_from_checkpoint(gb, tmp_path):
    checkpoint = tmp_path / "mc_seed.json"
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=2,
        unique_id=2,
        checkpoint_file=checkpoint,
    )

    resumed = _make_minimizer(gb, _make_energy_func(gb))
    resumed.run_MC(max_steps=10, checkpoint_file=checkpoint)

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["snapshot"]["run"]["seed"] == 0
    assert resumed.seed == 0


# --------------------------------------------------------------------------------------
# Checkpoint envelope validation (issue #86)
# --------------------------------------------------------------------------------------


def _make_mc_checkpoint(gb, tmp_path, name="mc_envelope.json"):
    checkpoint = tmp_path / name
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=2,
        unique_id="mc-envelope",
        checkpoint_file=checkpoint,
    )
    return checkpoint


def _rewrite_checkpoint(checkpoint, mutate):
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    mutate(saved)
    checkpoint.write_text(json.dumps(saved), encoding="utf-8")


def test_resume_rejects_checkpoint_missing_a_required_field(gb, tmp_path):
    checkpoint = _make_mc_checkpoint(gb, tmp_path)
    _rewrite_checkpoint(checkpoint, lambda saved: saved["snapshot"].pop("current_artifact"))

    with pytest.raises(GBMinimizerError, match="missing required field"):
        _make_minimizer(gb, _make_energy_func(gb)).run_MC(
            max_steps=4, checkpoint_file=checkpoint
        )


def test_resume_rejects_checkpoint_from_a_different_minimizer(gb, tmp_path):
    checkpoint = _make_mc_checkpoint(gb, tmp_path)
    _rewrite_checkpoint(
        checkpoint, lambda saved: saved.__setitem__(
            "minimizer", "GeneticAlgorithmMinimizer"
        )
    )

    with pytest.raises(GBMinimizerError, match="written by"):
        _make_minimizer(gb, _make_energy_func(gb)).run_MC(
            max_steps=4, checkpoint_file=checkpoint
        )


def test_resume_rejects_unsupported_schema_version(gb, tmp_path):
    checkpoint = _make_mc_checkpoint(gb, tmp_path)
    _rewrite_checkpoint(
        checkpoint, lambda saved: saved.__setitem__("schema_version", 99)
    )

    with pytest.raises(GBMinimizerError, match="unsupported .* schema version"):
        _make_minimizer(gb, _make_energy_func(gb)).run_MC(
            max_steps=4, checkpoint_file=checkpoint
        )


@pytest.mark.parametrize("bad_index", [-1, 1.5, "3"])
def test_resume_rejects_malformed_progress_index(gb, tmp_path, bad_index):
    checkpoint = _make_mc_checkpoint(gb, tmp_path)
    _rewrite_checkpoint(
        checkpoint,
        lambda saved: saved["snapshot"].__setitem__("completed_step", bad_index),
    )

    with pytest.raises(GBMinimizerError):
        _make_minimizer(gb, _make_energy_func(gb)).run_MC(
            max_steps=4, checkpoint_file=checkpoint
        )


def test_resume_rejects_a_corrupted_non_dict_envelope(gb, tmp_path):
    checkpoint = _make_mc_checkpoint(gb, tmp_path)
    checkpoint.write_text(json.dumps(["not", "an", "envelope"]), encoding="utf-8")

    with pytest.raises(GBMinimizerError, match="must be a dictionary"):
        _make_minimizer(gb, _make_energy_func(gb)).run_MC(
            max_steps=4, checkpoint_file=checkpoint
        )


# --------------------------------------------------------------------------------------
# Third-party operations via choices + registry (issue #80 AC7/AC8)
# --------------------------------------------------------------------------------------


class _IdentityUnaryOperation:
    """A minimal test-defined third-party unary operation: returns its input unchanged."""

    @property
    def name(self) -> str:
        return "identity_unary"

    @property
    def arity(self) -> int:
        return 1

    def execute(self, context):
        return ManipulationResult(children=context.parents)


class _TwoChildUnaryOperation:
    """A misbehaving third-party unary operation: returns two children."""

    @property
    def name(self) -> str:
        return "two_child_unary"

    @property
    def arity(self) -> int:
        return 1

    def execute(self, context):
        return ManipulationResult(children=(context.parents[0], context.parents[0]))


def test_third_party_unary_operation_participates_without_editing_optimizer_source(
    gb,
):
    registry = ManipulationRegistry()
    registry.register("identity_unary", _IdentityUnaryOperation())

    mc = MonteCarloMinimizer(
        gb,
        _make_energy_func(gb),
        ["identity_unary"],
        seed=0,
        registry=registry,
    )
    # A single step: the generic apply()/ManipulationContext boundary requires known
    # boundary-normal topology, which the initial GBMaker-built manipulator has but a
    # manipulator reloaded from a relaxed file via deprecated coordinate-based
    # inference does not -- see this class's own known-topology-dependent legacy
    # methods (make_translation_candidate, etc.) for the same established constraint.
    result = mc.run_MC(max_steps=1, unique_id=1)
    assert isinstance(result, float)
    assert mc.operation_list[1][0] == "identity_unary"


def test_third_party_multi_child_operation_is_rejected_not_retried(gb):
    registry = ManipulationRegistry()
    registry.register("two_child_unary", _TwoChildUnaryOperation())

    mc = MonteCarloMinimizer(
        gb,
        _make_energy_func(gb),
        ["two_child_unary"],
        seed=0,
        registry=registry,
    )
    with pytest.raises(GBMinimizerValueError, match="requires exactly one"):
        mc.run_MC(max_steps=2, unique_id=1)


# --------------------------------------------------------------------------------------
# Versioned MC/GA lifecycle events (issue #84)
# --------------------------------------------------------------------------------------


class _RecordingSink:
    def __init__(self):
        self.events: list[OptimizationEvent] = []

    def emit(self, event: OptimizationEvent) -> None:
        self.events.append(event)


class _ExplodingSink:
    def emit(self, event: OptimizationEvent) -> None:
        raise RuntimeError("sink is broken")


def test_default_event_sink_is_silent(gb, tmp_path):
    mc = _make_minimizer(gb, _make_energy_func(gb))

    result = mc.run_MC(max_steps=2, unique_id=1)

    assert isinstance(result, float)


def test_event_sink_type_is_validated(gb):
    with pytest.raises(GBMinimizerTypeError, match="event_sink"):
        MonteCarloMinimizer(
            gb, _make_energy_func(gb), ["translate_right_grain"], seed=0,
            event_sink=object(),
        )


def test_energy_tolerance_run_emits_expected_lifecycle_sequence(gb, tmp_path):
    root = tmp_path / "structures"
    energy_func = _make_sequence_energy_func([2.0, 1.99995], root)
    sink = _RecordingSink()
    mc = MonteCarloMinimizer(
        gb,
        energy_func,
        ["translate_right_grain"],
        seed=0,
        event_sink=sink,
        case_id="case-a",
        campaign_id="campaign-b",
    )

    mc.run_MC(max_steps=5, unique_id=99)

    event_types = [event.event_type for event in sink.events]
    assert event_types == [
        OptimizationEventType.RUN_STARTED,
        OptimizationEventType.INITIAL_EVALUATION,
        OptimizationEventType.PROPOSAL_EVALUATED,
        OptimizationEventType.CANDIDATE_ACCEPTED,
        OptimizationEventType.BEST_UPDATED,
        OptimizationEventType.RUN_TERMINATED,
    ]

    for event in sink.events:
        assert event.run.run_id == "99"
        assert event.run.seed == 0
        assert event.run.algorithm is OptimizationAlgorithm.MONTE_CARLO
        assert event.run.case_id == "case-a"
        assert event.run.campaign_id == "campaign-b"

    initial_event = sink.events[1]
    assert initial_event.status is EvaluationStatus.SUCCESS
    assert initial_event.energy == pytest.approx(2.0)

    proposal_event = sink.events[2]
    assert proposal_event.iteration == 1
    assert proposal_event.status is EvaluationStatus.SUCCESS
    assert proposal_event.energy == pytest.approx(1.99995)
    # translate_right_grain's own mutation-dispatch label describes the sampled shift
    # rather than repeating the operation's own name -- operation_name mirrors
    # whatever label Mutator.mutate() actually returns for the dispatched operation.
    assert proposal_event.operation_name.startswith("shift")

    terminated_event = sink.events[-1]
    assert terminated_event.termination_reason is TerminationReason.ENERGY_TOLERANCE


def test_max_steps_termination_reason(gb, tmp_path):
    sink = _RecordingSink()
    mc = MonteCarloMinimizer(
        gb, _make_energy_func(gb), ["translate_right_grain"], seed=0, event_sink=sink
    )

    mc.run_MC(max_steps=2, unique_id=1)

    terminated = [
        event
        for event in sink.events
        if event.event_type is OptimizationEventType.RUN_TERMINATED
    ]
    assert len(terminated) == 1
    assert terminated[0].termination_reason is TerminationReason.MAX_STEPS
    assert terminated[0].iteration == 2


def test_max_rejections_termination_reason(gb, tmp_path):
    root = tmp_path / "structures"
    # A strictly increasing sequence at T -> 0 is always rejected under the Metropolis
    # criterion used here, so every proposal after the initial one is a rejection.
    energy_func = _make_sequence_energy_func([2.0] + [100.0] * 5, root)
    sink = _RecordingSink()
    mc = MonteCarloMinimizer(
        gb, energy_func, ["translate_right_grain"], seed=0, event_sink=sink
    )

    mc.run_MC(max_steps=10, max_rejections=2, unique_id=1)

    rejected = [
        event
        for event in sink.events
        if event.event_type is OptimizationEventType.CANDIDATE_REJECTED
    ]
    terminated = [
        event
        for event in sink.events
        if event.event_type is OptimizationEventType.RUN_TERMINATED
    ]
    assert len(rejected) == 3
    assert len(terminated) == 1
    assert terminated[0].termination_reason is TerminationReason.MAX_REJECTIONS


def test_initial_evaluation_failure_emits_failed_events_then_raises(gb, tmp_path):
    def crashing_energy_func(GB, manipulator, atom_positions, unique_id):
        raise RuntimeError("boom")

    sink = _RecordingSink()
    mc = MonteCarloMinimizer(
        gb, crashing_energy_func, ["translate_right_grain"], seed=0, event_sink=sink
    )

    with pytest.raises(GBMinimizerError):
        mc.run_MC(max_steps=2, unique_id=1)

    event_types = [event.event_type for event in sink.events]
    assert event_types == [
        OptimizationEventType.RUN_STARTED,
        OptimizationEventType.INITIAL_EVALUATION,
        OptimizationEventType.RUN_FAILED,
    ]
    assert sink.events[1].status is EvaluationStatus.FAILED
    assert sink.events[2].failure_message is not None


def test_resume_emits_run_started_without_reemitting_initial_evaluation(gb, tmp_path):
    checkpoint = tmp_path / "mc.json"
    _make_minimizer(gb, _make_energy_func(gb)).run_MC(
        max_steps=1, unique_id=1, checkpoint_file=checkpoint
    )

    sink = _RecordingSink()
    resumed = MonteCarloMinimizer(
        gb, _make_energy_func(gb), ["translate_right_grain"], seed=0, event_sink=sink
    )
    resumed.run_MC(max_steps=2, checkpoint_file=checkpoint)

    event_types = [event.event_type for event in sink.events]
    assert event_types[0] is OptimizationEventType.RUN_STARTED
    assert OptimizationEventType.INITIAL_EVALUATION not in event_types


def test_a_broken_event_sink_does_not_abort_the_run(gb, tmp_path):
    mc = MonteCarloMinimizer(
        gb,
        _make_energy_func(gb),
        ["translate_right_grain"],
        seed=0,
        event_sink=_ExplodingSink(),
    )

    result = mc.run_MC(max_steps=2, unique_id=1)

    assert isinstance(result, float)


def test_event_emission_does_not_alter_rng_state_or_result(gb, tmp_path):
    silent = _make_minimizer(gb, _make_energy_func(gb))
    silent_result = silent.run_MC(max_steps=5, unique_id=1)

    observed = MonteCarloMinimizer(
        gb,
        _make_energy_func(gb),
        ["translate_right_grain"],
        seed=0,
        event_sink=_RecordingSink(),
    )
    observed_result = observed.run_MC(max_steps=5, unique_id=1)

    assert observed_result == silent_result
    assert (
        observed.local_random.bit_generator.state
        == silent.local_random.bit_generator.state
    )
    assert observed.operation_list == silent.operation_list
    assert observed.accepted_idx == silent.accepted_idx
