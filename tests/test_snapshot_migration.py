# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import json
import math
import pickle
from unittest.mock import patch

import numpy as np
import pytest

from GBOpt.BoundarySpec import CSLExactSpec
from GBOpt.Checkpoint import CheckpointStore
from GBOpt.evaluation import EvaluationStatus
from GBOpt.GBMaker import GBMaker
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL, GrainOwnership
from GBOpt.optimization.genetic import GeneticAlgorithmMinimizer
from GBOpt.optimization.monte_carlo import MonteCarloMinimizer
from GBOpt.snapshot import (
    GeneticAlgorithmSnapshot,
    MonteCarloSnapshot,
    SnapshotMigrationError,
    migrate_genetic_algorithm_checkpoint,
    migrate_monte_carlo_checkpoint,
)

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


def _make_energy_func(crash_after=None):
    call_count = 0

    def energy_func(GB, manipulator, atom_positions, unique_id):
        nonlocal call_count
        call_count += 1
        if crash_after is not None and call_count > crash_after:
            raise RuntimeError(f"Simulated crash at call {call_count}")
        path = f"{unique_id}_{call_count}.data"
        GB.write_lammps(path, atom_positions, manipulator.parents[0].box_dims)
        return 2.0 - call_count * 0.001, path

    return energy_func


def _install_mutate_crash(minimizer, crash_after):
    """Force a mid-run crash from the mutator so a checkpoint captures mid-run state."""
    original_mutate = minimizer.mutator.mutate
    call_count = 0

    def crashing_mutate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > crash_after:
            raise RuntimeError(f"Simulated crash at mutate call {call_count}")
        return original_mutate(*args, **kwargs)

    minimizer.mutator.mutate = crashing_mutate


def _v1_state_from_v2_envelope(v2_state: dict) -> dict:
    """Translate one real schema-v2 MC checkpoint envelope into an equivalent schema-v1
    envelope, for exercising the migrator against the on-disk shape it exists to read.

    ``MonteCarloMinimizer.run_MC`` itself only ever writes schema-v2 checkpoints (R28),
    so a real, pre-migration schema-v1 fixture can no longer come directly from a live
    run -- this maps a real run's own v2 output back onto the v1 field layout
    ``GBOpt.snapshot.migration._monte_carlo_snapshot_from_v1`` expects, keeping the
    fixture grounded in real RNG state, energies, and structure paths.
    """
    snap = v2_state["snapshot"]
    best_artifact = snap["best_artifact"]
    return {
        "schema_version": 1,
        "minimizer": "MonteCarloMinimizer",
        "progress_unit": "step",
        "progress_index": snap["completed_step"],
        "best_energy": snap["best_energy"],
        "best_dump": None if best_artifact is None else best_artifact["path"],
        "rng_state": snap["rng"]["state"],
        "run_params": {
            "E_accept": 0.1,
            "min_steps": snap["min_steps"],
            "max_steps": 10,
            "E_tol": 1e-4,
            "max_rejections": 20,
            "cooldown_rate": snap["cooldown_rate"],
            "unique_id": snap["run"]["run_id"],
            "seed": snap["run"]["seed"],
        },
        "state": {
            "T": snap["temperature"],
            "rejection_count": snap["rejection_count"],
            "prev_gbe": snap["previous_energy"],
            "current_structure_dump": snap["current_artifact"]["path"],
            "GBE_vals": snap["energy_history"],
            "accepted_idx": snap["accepted_steps"],
            "operation_list": [
                [entry["operation_name"], entry["accepted"]]
                for entry in snap["step_history"]
            ],
            "artifact_store": snap["retention_state"],
        },
    }


def _mc_v1_checkpoint(gb, tmp_path, *, fmt="json", name="mc", **run_kwargs):
    """Run a real MC crash-and-checkpoint, then rewrite its output as schema-v1."""
    mc = MonteCarloMinimizer(gb, _make_energy_func(), ["translate_right_grain"], seed=0)
    _install_mutate_crash(mc, crash_after=2)
    ext = "json" if fmt == "json" else "pkl"
    v2_checkpoint = tmp_path / f"{name}_v2.{ext}"
    run_kwargs.setdefault("unique_id", "mc-run")
    with pytest.raises(RuntimeError):
        mc.run_MC(
            max_steps=10,
            checkpoint_file=v2_checkpoint,
            checkpoint_format=fmt,
            **run_kwargs,
        )
    assert v2_checkpoint.exists()
    if fmt == "json":
        with open(v2_checkpoint, encoding="utf-8") as fh:
            v2_state = json.load(fh)
    else:
        with open(v2_checkpoint, "rb") as fh:
            v2_state = pickle.load(fh)
    v1_state = _v1_state_from_v2_envelope(v2_state)
    checkpoint = tmp_path / f"{name}.{ext}"
    if fmt == "json":
        checkpoint.write_text(json.dumps(v1_state), encoding="utf-8")
    else:
        with open(checkpoint, "wb") as fh:
            pickle.dump(v1_state, fh)
    return checkpoint


def _mc_checkpoint(gb, tmp_path, *, fmt="json"):
    return _mc_v1_checkpoint(gb, tmp_path, fmt=fmt)


def _run_ga_then_crash_after_first_save(minimizer, checkpoint, fmt, unique_id):
    """Run ``minimizer`` to completion of its first durable checkpoint commit, then crash.

    Patches ``CheckpointStore._save`` to raise immediately *after* delegating to the
    real save, so the checkpoint file on disk reflects one genuinely completed
    generation boundary -- the same technique
    ``tests/test_optimization_genetic.py``'s own checkpoint tests use, since crashing
    from the mutator instead can fire before a generation (or even the initial
    population) ever finishes, depending on population size and operation mix.
    """
    original_save = CheckpointStore._save
    call_count = 0

    def save_then_crash(self_store, state):
        nonlocal call_count
        original_save(self_store, state)
        call_count += 1
        if call_count >= 1:
            raise RuntimeError("simulated crash after first checkpoint commit")

    with (
        patch.object(CheckpointStore, "_save", save_then_crash),
        pytest.raises(RuntimeError),
    ):
        minimizer.run_GA(
            unique_id=unique_id, checkpoint_file=checkpoint, checkpoint_format=fmt,
        )
    assert checkpoint.exists()


def _legacy_ga_checkpoint(gb, tmp_path, *, fmt="json"):
    def fake_energy(GB, manipulator, atom_positions, unique_id):
        dump_file = tmp_path / f"{unique_id}.data"
        GB.write_lammps(str(dump_file), atom_positions, manipulator.parents[0].box_dims)
        return float(np.mean(atom_positions["x"])), str(dump_file)

    minimizer = GeneticAlgorithmMinimizer(
        gb, fake_energy, ["insert_atoms", "remove_atoms", "translate_right_grain"],
        seed=0, population_size=4, generations=3, keep_top_pct=25, intermediate_pct=75,
    )
    ext = "json" if fmt == "json" else "pkl"
    checkpoint = tmp_path / f"ga_legacy.{ext}"
    _run_ga_then_crash_after_first_save(minimizer, checkpoint, fmt, "ga-legacy-run")
    return checkpoint


def _write_owned_evaluator_output(path, atoms, box_dims):
    order = np.arange(len(atoms))[::-1]
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("Owned evaluator output\n\n")
        stream.write(f"{len(atoms)} atoms\n")
        stream.write(f"{len(set(atoms['name'].tolist()))} atom types\n")
        stream.writelines(
            f"{lower:.12f} {upper:.12f} {axis}lo {axis}hi\n"
            for axis, (lower, upper) in zip("xyz", box_dims, strict=True)
        )
        stream.write("\nAtoms\n\n")
        for row in order:
            atom = atoms[row]
            stream.write(
                f"{row + 1} {atom['name']} {atom['x']:.12f} "
                f"{atom['y']:.12f} {atom['z']:.12f}\n"
            )


@pytest.fixture
def owned_ga(tmp_path):
    gb = GBMaker.from_boundary_spec(
        3.52, "fcc", "Ni",
        CSLExactSpec(axis=(0, 0, 1), plane=(3, 1, 0), quat=(3, 0, 0, 1), sigma=5),
        mode="exact", gb_thickness=10.0, repeat_factor=2, x_dim_min=10.0,
        vacuum=0.0, interaction_distance=3.0,
    )
    seed_path = tmp_path / "owned_initial.data"
    gb.write_lammps(str(seed_path), type_as_int=False, precision=12)
    labels = np.hstack((
        np.full(len(gb.left_grain), LEFT_GRAIN_LABEL, dtype=np.int8),
        np.full(len(gb.right_grain), RIGHT_GRAIN_LABEL, dtype=np.int8),
    ))
    ownership = GrainOwnership(
        atom_ids=np.arange(1, len(gb.whole_system) + 1),
        labels=labels,
        gb_plane_x=gb.gb_plane_x,
        inplane_periodic=gb.inplane_periodic,
        left_grain_x_bounds=(gb.box_dims[0, 0], gb.gb_plane_x),
        right_grain_x_bounds=(gb.gb_plane_x, gb.box_dims[0, 1]),
        coordinate_tolerance=gb.epsilon,
        normal_topology=gb.normal_topology,
    )
    return gb, seed_path, ownership


def _owned_ga_checkpoint(owned_ga, tmp_path, *, fmt="json"):
    gb, seed_path, ownership = owned_ga

    def energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_owned_evaluator_output(
            output, atom_positions, manipulator.parents[0].box_dims
        )
        value = float(np.mean(atom_positions["x"]) + 0.001 * len(atom_positions))
        return value, str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb, energy, ["translate_right_grain"], seed=101,
        initial_structure=seed_path, initial_ownership=ownership,
        population_size=3, generations=3, keep_top_pct=25, intermediate_pct=100,
    )
    ext = "json" if fmt == "json" else "pkl"
    checkpoint = tmp_path / f"ga_owned.{ext}"
    _run_ga_then_crash_after_first_save(minimizer, checkpoint, fmt, "ga-owned-run")
    return checkpoint


class TestMigrateMonteCarloCheckpoint:
    @pytest.mark.parametrize("fmt", ["json", "pickle"])
    def test_migrates_real_checkpoint(self, gb, tmp_path, fmt):
        checkpoint = _mc_checkpoint(gb, tmp_path, fmt=fmt)
        snapshot = migrate_monte_carlo_checkpoint(checkpoint, fmt=fmt)
        assert isinstance(snapshot, MonteCarloSnapshot)
        assert snapshot.run.run_id == "mc-run"
        assert snapshot.run.seed == 0
        assert snapshot.completed_step >= 0
        assert snapshot.current_artifact.path
        assert len(snapshot.step_history) == snapshot.completed_step + 1
        assert snapshot.energy_history[0] > 0

    def test_rng_state_matches_source(self, gb, tmp_path):
        checkpoint = _mc_checkpoint(gb, tmp_path)
        with open(checkpoint, encoding="utf-8") as fh:
            raw = json.load(fh)
        snapshot = migrate_monte_carlo_checkpoint(checkpoint)
        assert dict(snapshot.rng.state) == raw["rng_state"]

    def test_restores_min_steps_and_cooldown_rate(self, gb, tmp_path):
        checkpoint = _mc_v1_checkpoint(
            gb,
            tmp_path,
            name="mc_ctrl",
            unique_id="mc-run-2",
            min_steps=5,
            cooldown_rate=0.8,
        )

        snapshot = migrate_monte_carlo_checkpoint(checkpoint)
        assert snapshot.min_steps == 5
        assert snapshot.cooldown_rate == pytest.approx(0.8)

    def test_wrong_algorithm_fails_explicitly(self, gb, tmp_path):
        checkpoint = _mc_checkpoint(gb, tmp_path)
        with pytest.raises(SnapshotMigrationError):
            migrate_genetic_algorithm_checkpoint(checkpoint)

    def test_missing_file_fails_explicitly(self, tmp_path):
        with pytest.raises(SnapshotMigrationError):
            migrate_monte_carlo_checkpoint(tmp_path / "does_not_exist.json")

    def test_unsupported_schema_version_fails_explicitly(self, gb, tmp_path):
        checkpoint = _mc_checkpoint(gb, tmp_path)
        with open(checkpoint, encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["schema_version"] = 99
        checkpoint.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SnapshotMigrationError, match="schema version"):
            migrate_monte_carlo_checkpoint(checkpoint)

    def test_missing_required_field_fails_explicitly(self, gb, tmp_path):
        checkpoint = _mc_checkpoint(gb, tmp_path)
        with open(checkpoint, encoding="utf-8") as fh:
            raw = json.load(fh)
        del raw["state"]["current_structure_dump"]
        checkpoint.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SnapshotMigrationError):
            migrate_monte_carlo_checkpoint(checkpoint)

    def test_semantically_invalid_reference_fails_explicitly(self, gb, tmp_path):
        checkpoint = _mc_checkpoint(gb, tmp_path)
        with open(checkpoint, encoding="utf-8") as fh:
            raw = json.load(fh)
        raw["state"]["current_structure_dump"] = ""
        checkpoint.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SnapshotMigrationError):
            migrate_monte_carlo_checkpoint(checkpoint)


class TestMigrateLegacyGeneticAlgorithmCheckpoint:
    @pytest.mark.parametrize("fmt", ["json", "pickle"])
    def test_migrates_real_checkpoint(self, gb, tmp_path, fmt):
        checkpoint = _legacy_ga_checkpoint(gb, tmp_path, fmt=fmt)
        snapshot = migrate_genetic_algorithm_checkpoint(checkpoint, fmt=fmt)
        assert isinstance(snapshot, GeneticAlgorithmSnapshot)
        assert snapshot.run.run_id == "ga-legacy-run"
        assert len(snapshot.population) == 4
        assert all(candidate.mapping is None for candidate in snapshot.population)
        assert snapshot.best.status is EvaluationStatus.SUCCESS
        assert snapshot.best.artifact is not None
        assert snapshot.generation_history

    def test_wrong_algorithm_fails_explicitly(self, gb, tmp_path):
        checkpoint = _legacy_ga_checkpoint(gb, tmp_path)
        with pytest.raises(SnapshotMigrationError):
            migrate_monte_carlo_checkpoint(checkpoint)


class TestMigrateOwnedGeneticAlgorithmCheckpoint:
    @pytest.mark.parametrize("fmt", ["json", "pickle"])
    def test_migrates_real_checkpoint(self, owned_ga, tmp_path, fmt):
        checkpoint = _owned_ga_checkpoint(owned_ga, tmp_path, fmt=fmt)
        snapshot = migrate_genetic_algorithm_checkpoint(checkpoint, fmt=fmt)
        assert isinstance(snapshot, GeneticAlgorithmSnapshot)
        assert snapshot.run.run_id == "ga-owned-run"
        assert len(snapshot.population) == 3
        assert all(
            candidate.mapping is not None for candidate in snapshot.population
        )
        assert snapshot.retention_lineages is not None
        assert len(snapshot.retention_lineages) == 3
        assert snapshot.best.status is EvaluationStatus.SUCCESS
        assert snapshot.last_generation_evaluations is not None
        assert len(snapshot.last_generation_evaluations) == 3

    def test_population_mapping_round_trips_fields(self, owned_ga, tmp_path):
        checkpoint = _owned_ga_checkpoint(owned_ga, tmp_path)
        snapshot = migrate_genetic_algorithm_checkpoint(checkpoint)
        with open(checkpoint, encoding="utf-8") as fh:
            raw = json.load(fh)
        raw_mapping = raw["state"]["population_candidates"][0]["mapping"]
        migrated_mapping = snapshot.population[0].mapping
        assert migrated_mapping is not None
        assert migrated_mapping.gb_plane_x == raw_mapping["gb_plane_x"]


class TestMigrationRejectsCorruptedEnvelope:
    def test_non_dict_envelope_fails_explicitly(self, tmp_path):
        checkpoint = tmp_path / "bad.json"
        checkpoint.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        with pytest.raises(SnapshotMigrationError):
            migrate_monte_carlo_checkpoint(checkpoint)

    def test_pickle_format_reads_same_as_json(self, gb, tmp_path):
        json_checkpoint = _mc_checkpoint(gb, tmp_path, fmt="json")
        with open(json_checkpoint, encoding="utf-8") as fh:
            raw = json.load(fh)
        pickle_checkpoint = tmp_path / "mc_from_json.pkl"
        with open(pickle_checkpoint, "wb") as fh:
            pickle.dump(raw, fh)
        snapshot = migrate_monte_carlo_checkpoint(pickle_checkpoint, fmt="pickle")
        assert snapshot.run.run_id == "mc-run"
