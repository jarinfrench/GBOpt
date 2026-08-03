# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from GBOpt.GBMaker import GBMaker
from GBOpt.GBMinimizer import GeneticAlgorithmMinimizer


# These tests use compact GBMaker fixtures; sizing-warning behavior is covered in
# the GBMaker test modules rather than the minimizer contract tests.
pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:Recommended repeat factor is at least 2\.:UserWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:Repeat factor in [yz] modified to \d+ to satisfy the "
        r"minimum in-plane dimension cutoff of .* A\.:UserWarning"
    ),
]


class TestGeneticAlgorithmMinimizer(unittest.TestCase):

    def setUp(self):
        theta = math.radians(36.869898)
        misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
        self.gb = GBMaker(
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
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_ga_returns_best_energy_and_dump(self):
        def fake_energy_func(GB, manipulator, atom_positions, unique_id):
            dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
            GB.write_lammps(
                str(dump_file),
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            energy = float(np.mean(atom_positions["x"]))
            return energy, str(dump_file)

        minimizer = GeneticAlgorithmMinimizer(
            self.gb,
            fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=2,
            keep_top_pct=25,
            intermediate_pct=75,
        )

        best_energy, best_dump = minimizer.run_GA(unique_id=1)

        self.assertIsInstance(best_energy, float)
        self.assertTrue(Path(best_dump).exists())
        self.assertEqual(len(minimizer.GBE_vals), minimizer.generations + 1)

    def test_history_populated_after_run(self):
        def fake_energy_func(GB, manipulator, atom_positions, unique_id):
            dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
            GB.write_lammps(
                str(dump_file),
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            return float(np.mean(atom_positions["x"])), str(dump_file)

        minimizer = GeneticAlgorithmMinimizer(
            self.gb,
            fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=2,
            keep_top_pct=25,
            intermediate_pct=75,
        )
        minimizer.run_GA(unique_id=2)

        # One entry per generation
        self.assertEqual(len(minimizer.history), minimizer.generations)

        for gen_idx, gen_history in enumerate(minimizer.history):
            # Each generation records population_size (lineage, energy) pairs
            self.assertEqual(len(gen_history), minimizer.population_size)

            for lineage, energy in gen_history:
                # Lineage is a non-empty list of strings
                self.assertIsInstance(lineage, list)
                self.assertGreater(len(lineage), 0)
                self.assertIsInstance(lineage[0], str)
                # First element is a known operation label
                op = lineage[0]
                self.assertTrue(
                    op in {"slice_and_merge", "carryover", "START"}
                    or op.startswith("shift") or op.startswith("add")
                    or op.startswith("remove"),
                    f"Unexpected operation label: {op!r}",
                )

            # Energies in history match the corresponding GBE_vals entry
            # (GBE_vals[0] is the initial eval, so gen 0 -> GBE_vals[1])
            self.assertEqual(
                [e for _, e in gen_history],
                minimizer.GBE_vals[gen_idx + 1],
            )

    def test_failed_generation_appends_to_history(self):
        def fake_energy_func(GB, manipulator, atom_positions, unique_id):
            # Force all generation-0 candidates to fail
            if "_g0_c" in str(unique_id):
                raise RuntimeError("Simulated failure")
            dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
            GB.write_lammps(
                str(dump_file),
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            return float(np.mean(atom_positions["x"])), str(dump_file)

        minimizer = GeneticAlgorithmMinimizer(
            self.gb,
            fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=2,
            keep_top_pct=25,
            intermediate_pct=75,
        )
        minimizer.run_GA(unique_id=3)

        # history still has one entry per generation despite the failure
        self.assertEqual(len(minimizer.history), minimizer.generations)

        # Generation 0 failed entirely — all energies should be PENALTY
        PENALTY = 1.0e30
        failed_gen = minimizer.history[0]
        self.assertEqual(len(failed_gen), minimizer.population_size)
        for lineage, energy in failed_gen:
            self.assertEqual(energy, PENALTY)

        # Generation 1 recovered and has real energies
        recovered_gen = minimizer.history[1]
        self.assertEqual(len(recovered_gen), minimizer.population_size)
        for _, energy in recovered_gen:
            self.assertLess(energy, PENALTY)

    def test_ga_history_never_exceeds_generations(self):
        def fake_energy_func(GB, manipulator, atom_positions, unique_id):
            dump_file = Path(self.tmpdir.name) / f"{unique_id}.data"
            GB.write_lammps(
                str(dump_file),
                atom_positions,
                manipulator.parents[0].box_dims,
            )
            return float(np.mean(atom_positions["x"])), str(dump_file)

        minimizer = GeneticAlgorithmMinimizer(
            self.gb,
            fake_energy_func,
            ["insert_atoms", "remove_atoms", "translate_right_grain"],
            seed=0,
            population_size=4,
            generations=2,
            keep_top_pct=25,
            intermediate_pct=75,
        )
        minimizer.run_GA(unique_id=2)
        self.assertEqual(len(minimizer.history), minimizer.generations)

        minimizer.run_GA(unique_id=2)
        self.assertEqual(len(minimizer.history), minimizer.generations)


if __name__ == "__main__":
    unittest.main()


def test_initial_owned_manipulator_preserves_counts_plane_and_runs_owned_ga(tmp_path):
    from GBOpt.FileGrainOwnership import (
        GrainOwnership,
        LEFT_GRAIN_LABEL,
        RIGHT_GRAIN_LABEL,
    )

    theta = math.radians(36.869898)
    gb = GBMaker(
        3.52,
        "fcc",
        10.0,
        np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0]),
        "Ni",
        repeat_factor=(1, 1),
        x_dim_min=12.0,
        vacuum=0.0,
        interaction_distance=4.0,
    )
    seed_path = tmp_path / "initial.data"
    gb.write_lammps(str(seed_path), type_as_int=True, precision=12)
    left_count = len(gb.left_grain)
    total_count = len(gb.whole_system)
    labels = np.full(total_count, RIGHT_GRAIN_LABEL, dtype=np.int8)
    labels[:left_count] = LEFT_GRAIN_LABEL
    ownership = GrainOwnership(
        atom_ids=np.arange(1, total_count + 1),
        labels=labels,
        gb_plane_x=gb.gb_plane_x,
        inplane_periodic=gb.inplane_periodic,
        right_grain_x_bounds=(gb.gb_plane_x, gb.box_dims[0, 1]),
        coordinate_tolerance=gb.epsilon,
        periodic_outer_x_interface=True,
    )

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        lambda *args: (0.0, str(seed_path)),
        ["translate_right_grain"],
        seed=0,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
    )
    parent = minimizer.manipulator.parents[0]
    assert len(parent.left_grain) == left_count
    assert len(parent.right_grain) == total_count - left_count
    assert parent.gb_plane_x == gb.gb_plane_x
    assert np.array_equal(parent.grain_labels, labels)

    best_energy, best_path = minimizer.run_GA(unique_id=1)
    assert best_energy == pytest.approx(0.0)
    assert Path(best_path) == seed_path
    assert minimizer.best_evaluation.success
    assert np.array_equal(
        minimizer.best_evaluation.manipulator.parents[0].grain_labels, labels
    )



def _write_preserved_id_candidate(path, atoms, ids, box_dims, *, change_species=False):
    order = np.arange(len(atoms))[::-1]
    output = atoms.copy()
    if change_species:
        output[0]["name"] = "Cu"
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        stream.write("GA fake evaluator output\n\n")
        stream.write(f"{len(output)} atoms\n")
        stream.write(f"{len(set(output['name'].tolist()))} atom types\n")
        for axis, (lower, upper) in zip("xyz", box_dims):
            stream.write(f"{lower:.12f} {upper:.12f} {axis}lo {axis}hi\n")
        stream.write("\nAtoms\n\n")
        for row in order:
            atom = output[row]
            stream.write(
                f"{int(ids[row])} {atom['name']} {atom['x']:.12f} "
                f"{atom['y']:.12f} {atom['z']:.12f}\n"
            )


def _owned_ga_fixture(tmp_path):
    from GBOpt.FileGrainOwnership import GrainOwnership

    theta = math.radians(36.869898)
    gb = GBMaker(
        3.52,
        "fcc",
        10.0,
        np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0]),
        "Ni",
        repeat_factor=(1, 1),
        x_dim_min=12.0,
        vacuum=0.0,
        interaction_distance=4.0,
    )
    seed_path = tmp_path / "owned_initial.data"
    gb.write_lammps(str(seed_path), type_as_int=False, precision=12)
    labels = np.hstack(
        (
            np.zeros(len(gb.left_grain), dtype=np.int8),
            np.ones(len(gb.right_grain), dtype=np.int8),
        )
    )
    ownership = GrainOwnership(
        atom_ids=np.arange(1, len(gb.whole_system) + 1),
        labels=labels,
        gb_plane_x=gb.gb_plane_x,
        inplane_periodic=gb.inplane_periodic,
        right_grain_x_bounds=(gb.gb_plane_x, gb.box_dims[0, 1]),
        coordinate_tolerance=gb.epsilon,
        periodic_outer_x_interface=True,
    )
    return gb, seed_path, ownership, labels


def test_owned_ga_smoke_preserves_crossing_atom_label_and_counts(tmp_path):
    gb, seed_path, ownership, labels = _owned_ga_fixture(tmp_path)
    crossing_events = []

    def fake_energy(GB, manipulator, atom_positions, unique_id):
        candidate = np.array(atom_positions, copy=True)
        candidate_labels = manipulator.candidate_grain_labels
        crossing_index = int(np.flatnonzero(candidate_labels == 0)[0])
        candidate[crossing_index]["x"] = GB.gb_plane_x + 0.25
        output = tmp_path / f"{unique_id}.data"
        ids = np.arange(1, len(candidate) + 1)
        _write_preserved_id_candidate(
            output, candidate, ids, manipulator.parents[0].box_dims
        )
        crossing_events.append((str(unique_id), crossing_index + 1, int(candidate_labels[crossing_index])))
        energy = 10.0 if "initial" in str(unique_id) else float(2 - int(str(unique_id).rsplit("c", 1)[-1]))
        return energy, str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        fake_energy,
        ["translate_right_grain"],
        seed=17,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
    )
    best_energy, best_path = minimizer.run_GA(unique_id=41)
    best = minimizer.best_evaluation
    assert best_energy == pytest.approx(1.0)
    assert Path(best_path).is_file()
    moved_id = [event[1] for event in crossing_events if event[0].endswith("c1")][0]
    assert best.manipulator.parents[0].grain_labels[moved_id - 1] == 0
    assert best.manipulator.parents[0].whole_system[moved_id - 1]["x"] > gb.gb_plane_x
    assert best.manipulator.parents[0].gb_plane_x == gb.gb_plane_x
    assert len(best.manipulator.parents[0].whole_system) == len(labels)
    assert np.count_nonzero(best.manipulator.parents[0].grain_labels == 0) == np.count_nonzero(labels == 0)
    assert np.count_nonzero(best.manipulator.parents[0].grain_labels == 1) == np.count_nonzero(labels == 1)


def test_owned_evaluator_failure_is_filtered_before_selection(tmp_path):
    gb, seed_path, ownership, _labels = _owned_ga_fixture(tmp_path)

    def fake_energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_preserved_id_candidate(
            output,
            np.array(atom_positions, copy=True),
            np.arange(1, len(atom_positions) + 1),
            manipulator.parents[0].box_dims,
            change_species=str(unique_id).endswith("c0"),
        )
        return 0.0, str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        fake_energy,
        ["translate_right_grain"],
        seed=3,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
    )
    minimizer.run_GA(unique_id=7)
    failed, valid = minimizer.last_generation_evaluations
    assert failed.success is False
    assert failed.energy == 1.0e30
    assert failed.structure_path is None
    assert failed.manipulator is None
    assert "changed species" in failed.failure_reason
    assert valid.success is True


def test_owned_batch_requires_one_structure_path_per_candidate(tmp_path):
    gb, seed_path, ownership, _labels = _owned_ga_fixture(tmp_path)

    def scalar_energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_preserved_id_candidate(
            output,
            np.array(atom_positions, copy=True),
            np.arange(1, len(atom_positions) + 1),
            manipulator.parents[0].box_dims,
        )
        return 0.0, str(output)

    def scalar_only_batch(*args):
        structures = args[2]
        return [{"energy": 0.0} for _ in structures]

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        scalar_energy,
        ["translate_right_grain"],
        seed=3,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
        gb_batch_energy_func=scalar_only_batch,
    )
    with pytest.raises(ValueError, match="final_dump"):
        minimizer.run_GA(unique_id=8)


def test_owned_supported_batch_keeps_result_mapping_alignment(tmp_path):
    gb, seed_path, ownership, _labels = _owned_ga_fixture(tmp_path)

    def scalar_energy(GB, manipulator, atom_positions, unique_id):
        output = tmp_path / f"{unique_id}.data"
        _write_preserved_id_candidate(
            output,
            np.array(atom_positions, copy=True),
            np.arange(1, len(atom_positions) + 1),
            manipulator.parents[0].box_dims,
        )
        return 5.0, str(output)

    def batch_energy(GB, manipulators, structures, lineages, unique_ids):
        results = []
        for index, (manipulator, atoms, candidate_id) in enumerate(
            zip(manipulators, structures, unique_ids)
        ):
            output = tmp_path / f"batch_{candidate_id}.data"
            _write_preserved_id_candidate(
                output,
                np.array(atoms, copy=True),
                np.arange(1, len(atoms) + 1),
                manipulator.parents[0].box_dims,
                change_species=index == 0,
            )
            results.append({"energy": float(index), "final_dump": str(output)})
        return results

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        scalar_energy,
        ["translate_right_grain"],
        seed=5,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
        gb_batch_energy_func=batch_energy,
    )
    best_energy, _ = minimizer.run_GA(unique_id=9)
    first, second = minimizer.last_generation_evaluations
    assert first.input_index == 0 and first.success is False
    assert second.input_index == 1 and second.success is True
    assert second.energy == pytest.approx(1.0)
    assert np.array_equal(
        second.mapping.labels,
        second.manipulator.parents[0].grain_labels,
    )
    assert best_energy == pytest.approx(1.0)


def test_direct_owned_file_reload_path_is_blocked(tmp_path):
    gb, seed_path, ownership, _labels = _owned_ga_fixture(tmp_path)
    minimizer = GeneticAlgorithmMinimizer(
        gb,
        lambda *args: (0.0, str(seed_path)),
        ["translate_right_grain"],
        seed=0,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
    )
    with pytest.raises(RuntimeError, match="must use reload_explicit_manipulator"):
        minimizer._make_manipulator_from_file(str(seed_path))


def test_energy_selection_is_stable_for_ties(tmp_path):
    gb, seed_path, ownership, _labels = _owned_ga_fixture(tmp_path)
    minimizer = GeneticAlgorithmMinimizer(
        gb,
        lambda *args: (0.0, str(seed_path)),
        ["translate_right_grain"],
        seed=0,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=4,
        generations=1,
        keep_top_pct=50,
        intermediate_pct=100,
    )
    lowest, intermediate = minimizer._select_indices_by_energy([2.0, 1.0, 1.0, 3.0])
    assert lowest == [1, 2]
    assert intermediate == [1, 2, 0, 3]
