# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""End-to-end coverage of the full refactored stack for both boundary topologies.

Exercises build (``GBMaker``) -> manipulate (``GBManipulator``) -> write
(``CandidateLoader``/``LammpsDataWriter``) -> evaluate/reload
(``CandidateLoader``/legacy reload) -> optimize (``MonteCarloMinimizer``/
``GeneticAlgorithmMinimizer``) as one coherent claim, for both
``BoundaryNormalTopology.PERIODIC_BICRYSTAL`` (``vacuum=0``) and
``BoundaryNormalTopology.SINGLE_INTERFACE_SLAB`` (``vacuum>0``) --
``GBOpt.gbmaker.dimension._normalize_vacuum_topology``'s own boundary condition.

This does not duplicate coverage: the owned-mode (explicit-ownership) genetic-algorithm
suite (``tests/test_optimization_genetic.py``'s ``owned_ga`` fixture) already runs a full
periodic-topology owned-mode GA loop through ``CandidateLoader``, and the Monte Carlo
suite (``tests/test_optimization_monte_carlo.py``'s ``gb`` fixture) already runs a full
slab-topology legacy-reload MC loop -- both incidentally, as fixtures for other
acceptance criteria. This file makes the same composition claim explicit and first-class,
parametrized directly over topology, rather than leaving it to be inferred from two
unrelated fixtures' incidental vacuum values.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from GBOpt.BoundarySpec import CSLExactSpec
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.CandidateLoader import CandidateLoader
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import GBManipulator
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL, GrainOwnership
from GBOpt.observability import CompositeEventSink, JsonlEventSink, LoggingEventSink
from GBOpt.optimization.genetic import GeneticAlgorithmMinimizer
from GBOpt.optimization.monte_carlo import MonteCarloMinimizer

pytestmark = pytest.mark.filterwarnings(
    "ignore:File-backed Parent initialization without explicit grain ownership is "
    "deprecated.*:DeprecationWarning"
)


def _build_gb(vacuum: float) -> GBMaker:
    theta = math.radians(36.869898)
    misorientation = np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0])
    return GBMaker(
        3.52,
        "fcc",
        10.0,
        misorientation,
        "Ni",
        repeat_factor=(2, 5),
        x_dim_min=20.0,
        vacuum=vacuum,
        interaction_distance=8.0,
    )


@pytest.mark.parametrize(
    ("vacuum", "expected_topology"),
    [
        pytest.param(0.0, BoundaryNormalTopology.PERIODIC_BICRYSTAL, id="periodic"),
        pytest.param(8.0, BoundaryNormalTopology.SINGLE_INTERFACE_SLAB, id="slab"),
    ],
)
def test_build_manipulate_write_reload_roundtrip(
    tmp_path, vacuum, expected_topology
) -> None:
    """Build -> manipulate -> write -> reload preserves atom count and topology."""
    gb = _build_gb(vacuum)
    assert gb.normal_topology == expected_topology

    manipulator = GBManipulator(gb)
    manipulator.rng = np.random.default_rng(0)
    manipulator.translate_right_grain(0.5, 0.0)

    atoms = manipulator.parents[0].whole_system
    labels = np.hstack(
        (
            np.full(len(gb.left_grain), LEFT_GRAIN_LABEL, dtype=np.int8),
            np.full(len(gb.right_grain), RIGHT_GRAIN_LABEL, dtype=np.int8),
        )
    )
    loader = CandidateLoader()
    path = tmp_path / "candidate.data"
    result, mapping = loader.write_candidate(
        path,
        atoms,
        labels,
        box_dims=gb.box_dims,
        gb_plane_x=gb.gb_plane_x,
        inplane_periodic=gb.inplane_periodic,
        right_grain_x_bounds=(gb.gb_plane_x, gb.box_dims[0, 1]),
        left_grain_x_bounds=(gb.box_dims[0, 0], gb.gb_plane_x),
        normal_topology=expected_topology,
        type_map=gb.unit_cell.type_map,
        coordinate_tolerance=gb.epsilon,
    )

    # WriteResult's atom_ids are candidate-local (see GBOpt/io/types.py's own docstring)
    # -- the mapping built from them carries exactly as many entries as atoms written.
    assert len(result.atom_ids) == len(atoms)
    assert len(mapping.atom_ids) == len(atoms)

    reloaded = loader.reload(
        path,
        candidate_mapping=mapping,
        unit_cell=gb.unit_cell,
        gb_thickness=gb.gb_thickness,
    )
    assert len(reloaded.parents[0].whole_system) == len(atoms)
    assert reloaded.parents[0].normal_topology == expected_topology


@pytest.mark.parametrize("vacuum", [0.0, 8.0], ids=["periodic", "slab"])
def test_end_to_end_monte_carlo_optimize(tmp_path, vacuum) -> None:
    """A short MC run composes build/manipulate/write/reload/optimize end to end."""
    gb = _build_gb(vacuum)
    call_count = 0

    def energy_func(GB, manipulator, atom_positions, unique_id):
        nonlocal call_count
        call_count += 1
        path = tmp_path / f"{unique_id}_{call_count}.data"
        GB.write_lammps(str(path), atom_positions, manipulator.parents[0].box_dims)
        return 2.0 - call_count * 0.001, str(path)

    minimizer = MonteCarloMinimizer(
        gb, energy_func, ["translate_right_grain"], seed=0
    )
    minimizer.run_MC(
        max_steps=3,
        unique_id=1,
        checkpoint_file=str(tmp_path / "mc_checkpoint.json"),
    )
    assert call_count >= 1
    assert len(minimizer.GBE_vals) >= 1


def test_end_to_end_owned_ga_optimize_periodic(tmp_path) -> None:
    """A short owned-mode GA run reloads every candidate through CandidateLoader."""
    gb = GBMaker.from_boundary_spec(
        3.52,
        "fcc",
        "Ni",
        CSLExactSpec(axis=(0, 0, 1), plane=(3, 1, 0), quat=(3, 0, 0, 1), sigma=5),
        mode="exact",
        gb_thickness=10.0,
        repeat_factor=2,
        x_dim_min=10.0,
        vacuum=0.0,
        interaction_distance=3.0,
    )
    assert gb.normal_topology == BoundaryNormalTopology.PERIODIC_BICRYSTAL

    seed_path = tmp_path / "owned_initial.data"
    gb.write_lammps(str(seed_path), type_as_int=False, precision=12)
    labels = np.hstack(
        (
            np.full(len(gb.left_grain), LEFT_GRAIN_LABEL, dtype=np.int8),
            np.full(len(gb.right_grain), RIGHT_GRAIN_LABEL, dtype=np.int8),
        )
    )
    ownership = GrainOwnership(
        atom_ids=np.arange(1, len(gb.whole_system) + 1),
        labels=labels,
        gb_plane_x=gb.gb_plane_x,
        inplane_periodic=gb.inplane_periodic,
        left_grain_x_bounds=(gb.box_dims[0, 0], gb.gb_plane_x),
        right_grain_x_bounds=(gb.gb_plane_x, gb.box_dims[0, 1]),
        normal_topology=gb.normal_topology,
        coordinate_tolerance=gb.epsilon,
    )

    def energy(GB, manipulator, atom_positions, unique_id):
        box = np.asarray(manipulator.parents[0].box_dims, dtype=float)
        output = tmp_path / f"{unique_id}.data"
        GB.write_lammps(
            str(output), atom_positions, box, type_as_int=False, precision=12
        )
        value = float(np.mean(atom_positions["x"]))
        return value, str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        energy,
        ["translate_right_grain"],
        seed=101,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=3,
        generations=2,
        keep_top_pct=34,
        intermediate_pct=100,
    )
    minimizer.run_GA(unique_id=1)

    assert len(minimizer.history) == 2
    assert len(minimizer.history[-1]) > 0


def test_simultaneous_logging_journal_checkpoint_does_not_change_numerical_history(
    tmp_path,
) -> None:
    """Enabling logging + a JSONL journal + checkpointing together is numerically inert.

    Runs the same fixed-seed MC sequence twice: once with only checkpointing enabled
    (the baseline every other test in this suite already exercises implicitly), once
    with a `LoggingEventSink` and a `JsonlEventSink` also wired in through a
    `CompositeEventSink`. Only the energy history is compared -- per
    `GBOpt.observability`'s own contract, logging/journaling are observers of already-
    decided optimizer state, never inputs to it, so wiring them in must not perturb the
    accept/reject sequence a fixed seed produces.
    """
    gb = _build_gb(vacuum=0.0)

    def make_energy_func():
        count = 0

        def energy_func(GB, manipulator, atom_positions, unique_id):
            nonlocal count
            count += 1
            path = tmp_path / f"{unique_id}_{count}.data"
            GB.write_lammps(str(path), atom_positions, manipulator.parents[0].box_dims)
            return 2.0 - count * 0.001, str(path)

        return energy_func

    baseline = MonteCarloMinimizer(
        gb, make_energy_func(), ["translate_right_grain"], seed=7
    )
    baseline.run_MC(
        max_steps=4,
        unique_id=1,
        checkpoint_file=str(tmp_path / "baseline_checkpoint.json"),
    )

    journal_path = tmp_path / "run.jsonl"
    logger = logging.getLogger("test_simultaneous_sinks")
    with JsonlEventSink(journal_path) as jsonl_sink:
        sink = CompositeEventSink(
            (LoggingEventSink(logger), jsonl_sink)
        )
        instrumented = MonteCarloMinimizer(
            gb,
            make_energy_func(),
            ["translate_right_grain"],
            seed=7,
            event_sink=sink,
        )
        instrumented.run_MC(
            max_steps=4,
            unique_id=2,
            checkpoint_file=str(tmp_path / "instrumented_checkpoint.json"),
        )

    assert instrumented.GBE_vals == baseline.GBE_vals
    assert journal_path.exists()
    assert journal_path.stat().st_size > 0

    # Journal files cannot be loaded as checkpoints: a JSONL event stream has none of
    # a checkpoint envelope's required fields (schema_version/minimizer/progress_unit),
    # so CheckpointStore.load rejects it rather than silently treating it as restart
    # state.
    from GBOpt.Checkpoint import CheckpointError, CheckpointStore

    with pytest.raises((CheckpointError, ValueError)):
        CheckpointStore.from_optional(journal_path).load()
