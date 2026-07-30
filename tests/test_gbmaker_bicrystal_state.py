# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Integration tests for GBMaker's generation-time BicrystalState contract."""

from pathlib import Path

import numpy as np
import pytest

from GBOpt.BicrystalState import BicrystalState
from GBOpt.BoundarySpec import FiveDOFSpec, PQSpec
from GBOpt.GBMaker import GBMaker, GBMakerValueError


IDENTITY_FIVE_DOF = FiveDOFSpec(params=(0.0, 0.0, 0.0, 0.0, 0.0))
SIGMA5_TILT_PQ = PQSpec(
    P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]],
    basis_mode="supplied",
)


def _build_approximate(
    *,
    vacuum: float,
    topology: str,
    boundary_conditions: tuple[str, str, str],
    provenance=None,
    termination_ids=(0, 0),
) -> GBMaker:
    return GBMaker.from_boundary_spec(
        3.615,
        "fcc",
        "Cu",
        IDENTITY_FIVE_DOF,
        mode="approximate",
        gb_thickness=0.0,
        repeat_factor=2,
        x_dim_min=8.0,
        vacuum=vacuum,
        interaction_distance=1.0,
        topology=topology,
        boundary_conditions=boundary_conditions,
        termination_ids=termination_ids,
        provenance=provenance,
    )


def test_gbmaker_periodic_seed_has_explicit_ids_and_two_interfaces():
    gb = _build_approximate(
        vacuum=0.0,
        topology="periodic_bicrystal",
        boundary_conditions=("periodic", "periodic", "periodic"),
    )
    state = gb.bicrystal_state

    assert state.topology == "periodic_bicrystal"
    assert state.boundary_conditions == ("periodic", "periodic", "periodic")
    assert len(state.interfaces) == 2
    assert {item.location for item in state.interfaces} == {
        "interior",
        "periodic_boundary",
    }
    np.testing.assert_array_equal(
        state.atom_ids,
        np.arange(1, len(gb.whole_system) + 1, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        state.grain_ids[: len(gb.left_grain)],
        np.zeros(len(gb.left_grain), dtype=np.int8),
    )
    np.testing.assert_array_equal(
        state.grain_ids[len(gb.left_grain) :],
        np.ones(len(gb.right_grain), dtype=np.int8),
    )
    assert gb.atom_ids is state.atom_ids
    assert gb.grain_ids is state.grain_ids


def test_gbmaker_slab_seed_has_one_interface_surfaces_vacuum_and_provenance():
    provenance = {
        "case_id": "zhang_001_ST_100",
        "source_row": 1,
        "source_table": "gb_data_gbopt.csv",
        "source_sha256": "abc123",
    }
    gb = _build_approximate(
        vacuum=3.0,
        topology="single_interface_slab",
        boundary_conditions=("fixed", "periodic", "periodic"),
        provenance=provenance,
        termination_ids=(2, 3),
    )
    state = gb.bicrystal_state

    assert len(state.interfaces) == 1
    assert state.interfaces[0].location == "interior"
    assert len(state.external_surfaces) == 2
    assert len(state.vacuum_regions) == 2
    assert state.fixed_regions == ()
    assert state.buffer_regions == ()
    assert state.termination_ids == (2, 3)
    assert dict(state.metadata["provenance"]) == provenance
    assert state.metadata["boundary_spec"]["type"] == "FiveDOFSpec"
    assert state.metadata["topology_source"] == "explicit"
    assert state.metadata["boundary_conditions_source"] == "explicit"


def test_gbmaker_state_round_trip_reconstructs_real_seed(tmp_path: Path):
    gb = _build_approximate(
        vacuum=3.0,
        topology="single_interface_slab",
        boundary_conditions=("fixed", "periodic", "periodic"),
        provenance={"case_id": "roundtrip"},
    )
    target = tmp_path / "bicrystal_state"

    gb.bicrystal_state.save(target)
    restored = BicrystalState.load(target)

    assert restored.structure_hash == gb.bicrystal_state.structure_hash
    assert restored.state_hash == gb.bicrystal_state.state_hash
    np.testing.assert_array_equal(restored.atoms, gb.whole_system)
    np.testing.assert_array_equal(restored.atom_ids, gb.atom_ids)
    np.testing.assert_array_equal(restored.grain_ids, gb.grain_ids)


def test_repeated_generation_produces_identical_state_hashes():
    kwargs = {
        "vacuum": 0.0,
        "topology": "periodic_bicrystal",
        "boundary_conditions": ("periodic", "periodic", "periodic"),
        "provenance": {"case_id": "deterministic", "source_row": 4},
    }

    first = _build_approximate(**kwargs).bicrystal_state
    second = _build_approximate(**kwargs).bicrystal_state

    assert first.structure_hash == second.structure_hash
    assert first.state_hash == second.state_hash


def test_real_seed_grain_ids_survive_right_grain_translation_across_midpoint():
    state = _build_approximate(
        vacuum=0.0,
        topology="periodic_bicrystal",
        boundary_conditions=("periodic", "periodic", "periodic"),
    ).bicrystal_state
    translated_atoms = state.atoms.copy()
    right_mask = state.grain_ids == 1
    translated_atoms["x"][right_mask] -= 0.35 * (
        state.box_dims[0, 1] - state.box_dims[0, 0]
    )

    translated = state.with_atoms(translated_atoms)

    np.testing.assert_array_equal(translated.atom_ids, state.atom_ids)
    np.testing.assert_array_equal(translated.grain_ids, state.grain_ids)
    midpoint = float(np.mean(state.box_dims[0]))
    assert np.any(translated.atoms["x"][right_mask] < midpoint)


def test_exact_boundary_state_retains_resolved_pq_seed_metadata():
    gb = GBMaker.from_boundary_spec(
        3.615,
        "fcc",
        "Cu",
        SIGMA5_TILT_PQ,
        mode="exact",
        gb_thickness=0.0,
        repeat_factor=2,
        x_dim_min=8.0,
        vacuum=0.0,
        interaction_distance=1.0,
        topology="periodic_bicrystal",
        boundary_conditions=("periodic", "periodic", "periodic"),
        provenance={"case_id": "sigma5_exact"},
    )
    metadata = gb.bicrystal_state.metadata

    assert gb.uses_exact_construction
    assert metadata["construction_mode"] == "exact"
    assert metadata["boundary_spec"]["type"] == "PQSpec"
    assert metadata["embedding"]["exact"] is True
    np.testing.assert_array_equal(metadata["embedding"]["P"], SIGMA5_TILT_PQ.P)
    np.testing.assert_array_equal(metadata["embedding"]["Q"], SIGMA5_TILT_PQ.Q)


@pytest.mark.parametrize(
    ("vacuum", "topology", "boundary_conditions", "match"),
    [
        pytest.param(
            2.0,
            "periodic_bicrystal",
            ("periodic", "periodic", "periodic"),
            "requires vacuum=0",
            id="periodic-with-vacuum",
        ),
        pytest.param(
            0.0,
            "single_interface_slab",
            ("periodic", "periodic", "periodic"),
            "requires x boundary condition 'fixed'",
            id="slab-with-periodic-x",
        ),
    ],
)
def test_inconsistent_explicit_topology_is_rejected(
    vacuum,
    topology,
    boundary_conditions,
    match,
):
    with pytest.raises(GBMakerValueError, match=match):
        _build_approximate(
            vacuum=vacuum,
            topology=topology,
            boundary_conditions=boundary_conditions,
        )


def test_legacy_vacuum_compatibility_is_resolved_and_recorded_once():
    gb = GBMaker.from_boundary_spec(
        3.615,
        "fcc",
        "Cu",
        IDENTITY_FIVE_DOF,
        mode="approximate",
        gb_thickness=0.0,
        repeat_factor=2,
        x_dim_min=8.0,
        vacuum=0.0,
        interaction_distance=1.0,
    )

    assert gb.topology == "periodic_bicrystal"
    assert gb.boundary_conditions == ("periodic", "periodic", "periodic")
    assert gb.bicrystal_state.metadata["topology_source"] == (
        "legacy_vacuum_inference"
    )
    assert gb.bicrystal_state.metadata["boundary_conditions_source"] == (
        "construction_default"
    )


def test_fixed_inplane_conditions_emit_side_surface_descriptors():
    gb = _build_approximate(
        vacuum=0.0,
        topology="periodic_bicrystal",
        boundary_conditions=("periodic", "fixed", "fixed"),
    )

    surfaces = gb.bicrystal_state.external_surfaces
    assert len(surfaces) == 4
    assert {surface.axis for surface in surfaces} == {1, 2}
    assert {surface.surface_id for surface in surfaces} == {
        "y_lower_surface",
        "y_upper_surface",
        "z_lower_surface",
        "z_upper_surface",
    }


def test_slab_fixed_and_surface_buffer_regions_are_physical_state_descriptors():
    gb = GBMaker.from_boundary_spec(
        3.615,
        "fcc",
        "Cu",
        IDENTITY_FIVE_DOF,
        mode="approximate",
        gb_thickness=0.0,
        repeat_factor=2,
        x_dim_min=8.0,
        vacuum=3.0,
        fixed_region_thickness=1.0,
        surface_buffer_thickness=1.5,
        interaction_distance=1.0,
        topology="single_interface_slab",
        boundary_conditions=("fixed", "periodic", "periodic"),
    )
    state = gb.bicrystal_state
    surfaces = {surface.surface_id: surface for surface in state.external_surfaces}
    fixed = {region.region_id: region for region in state.fixed_regions}
    buffers = {region.region_id: region for region in state.buffer_regions}

    assert set(fixed) == {"left_fixed", "right_fixed"}
    assert set(buffers) == {"left_surface_buffer", "right_surface_buffer"}
    assert fixed["left_fixed"].lower == surfaces["left_surface"].position
    assert fixed["left_fixed"].upper == buffers["left_surface_buffer"].lower
    assert buffers["right_surface_buffer"].upper == fixed["right_fixed"].lower
    assert fixed["right_fixed"].upper == surfaces["right_surface"].position
    assert fixed["left_fixed"].grain_ids == (0,)
    assert fixed["right_fixed"].grain_ids == (1,)
    assert state.metadata["fixed_region_thickness"] == 1.0
    assert state.metadata["surface_buffer_thickness"] == 1.5


@pytest.mark.parametrize(
    ("topology", "vacuum", "fixed", "buffer", "match"),
    [
        pytest.param(
            "periodic_bicrystal",
            0.0,
            1.0,
            0.0,
            "does not support slab fixed",
            id="periodic-fixed-region",
        ),
        pytest.param(
            "single_interface_slab",
            3.0,
            100.0,
            1.0,
            "exceeds the available solid thickness",
            id="slab-region-too-wide",
        ),
    ],
)
def test_invalid_slab_region_construction_is_rejected(
    topology, vacuum, fixed, buffer, match
):
    conditions = (
        ("periodic", "periodic", "periodic")
        if topology == "periodic_bicrystal"
        else ("fixed", "periodic", "periodic")
    )
    with pytest.raises(GBMakerValueError, match=match):
        GBMaker.from_boundary_spec(
            3.615,
            "fcc",
            "Cu",
            IDENTITY_FIVE_DOF,
            mode="approximate",
            gb_thickness=0.0,
            repeat_factor=2,
            x_dim_min=8.0,
            vacuum=vacuum,
            fixed_region_thickness=fixed,
            surface_buffer_thickness=buffer,
            interaction_distance=1.0,
            topology=topology,
            boundary_conditions=conditions,
        )
