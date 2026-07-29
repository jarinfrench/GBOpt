# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

from pathlib import Path

import numpy as np
import pytest

from GBOpt.BicrystalState import (
    BicrystalState,
    BicrystalStateValueError,
    InterfaceDescriptor,
    RegionDescriptor,
    SurfaceDescriptor,
)

ATOM_DTYPE = np.dtype(
    [("name", "U2"), ("x", np.float64), ("y", np.float64), ("z", np.float64)]
)


def _atoms() -> np.ndarray:
    return np.array(
        [
            ("U", 2.0, 1.0, 1.0),
            ("O", 3.0, 2.0, 2.0),
            ("U", 7.0, 1.0, 1.0),
            ("O", 8.0, 2.0, 2.0),
        ],
        dtype=ATOM_DTYPE,
    )


def _periodic_state(**overrides) -> BicrystalState:
    kwargs = {
        "atoms": _atoms(),
        "box_dims": np.array(((0.0, 10.0), (0.0, 5.0), (0.0, 5.0))),
        "topology": "periodic_bicrystal",
        "boundary_conditions": ("periodic", "periodic", "periodic"),
        "atom_ids": np.arange(1, 5, dtype=np.int64),
        "grain_ids": np.array((0, 0, 1, 1), dtype=np.int8),
        "interfaces": (
            InterfaceDescriptor(
                interface_id="central_gb",
                axis=0,
                location="interior",
                position=5.0,
                minus_grain_id=0,
                plus_grain_id=1,
                normal_lab=(1.0, 0.0, 0.0),
            ),
            InterfaceDescriptor(
                interface_id="periodic_gb",
                axis=0,
                location="periodic_boundary",
                position=0.0,
                periodic_partner_position=10.0,
                minus_grain_id=1,
                plus_grain_id=0,
                normal_lab=(1.0, 0.0, 0.0),
            ),
        ),
        "metadata": {
            "case_id": "zhang_001_ST_100",
            "source_row": 1,
            "P": [[0, 18, -1], [0, 1, 18], [1, 0, 0]],
        },
    }
    kwargs.update(overrides)
    return BicrystalState(**kwargs)


def _slab_state() -> BicrystalState:
    atoms = _atoms()
    atoms["x"] += 2.0
    return BicrystalState(
        atoms=atoms,
        box_dims=np.array(((0.0, 14.0), (0.0, 5.0), (0.0, 5.0))),
        topology="single_interface_slab",
        boundary_conditions=("fixed", "periodic", "periodic"),
        atom_ids=np.arange(1, 5, dtype=np.int64),
        grain_ids=np.array((0, 0, 1, 1), dtype=np.int8),
        interfaces=(
            InterfaceDescriptor(
                interface_id="central_gb",
                axis=0,
                location="interior",
                position=7.0,
                minus_grain_id=0,
                plus_grain_id=1,
                normal_lab=(1.0, 0.0, 0.0),
            ),
        ),
        external_surfaces=(
            SurfaceDescriptor(
                surface_id="left_surface",
                axis=0,
                position=2.0,
                outward_normal_lab=(-1.0, 0.0, 0.0),
                grain_ids=(0,),
            ),
            SurfaceDescriptor(
                surface_id="right_surface",
                axis=0,
                position=12.0,
                outward_normal_lab=(1.0, 0.0, 0.0),
                grain_ids=(1,),
            ),
        ),
        vacuum_regions=(
            RegionDescriptor(
                region_id="lower_vacuum",
                kind="vacuum",
                axis=0,
                lower=0.0,
                upper=2.0,
            ),
            RegionDescriptor(
                region_id="upper_vacuum",
                kind="vacuum",
                axis=0,
                lower=12.0,
                upper=14.0,
            ),
        ),
        metadata={"case_id": "synthetic_slab", "source_row": 1},
    )


def test_periodic_bicrystal_has_two_active_interfaces():
    state = _periodic_state()

    assert len(state.interfaces) == 2
    assert {interface.location for interface in state.interfaces} == {
        "interior",
        "periodic_boundary",
    }
    assert state.external_surfaces == ()
    assert state.vacuum_regions == ()


def test_single_interface_slab_has_one_gb_and_separate_surfaces_and_vacuum():
    state = _slab_state()

    assert len(state.interfaces) == 1
    assert state.interfaces[0].location == "interior"
    assert len(state.external_surfaces) == 2
    assert len(state.vacuum_regions) == 2
    assert all(region.kind == "vacuum" for region in state.vacuum_regions)


def test_state_arrays_are_defensive_read_only_copies():
    atoms = _atoms()
    state = _periodic_state(atoms=atoms)
    atoms["x"] += 1.0

    assert state.atoms[0]["x"] == 2.0
    assert not state.atoms.flags.writeable
    assert not state.box_dims.flags.writeable
    assert not state.atom_ids.flags.writeable
    assert not state.grain_ids.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        state.atom_ids[0] = 99


def test_coordinate_change_does_not_reclassify_grains_or_atom_ids():
    state = _periodic_state()
    translated_atoms = state.atoms.copy()
    translated_atoms["x"][state.grain_ids == 1] -= 4.0

    translated = state.with_atoms(translated_atoms)

    np.testing.assert_array_equal(translated.atom_ids, state.atom_ids)
    np.testing.assert_array_equal(translated.grain_ids, state.grain_ids)
    assert np.any(translated.atoms["x"][translated.grain_ids == 1] < 5.0)


def test_sub_microangstrom_box_excursion_is_accepted_without_clipping():
    atoms = _atoms()
    atoms["z"][0] = -2.2e-7

    state = _periodic_state(atoms=atoms)

    assert state.atoms["z"][0] == pytest.approx(-2.2e-7, abs=0.0)


def test_atom_materially_outside_box_is_rejected():
    atoms = _atoms()
    atoms["z"][0] = -1.0e-4

    with pytest.raises(BicrystalStateValueError, match="extends outside box axis 2"):
        _periodic_state(atoms=atoms)


def test_round_trip_preserves_complete_state(tmp_path: Path):
    state = _slab_state()
    target = tmp_path / "seed_state"

    state.save(target)
    loaded = BicrystalState.load(target)

    np.testing.assert_array_equal(loaded.atoms, state.atoms)
    np.testing.assert_array_equal(loaded.box_dims, state.box_dims)
    np.testing.assert_array_equal(loaded.atom_ids, state.atom_ids)
    np.testing.assert_array_equal(loaded.grain_ids, state.grain_ids)
    assert loaded.boundary_conditions == state.boundary_conditions
    assert loaded.interfaces == state.interfaces
    assert loaded.external_surfaces == state.external_surfaces
    assert loaded.vacuum_regions == state.vacuum_regions
    assert loaded.relative_translation_lab == state.relative_translation_lab
    assert loaded.termination_ids == state.termination_ids
    assert dict(loaded.metadata) == dict(state.metadata)
    assert loaded.structure_hash == state.structure_hash
    assert loaded.state_hash == state.state_hash


def test_serialization_and_hashes_are_deterministic(tmp_path: Path):
    first = _periodic_state()
    second = _periodic_state()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first.save(first_dir)
    second.save(second_dir)

    assert first.structure_hash == second.structure_hash
    assert first.state_hash == second.state_hash
    for filename in (
        "atoms.npy",
        "box_dims.npy",
        "atom_ids.npy",
        "grain_ids.npy",
        "state.json",
    ):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()


def test_periodic_topology_rejects_missing_periodic_interface():
    central_only = (_periodic_state().interfaces[0],)

    with pytest.raises(
        BicrystalStateValueError,
        match="one interior and one periodic",
    ):
        _periodic_state(interfaces=central_only)


def test_slab_topology_rejects_periodic_normal_boundary_condition():
    slab = _slab_state()

    with pytest.raises(
        BicrystalStateValueError,
        match="nonperiodic normal axis",
    ):
        BicrystalState(
            atoms=slab.atoms,
            box_dims=slab.box_dims,
            topology=slab.topology,
            boundary_conditions=("periodic", "periodic", "periodic"),
            atom_ids=slab.atom_ids,
            grain_ids=slab.grain_ids,
            interfaces=slab.interfaces,
            external_surfaces=slab.external_surfaces,
            vacuum_regions=slab.vacuum_regions,
            metadata=slab.metadata,
        )


def test_periodic_normal_topology_allows_surfaces_on_fixed_inplane_axis():
    state = _periodic_state(
        boundary_conditions=("periodic", "fixed", "periodic"),
        external_surfaces=(
            SurfaceDescriptor(
                surface_id="y_lower_surface",
                axis=1,
                position=0.0,
                outward_normal_lab=(0.0, -1.0, 0.0),
                grain_ids=(0, 1),
            ),
            SurfaceDescriptor(
                surface_id="y_upper_surface",
                axis=1,
                position=5.0,
                outward_normal_lab=(0.0, 1.0, 0.0),
                grain_ids=(0, 1),
            ),
        ),
    )

    assert len(state.interfaces) == 2
    assert len(state.external_surfaces) == 2
    assert all(surface.axis == 1 for surface in state.external_surfaces)
