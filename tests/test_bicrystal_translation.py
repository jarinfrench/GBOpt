# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for the topology-aware rigid BicrystalState translation primitive."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from GBOpt.BicrystalState import (
    RIGHT_GRAIN_ID,
    TRANSLATION_HISTORY_KEY,
    BicrystalState,
    BicrystalStateTypeError,
    BicrystalStateValueError,
    InterfaceDescriptor,
    RegionDescriptor,
    SurfaceDescriptor,
    translate_grain,
)

ATOM_DTYPE = np.dtype(
    [("name", "U2"), ("x", np.float64), ("y", np.float64), ("z", np.float64)]
)


@pytest.fixture
def periodic_state() -> BicrystalState:
    atoms = np.array(
        [
            ("U", -1.0, 11.0, -6.0),
            ("O", 0.5, 14.0, -2.0),
            ("U", 4.0, 14.5, -1.5),
            ("O", 7.5, 10.5, -6.5),
        ],
        dtype=ATOM_DTYPE,
    )
    return BicrystalState(
        atoms=atoms,
        box_dims=np.array(((-2.0, 8.0), (10.0, 15.0), (-7.0, -1.0))),
        topology="periodic_bicrystal",
        boundary_conditions=("periodic", "periodic", "periodic"),
        atom_ids=np.array((11, 12, 21, 22), dtype=np.int64),
        grain_ids=np.array((0, 0, 1, 1), dtype=np.int8),
        interfaces=(
            InterfaceDescriptor(
                "central_gb", 0, "interior", 3.0, 0, 1, (1.0, 0.0, 0.0)
            ),
            InterfaceDescriptor(
                "periodic_gb",
                0,
                "periodic_boundary",
                -2.0,
                1,
                0,
                (1.0, 0.0, 0.0),
                periodic_partner_position=8.0,
            ),
        ),
        termination_ids=(2, 4),
        metadata={
            "case_id": "synthetic_periodic",
            "source": {"row": 7, "P": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        },
    )


@pytest.fixture
def slab_state() -> BicrystalState:
    atoms = np.array(
        [
            ("U", -1.0, 11.0, -6.0),
            ("O", 1.0, 14.0, -2.0),
            ("U", 5.0, 14.5, -1.5),
            ("O", 8.0, 10.5, -6.5),
        ],
        dtype=ATOM_DTYPE,
    )
    return BicrystalState(
        atoms=atoms,
        box_dims=np.array(((-4.0, 12.0), (10.0, 15.0), (-7.0, -1.0))),
        topology="single_interface_slab",
        boundary_conditions=("fixed", "periodic", "periodic"),
        atom_ids=np.array((11, 12, 21, 22), dtype=np.int64),
        grain_ids=np.array((0, 0, 1, 1), dtype=np.int8),
        interfaces=(
            InterfaceDescriptor(
                "central_gb", 0, "interior", 3.0, 0, 1, (1.0, 0.0, 0.0)
            ),
        ),
        external_surfaces=(
            SurfaceDescriptor("left_surface", 0, -2.0, (-1.0, 0.0, 0.0), (0,)),
            SurfaceDescriptor("right_surface", 0, 10.0, (1.0, 0.0, 0.0), (1,)),
        ),
        vacuum_regions=(
            RegionDescriptor("left_vacuum", "vacuum", 0, -4.0, -2.0),
            RegionDescriptor("right_vacuum", "vacuum", 0, 10.0, 12.0),
        ),
        fixed_regions=(
            RegionDescriptor("left_fixed", "fixed", 0, -2.0, -0.5, (0,)),
        ),
        buffer_regions=(
            RegionDescriptor("right_buffer", "buffer", 0, 8.5, 10.0, (1,)),
        ),
        termination_ids=(3, 5),
        metadata={"case_id": "synthetic_slab", "source": {"row": 8}},
    )


def _positions(state: BicrystalState) -> np.ndarray:
    return np.column_stack((state.atoms["x"], state.atoms["y"], state.atoms["z"]))


def _history(state: BicrystalState) -> list[dict[str, object]]:
    return state.manifest()["metadata"][TRANSLATION_HISTORY_KEY]


def test_translation_returns_new_state_without_mutating_input(periodic_state) -> None:
    atoms_before = periodic_state.atoms.copy()
    metadata_before = periodic_state.manifest()["metadata"]
    structure_hash_before = periodic_state.structure_hash
    state_hash_before = periodic_state.state_hash

    translated = translate_grain(
        periodic_state, displacement=(1.0, 0.5, -0.25), grain="right"
    )

    assert translated is not periodic_state
    np.testing.assert_array_equal(periodic_state.atoms, atoms_before)
    assert periodic_state.manifest()["metadata"] == metadata_before
    assert periodic_state.structure_hash == structure_hash_before
    assert periodic_state.state_hash == state_hash_before
    assert not periodic_state.atoms.flags.writeable
    assert not translated.atoms.flags.writeable


def test_only_selected_right_grain_moves_and_identity_is_preserved(
    periodic_state,
) -> None:
    translated = translate_grain(
        periodic_state, grain=RIGHT_GRAIN_ID, displacement=(1.0, 0.5, 0.25)
    )
    left = periodic_state.grain_ids == 0
    right = periodic_state.grain_ids == 1

    np.testing.assert_array_equal(translated.atoms[left], periodic_state.atoms[left])
    assert not np.array_equal(translated.atoms[right], periodic_state.atoms[right])
    np.testing.assert_array_equal(translated.atom_ids, periodic_state.atom_ids)
    np.testing.assert_array_equal(translated.grain_ids, periodic_state.grain_ids)
    np.testing.assert_array_equal(
        translated.atoms["name"], periodic_state.atoms["name"]
    )
    assert translated.termination_ids == periodic_state.termination_ids
    assert translated.interfaces == periodic_state.interfaces
    assert translated.external_surfaces == periodic_state.external_surfaces
    assert translated.vacuum_regions == periodic_state.vacuum_regions
    assert translated.fixed_regions == periodic_state.fixed_regions
    assert translated.buffer_regions == periodic_state.buffer_regions


def test_arbitrary_lab_frame_displacement_uses_asymmetric_actual_bounds(
    periodic_state,
) -> None:
    translated = translate_grain(
        periodic_state, displacement=(2.5, 1.0, -2.0)
    )
    right = periodic_state.grain_ids == 1

    expected = np.array(
        [
            [6.5, 10.5, -3.5],
            [0.0, 11.5, -2.5],
        ]
    )
    np.testing.assert_allclose(
        _positions(translated)[right], expected, atol=0.0, rtol=0.0
    )
    assert translated.relative_translation_lab == (2.5, 1.0, 4.0)


def test_translation_along_periodic_interface_normal_wraps(periodic_state) -> None:
    translated = translate_grain(periodic_state, displacement=(3.0, 0.0, 0.0))
    right = periodic_state.grain_ids == 1

    np.testing.assert_allclose(translated.atoms["x"][right], (7.0, 0.5))
    assert translated.relative_translation_lab == (3.0, 0.0, 0.0)


def test_periodic_axes_wrap_but_fixed_axis_is_unwrapped(slab_state) -> None:
    translated = translate_grain(slab_state, displacement=(1.25, 1.0, -2.0))
    right = slab_state.grain_ids == 1

    np.testing.assert_allclose(translated.atoms["x"][right], (6.25, 9.25))
    np.testing.assert_allclose(translated.atoms["y"][right], (10.5, 11.5))
    np.testing.assert_allclose(translated.atoms["z"][right], (-3.5, -2.5))
    assert translated.relative_translation_lab == (1.25, 1.0, 4.0)
    assert translated.boundary_conditions == ("fixed", "periodic", "periodic")


def test_fixed_axis_out_of_box_translation_is_not_periodically_wrapped(
    slab_state,
) -> None:
    with pytest.raises(BicrystalStateValueError, match="outside box axis 0"):
        translate_grain(slab_state, displacement=(5.0, 0.0, 0.0))

    np.testing.assert_allclose(
        slab_state.atoms["x"][slab_state.grain_ids == 1], (5.0, 8.0)
    )


def test_sequential_translations_accumulate_deterministically(periodic_state) -> None:
    first = translate_grain(periodic_state, displacement=(1.5, 2.0, -1.0))
    second = translate_grain(first, displacement=(2.5, 4.0, 3.0))
    combined = translate_grain(periodic_state, displacement=(4.0, 6.0, 2.0))

    assert second.relative_translation_lab == (4.0, 1.0, 2.0)
    np.testing.assert_array_equal(second.atoms, combined.atoms)
    assert second.structure_hash == combined.structure_hash
    assert second.state_hash != combined.state_hash
    assert len(_history(second)) == 2
    assert len(_history(combined)) == 1


def test_periodically_equivalent_displacements_canonicalize_identically(
    periodic_state,
) -> None:
    first = translate_grain(periodic_state, displacement=(1.0, 0.5, -0.25))
    equivalent = translate_grain(periodic_state, displacement=(11.0, 5.5, 5.75))

    np.testing.assert_array_equal(first.atoms, equivalent.atoms)
    assert first.relative_translation_lab == equivalent.relative_translation_lab
    assert first.structure_hash == equivalent.structure_hash
    assert first.state_hash == equivalent.state_hash
    assert first.manifest() == equivalent.manifest()


def test_zero_and_periodic_box_vector_have_identical_canonical_provenance(
    periodic_state,
) -> None:
    zero = translate_grain(periodic_state, displacement=(0.0, 0.0, 0.0))
    box_vector = translate_grain(periodic_state, displacement=(10.0, -5.0, 12.0))

    np.testing.assert_array_equal(zero.atoms, periodic_state.atoms)
    assert zero.structure_hash == periodic_state.structure_hash
    assert zero.state_hash != periodic_state.state_hash
    assert zero.state_hash == box_vector.state_hash
    assert _history(zero)[0]["displacement_lab"] == [0.0, 0.0, 0.0]


def test_provenance_contains_reproducible_operation_inputs(periodic_state) -> None:
    translated = translate_grain(
        periodic_state, grain="right", displacement=(1.0, 6.0, -0.5)
    )
    operation = _history(translated)[0]

    assert operation == {
        "operation": "translate_grain",
        "schema_version": 1,
        "grain": "right",
        "grain_id": 1,
        "coordinates": "lab",
        "displacement_lab": [1.0, 1.0, 5.5],
        "periodic_axes": [0, 1, 2],
        "box_bounds_lab": [[-2.0, 8.0], [10.0, 15.0], [-7.0, -1.0]],
        "relative_translation_before_lab": [0.0, 0.0, 0.0],
        "relative_translation_after_lab": [1.0, 1.0, 5.5],
        "input_structure_hash": periodic_state.structure_hash,
        "input_state_hash": periodic_state.state_hash,
    }
    assert translated.manifest()["metadata"]["case_id"] == "synthetic_periodic"
    assert translated.manifest()["metadata"]["source"] == {
        "P": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "row": 7,
    }


def test_serialization_and_hashes_are_deterministic(
    periodic_state, tmp_path: Path
) -> None:
    first = translate_grain(periodic_state, displacement=(1.0, 2.0, 3.0))
    second = translate_grain(periodic_state, displacement=(1.0, 2.0, 3.0))
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"

    first.save(first_path)
    second.save(second_path)
    loaded = BicrystalState.load(first_path)

    assert first.structure_hash == second.structure_hash == loaded.structure_hash
    assert first.state_hash == second.state_hash == loaded.state_hash
    assert first.manifest() == second.manifest() == loaded.manifest()
    for filename in (
        "atoms.npy",
        "box_dims.npy",
        "atom_ids.npy",
        "grain_ids.npy",
        "state.json",
    ):
        assert (first_path / filename).read_bytes() == (
            second_path / filename
        ).read_bytes()
    json.loads((first_path / "state.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "grain",
    [
        pytest.param("left", id="left-name"),
        pytest.param(0, id="left-id"),
        pytest.param("RIGHT", id="wrong-case"),
        pytest.param(2, id="unknown-id"),
    ],
)
def test_invalid_grain_values_are_rejected(periodic_state, grain) -> None:
    with pytest.raises(BicrystalStateValueError, match="documented moving grain"):
        translate_grain(periodic_state, grain=grain, displacement=(0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "grain",
    [
        pytest.param(True, id="boolean"),
        pytest.param(1.0, id="float"),
        pytest.param(None, id="none"),
    ],
)
def test_invalid_grain_types_are_rejected(periodic_state, grain) -> None:
    with pytest.raises(BicrystalStateTypeError, match="grain must be"):
        translate_grain(periodic_state, grain=grain, displacement=(0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    ("displacement", "error", "match"),
    [
        pytest.param((1.0, 2.0), BicrystalStateValueError, "exactly three", id="short"),
        pytest.param(
            (1.0, 2.0, 3.0, 4.0),
            BicrystalStateValueError,
            "exactly three",
            id="long",
        ),
        pytest.param(1.0, BicrystalStateTypeError, "numeric sequence", id="scalar"),
        pytest.param("1,2,3", BicrystalStateTypeError, "numeric sequence", id="string"),
        pytest.param(
            (1.0, True, 3.0),
            BicrystalStateTypeError,
            "finite float",
            id="boolean",
        ),
        pytest.param(
            (1.0, "bad", 3.0),
            BicrystalStateTypeError,
            "finite float",
            id="text",
        ),
        pytest.param((1.0, np.nan, 3.0), BicrystalStateValueError, "finite", id="nan"),
        pytest.param(
            (1.0, np.inf, 3.0),
            BicrystalStateValueError,
            "finite",
            id="infinity",
        ),
        pytest.param(
            np.array([[1.0], [2.0], [3.0]]),
            BicrystalStateTypeError,
            "scalar finite float",
            id="nested-array",
        ),
    ],
)
def test_malformed_or_nonfinite_displacements_are_rejected(
    periodic_state, displacement, error, match
) -> None:
    with pytest.raises(error, match=match):
        translate_grain(periodic_state, displacement=displacement)


def test_only_lab_coordinates_are_supported(periodic_state) -> None:
    with pytest.raises(BicrystalStateValueError, match="must be 'lab'"):
        translate_grain(
            periodic_state,
            displacement=(0.0, 0.0, 0.0),
            coordinates="fractional",  # type: ignore[arg-type]
        )


def test_malformed_existing_translation_history_is_rejected(periodic_state) -> None:
    malformed = BicrystalState(
        atoms=periodic_state.atoms,
        box_dims=periodic_state.box_dims,
        topology=periodic_state.topology,
        boundary_conditions=periodic_state.boundary_conditions,
        atom_ids=periodic_state.atom_ids,
        grain_ids=periodic_state.grain_ids,
        interfaces=periodic_state.interfaces,
        termination_ids=periodic_state.termination_ids,
        metadata={TRANSLATION_HISTORY_KEY: "not-a-sequence"},
    )

    with pytest.raises(BicrystalStateValueError, match=TRANSLATION_HISTORY_KEY):
        translate_grain(malformed, displacement=(0.0, 0.0, 0.0))


def test_legacy_zero_based_inplane_behavior_is_compatible() -> None:
    atoms = np.array(
        [
            ("U", 1.0, 1.0, 1.0),
            ("O", 2.0, 2.0, 2.0),
            ("U", 6.0, 4.5, 5.5),
            ("O", 8.0, 0.5, 0.5),
        ],
        dtype=ATOM_DTYPE,
    )
    state = BicrystalState(
        atoms=atoms,
        box_dims=np.array(((0.0, 10.0), (0.0, 5.0), (0.0, 6.0))),
        topology="periodic_bicrystal",
        boundary_conditions=("periodic", "periodic", "periodic"),
        atom_ids=np.arange(1, 5),
        grain_ids=np.array((0, 0, 1, 1)),
        interfaces=(
            InterfaceDescriptor("central", 0, "interior", 5.0, 0, 1, (1.0, 0.0, 0.0)),
            InterfaceDescriptor(
                "periodic", 0, "periodic_boundary", 0.0, 1, 0, (1.0, 0.0, 0.0), 10.0
            ),
        ),
    )
    dy, dz = 1.5, 2.0

    translated = translate_grain(state, displacement=(0.0, dy, dz))
    expected = state.atoms.copy()
    right = state.grain_ids == 1
    expected["y"][right] = (expected["y"][right] + dy) % 5.0
    expected["z"][right] = (expected["z"][right] + dz) % 6.0

    np.testing.assert_array_equal(translated.atoms, expected)
