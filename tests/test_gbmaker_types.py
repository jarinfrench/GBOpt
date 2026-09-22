# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
from typing import Any

import numpy as np
import pytest

from GBOpt.BoundarySpec import BoundaryEmbedding
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.gbmaker.types import (
    AxisAccommodation,
    BicrystalResult,
    DimensionPlan,
    GBBuildConfig,
    GBMakerConstructionTypeError,
    GBMakerConstructionValueError,
    GrainBuildRequest,
    GrainBuildResult,
    MaterialState,
    OrientationState,
    ResolvedBoundaryInput,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _material(**overrides: Any) -> MaterialState:
    kwargs: dict[str, Any] = {"a0": 3.5, "structure": "fcc", "atom_types": "Ni"}
    kwargs.update(overrides)
    return MaterialState(**kwargs)


def _embedding(**overrides: Any) -> BoundaryEmbedding:
    kwargs: dict[str, Any] = {
        "P": np.eye(3, dtype=int),
        "Q": np.eye(3, dtype=int),
        "R_left": np.eye(3),
        "R_right": np.eye(3),
        "exact": True,
        "coherent": True,
        "source": "pq",
    }
    kwargs.update(overrides)
    return BoundaryEmbedding(**kwargs)


def _box_dims() -> np.ndarray:
    return np.array([[0.0, 70.0], [0.0, 30.0], [0.0, 30.0]])


def _axis_accommodation_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "left_repeats": 2,
        "right_repeats": 3,
        "left_unstrained_length": 10.0,
        "right_unstrained_length": 15.0,
        "box_length": 30.0,
        "left_scale": 3.0,
        "right_scale": 2.0,
        "mismatch": 0.05,
    }
    kwargs.update(overrides)
    return kwargs


def _atoms(count: int = 4) -> np.ndarray:
    return np.zeros(count, dtype=[("name", "U2"), ("x", "f8"), ("y", "f8"), ("z", "f8")])


# --------------------------------------------------------------------------------------
# MaterialState
# --------------------------------------------------------------------------------------


def test_material_state_normalizes_valid_fields() -> None:
    material = _material(a0=3.5, structure="fcc", atom_types=("Ni", "Al"))
    assert material.a0 == 3.5
    assert material.structure == "fcc"
    assert material.atom_types == ("Ni", "Al")


def test_material_state_rejects_nonpositive_a0() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        _material(a0=0.0)


def test_material_state_rejects_empty_atom_types() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        _material(atom_types="")


def test_material_state_rejects_non_string_atom_type_element() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        _material(atom_types=(1,))


# --------------------------------------------------------------------------------------
# GBBuildConfig
# --------------------------------------------------------------------------------------


def test_gb_build_config_normalizes_scalar_repeat_factor() -> None:
    config = GBBuildConfig(material=_material(), gb_thickness=0.0, repeat_factor=3)
    assert config.repeat_factor == (3, 3)
    assert config.mismatch_tol is None


def test_gb_build_config_accepts_pair_repeat_factor_and_mismatch_tol() -> None:
    config = GBBuildConfig(
        material=_material(),
        gb_thickness=2.0,
        repeat_factor=(2, 4),
        mismatch_tol=0.01,
        strain_grain="left",
    )
    assert config.repeat_factor == (2, 4)
    assert config.mismatch_tol == 0.01
    assert config.strain_grain == "left"


def test_gb_build_config_rejects_non_material_state() -> None:
    with pytest.raises(GBMakerConstructionTypeError):
        GBBuildConfig(material="fcc", gb_thickness=0.0)  # type: ignore[arg-type]


def test_gb_build_config_rejects_negative_gb_thickness() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        GBBuildConfig(material=_material(), gb_thickness=-1.0)


def test_gb_build_config_rejects_invalid_strain_grain() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        GBBuildConfig(material=_material(), gb_thickness=0.0, strain_grain="both_sides")


def test_gb_build_config_rejects_invalid_repeat_factor() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        GBBuildConfig(material=_material(), gb_thickness=0.0, repeat_factor=(1, 2, 3))


# --------------------------------------------------------------------------------------
# ResolvedBoundaryInput
# --------------------------------------------------------------------------------------


def test_resolved_boundary_input_normalizes_valid_fields() -> None:
    resolved = ResolvedBoundaryInput(
        embedding=_embedding(),
        mode="exact",
        max_primitive_area_index=10_000,
        max_pq_determinant=10_000,
    )
    assert resolved.mode == "exact"


def test_resolved_boundary_input_rejects_non_embedding() -> None:
    with pytest.raises(GBMakerConstructionTypeError):
        ResolvedBoundaryInput(
            embedding=object(),  # type: ignore[arg-type]
            mode="exact",
            max_primitive_area_index=1,
            max_pq_determinant=1,
        )


def test_resolved_boundary_input_rejects_invalid_mode() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        ResolvedBoundaryInput(
            embedding=_embedding(),
            mode="fuzzy",  # type: ignore[arg-type]
            max_primitive_area_index=1,
            max_pq_determinant=1,
        )


# --------------------------------------------------------------------------------------
# OrientationState
# --------------------------------------------------------------------------------------


def test_orientation_state_defaults() -> None:
    state = OrientationState(embedding=_embedding())
    assert state.inplane_periodic == (True, True)
    assert state.normal_topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL


def test_orientation_state_rejects_bad_inplane_periodic_length() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        OrientationState(embedding=_embedding(), inplane_periodic=(True,))  # type: ignore[arg-type]


def test_orientation_state_rejects_non_topology() -> None:
    with pytest.raises(GBMakerConstructionTypeError):
        OrientationState(
            embedding=_embedding(), normal_topology="periodic_bicrystal"  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------------------
# AxisAccommodation
# --------------------------------------------------------------------------------------


def test_axis_accommodation_resized_scales_lengths_and_repeats() -> None:
    accommodation = AxisAccommodation(**_axis_accommodation_kwargs())
    resized = accommodation.resized(2)
    assert resized.left_repeats == 4
    assert resized.right_repeats == 6
    assert resized.box_length == 60.0
    assert resized.left_scale == accommodation.left_scale
    assert resized.mismatch == accommodation.mismatch


def test_axis_accommodation_resized_rejects_nonpositive_factor() -> None:
    accommodation = AxisAccommodation(**_axis_accommodation_kwargs())
    with pytest.raises(GBMakerConstructionValueError):
        accommodation.resized(0)


def test_axis_accommodation_rejects_negative_mismatch() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        AxisAccommodation(**_axis_accommodation_kwargs(mismatch=-0.1))


def test_axis_accommodation_rejects_nonpositive_repeats() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        AxisAccommodation(**_axis_accommodation_kwargs(left_repeats=0))


# --------------------------------------------------------------------------------------
# DimensionPlan
# --------------------------------------------------------------------------------------


def test_dimension_plan_normalizes_spacing_and_accommodation() -> None:
    accommodation = AxisAccommodation(**_axis_accommodation_kwargs())
    plan = DimensionPlan(
        box_dims=_box_dims(),
        vacuum_thickness=10.0,
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        periodic_spacing={"y": 3.5},
        accommodation={"y": accommodation},
    )
    assert plan.periodic_spacing == {"y": 3.5}
    assert plan.accommodation["y"] is accommodation
    with pytest.raises(TypeError):
        plan.periodic_spacing["z"] = 1.0  # type: ignore[index]


def test_dimension_plan_rejects_inverted_box_bounds() -> None:
    bad_box = np.array([[70.0, 0.0], [0.0, 30.0], [0.0, 30.0]])
    with pytest.raises(GBMakerConstructionValueError):
        DimensionPlan(
            box_dims=bad_box,
            vacuum_thickness=10.0,
            normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        )


def test_dimension_plan_rejects_bad_accommodation_key() -> None:
    accommodation = AxisAccommodation(**_axis_accommodation_kwargs())
    with pytest.raises(GBMakerConstructionValueError):
        DimensionPlan(
            box_dims=_box_dims(),
            vacuum_thickness=10.0,
            normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            accommodation={"x": accommodation},
        )


def test_dimension_plan_rejects_wrong_accommodation_value_type() -> None:
    with pytest.raises(GBMakerConstructionTypeError):
        DimensionPlan(
            box_dims=_box_dims(),
            vacuum_thickness=10.0,
            normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            accommodation={"y": "not-an-accommodation"},
        )


# --------------------------------------------------------------------------------------
# GrainBuildRequest
# --------------------------------------------------------------------------------------


def test_grain_build_request_normalizes_exact_orientation() -> None:
    request = GrainBuildRequest(
        material=_material(),
        orientation=np.eye(3, dtype=int),
        grain_side="left",
        x_length=25.0,
        box_dims=_box_dims(),
        exact=True,
    )
    assert request.orientation.dtype.kind == "i"
    assert not request.orientation.flags.writeable


def test_grain_build_request_rejects_bad_orientation_shape() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        GrainBuildRequest(
            material=_material(),
            orientation=np.eye(2),
            grain_side="left",
            x_length=25.0,
            box_dims=_box_dims(),
        )


def test_grain_build_request_rejects_bad_grain_side() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        GrainBuildRequest(
            material=_material(),
            orientation=np.eye(3),
            grain_side="middle",  # type: ignore[arg-type]
            x_length=25.0,
            box_dims=_box_dims(),
        )


# --------------------------------------------------------------------------------------
# GrainBuildResult
# --------------------------------------------------------------------------------------


def test_grain_build_result_accepts_parallel_arrays() -> None:
    atoms = _atoms(4)
    origin_ids = np.array([0, 0, 1, 1])
    result = GrainBuildResult(
        grain_side="right", atoms=atoms, origin_ids=origin_ids, basis_size=2
    )
    assert result.basis_size == 2
    assert result.atoms is atoms
    assert result.atoms.flags.writeable


def test_grain_build_result_rejects_mismatched_origin_ids() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        GrainBuildResult(
            grain_side="left",
            atoms=_atoms(4),
            origin_ids=np.array([0, 0, 1]),
            basis_size=2,
        )


def test_grain_build_result_rejects_nonpositive_basis_size() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        GrainBuildResult(
            grain_side="left",
            atoms=_atoms(4),
            origin_ids=np.array([0, 0, 1, 1]),
            basis_size=0,
        )


# --------------------------------------------------------------------------------------
# BicrystalResult
# --------------------------------------------------------------------------------------


def test_bicrystal_result_normalizes_valid_fields() -> None:
    result = BicrystalResult(
        atoms=_atoms(4),
        box_dims=_box_dims(),
        normal_topology=BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
        gb_id=3,
    )
    assert result.gb_id == 3
    assert not result.box_dims.flags.writeable


def test_bicrystal_result_rejects_nonpositive_gb_id() -> None:
    with pytest.raises(GBMakerConstructionValueError):
        BicrystalResult(
            atoms=_atoms(4),
            box_dims=_box_dims(),
            normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            gb_id=0,
        )
