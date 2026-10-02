# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Direct unit tests for ``GBOpt.gbmaker.approximate_grain``.

Calls ``build_approximate_grain``, ``filter_grain_result_complete_origins``, and
``trim_grain_result_to_upper_x`` directly against hand-built ``GrainBuildRequest``
fixtures, rather than relying only on indirect coverage through GBMaker's real
construction in ``tests/test_gbmaker.py`` and ``tests/test_gbmaker_exact_path.py``.
"""

from typing import Any

import numpy as np
import pytest

from GBOpt.gbmaker.approximate_grain import (
    build_approximate_grain,
    filter_grain_result_complete_origins,
    trim_grain_result_to_upper_x,
)
from GBOpt.gbmaker.material import resolve_material_state
from GBOpt.gbmaker.types import (
    GBMakerConstructionValueError,
    GrainBuildRequest,
    MaterialState,
)

A0_FCC = 3.615


def _fcc_material() -> MaterialState:
    return resolve_material_state(A0_FCC, "fcc", "Cu")


def _request(material: MaterialState, **overrides: Any) -> GrainBuildRequest:
    a0 = material.a0
    kwargs: dict[str, Any] = {
        "material": material,
        "rotation": np.eye(3),
        "periodic_matrix": np.eye(3, dtype=int),
        "grain_side": "left",
        "x_offset": 0.0,
        "x_length": a0,
        "inplane_periodic": (True, True),
        "inplane_box_lengths": (2 * a0, 2 * a0),
        "epsilon": 1e-10,
        "exact": False,
    }
    kwargs.update(overrides)
    return GrainBuildRequest(**kwargs)


# --------------------------------------------------------------------------------------
# build_approximate_grain
# --------------------------------------------------------------------------------------


def test_build_approximate_grain_returns_complete_fcc_origins():
    material = _fcc_material()
    result = build_approximate_grain(_request(material))

    assert result.grain_side == "left"
    assert result.basis_size == 4
    assert len(result.atoms) == 16
    assert np.all(result.atoms["name"] == "Cu")


def test_build_approximate_grain_origin_ids_group_complete_basis_blocks():
    material = _fcc_material()
    result = build_approximate_grain(_request(material))

    for origin_id in np.unique(result.origin_ids):
        assert np.count_nonzero(result.origin_ids == origin_id) == result.basis_size


def test_build_approximate_grain_preserves_fluorite_stoichiometry():
    material = resolve_material_state(5.47, "fluorite", ("U", "O"))
    a0 = material.a0
    request = _request(material, x_length=a0, inplane_box_lengths=(a0, a0))

    result = build_approximate_grain(request)

    uranium_count = int(np.count_nonzero(result.atoms["name"] == "U"))
    oxygen_count = int(np.count_nonzero(result.atoms["name"] == "O"))
    assert result.basis_size == 12
    assert uranium_count == 4
    assert oxygen_count == 8
    assert uranium_count + oxygen_count == len(result.atoms)


def test_build_approximate_grain_atoms_lie_within_half_open_box():
    material = _fcc_material()
    request = _request(material)
    result = build_approximate_grain(request)

    assert np.all(result.atoms["x"] >= request.x_offset)
    assert np.all(result.atoms["x"] < request.x_offset + request.x_length)
    y_dim, z_dim = request.inplane_box_lengths
    assert np.all(result.atoms["y"] >= 0.0)
    assert np.all(result.atoms["y"] < y_dim)
    assert np.all(result.atoms["z"] >= 0.0)
    assert np.all(result.atoms["z"] < z_dim)


def test_build_approximate_grain_non_periodic_axes_do_not_select_complete_origins():
    material = _fcc_material()
    a0 = material.a0
    request = _request(
        material,
        inplane_periodic=(False, False),
        inplane_box_lengths=(a0, a0),
    )

    result = build_approximate_grain(request)

    assert len(result.atoms) > 0
    assert np.all(result.atoms["y"] >= 0.0)
    assert np.all(result.atoms["y"] < a0)
    assert np.all(result.atoms["z"] >= 0.0)
    assert np.all(result.atoms["z"] < a0)


def test_build_approximate_grain_requires_resolved_unit_cell():
    bare_material = MaterialState(a0=A0_FCC, structure="fcc", atom_types="Cu")
    request = _request(bare_material)

    with pytest.raises(GBMakerConstructionValueError, match="unit_cell"):
        build_approximate_grain(request)


# --------------------------------------------------------------------------------------
# filter_grain_result_complete_origins
# --------------------------------------------------------------------------------------


def test_filter_grain_result_complete_origins_keeps_only_fully_passing_origins():
    material = _fcc_material()
    result = build_approximate_grain(_request(material))

    first_origin_id = result.origin_ids[0]
    mask = result.origin_ids != first_origin_id

    filtered = filter_grain_result_complete_origins(result, mask)

    assert first_origin_id not in filtered.origin_ids
    assert len(filtered.atoms) == len(result.atoms) - result.basis_size
    assert filtered.basis_size == result.basis_size


def test_filter_grain_result_complete_origins_passthrough_mask_keeps_all_atoms():
    material = _fcc_material()
    result = build_approximate_grain(_request(material))

    filtered = filter_grain_result_complete_origins(
        result, np.ones(len(result.atoms), dtype=bool)
    )

    np.testing.assert_array_equal(filtered.atoms, result.atoms)
    np.testing.assert_array_equal(filtered.origin_ids, result.origin_ids)


# --------------------------------------------------------------------------------------
# trim_grain_result_to_upper_x
# --------------------------------------------------------------------------------------


def test_trim_grain_result_to_upper_x_removes_only_origins_beyond_bound():
    material = _fcc_material()
    a0 = material.a0
    request = _request(material, x_length=2 * a0)
    result = build_approximate_grain(request)

    trimmed = trim_grain_result_to_upper_x(result, upper_x=a0, epsilon=1e-10)

    assert len(trimmed.atoms) < len(result.atoms)
    assert np.all(trimmed.atoms["x"] < a0)


def test_trim_grain_result_to_upper_x_rejects_non_finite_bound():
    material = _fcc_material()
    result = build_approximate_grain(_request(material))

    with pytest.raises(GBMakerConstructionValueError, match="finite"):
        trim_grain_result_to_upper_x(result, upper_x=float("nan"), epsilon=1e-10)
