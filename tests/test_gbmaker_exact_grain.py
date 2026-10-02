# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Direct unit tests for ``GBOpt.gbmaker.exact_grain.build_exact_grain``.

Calls ``build_exact_grain`` directly against hand-built ``GrainBuildRequest``
fixtures, rather than relying only on indirect coverage through GBMaker's real
construction in ``tests/test_gbmaker.py`` and ``tests/test_gbmaker_exact_path.py``.
"""

from typing import Any

import numpy as np
import pytest

from GBOpt.gbmaker.exact_grain import build_exact_grain
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
        "exact": True,
    }
    kwargs.update(overrides)
    return GrainBuildRequest(**kwargs)


# --------------------------------------------------------------------------------------
# Successful construction
# --------------------------------------------------------------------------------------


def test_build_exact_grain_returns_complete_fcc_supercell():
    material = _fcc_material()
    result = build_exact_grain(_request(material))

    assert result.grain_side == "left"
    assert result.basis_size == 4
    assert len(result.atoms) == 16
    assert np.all(result.atoms["name"] == "Cu")


def test_build_exact_grain_origin_ids_group_contiguous_basis_blocks():
    material = _fcc_material()
    result = build_exact_grain(_request(material))

    expected = np.repeat(np.arange(len(result.atoms) // result.basis_size), result.basis_size)
    np.testing.assert_array_equal(result.origin_ids, expected)


def test_build_exact_grain_preserves_fluorite_stoichiometry():
    material = resolve_material_state(5.47, "fluorite", ("U", "O"))
    a0 = material.a0
    request = _request(
        material,
        x_offset=0.0,
        x_length=a0,
        inplane_box_lengths=(a0, a0),
    )

    result = build_exact_grain(request)

    uranium_count = int(np.count_nonzero(result.atoms["name"] == "U"))
    oxygen_count = int(np.count_nonzero(result.atoms["name"] == "O"))
    assert result.basis_size == 12
    assert uranium_count == 4
    assert oxygen_count == 8
    assert uranium_count + oxygen_count == len(result.atoms)


def test_build_exact_grain_atoms_lie_within_half_open_box():
    material = _fcc_material()
    request = _request(material)
    result = build_exact_grain(request)

    assert np.all(result.atoms["x"] >= request.x_offset)
    assert np.all(result.atoms["x"] < request.x_offset + request.x_length)
    y_dim, z_dim = request.inplane_box_lengths
    assert np.all(result.atoms["y"] >= 0.0)
    assert np.all(result.atoms["y"] < y_dim)
    assert np.all(result.atoms["z"] >= 0.0)
    assert np.all(result.atoms["z"] < z_dim)


def test_build_exact_grain_explicit_repeats_override_commensurate_search():
    material = _fcc_material()
    a0 = material.a0
    request = _request(
        material,
        y_repeats=3,
        z_repeats=1,
        inplane_box_lengths=(2 * a0, 2 * a0),
    )

    result = build_exact_grain(request)

    # repeat_x=1 (from x_length == a0), explicit y_repeats=3, z_repeats=1, basis_size=4.
    assert len(result.atoms) == 1 * 3 * 1 * 4


def test_build_exact_grain_is_deterministic():
    material = _fcc_material()
    request = _request(material)

    first = build_exact_grain(request)
    second = build_exact_grain(request)

    np.testing.assert_array_equal(first.atoms, second.atoms)
    np.testing.assert_array_equal(first.origin_ids, second.origin_ids)


# --------------------------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------------------------


def test_build_exact_grain_requires_rational_basis():
    bare_material = MaterialState(a0=A0_FCC, structure="fcc", atom_types="Cu")
    request = _request(bare_material)

    with pytest.raises(GBMakerConstructionValueError, match="rational_basis"):
        build_exact_grain(request)


def test_build_exact_grain_rejects_incommensurate_x_length():
    material = _fcc_material()
    request = _request(material, x_length=material.a0 * 1.3)

    with pytest.raises(GBMakerConstructionValueError, match="integer multiple"):
        build_exact_grain(request)


def test_build_exact_grain_rejects_incommensurate_inplane_dimension():
    material = _fcc_material()
    request = _request(material, inplane_box_lengths=(material.a0 * 1.5, material.a0 * 2))

    with pytest.raises(GBMakerConstructionValueError, match="integer multiple"):
        build_exact_grain(request)
