# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

import math

import numpy as np
import pytest

from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.gbmaker.dimension import _find_commensurate_pair, _normalize_vacuum_topology
from GBOpt.gbmaker.types import GBMakerConstructionValueError

# --------------------------------------------------------------------------------------
# Commensurate-pair search
# --------------------------------------------------------------------------------------


def _brute_force_commensurate_pair(
    d1: float,
    d2: float,
    tol: float,
    max_n: int,
) -> tuple[int, int, float, float] | None:
    best = None
    best_key = None

    for n1 in range(1, max_n + 1):
        for n2 in range(1, max_n + 1):
            l1 = n1 * d1
            l2 = n2 * d2
            size = max(l1, l2)
            mismatch = abs(l1 - l2) / size
            if mismatch > tol:
                continue

            key = (size, mismatch, n1 + n2, n1, n2)
            if best_key is None or key < best_key:
                best = (n1, n2, l1, l2)
                best_key = key

    return best


@pytest.mark.parametrize(
    ("d1", "d2", "tol", "max_n", "expected"),
    [
        pytest.param(3.0, 3.0, 0.005, 10, (1, 1, 3.0, 3.0), id="identical"),
        pytest.param(3.0, 6.0, 0.0, 10, (2, 1, 6.0, 6.0), id="exact-multiple"),
        pytest.param(4.0, 2.0, 0.0, 10, (1, 2, 4.0, 4.0), id="smallest-box"),
        pytest.param(
            1.0,
            2.0,
            0.5,
            3,
            (2, 1, 2.0, 2.0),
            id="equal-size-prefers-lower-mismatch",
        ),
    ],
)
def test_find_commensurate_pair_returns_expected_best_candidate(
    d1,
    d2,
    tol,
    max_n,
    expected,
):
    result = _find_commensurate_pair(d1, d2, tol=tol, max_n=max_n)

    assert result is not None
    assert result[:2] == expected[:2]
    np.testing.assert_allclose(result[2:], expected[2:], atol=1e-12, rtol=0.0)


def test_find_commensurate_pair_accepts_pair_within_tolerance():
    result = _find_commensurate_pair(5.0, 7.48, tol=0.005, max_n=20)

    assert result is not None
    n1, n2, l1, l2 = result
    assert (n1, n2) == (3, 2)
    assert l1 == pytest.approx(15.0, abs=1e-12, rel=0.0)
    assert l2 == pytest.approx(14.96, abs=1e-12, rel=0.0)
    assert abs(l1 - l2) / max(l1, l2) <= 0.005


def test_find_commensurate_pair_returns_none_when_no_pair_is_within_tolerance():
    result = _find_commensurate_pair(1.0, math.pi, tol=0.00001, max_n=5)

    assert result is None


def test_find_commensurate_pair_accepts_mismatch_exactly_at_tolerance():
    d1 = 10.0
    d2 = 10.1
    tol = abs(d1 - d2) / max(d1, d2)

    result = _find_commensurate_pair(d1, d2, tol=tol, max_n=5)

    assert result is not None
    assert result[:2] == (1, 1)
    assert abs(result[2] - result[3]) / max(result[2], result[3]) == pytest.approx(
        tol,
        abs=1e-15,
        rel=0.0,
    )


@pytest.mark.parametrize(
    ("d1", "d2", "tol", "max_n"),
    [
        pytest.param(5.0, 7.48, 0.005, 20, id="near-3-2"),
        pytest.param(1.0, math.sqrt(2.0), 0.001, 50, id="sqrt2"),
        pytest.param(
            3.615 * math.sqrt(29.0),
            3.615,
            0.005,
            50,
            id="sigma29-row",
        ),
        pytest.param(
            10.0,
            10.1,
            abs(10.0 - 10.1) / 10.1,
            5,
            id="tolerance-boundary",
        ),
        pytest.param(1.0, 2.0, 0.5, 3, id="size-mismatch-tie-break"),
    ],
)
def test_find_commensurate_pair_matches_full_brute_force_ordering(
    d1,
    d2,
    tol,
    max_n,
):
    result = _find_commensurate_pair(d1, d2, tol=tol, max_n=max_n)
    brute_force = _brute_force_commensurate_pair(d1, d2, tol, max_n)

    assert (result is None) == (brute_force is None)
    if result is None or brute_force is None:
        return

    assert result[:2] == brute_force[:2]
    np.testing.assert_allclose(
        result[2:],
        brute_force[2:],
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("argument", "value", "match"),
    [
        pytest.param("d1", 0.0, r"d1 must be a finite positive period", id="d1-zero"),
        pytest.param("d1", -1.0, r"d1 must be a finite positive period",
                     id="d1-negative"),
        pytest.param("d1", np.nan, r"d1 must be a finite positive period", id="d1-nan"),
        pytest.param("d1", np.inf, r"d1 must be a finite positive period", id="d1-inf"),
        pytest.param("d1", True, r"d1 must be a finite positive period", id="d1-bool"),
        pytest.param(
            "d1", "bad", r"d1 and d2 must be finite positive periods", id="d1-string"),
        pytest.param("d2", 0.0, r"d2 must be a finite positive period", id="d2-zero"),
        pytest.param("d2", -1.0, r"d2 must be a finite positive period",
                     id="d2-negative"),
        pytest.param("d2", np.nan, r"d2 must be a finite positive period", id="d2-nan"),
        pytest.param("d2", np.inf, r"d2 must be a finite positive period", id="d2-inf"),
        pytest.param("d2", np.bool_(True),
                     r"d2 must be a finite positive period", id="d2-bool"),
        pytest.param("d2", object(),
                     r"d1 and d2 must be finite positive periods", id="d2-object"),
    ],
)
def test_find_commensurate_pair_rejects_invalid_periods(argument, value, match):
    kwargs = {"d1": 1.0, "d2": 1.0, argument: value}

    with pytest.raises(GBMakerConstructionValueError, match=match):
        _find_commensurate_pair(**kwargs)


@pytest.mark.parametrize(
    "tol",
    [
        pytest.param(-0.001, id="negative"),
        pytest.param(np.nan, id="nan"),
        pytest.param(np.inf, id="inf"),
        pytest.param(True, id="bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param("bad", id="string"),
    ],
)
def test_find_commensurate_pair_rejects_invalid_tolerance(tol):
    with pytest.raises(
        GBMakerConstructionValueError,
        match=r"tol must be finite and non-negative",
    ):
        _find_commensurate_pair(1.0, 1.0, tol=tol)


@pytest.mark.parametrize(
    "max_n",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="bool"),
        pytest.param(np.bool_(True), id="numpy-bool"),
        pytest.param("10", id="string"),
    ],
)
def test_find_commensurate_pair_rejects_invalid_max_n(max_n):
    with pytest.raises(
        GBMakerConstructionValueError,
        match=r"max_n must be a positive integer",
    ):
        _find_commensurate_pair(1.0, 1.0, max_n=max_n)


# --------------------------------------------------------------------------------------
# Vacuum/topology normalization
# --------------------------------------------------------------------------------------


def test_normalize_vacuum_topology_zero_vacuum_is_periodic_bicrystal():
    vacuum, topology = _normalize_vacuum_topology(0.0, tolerance=1e-10)
    assert vacuum == 0.0
    assert topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL


def test_normalize_vacuum_topology_within_tolerance_snaps_to_zero():
    vacuum, topology = _normalize_vacuum_topology(5e-11, tolerance=1e-10)
    assert vacuum == 0.0
    assert topology is BoundaryNormalTopology.PERIODIC_BICRYSTAL


def test_normalize_vacuum_topology_nonzero_vacuum_is_single_interface_slab():
    vacuum, topology = _normalize_vacuum_topology(10.0, tolerance=1e-10)
    assert vacuum == 10.0
    assert topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB


def test_normalize_vacuum_topology_just_outside_tolerance_is_single_interface_slab():
    vacuum, topology = _normalize_vacuum_topology(2e-10, tolerance=1e-10)
    assert vacuum == 2e-10
    assert topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB
