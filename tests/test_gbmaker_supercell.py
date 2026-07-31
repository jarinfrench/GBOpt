# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for exact integer supercell construction and enumeration."""

from collections import Counter
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from GBOpt.crystallography.integer import integer_adj3, integer_det3
from GBOpt.gbmaker_supercell import (
    SupercellSites,
    _integer_membership,
    build_supercell_matrix,
    enumerate_supercell_origins,
    enumerate_supercell_sites,
    supercell_axis_numerators,
)
from GBOpt.UnitCell import RationalBasis, UnitCell

# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------

IDENTITY_ROWS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
SIGMA5_RIGHT_GRAIN_ROWS = ((4, -3, 0), (3, 4, 0), (0, 0, 1))
OBLIQUE_INDEX2_ROWS = ((1, -1, 0), (1, 1, 0), (0, 0, 1))

INVALID_3X3_INTEGER_MATRICES = [
    pytest.param(
        [[1.5, 0, 0], [0, 1, 0], [0, 0, 1]],
        "integer-valued",
        id="non-integer-entry",
    ),
    pytest.param(
        [[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]],
        "finite",
        id="non-finite-entry",
    ),
    pytest.param(
        [[True, 0, 0], [0, 1, 0], [0, 0, 1]],
        "not an integer",
        id="boolean-entry",
    ),
    pytest.param(
        [[1, 0], [0, 1]],
        "shape",
        id="wrong-shape",
    ),
]


def _int_matrix(rows) -> np.ndarray:
    """Return shared matrix rows as a fresh object-dtype array."""
    return np.array(rows, dtype=object)


def _origin_set(origins: np.ndarray) -> set[tuple[int, int, int]]:
    """Return integer origins as an order-independent set of tuples."""
    return {tuple(int(value) for value in row) for row in origins}


# ---------------------------------------------------------------------------
# _integer_membership
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        pytest.param((0, 0, 0), True, id="lower-corner"),
        pytest.param((2, 1, 0), True, id="last-interior-origin"),
        pytest.param((-1, 0, 0), False, id="below-x"),
        pytest.param((0, -1, 0), False, id="below-y"),
        pytest.param((0, 0, -1), False, id="below-z"),
        pytest.param((3, 0, 0), False, id="exclusive-x-upper-bound"),
        pytest.param((0, 2, 0), False, id="exclusive-y-upper-bound"),
        pytest.param((0, 0, 1), False, id="exclusive-z-upper-bound"),
    ],
)
def test_integer_membership_uses_half_open_repeated_identity_cell(origin, expected):
    identity = _int_matrix(IDENTITY_ROWS)

    assert (
        _integer_membership(
            origin,
            integer_adj3(identity),
            integer_det3(identity),
            3,
            2,
            1,
        )
        is expected
    )


def test_integer_membership_normalizes_negative_determinant_numerators():
    supercell = _int_matrix(((0, 1, 0), (2, 0, 0), (0, 0, 1)))
    determinant = integer_det3(supercell)

    assert determinant == -2
    assert _integer_membership(
        (1, 0, 0),
        integer_adj3(supercell),
        determinant,
        1,
        1,
        1,
    )


# ---------------------------------------------------------------------------
# build_supercell_matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "orientation_rows",
    [
        pytest.param(IDENTITY_ROWS, id="identity"),
        pytest.param(SIGMA5_RIGHT_GRAIN_ROWS, id="sigma5-right-grain"),
        pytest.param(OBLIQUE_INDEX2_ROWS, id="oblique-index-2"),
    ],
)
def test_build_supercell_matrix_returns_canonical_orientation_rows(orientation_rows):
    orientation = np.array(orientation_rows, dtype=float)

    result = build_supercell_matrix(orientation)

    np.testing.assert_array_equal(result, _int_matrix(orientation_rows))


@pytest.mark.parametrize(
    ("bad_matrix", "match"),
    INVALID_3X3_INTEGER_MATRICES,
)
def test_build_supercell_matrix_rejects_invalid_matrix(bad_matrix, match):
    with pytest.raises(ValueError, match=match):
        build_supercell_matrix(bad_matrix)


def test_build_supercell_matrix_rejects_singular_array_like():
    singular = [[1, 0, 0], [1, 0, 0], [0, 1, 0]]

    with pytest.raises(ValueError, match="singular"):
        build_supercell_matrix(singular)  # type: ignore[ty:invalid-argument-type]


@pytest.mark.parametrize(
    "noncanonical_rows",
    [
        pytest.param(
            ((2, 0, 0), (0, 1, 0), (0, 0, 1)),
            id="nonprimitive-normal",
        ),
        pytest.param(
            ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
            id="left-handed",
        ),
        pytest.param(
            ((1, 1, 0), (0, 1, 0), (0, 0, 1)),
            id="normal-does-not-match-inplane-cross-product",
        ),
    ],
)
def test_build_supercell_matrix_rejects_noncanonical_orientation_rows(
    noncanonical_rows,
):
    with pytest.raises(ValueError, match="canonical and right-handed"):
        build_supercell_matrix(noncanonical_rows)


# ---------------------------------------------------------------------------
# enumerate_supercell_origins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("supercell_rows", "repeats", "expected"),
    [
        pytest.param(
            IDENTITY_ROWS,
            (2, 2, 2),
            set(product(range(2), repeat=3)),
            id="identity-2x2x2",
        ),
        pytest.param(
            OBLIQUE_INDEX2_ROWS,
            (1, 1, 1),
            {(0, 0, 0), (1, 0, 0)},
            id="oblique-index-2",
        ),
    ],
)
def test_enumerate_supercell_origins_returns_known_origin_set(
    supercell_rows,
    repeats,
    expected,
):
    origins = enumerate_supercell_origins(
        _int_matrix(supercell_rows),
        *repeats,
    )

    assert origins.shape == (len(expected), 3)
    assert _origin_set(origins) == expected


@pytest.mark.parametrize(
    "repeats",
    [
        pytest.param((1, 1, 1), id="unit-repeat"),
        pytest.param((2, 1, 1), id="repeated-normal"),
        pytest.param((1, 2, 3), id="repeated-inplane"),
    ],
)
def test_enumerate_supercell_origins_satisfies_count_uniqueness_and_membership(
    repeats,
):
    supercell = _int_matrix(SIGMA5_RIGHT_GRAIN_ROWS)
    origins = enumerate_supercell_origins(supercell, *repeats)
    determinant = integer_det3(supercell)
    adjugate = np.asarray(integer_adj3(supercell), dtype=object)
    numerators = np.asarray(origins, dtype=object) @ adjugate
    upper_bounds = np.asarray(repeats, dtype=object) * determinant

    assert origins.shape == (int(np.prod(repeats)) * determinant, 3)
    assert len(np.unique(origins, axis=0)) == len(origins)
    assert np.all(numerators >= 0)
    assert np.all(numerators < upper_bounds)


@pytest.mark.parametrize(
    ("bad_supercell", "match"),
    INVALID_3X3_INTEGER_MATRICES,
)
def test_enumerate_supercell_origins_rejects_invalid_supercell(
    bad_supercell,
    match,
):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_origins(bad_supercell, 1, 1, 1)


@pytest.mark.parametrize(
    ("supercell", "match"),
    [
        pytest.param(
            [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
            "non-singular",
            id="singular",
        ),
        pytest.param(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
            "positive determinant",
            id="negative-determinant",
        ),
    ],
)
def test_enumerate_supercell_origins_rejects_invalid_determinant(
    supercell,
    match,
):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_origins(supercell, 1, 1, 1)


@pytest.mark.parametrize(
    ("repeats", "match"),
    [
        pytest.param((0, 1, 1), "repeat_x", id="zero-x"),
        pytest.param((1, -1, 1), "repeat_y", id="negative-y"),
        pytest.param((1, 1, 1.5), "repeat_z", id="float-z"),
        pytest.param((True, 1, 1), "repeat_x", id="boolean-x"),
        pytest.param((1, np.bool_(True), 1), "repeat_y", id="numpy-boolean-y"),
    ],
)
def test_enumerate_supercell_origins_rejects_invalid_repeat_count(repeats, match):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_origins(_int_matrix(IDENTITY_ROWS), *repeats)


def test_enumerate_supercell_origins_accepts_numpy_integer_repeat_counts():
    origins = enumerate_supercell_origins(
        _int_matrix(IDENTITY_ROWS),
        np.int64(2),  # type: ignore[ty:invalid-argument-type]
        np.int64(1),  # type: ignore[ty:invalid-argument-type]
        np.int64(1),  # type: ignore[ty:invalid-argument-type]
    )

    assert _origin_set(origins) == {(0, 0, 0), (1, 0, 0)}


def test_enumerate_supercell_origins_preserves_known_oblique_order():
    origins = enumerate_supercell_origins(
        _int_matrix(OBLIQUE_INDEX2_ROWS),
        1,
        1,
        1,
    )

    np.testing.assert_array_equal(
        origins,
        np.array(((0, 0, 0), (1, 0, 0)), dtype=int),
    )


# ---------------------------------------------------------------------------
# enumerate_supercell_sites
# ---------------------------------------------------------------------------


NEGATIVE_DETERMINANT_ROWS = ((0, 1, 0), (2, 0, 0), (0, 0, 1))
SHEARED_INDEX2_ROWS = ((1, -1, 0), (1, 1, 0), (0, 0, 1))
ZHANG_001_LEFT_ROWS = ((0, 18, -1), (0, 1, 18), (1, 0, 0))


def _one_site_basis() -> RationalBasis:
    return RationalBasis(
        names=("A",),
        numerators=np.array(((0, 0, 0),), dtype=object),
        denominator=1,
    )


def _fluorite_basis() -> RationalBasis:
    cell = UnitCell()
    cell.init_by_structure("fluorite", 5.454, ("U", "O"))
    basis = cell.rational_basis
    assert basis is not None
    return basis


def _decorated_site_keys(sites: SupercellSites) -> set[tuple[tuple[int, int, int], int]]:
    return {
        (tuple(int(value) for value in row), int(basis_index))
        for row, basis_index in zip(
            sites.coordinate_numerators,
            sites.basis_indices,
        )
    }


def _assert_site_invariants(
    sites: SupercellSites,
    supercell_rows,
    repeats,
    *,
    basis_size: int,
) -> None:
    determinant = abs(integer_det3(_int_matrix(supercell_rows)))
    expected_per_basis = determinant * int(np.prod(repeats))
    expected_count = basis_size * expected_per_basis
    upper_bounds = np.asarray(repeats, dtype=object) * sites.coordinate_denominator

    assert sites.site_count == expected_count
    assert sites.coordinate_numerators.shape == (expected_count, 3)
    assert sites.basis_indices.shape == (expected_count,)
    assert sites.repeats == tuple(repeats)
    assert sites.supercell_index == determinant
    assert np.all(sites.coordinate_numerators >= 0)
    assert np.all(sites.coordinate_numerators < upper_bounds)
    assert len(_decorated_site_keys(sites)) == expected_count
    assert Counter(int(index) for index in sites.basis_indices) == {
        index: expected_per_basis for index in range(basis_size)
    }


def test_enumerate_supercell_sites_identity_one_site_basis():
    sites = enumerate_supercell_sites(
        _int_matrix(IDENTITY_ROWS),
        1,
        1,
        1,
        rational_basis=_one_site_basis(),
    )

    np.testing.assert_array_equal(
        sites.coordinate_numerators,
        np.array(((0, 0, 0),), dtype=object),
    )
    np.testing.assert_array_equal(sites.basis_indices, np.array((0,)))
    assert sites.coordinate_denominator == 1
    _assert_site_invariants(sites, IDENTITY_ROWS, (1, 1, 1), basis_size=1)


def test_enumerate_supercell_sites_fluorite_identity_population_and_species():
    basis = _fluorite_basis()

    sites = enumerate_supercell_sites(
        _int_matrix(IDENTITY_ROWS),
        1,
        1,
        1,
        rational_basis=basis,
    )

    assert sites.site_count == 12
    np.testing.assert_array_equal(sites.basis_indices, np.arange(12))
    recovered_species = [basis.names[index] for index in sites.basis_indices]
    assert Counter(recovered_species) == {"U": 4, "O": 8}
    _assert_site_invariants(sites, IDENTITY_ROWS, (1, 1, 1), basis_size=12)


def test_enumerate_supercell_sites_preserves_zhang_001_fluorite_count():
    basis = _fluorite_basis()

    sites = enumerate_supercell_sites(
        _int_matrix(ZHANG_001_LEFT_ROWS),
        1,
        1,
        5,
        rational_basis=basis,
    )

    assert sites.site_count == 19_500
    _assert_site_invariants(
        sites,
        ZHANG_001_LEFT_ROWS,
        (1, 1, 5),
        basis_size=12,
    )


@pytest.mark.parametrize(
    ("supercell_rows", "repeats"),
    [
        pytest.param(IDENTITY_ROWS, (2, 3, 4), id="identity-repeats-all-axes"),
        pytest.param(SHEARED_INDEX2_ROWS, (2, 1, 3), id="sheared-index-two"),
        pytest.param(SIGMA5_RIGHT_GRAIN_ROWS, (1, 2, 1), id="sigma-five"),
        pytest.param(NEGATIVE_DETERMINANT_ROWS, (1, 2, 2), id="negative-determinant"),
    ],
)
def test_enumerate_supercell_sites_general_count_population_and_uniqueness(
    supercell_rows,
    repeats,
):
    basis = RationalBasis(
        names=("A", "B", "C"),
        numerators=np.array(((0, 0, 0), (1, 2, 3), (3, 1, 2)), dtype=object),
        denominator=4,
    )

    sites = enumerate_supercell_sites(
        _int_matrix(supercell_rows),
        *repeats,
        rational_basis=basis,
    )

    _assert_site_invariants(sites, supercell_rows, repeats, basis_size=3)


def test_enumerate_supercell_sites_order_is_origin_then_basis_row():
    basis = RationalBasis(
        names=("A", "B"),
        numerators=np.array(((0, 0, 0), (1, 1, 1)), dtype=object),
        denominator=2,
    )

    sites = enumerate_supercell_sites(
        _int_matrix(IDENTITY_ROWS),
        2,
        1,
        1,
        rational_basis=basis,
    )

    np.testing.assert_array_equal(sites.basis_indices, np.array((0, 1, 0, 1)))
    np.testing.assert_array_equal(
        sites.coordinate_numerators,
        np.array(
            ((0, 0, 0), (1, 1, 1), (2, 0, 0), (3, 1, 1)),
            dtype=object,
        ),
    )


def test_enumerate_supercell_sites_is_deterministic():
    basis = _fluorite_basis()
    arguments = (_int_matrix(SHEARED_INDEX2_ROWS), 2, 2, 1)

    first = enumerate_supercell_sites(*arguments, rational_basis=basis)
    second = enumerate_supercell_sites(*arguments, rational_basis=basis)

    np.testing.assert_array_equal(
        first.coordinate_numerators,
        second.coordinate_numerators,
    )
    np.testing.assert_array_equal(first.basis_indices, second.basis_indices)
    np.testing.assert_array_equal(first.supercell_matrix, second.supercell_matrix)
    assert first.coordinate_denominator == second.coordinate_denominator
    assert first.repeats == second.repeats


def test_enumerate_supercell_sites_returns_defensively_immutable_arrays():
    sites = enumerate_supercell_sites(
        _int_matrix(IDENTITY_ROWS),
        1,
        1,
        1,
        rational_basis=_one_site_basis(),
    )

    coordinates = sites.coordinate_numerators
    basis_indices = sites.basis_indices
    supercell = sites.supercell_matrix
    assert not coordinates.flags.writeable
    assert not basis_indices.flags.writeable
    assert not supercell.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        coordinates[0, 0] = 99
    with pytest.raises(ValueError, match="read-only"):
        basis_indices[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        supercell[0, 0] = 99

    coordinates.setflags(write=True)
    basis_indices.setflags(write=True)
    supercell.setflags(write=True)
    coordinates[0, 0] = 99
    basis_indices[0] = 99
    supercell[0, 0] = 99

    np.testing.assert_array_equal(
        sites.coordinate_numerators,
        np.array(((0, 0, 0),), dtype=object),
    )
    np.testing.assert_array_equal(sites.basis_indices, np.array((0,)))
    np.testing.assert_array_equal(sites.supercell_matrix, _int_matrix(IDENTITY_ROWS))


def test_enumerate_supercell_sites_wraps_crossing_decorated_sites_without_loss():
    basis = RationalBasis(
        names=("A", "B"),
        numerators=np.array(((0, 0, 0), (0, 1, 0)), dtype=object),
        denominator=4,
    )
    supercell = _int_matrix(SHEARED_INDEX2_ROWS)
    origins = enumerate_supercell_origins(supercell, 1, 1, 1)
    adjugate = np.asarray(integer_adj3(supercell), dtype=object)
    determinant = integer_det3(supercell)
    crossing_unwrapped = []
    for origin in origins:
        decorated_origin = (
            4 * np.asarray(origin, dtype=object)
            + np.array((0, 1, 0), dtype=object)
        )
        unwrapped = decorated_origin @ adjugate
        crossing_unwrapped.append(tuple(int(value) for value in unwrapped))
    assert any(
        value < 0 or value >= 4 * determinant
        for row in crossing_unwrapped
        for value in row
    )

    sites = enumerate_supercell_sites(
        supercell,
        1,
        1,
        1,
        rational_basis=basis,
    )

    _assert_site_invariants(sites, SHEARED_INDEX2_ROWS, (1, 1, 1), basis_size=2)
    assert Counter(int(index) for index in sites.basis_indices) == {0: 2, 1: 2}


def test_enumerate_supercell_sites_allows_same_coordinate_for_different_species_rows():
    basis = RationalBasis(
        names=("A", "B"),
        numerators=np.array(((0, 0, 0), (0, 0, 0)), dtype=object),
        denominator=1,
    )

    sites = enumerate_supercell_sites(
        _int_matrix(IDENTITY_ROWS),
        1,
        1,
        1,
        rational_basis=basis,
    )

    assert sites.site_count == 2
    assert len(_decorated_site_keys(sites)) == 2


@pytest.mark.parametrize(
    ("bad_supercell", "match"),
    INVALID_3X3_INTEGER_MATRICES,
)
def test_enumerate_supercell_sites_rejects_malformed_supercell(bad_supercell, match):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_sites(
            bad_supercell,
            1,
            1,
            1,
            rational_basis=_one_site_basis(),
        )


def test_enumerate_supercell_sites_rejects_singular_supercell():
    with pytest.raises(ValueError, match="non-singular"):
        enumerate_supercell_sites(
            [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
            1,
            1,
            1,
            rational_basis=_one_site_basis(),
        )


@pytest.mark.parametrize(
    ("repeats", "match"),
    [
        pytest.param((0, 1, 1), "repeat_x", id="zero-x"),
        pytest.param((1, -1, 1), "repeat_y", id="negative-y"),
        pytest.param((1, 1, 1.5), "repeat_z", id="float-z"),
        pytest.param((True, 1, 1), "repeat_x", id="boolean-x"),
        pytest.param((1, np.bool_(True), 1), "repeat_y", id="numpy-boolean-y"),
    ],
)
def test_enumerate_supercell_sites_rejects_invalid_repeats(repeats, match):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_sites(
            _int_matrix(IDENTITY_ROWS),
            *repeats,
            rational_basis=_one_site_basis(),
        )


def test_enumerate_supercell_sites_rejects_missing_rational_metadata():
    cell = UnitCell()
    assert cell.rational_basis is None

    with pytest.raises(ValueError, match="requires UnitCell.rational_basis"):
        enumerate_supercell_sites(
            _int_matrix(IDENTITY_ROWS),
            1,
            1,
            1,
            rational_basis=cell.rational_basis,
        )


@pytest.mark.parametrize(
    ("rational_basis", "match"),
    [
        pytest.param(
            SimpleNamespace(names=("A",), numerators=[[0, 0]], denominator=1),
            "shape",
            id="wrong-coordinate-shape",
        ),
        pytest.param(
            SimpleNamespace(names=("A",), numerators=[[0.5, 0, 0]], denominator=1),
            "only integers",
            id="noninteger-coordinate",
        ),
        pytest.param(
            SimpleNamespace(names=("A",), numerators=[[0, 0, 0]], denominator=0),
            "denominator",
            id="nonpositive-denominator",
        ),
        pytest.param(
            SimpleNamespace(names=("A", "B"), numerators=[[0, 0, 0]], denominator=1),
            "equal lengths",
            id="species-coordinate-mismatch",
        ),
        pytest.param(
            SimpleNamespace(
                names=("A", "A"),
                numerators=[[0, 0, 0], [0, 0, 0]],
                denominator=1,
            ),
            "duplicate decorated",
            id="duplicate-decorated-row",
        ),
        pytest.param(
            SimpleNamespace(names=("A",), numerators=[[-1, 0, 0]], denominator=2),
            "canonical half-open",
            id="negative-coordinate",
        ),
        pytest.param(
            SimpleNamespace(names=("A",), numerators=[[2, 0, 0]], denominator=2),
            "canonical half-open",
            id="coordinate-at-denominator",
        ),
    ],
)
def test_enumerate_supercell_sites_rejects_bypassed_rational_basis_contract(
    rational_basis,
    match,
):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_sites(
            _int_matrix(IDENTITY_ROWS),
            1,
            1,
            1,
            rational_basis=rational_basis,
        )


def test_supercell_sites_rejects_internal_count_mismatch():
    with pytest.raises(ValueError, match="expected 2 sites"):
        SupercellSites(
            coordinate_numerators=np.array(((0, 0, 0),), dtype=object),
            basis_denominator=1,
            basis_indices=np.array((0,)),
            supercell_matrix=_int_matrix(OBLIQUE_INDEX2_ROWS),
            repeats=(1, 1, 1),
            basis_size=1,
        )


def test_supercell_sites_rejects_internal_duplicate_representatives():
    with pytest.raises(ValueError, match="duplicate wrapped exact representatives"):
        SupercellSites(
            coordinate_numerators=np.array(((0, 0, 0), (0, 0, 0)), dtype=object),
            basis_denominator=1,
            basis_indices=np.array((0, 0)),
            supercell_matrix=_int_matrix(OBLIQUE_INDEX2_ROWS),
            repeats=(1, 1, 1),
            basis_size=1,
        )


# ---------------------------------------------------------------------------
# supercell_axis_numerators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        pytest.param(0, [0, 1], id="axis-0"),
        pytest.param(1, [0, 1], id="axis-1"),
        pytest.param(2, [0, 0], id="axis-2"),
    ],
)
def test_supercell_axis_numerators_returns_known_oblique_numerators(axis, expected):
    origins = np.array([[0, 0, 0], [1, 0, 0]], dtype=object)

    numerators = supercell_axis_numerators(
        _int_matrix(OBLIQUE_INDEX2_ROWS),
        origins,
        axis=axis,
    )

    np.testing.assert_array_equal(numerators, np.array(expected, dtype=object))
    assert numerators.dtype == object
    assert all(type(value) is int for value in numerators)


def test_supercell_axis_numerators_normalizes_negative_determinant_sign():
    supercell = _int_matrix(((0, 1, 0), (2, 0, 0), (0, 0, 1)))

    numerators = supercell_axis_numerators(
        supercell,
        [[1, 0, 0]],  # type: ignore[ty:invalid-argument-type]
        axis=1,
    )

    np.testing.assert_array_equal(numerators, np.array([1], dtype=object))


def test_supercell_axis_numerators_preserves_large_exact_integers():
    large = 10**20

    numerators = supercell_axis_numerators(
        _int_matrix(IDENTITY_ROWS),
        [[large, 0, 0]],  # type: ignore[ty:invalid-argument-type]
        axis=0,
    )

    np.testing.assert_array_equal(numerators, np.array([large], dtype=object))
    assert type(numerators[0]) is int


@pytest.mark.parametrize(
    ("bad_supercell", "match"),
    INVALID_3X3_INTEGER_MATRICES,
)
def test_supercell_axis_numerators_rejects_invalid_supercell(
    bad_supercell,
    match,
):
    with pytest.raises(ValueError, match=match):
        supercell_axis_numerators(
            bad_supercell,
            [[0, 0, 0]],  # type: ignore[ty:invalid-argument-type]
        )


@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(-1, id="negative"),
        pytest.param(3, id="too-large"),
        pytest.param(1.5, id="float"),
        pytest.param(True, id="boolean"),
        pytest.param(np.bool_(False), id="numpy-boolean"),
    ],
)
def test_supercell_axis_numerators_rejects_invalid_axis(axis):
    with pytest.raises(ValueError, match="axis must be 0, 1, or 2"):
        supercell_axis_numerators(
            _int_matrix(IDENTITY_ROWS),
            [[0, 0, 0]],  # type: ignore[ty:invalid-argument-type]
            axis=axis,
        )


def test_supercell_axis_numerators_accepts_numpy_integer_axis():
    numerators = supercell_axis_numerators(
        _int_matrix(IDENTITY_ROWS),
        [[1, 2, 3]],  # type: ignore[ty:invalid-argument-type]
        axis=np.int64(2),  # type: ignore[ty:invalid-argument-type]
    )

    np.testing.assert_array_equal(numerators, np.array([3], dtype=object))


@pytest.mark.parametrize(
    ("bad_origins", "match"),
    [
        pytest.param([0, 0, 0], r"shape \(N, 3\)", id="one-dimensional"),
        pytest.param([[0, 0], [1, 1]], r"shape \(N, 3\)", id="wrong-width"),
        pytest.param(
            [[0.5, 0, 0]],
            "integer-valued",
            id="non-integer-entry",
        ),
        pytest.param(
            [[True, 0, 0]],
            "not an integer",
            id="boolean-entry",
        ),
    ],
)
def test_supercell_axis_numerators_rejects_invalid_origins(bad_origins, match):
    with pytest.raises(ValueError, match=match):
        supercell_axis_numerators(
            _int_matrix(IDENTITY_ROWS),
            bad_origins,
        )


def test_supercell_axis_numerators_rejects_singular_supercell():
    singular = [[1, 0, 0], [1, 0, 0], [0, 0, 1]]

    with pytest.raises(ValueError, match="non-singular"):
        supercell_axis_numerators(
            singular,  # type: ignore[ty:invalid-argument-type]
            [[0, 0, 0]],  # type: ignore[ty:invalid-argument-type]
        )
