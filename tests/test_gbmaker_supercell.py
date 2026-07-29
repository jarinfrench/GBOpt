# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for exact integer supercell construction and enumeration."""

from itertools import product

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
from GBOpt.UnitCell import UnitCell

# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------

IDENTITY_ROWS = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
SIGMA5_RIGHT_GRAIN_ROWS = ((4, -3, 0), (3, 4, 0), (0, 0, 1))
OBLIQUE_INDEX2_ROWS = ((1, -1, 0), (1, 1, 0), (0, 0, 1))
ZHANG_001_LEFT_ROWS = ((0, 18, -1), (0, 1, 18), (1, 0, 0))
ZHANG_041_LEFT_ROWS = ((0, 0, 1), (4, 1, 0), (-1, 4, 0))
NEGATIVE_DETERMINANT_ROWS = ((0, 1, 0), (2, 0, 0), (0, 0, 1))

FLUORITE_BASIS_NUMERATORS = (
    (0, 0, 0),
    (0, 2, 2),
    (2, 0, 2),
    (2, 2, 0),
    (1, 1, 1),
    (1, 1, 3),
    (1, 3, 1),
    (1, 3, 3),
    (3, 1, 1),
    (3, 1, 3),
    (3, 3, 1),
    (3, 3, 3),
)

SUPPORTED_RATIONAL_BASIS_CASES = (
    pytest.param("sc", "Cu", 1, id="sc"),
    pytest.param("bcc", "Fe", 2, id="bcc"),
    pytest.param("fcc", "Cu", 4, id="fcc"),
    pytest.param("diamond", "C", 8, id="diamond"),
    pytest.param("fluorite", ("U", "O"), 12, id="fluorite"),
    pytest.param("rocksalt", ("Na", "Cl"), 8, id="rocksalt"),
    pytest.param("zincblende", ("Zn", "S"), 8, id="zincblende"),
)

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


def _assert_exact_site_invariants(
    sites: SupercellSites,
    supercell_rows,
    repeats,
    *,
    basis_size: int,
) -> None:
    """Assert exact count, wrapping, uniqueness, and reconstruction invariants."""
    supercell = _int_matrix(supercell_rows)
    determinant = integer_det3(supercell)
    absolute_determinant = abs(determinant)
    expected_origins = int(np.prod(repeats)) * absolute_determinant
    expected_sites = expected_origins * basis_size

    assert sites.crystal_numerators.shape == (expected_sites, 3)
    assert sites.supercell_coordinate_numerators.shape == (expected_sites, 3)
    assert sites.basis_indices.shape == (expected_sites,)
    assert sites.supercell_index == absolute_determinant
    assert sites.supercell_coordinate_denominator == (
        sites.denominator * sites.supercell_index
    )

    upper_bounds = (
        np.asarray(repeats, dtype=object)
        * sites.supercell_coordinate_denominator
    )
    assert np.all(sites.supercell_coordinate_numerators >= 0)
    assert np.all(sites.supercell_coordinate_numerators < upper_bounds)

    reconstructed_scaled = (
        sites.supercell_coordinate_numerators @ supercell
    )
    np.testing.assert_array_equal(
        reconstructed_scaled,
        sites.crystal_numerators * absolute_determinant,
    )

    assert len(
        {
            tuple(int(value) for value in row)
            for row in sites.crystal_numerators
        }
    ) == expected_sites
    np.testing.assert_array_equal(
        np.bincount(sites.basis_indices, minlength=basis_size),
        np.full(basis_size, expected_origins),
    )


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


# ---------------------------------------------------------------------------
# enumerate_supercell_sites
# ---------------------------------------------------------------------------


def test_enumerate_supercell_sites_returns_known_identity_sites_in_origin_order():
    sites = enumerate_supercell_sites(
        _int_matrix(IDENTITY_ROWS),
        2,
        1,
        1,
        basis_numerators=np.array(
            ((0, 0, 0), (1, 1, 1)),
            dtype=object,
        ),
        basis_denominator=2,
    )

    np.testing.assert_array_equal(
        sites.crystal_numerators,
        np.array(
            (
                (0, 0, 0),
                (1, 1, 1),
                (2, 0, 0),
                (3, 1, 1),
            ),
            dtype=object,
        ),
    )
    np.testing.assert_array_equal(sites.basis_indices, np.array((0, 1, 0, 1)))
    np.testing.assert_array_equal(
        sites.supercell_coordinate_numerators,
        sites.crystal_numerators,
    )
    assert sites.denominator == 2
    assert sites.supercell_index == 1
    assert sites.supercell_coordinate_denominator == 2


def test_supercell_sites_derives_supercell_coordinate_denominator():
    sites = SupercellSites(
        crystal_numerators=np.array(((0, 0, 0),), dtype=object),
        denominator=4,
        basis_indices=np.array((0,)),
        supercell_coordinate_numerators=np.array(((0, 0, 0),), dtype=object),
        supercell_index=325,
    )

    assert sites.supercell_coordinate_denominator == 1300


@pytest.mark.parametrize(
    ("supercell_rows", "repeats", "expected_origins"),
    [
        pytest.param(
            ZHANG_001_LEFT_ROWS,
            (1, 1, 5),
            1625,
            id="zhang-001-st-100",
        ),
        pytest.param(
            ZHANG_041_LEFT_ROWS,
            (12, 1, 1),
            204,
            id="zhang-041-tw-100",
        ),
    ],
)
def test_enumerate_supercell_sites_preserves_complete_fluorite_basis_population(
    supercell_rows,
    repeats,
    expected_origins,
):
    sites = enumerate_supercell_sites(
        _int_matrix(supercell_rows),
        *repeats,
        basis_numerators=np.array(FLUORITE_BASIS_NUMERATORS, dtype=object),
        basis_denominator=4,
    )

    assert len(sites.crystal_numerators) == expected_origins * 12
    _assert_exact_site_invariants(
        sites,
        supercell_rows,
        repeats,
        basis_size=12,
    )


@pytest.mark.parametrize(
    ("structure", "atom_types", "expected_basis_size"),
    SUPPORTED_RATIONAL_BASIS_CASES,
)
def test_enumerate_supercell_sites_supports_every_builtin_rational_basis(
    structure,
    atom_types,
    expected_basis_size,
):
    unit_cell = UnitCell()
    unit_cell.init_by_structure(structure, 1.0, atom_types)
    basis = unit_cell.rational_basis

    assert basis is not None
    assert len(basis.names) == expected_basis_size

    sites = enumerate_supercell_sites(
        _int_matrix(OBLIQUE_INDEX2_ROWS),
        2,
        1,
        1,
        basis_numerators=basis.numerators,
        basis_denominator=basis.denominator,
    )

    assert sites.denominator == basis.denominator
    _assert_exact_site_invariants(
        sites,
        OBLIQUE_INDEX2_ROWS,
        (2, 1, 1),
        basis_size=expected_basis_size,
    )


def test_enumerate_supercell_sites_is_deterministic():
    kwargs = {
        "basis_numerators": np.array(FLUORITE_BASIS_NUMERATORS, dtype=object),
        "basis_denominator": 4,
    }

    first = enumerate_supercell_sites(
        _int_matrix(ZHANG_041_LEFT_ROWS),
        2,
        1,
        1,
        **kwargs,
    )
    second = enumerate_supercell_sites(
        _int_matrix(ZHANG_041_LEFT_ROWS),
        2,
        1,
        1,
        **kwargs,
    )

    np.testing.assert_array_equal(
        first.crystal_numerators,
        second.crystal_numerators,
    )
    np.testing.assert_array_equal(first.basis_indices, second.basis_indices)
    np.testing.assert_array_equal(
        first.supercell_coordinate_numerators,
        second.supercell_coordinate_numerators,
    )
    assert first.denominator == second.denominator
    assert first.supercell_index == second.supercell_index


def test_enumerate_supercell_sites_preserves_large_shear_exactly():
    large_shear = 10_000
    supercell_rows = (
        (1, large_shear, 0),
        (0, 1, 0),
        (0, 0, 1),
    )

    sites = enumerate_supercell_sites(
        _int_matrix(supercell_rows),
        1,
        1,
        1,
        basis_numerators=np.array(((0, 0, 0), (1, 3, 2)), dtype=object),
        basis_denominator=4,
    )

    _assert_exact_site_invariants(
        sites,
        supercell_rows,
        (1, 1, 1),
        basis_size=2,
    )
    np.testing.assert_array_equal(
        sites.supercell_coordinate_numerators,
        np.array(((0, 0, 0), (1, 3, 2)), dtype=object),
    )
    np.testing.assert_array_equal(
        sites.crystal_numerators,
        np.array(((0, 0, 0), (1, large_shear + 3, 2)), dtype=object),
    )


def test_enumerate_supercell_sites_supports_negative_determinant_supercell():
    sites = enumerate_supercell_sites(
        _int_matrix(NEGATIVE_DETERMINANT_ROWS),
        1,
        1,
        1,
        basis_numerators=np.array(((0, 0, 0), (1, 1, 1)), dtype=object),
        basis_denominator=2,
    )

    _assert_exact_site_invariants(
        sites,
        NEGATIVE_DETERMINANT_ROWS,
        (1, 1, 1),
        basis_size=2,
    )


def test_enumerate_supercell_sites_returns_immutable_exact_arrays():
    sites = enumerate_supercell_sites(
        _int_matrix(IDENTITY_ROWS),
        1,
        1,
        1,
        basis_numerators=np.array(((0, 0, 0),), dtype=object),
        basis_denominator=1,
    )

    assert sites.crystal_numerators.dtype == object
    assert sites.supercell_coordinate_numerators.dtype == object
    assert not sites.crystal_numerators.flags.writeable
    assert not sites.supercell_coordinate_numerators.flags.writeable
    assert not sites.basis_indices.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        sites.crystal_numerators[0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        sites.supercell_coordinate_numerators[0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        sites.basis_indices[0] = 1


@pytest.mark.parametrize(
    ("bad_supercell", "match"),
    INVALID_3X3_INTEGER_MATRICES,
)
def test_enumerate_supercell_sites_rejects_invalid_supercell(
    bad_supercell,
    match,
):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_sites(
            bad_supercell,
            1,
            1,
            1,
            basis_numerators=np.array(((0, 0, 0),), dtype=object),
            basis_denominator=1,
        )


def test_enumerate_supercell_sites_rejects_singular_supercell():
    with pytest.raises(ValueError, match="non-singular"):
        enumerate_supercell_sites(
            [[1, 0, 0], [1, 0, 0], [0, 0, 1]],
            1,
            1,
            1,
            basis_numerators=np.array(((0, 0, 0),), dtype=object),
            basis_denominator=1,
        )


@pytest.mark.parametrize(
    ("repeats", "match"),
    [
        pytest.param((0, 1, 1), "repeat_x", id="zero-x"),
        pytest.param((1, -1, 1), "repeat_y", id="negative-y"),
        pytest.param((1, 1, 1.5), "repeat_z", id="float-z"),
        pytest.param((True, 1, 1), "repeat_x", id="boolean-x"),
    ],
)
def test_enumerate_supercell_sites_rejects_invalid_repeat_count(repeats, match):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_sites(
            _int_matrix(IDENTITY_ROWS),
            *repeats,
            basis_numerators=np.array(((0, 0, 0),), dtype=object),
            basis_denominator=1,
        )


@pytest.mark.parametrize(
    ("basis_numerators", "basis_denominator", "match"),
    [
        pytest.param(
            np.empty((0, 3), dtype=object),
            1,
            "at least one",
            id="empty-basis",
        ),
        pytest.param(
            [[0, 0]],
            1,
            "shape",
            id="wrong-width",
        ),
        pytest.param(
            [[0.5, 0, 0]],
            1,
            "integer-valued",
            id="noninteger-numerator",
        ),
        pytest.param(
            [[True, 0, 0]],
            1,
            "not an integer",
            id="boolean-numerator",
        ),
        pytest.param(
            [[-1, 0, 0]],
            2,
            "half-open interval",
            id="negative-numerator",
        ),
        pytest.param(
            [[2, 0, 0]],
            2,
            "half-open interval",
            id="numerator-at-denominator",
        ),
        pytest.param(
            [[0, 0, 0], [0, 0, 0]],
            1,
            "duplicate",
            id="duplicate-site",
        ),
        pytest.param(
            [[0, 0, 0]],
            0,
            "basis_denominator",
            id="zero-denominator",
        ),
        pytest.param(
            [[0, 0, 0]],
            1.5,
            "basis_denominator",
            id="noninteger-denominator",
        ),
        pytest.param(
            [[0, 0, 0]],
            True,
            "basis_denominator",
            id="boolean-denominator",
        ),
    ],
)
def test_enumerate_supercell_sites_rejects_invalid_basis(
    basis_numerators,
    basis_denominator,
    match,
):
    with pytest.raises(ValueError, match=match):
        enumerate_supercell_sites(
            _int_matrix(IDENTITY_ROWS),
            1,
            1,
            1,
            basis_numerators=basis_numerators,
            basis_denominator=basis_denominator,
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
