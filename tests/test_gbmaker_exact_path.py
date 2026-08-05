# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Integration tests for GBMaker's exact integer grain-construction path."""

import importlib
import numpy as np
import pytest
from scipy.spatial import KDTree

from GBOpt.Atom import Atom
from GBOpt.BoundarySpec import CSLExactSpec, PQSpec
from GBOpt.crystallography import pq_spec_to_embedding
from GBOpt.GBMaker import GBMaker, GBMakerValueError
from GBOpt.gbmaker_supercell import SupercellSites
from GBOpt.UnitCell import UnitCell
from tests.data.zhang_2022_uo2_ceo2_gb_energies import BOUNDARIES

# --------------------------------------------------------------------------------------
# Shared boundary specifications and material data
# --------------------------------------------------------------------------------------

SIGMA5_TILT_P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
SIGMA5_TILT_Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]
SIGMA5_TILT_EXACT_SPEC = CSLExactSpec(
    axis=[0, 0, 1],
    plane=[1, 0, 0],
    quat=[3, 0, 0, 1],
)
SIGMA5_TILT_PQ_SPEC = PQSpec(
    P=SIGMA5_TILT_P,
    Q=SIGMA5_TILT_Q,
    basis_mode="supplied",
)
NONCOMMENSURATE_PQ_SPEC = PQSpec(
    P=SIGMA5_TILT_P,
    Q=[[1, 0, 0], [0, 1, 1], [0, -1, 1]],
    basis_mode="supplied",
)

A0_FCC = 3.615
STRUCTURE_FCC = "fcc"
ATOM_TYPES_FCC = "Cu"

EXACT_BOX_CASES = [
    pytest.param(
        SIGMA5_TILT_PQ_SPEC,
        A0_FCC,
        STRUCTURE_FCC,
        ATOM_TYPES_FCC,
        id="pq-fcc",
    ),
    pytest.param(
        SIGMA5_TILT_EXACT_SPEC,
        A0_FCC,
        STRUCTURE_FCC,
        ATOM_TYPES_FCC,
        id="csl-exact-fcc",
    ),
    pytest.param(
        SIGMA5_TILT_PQ_SPEC,
        5.47,
        "fluorite",
        ("U", "O"),
        id="pq-fluorite",
    ),
]

VACUUM_ZERO_BOX_CASES = [
    pytest.param(
        A0_FCC,
        STRUCTURE_FCC,
        ATOM_TYPES_FCC,
        {"interaction_distance": A0_FCC, "repeat_factor": 2},
        id="fcc-default-thickness",
    ),
    pytest.param(
        5.47,
        "fluorite",
        ("U", "O"),
        {"interaction_distance": 5.47, "repeat_factor": 2},
        id="fluorite-default-thickness",
    ),
    pytest.param(
        5.47,
        "fluorite",
        ("U", "O"),
        {"x_dim_min": 20.0, "interaction_distance": 1.0, "repeat_factor": 2},
        id="fluorite-reduced-thickness",
    ),
]

FLUORITE_EXACT_SPECS = [
    pytest.param(SIGMA5_TILT_PQ_SPEC, id="pq"),
    pytest.param(SIGMA5_TILT_EXACT_SPEC, id="csl-exact"),
]

ZHANG_001_CASE = (
    "zhang_001_ST_100",
    [[0, 18, -1], [0, 1, 18], [1, 0, 0]],
    [[0, 1, -18], [0, 18, 1], [1, 0, 0]],
    19_500,
    19_500,
)
ZHANG_031_CASE = (
    "zhang_031_AT_100",
    [[0, -5, 14], [1, 0, 0], [0, 14, 5]],
    [[0, 10, 11], [1, 0, 0], [0, 11, -10]],
    13_260,
    13_260,
)
ZHANG_041_CASE = (
    "zhang_041_TW_100",
    [[0, 0, 1], [4, 1, 0], [-1, 4, 0]],
    [[0, 0, 1], [4, -1, 0], [1, 4, 0]],
    2_448,
    2_448,
)
ZHANG_086_CASE = (
    "zhang_086_AT_110",
    [[-1, -1, 6], [1, -1, 0], [3, 3, 1]],
    [[1, 1, 12], [1, -1, 0], [6, 6, -1]],
    112_176,
    220_752,
)

REPRESENTATIVE_CASES = [
    pytest.param(*ZHANG_001_CASE, id="zhang-001-ST-100"),
    pytest.param(*ZHANG_031_CASE, id="zhang-031-AT-100"),
    pytest.param(*ZHANG_041_CASE, id="zhang-041-TW-100"),
    pytest.param(
        *ZHANG_086_CASE,
        marks=pytest.mark.slow,
        id="zhang-086-AT-110",
    ),
]


# --------------------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------------------


@pytest.fixture
def build_gb():
    """Return a function-scoped GBMaker factory with compact exact-path defaults."""

    def _build(
        boundary=SIGMA5_TILT_PQ_SPEC,
        *,
        a0=A0_FCC,
        structure=STRUCTURE_FCC,
        atom_types=ATOM_TYPES_FCC,
        mode="exact",
        **overrides,
    ):
        kwargs = {
            "gb_thickness": 0.0,
            "repeat_factor": 2,
            "interaction_distance": a0,
        }
        kwargs.update(overrides)
        return GBMaker.from_boundary_spec(
            a0,
            structure,
            atom_types,
            boundary,
            mode=mode,
            **kwargs,
        )

    return _build


def _positions(atoms):
    """Return structured atom coordinates as an ``(N, 3)`` float array."""
    return np.column_stack((atoms["x"], atoms["y"], atoms["z"]))


def _assert_fluorite_stoichiometry(atoms, *, label):
    """Assert a nonempty atom collection has the expected UO2 species ratio."""
    uranium_count = int(np.count_nonzero(atoms["name"] == "U"))
    oxygen_count = int(np.count_nonzero(atoms["name"] == "O"))

    assert uranium_count > 0, f"{label} contains no U atoms"
    assert oxygen_count == 2 * uranium_count, (
        f"{label} stoichiometry is {uranium_count} U to {oxygen_count} O; "
        "expected UO2"
    )


def _assert_exact_fluorite_counts(atoms, expected_count, *, label):
    """Assert complete fluorite population from its four-U/eight-O basis."""
    assert len(atoms) == expected_count
    assert expected_count % 12 == 0

    conventional_cells = expected_count // 12
    uranium_count = int(np.count_nonzero(atoms["name"] == "U"))
    oxygen_count = int(np.count_nonzero(atoms["name"] == "O"))

    assert uranium_count == 4 * conventional_cells, (
        f"{label} contains {uranium_count} U atoms; expected "
        f"{4 * conventional_cells}"
    )
    assert oxygen_count == 8 * conventional_cells, (
        f"{label} contains {oxygen_count} O atoms; expected "
        f"{8 * conventional_cells}"
    )


def _build_representative_boundary(P, Q):
    """Build one representative case with the original campaign conventions.

    Callers filter the expected repeat recommendation and automatic commensurate
    resize warnings; those warning contracts are exercised in focused GBMaker tests.
    """
    boundary = PQSpec(P=P, Q=Q, basis_mode="supplied")
    common = {
        "a0": 5.454,
        "structure": "fluorite",
        "atom_types": ("U", "O"),
        "boundary": boundary,
        "mode": "exact",
        "repeat_factor": (1, 1),
        "x_dim_min": 60.0,
        "vacuum": 0.0,
        "interaction_distance": 11.0,
        "mismatch_tol": 0.005,
        "mismatch_max_cells": 50,
        "strain_grain": "both",
    }

    probe = GBMaker.from_boundary_spec(gb_thickness=5.454, **common)
    gb_thickness = 2.0 * max(
        float(probe.spacing["x"]["left"]),
        float(probe.spacing["x"]["right"]),
    )
    return GBMaker.from_boundary_spec(gb_thickness=gb_thickness, **common)


# --------------------------------------------------------------------------------------
# Exact-path dispatch and commensurability
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("grain_rows", "row_index", "dimension_name"),
    [
        pytest.param(SIGMA5_TILT_P, 1, "y_dim", id="left-y"),
        pytest.param(SIGMA5_TILT_P, 2, "z_dim", id="left-z"),
        pytest.param(SIGMA5_TILT_Q, 1, "y_dim", id="right-y"),
        pytest.param(SIGMA5_TILT_Q, 2, "z_dim", id="right-z"),
    ],
)
def test_exact_inplane_dimensions_are_integer_multiples_of_both_grain_periods(
    build_gb,
    grain_rows,
    row_index,
    dimension_name,
):
    gb = build_gb()

    period = A0_FCC * np.linalg.norm(
        np.asarray(grain_rows[row_index], dtype=float)
    )
    repeat_count = getattr(gb, dimension_name) / period

    assert repeat_count == pytest.approx(round(repeat_count), abs=1e-6, rel=0.0)


def test_exact_embedding_uses_integer_rows_without_float_approximation(
    monkeypatch,
    build_gb,
):
    embedding = pq_spec_to_embedding(
        PQSpec(P=SIGMA5_TILT_P, Q=SIGMA5_TILT_P, basis_mode="supplied")
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            "The exact embedding path must not approximate rotation rows as integers."
        )

    monkeypatch.setattr(
        GBMaker,
        "_GBMaker__approximate_rotation_matrix_as_int",
        fail_if_called,
    )

    gb = GBMaker._from_boundary_embedding(
        embedding,
        a0=A0_FCC,
        structure=STRUCTURE_FCC,
        atom_types=ATOM_TYPES_FCC,
        gb_thickness=0.0,
        repeat_factor=2,
        interaction_distance=A0_FCC,
    )

    assert gb.whole_system.size > 0


# --------------------------------------------------------------------------------------
# Exact decorated-site construction
# --------------------------------------------------------------------------------------


def test_exact_builder_returns_atom_dtype_and_complete_decorated_populations(
    build_gb,
    monkeypatch,
):
    gbmaker_module = importlib.import_module("GBOpt.GBMaker")
    original_enumerator = gbmaker_module.enumerate_supercell_sites
    enumerated_sites = []

    def capture_sites(*args, **kwargs):
        sites = original_enumerator(*args, **kwargs)
        enumerated_sites.append(sites)
        return sites

    monkeypatch.setattr(
        gbmaker_module,
        "enumerate_supercell_sites",
        capture_sites,
    )

    gb = build_gb(
        a0=5.47,
        structure="fluorite",
        atom_types=("U", "O"),
        vacuum=0.0,
    )
    rational_basis = gb.unit_cell.rational_basis

    assert rational_basis is not None
    assert gb.whole_system.dtype == Atom.atom_dtype
    assert gb.whole_system.size == gb.left_grain.size + gb.right_grain.size
    assert len(enumerated_sites) == 2

    for label, grain, sites in zip(
        ("left", "right"),
        (gb.left_grain, gb.right_grain),
        enumerated_sites,
    ):
        expected_per_basis = sites.supercell_index * np.prod(sites.repeats)
        expected_count = len(rational_basis.names) * expected_per_basis
        populations = np.bincount(
            sites.basis_indices,
            minlength=len(rational_basis.names),
        )
        decorated_keys = {
            (tuple(row), int(basis_index))
            for row, basis_index in zip(
                sites.coordinate_numerators,
                sites.basis_indices,
            )
        }
        expected_names = np.asarray(rational_basis.names)[sites.basis_indices]

        assert len(grain) == expected_count
        assert np.all(populations == expected_per_basis)
        assert len(decorated_keys) == sites.site_count
        assert np.array_equal(grain["name"], expected_names), label


def test_exact_builder_returns_complete_small_fcc_grains(build_gb):
    gb = build_gb()
    basis_size = len(gb.unit_cell.asarray())

    for label, grain in (("left", gb.left_grain), ("right", gb.right_grain)):
        assert grain.size > 0, f"{label} grain is empty"
        assert grain.size % basis_size == 0, (
            f"{label} grain contains {grain.size} atoms, which is not divisible by "
            f"the conventional-cell basis size {basis_size}"
        )


def test_exact_builder_rejects_unrepresentable_integer_coordinates(
    build_gb,
    monkeypatch,
):
    gb = build_gb()
    supercell = np.array(
        [
            [1, 10**400, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=object,
    )
    sites = SupercellSites(
        coordinate_numerators=np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 1, 1],
            ],
            dtype=object,
        ),
        basis_denominator=2,
        basis_indices=np.arange(4, dtype=np.intp),
        supercell_matrix=supercell,
        repeats=(1, 1, 1),
        basis_size=4,
    )

    monkeypatch.setattr(
        GBMaker,
        "_GBMaker__exact_grain_repeats",
        lambda _self, _P_or_Q, _x_length, _grain_side: (
            supercell,
            1,
            1,
            1,
        ),
    )
    monkeypatch.setattr(
        importlib.import_module("GBOpt.GBMaker"),
        "enumerate_supercell_sites",
        lambda *_args, **_kwargs: sites,
    )

    with pytest.raises(
        GBMakerValueError,
        match=r"cannot be represented as finite Cartesian values",
    ):
        gb._GBMaker__generate_grain_exact(
            np.eye(3),
            np.eye(3, dtype=object),
            gb.a0,
            0.0,
            "left",
        )


def test_exact_builder_canonicalizes_rounded_upper_x_face_into_half_open_slab(
    build_gb,
    monkeypatch,
):
    gb = build_gb()
    denominator = 10**20
    supercell = np.eye(3, dtype=object)
    sites = SupercellSites(
        coordinate_numerators=np.array(
            [
                [denominator - 1, 0, 0],
                [0, 0, 1],
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=object,
        ),
        basis_denominator=denominator,
        basis_indices=np.arange(4, dtype=np.intp),
        supercell_matrix=supercell,
        repeats=(1, 1, 1),
        basis_size=4,
    )

    monkeypatch.setattr(
        GBMaker,
        "_GBMaker__exact_grain_repeats",
        lambda _self, _P_or_Q, _x_length, _grain_side: (
            supercell,
            1,
            1,
            1,
        ),
    )
    monkeypatch.setattr(
        importlib.import_module("GBOpt.GBMaker"),
        "enumerate_supercell_sites",
        lambda *_args, **_kwargs: sites,
    )

    atoms = gb._GBMaker__generate_grain_exact(
        np.eye(3),
        np.eye(3, dtype=object),
        gb.a0,
        0.0,
        "left",
    )

    assert np.max(atoms["x"]) < gb.a0
    assert np.max(atoms["x"]) == np.nextafter(gb.a0, 0.0)


@pytest.mark.filterwarnings(
    r"ignore:Commensurate repeat pair in [yz] multiplied by \d+ to satisfy the "
    r"minimum in-plane dimension cutoff of .* A\.:UserWarning"
)
@pytest.mark.filterwarnings(
    r"ignore:Recommended repeat factor is at least 2\.:UserWarning"
)
@pytest.mark.parametrize(
    ("case_id", "P", "Q", "left_expected", "right_expected"),
    REPRESENTATIVE_CASES,
)
def test_representative_exact_counts_and_species_are_complete(
    case_id,
    P,
    Q,
    left_expected,
    right_expected,
):
    gb = _build_representative_boundary(P, Q)

    _assert_exact_fluorite_counts(
        gb.left_grain,
        left_expected,
        label=f"{case_id} left grain",
    )
    _assert_exact_fluorite_counts(
        gb.right_grain,
        right_expected,
        label=f"{case_id} right grain",
    )
    _assert_exact_fluorite_counts(
        gb.whole_system,
        left_expected + right_expected,
        label=f"{case_id} whole system",
    )

    assert len(gb.whole_system) == len(gb.left_grain) + len(gb.right_grain)
    assert np.all(np.isfinite(_positions(gb.whole_system)))


@pytest.mark.filterwarnings(
    r"ignore:Commensurate repeat pair in [yz] multiplied by \d+ to satisfy the "
    r"minimum in-plane dimension cutoff of .* A\.:UserWarning"
)
@pytest.mark.filterwarnings(
    r"ignore:Recommended repeat factor is at least 2\.:UserWarning"
)
def test_high_index_exact_boundary_preserves_all_decorated_sites_near_x_boundary():
    _, P, Q, left_expected, right_expected = ZHANG_001_CASE
    gb = _build_representative_boundary(P, Q)

    assert len(gb.left_grain) == left_expected
    assert len(gb.right_grain) == right_expected
    assert len(gb.whole_system) == left_expected + right_expected

    tolerance = max(1e-8, 100.0 * gb.epsilon)
    assert np.min(gb.left_grain["x"]) >= -tolerance
    assert np.max(gb.left_grain["x"]) < gb.gb_plane_x + tolerance
    assert np.min(gb.right_grain["x"]) >= gb.gb_plane_x - tolerance
    assert np.max(gb.right_grain["x"]) < gb.x_dim + tolerance


@pytest.mark.filterwarnings(
    r"ignore:Commensurate repeat pair in [yz] multiplied by \d+ to satisfy the "
    r"minimum in-plane dimension cutoff of .* A\.:UserWarning"
)
@pytest.mark.filterwarnings(
    r"ignore:Recommended repeat factor is at least 2\.:UserWarning"
)
def test_exact_construction_is_deterministic_for_names_and_coordinate_order():
    _, P, Q, _, _ = ZHANG_041_CASE
    first = _build_representative_boundary(P, Q)
    second = _build_representative_boundary(P, Q)

    assert np.array_equal(first.whole_system["name"], second.whole_system["name"])
    assert np.array_equal(first.whole_system, second.whole_system)


def test_exact_construction_rejects_missing_rational_basis(monkeypatch):
    monkeypatch.setattr(
        UnitCell,
        "rational_basis",
        property(lambda _self: None),
    )

    with pytest.raises(
        GBMakerValueError,
        match=r"Exact grain generation requires UnitCell\.rational_basis",
    ):
        GBMaker.from_boundary_spec(
            A0_FCC,
            STRUCTURE_FCC,
            ATOM_TYPES_FCC,
            SIGMA5_TILT_PQ_SPEC,
            mode="exact",
            gb_thickness=0.0,
            repeat_factor=2,
            interaction_distance=A0_FCC,
        )


# --------------------------------------------------------------------------------------
# Cartesian box and central-interface bounds
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "a0", "structure", "atom_types"),
    EXACT_BOX_CASES,
)
def test_exact_atoms_are_within_periodic_yz_box(
    build_gb,
    spec,
    a0,
    structure,
    atom_types,
):
    gb = build_gb(
        spec,
        a0=a0,
        structure=structure,
        atom_types=atom_types,
    )
    atoms = gb.whole_system
    tolerance = max(1e-8, 100.0 * gb.epsilon)

    assert np.min(atoms["y"]) >= -tolerance
    assert np.max(atoms["y"]) < gb.y_dim + tolerance
    assert np.min(atoms["z"]) >= -tolerance
    assert np.max(atoms["z"]) < gb.z_dim + tolerance


def test_exact_grains_do_not_cross_central_boundary_plane():
    spec = PQSpec(
        P=[[3, 1, 0], [0, 0, 2], [1, -3, 0]],
        Q=[[3, 1, 0], [0, 0, -2], [-1, 3, 0]],
    )

    with pytest.warns(UserWarning, match=r"Recommended repeat factor is at least 2\."):
        gb = GBMaker.from_boundary_spec(
            3.52,
            "fcc",
            "Ni",
            spec,
            mode="exact",
            gb_thickness=0.0,
            repeat_factor=(1, 3),
            x_dim_min=20.0,
            vacuum=0.0,
            interaction_distance=5.0,
        )

    tolerance = 1e-4 * gb.a0
    assert np.max(gb.left_grain["x"]) <= gb.gb_plane_x + tolerance
    assert np.min(gb.right_grain["x"]) >= gb.gb_plane_x - tolerance


@pytest.mark.parametrize(
    ("a0", "structure", "atom_types", "kwargs"),
    VACUUM_ZERO_BOX_CASES,
)
def test_vacuum_zero_exact_atoms_are_within_x_box(
    build_gb,
    a0,
    structure,
    atom_types,
    kwargs,
):
    gb = build_gb(
        a0=a0,
        structure=structure,
        atom_types=atom_types,
        vacuum=0.0,
        **kwargs,
    )
    atoms = gb.whole_system
    tolerance = max(1e-8, 100.0 * gb.epsilon)

    assert np.min(atoms["x"]) >= -tolerance
    assert np.max(atoms["x"]) < gb.x_dim + tolerance


# --------------------------------------------------------------------------------------
# Vacuum-zero periodic-interface regressions
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("central_gap", "periodic_gap"),
    [
        pytest.param(-1e-3, 0.25, id="central-overlap"),
        pytest.param(0.25, -1e-3, id="periodic-overlap"),
    ],
)
def test_exact_gap_handling_rejects_negative_gap_without_layer_deletion(
    build_gb,
    monkeypatch,
    central_gap,
    periodic_gap,
):
    gb = build_gb(vacuum=0.0)
    left_before = gb.left_grain.copy()
    right_before = gb.right_grain.copy()
    whole_before = gb.whole_system.copy()

    monkeypatch.setattr(
        GBMaker,
        "_GBMaker__current_gap_metrics",
        lambda _self, _left_bounds, _right_bounds: (
            central_gap,
            periodic_gap,
            0.0,
            gb.x_dim,
        ),
    )

    with pytest.raises(
        GBMakerValueError,
        match=r"invalid x-boundary overlap",
    ):
        gb._GBMaker__equalize_periodic_gap(
            left_bounds=np.array([0.0, gb.gb_plane_x]),
            right_effective_bounds=np.array([gb.gb_plane_x, gb.x_dim]),
            use_exact=True,
            right_float_result=None,
            vacuum0_trim_applied=False,
            x_period_right=None,
        )

    np.testing.assert_array_equal(gb.left_grain, left_before)
    np.testing.assert_array_equal(gb.right_grain, right_before)
    np.testing.assert_array_equal(gb.whole_system, whole_before)


def test_vacuum_zero_exact_gap_metrics_are_diagnostic_only(build_gb):
    gb = build_gb(vacuum=0.0)

    central_gap = float(
        np.min(gb.right_grain["x"]) - np.max(gb.left_grain["x"])
    )
    periodic_gap = float(
        (gb.x_dim - np.max(gb.right_grain["x"]))
        + np.min(gb.left_grain["x"])
    )

    assert np.isfinite(central_gap)
    assert np.isfinite(periodic_gap)
    assert central_gap >= -gb.epsilon
    assert periodic_gap >= -gb.epsilon
    assert periodic_gap < central_gap - gb.epsilon

    # The exact path must retain both complete 4-site FCC populations instead of
    # deleting a right-grain layer to reverse this diagnostic gap ordering.
    assert len(gb.left_grain) == 1_200
    assert len(gb.right_grain) == 1_200
    assert len(gb.whole_system) == 2_400


@pytest.mark.parametrize("spec", FLUORITE_EXACT_SPECS)
def test_vacuum_zero_has_no_coincident_atoms_across_periodic_images(build_gb, spec):
    gb = build_gb(
        spec,
        a0=5.47,
        structure="fluorite",
        atom_types=("U", "O"),
        vacuum=0.0,
    )
    box_lengths = np.array([gb.x_dim, gb.y_dim, gb.z_dim], dtype=float)
    left_positions = np.mod(_positions(gb.left_grain), box_lengths)
    right_positions = np.mod(_positions(gb.right_grain), box_lengths)

    tree = KDTree(right_positions, boxsize=box_lengths)
    nearest_distances, _ = tree.query(left_positions, k=1)
    coincident_count = int(np.count_nonzero(nearest_distances <= 1e-4))

    assert coincident_count == 0, (
        f"detected {coincident_count} coincident left/right atom pairs under periodic "
        "boundary conditions"
    )


@pytest.mark.parametrize("spec", FLUORITE_EXACT_SPECS)
def test_vacuum_zero_preserves_fluorite_stoichiometry_in_each_grain_and_system(
    build_gb,
    spec,
):
    gb = build_gb(
        spec,
        a0=5.47,
        structure="fluorite",
        atom_types=("U", "O"),
        vacuum=0.0,
    )

    _assert_fluorite_stoichiometry(gb.left_grain, label="left grain")
    _assert_fluorite_stoichiometry(gb.right_grain, label="right grain")
    _assert_fluorite_stoichiometry(gb.whole_system, label="whole system")


@pytest.mark.parametrize(
    "spec",
    [
        pytest.param(SIGMA5_TILT_PQ_SPEC, id="pq"),
        pytest.param(SIGMA5_TILT_EXACT_SPEC, id="csl-exact"),
    ],
)
def test_vacuum_zero_preserves_rocksalt_stoichiometry_in_each_grain_and_system(
    build_gb,
    spec,
):
    gb = build_gb(
        spec,
        a0=4.0,
        structure="rocksalt",
        atom_types=("Na", "Cl"),
        vacuum=0.0,
    )

    for label, atoms in (
        ("left grain", gb.left_grain),
        ("right grain", gb.right_grain),
        ("whole system", gb.whole_system),
    ):
        sodium_count = int(np.count_nonzero(atoms["name"] == "Na"))
        chlorine_count = int(np.count_nonzero(atoms["name"] == "Cl"))

        assert sodium_count > 0, f"{label} contains no Na atoms"
        assert chlorine_count == sodium_count, (
            f"{label} stoichiometry is {sodium_count} Na to "
            f"{chlorine_count} Cl; expected NaCl"
        )


def test_zhang_sigma53_vacuum_zero_regression_preserves_box_gap_and_stoichiometry():
    """Cover the external fluorite case that previously leaked basis offsets in x."""

    entry = BOUNDARIES["sigma53_100_0_7_2bar_0_2bar_7_STGB"]
    spec = PQSpec(P=entry["P"], Q=entry["Q"])

    with pytest.warns(UserWarning, match=r"Recommended repeat factor is at least 2\."):
        gb = GBMaker.from_boundary_spec(
            5.454,
            "fluorite",
            ("U", "O"),
            spec,
            mode="exact",
            gb_thickness=0.0,
            vacuum=0.0,
            repeat_factor=[1, 1],
            x_dim_min=20.0,
            interaction_distance=1.0,
        )

    tolerance = max(1e-8, 100.0 * gb.epsilon)
    assert np.min(gb.whole_system["x"]) >= -tolerance
    assert np.max(gb.whole_system["x"]) < gb.x_dim + tolerance

    central_gap = float(
        np.min(gb.right_grain["x"]) - np.max(gb.left_grain["x"])
    )
    periodic_gap = float(
        (gb.x_dim - np.max(gb.right_grain["x"]))
        + np.min(gb.left_grain["x"])
    )
    assert np.isfinite(central_gap)
    assert np.isfinite(periodic_gap)
    assert central_gap >= -gb.epsilon
    assert periodic_gap >= -gb.epsilon

    _assert_fluorite_stoichiometry(gb.left_grain, label="left grain")
    _assert_fluorite_stoichiometry(gb.right_grain, label="right grain")
    _assert_fluorite_stoichiometry(gb.whole_system, label="whole system")
