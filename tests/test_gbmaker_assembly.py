# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for ``GBOpt.gbmaker.assembly``'s pure bicrystal-assembly stage functions and
the ``build_bicrystal``/``assemble_bicrystal`` construction pipeline.
"""

import sys

import numpy as np
import pytest

from GBOpt.BoundarySpec import BoundaryEmbedding, PQSpec
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GBMaker import GBMaker
from GBOpt.gbmaker.assembly import (
    _current_gap_metrics,
    _grain_strain_scales,
    _grain_x_bounds,
    _select_gb_region,
    _use_exact_grain_generation,
    assemble_bicrystal,
    build_bicrystal,
)
from GBOpt.gbmaker.material import resolve_material_state
from GBOpt.gbmaker.types import (
    AxisAccommodation,
    GBMakerConstructionValueError,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _atoms(x_values):
    atoms = np.zeros(
        len(x_values), dtype=[("name", "U2"), ("x", "f8"), ("y", "f8"), ("z", "f8")]
    )
    atoms["name"] = "Ni"
    atoms["x"] = x_values
    return atoms


def _accommodation(**overrides) -> AxisAccommodation:
    kwargs = {
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
    return AxisAccommodation(**kwargs)


def _build_bicrystal_from_gbmaker(gb: GBMaker):
    """Rebuild the same bicrystal ``gb`` holds, but through ``build_bicrystal`` alone.

    Pulls the same resolved construction knobs ``gb`` normalized in its constructor,
    without touching ``gb`` itself, so the two results can be compared for exact
    equality.
    """
    return build_bicrystal(
        material=gb._GBMaker__material_state(),
        misorientation=np.hstack(
            (gb._GBMaker__misorientation, gb._GBMaker__inclination)
        ),
        embedding=gb._GBMaker__embedding,
        x_dim_min=gb._GBMaker__x_dim_min,
        vacuum_thickness=gb._GBMaker__vacuum_thickness,
        normal_topology=gb._GBMaker__normal_topology,
        interaction_distance=gb._GBMaker__interaction_distance,
        gb_thickness=gb._GBMaker__gb_thickness,
        gb_id=gb._GBMaker__id,
        epsilon=gb._GBMaker__epsilon,
        repeat_factor=tuple(gb._GBMaker__repeat_factor),
        mismatch_tol=gb._GBMaker__mismatch_tol,
        mismatch_max_cells=gb._GBMaker__mismatch_max_cells,
        strain_grain=gb._GBMaker__strain_grain,
    )


# --------------------------------------------------------------------------------------
# _grain_x_bounds
# --------------------------------------------------------------------------------------


def test_grain_x_bounds_places_grains_adjacent_at_vacuum_offset():
    left_bounds, right_bounds = _grain_x_bounds(
        left_x=10.0, right_x=15.0, x_dim=25.0, vacuum_thickness=2.0
    )
    assert left_bounds.tolist() == [2.0, 12.0]
    assert right_bounds.tolist() == [12.0, 27.0]


def test_grain_x_bounds_zero_vacuum():
    left_bounds, right_bounds = _grain_x_bounds(
        left_x=5.0, right_x=5.0, x_dim=10.0, vacuum_thickness=0.0
    )
    assert left_bounds.tolist() == [0.0, 5.0]
    assert right_bounds.tolist() == [5.0, 10.0]


# --------------------------------------------------------------------------------------
# _use_exact_grain_generation
# --------------------------------------------------------------------------------------


def test_use_exact_grain_generation_none_embedding_is_false():
    assert _use_exact_grain_generation(None) is False


def test_use_exact_grain_generation_non_exact_embedding_is_false():
    embedding = BoundaryEmbedding(
        P=None,
        Q=None,
        R_left=np.eye(3),
        R_right=np.eye(3),
        exact=False,
        coherent=True,
        source="pq",
    )
    assert _use_exact_grain_generation(embedding) is False


def test_use_exact_grain_generation_incoherent_embedding_is_false():
    embedding = BoundaryEmbedding(
        P=np.eye(3, dtype=int),
        Q=np.eye(3, dtype=int),
        R_left=np.eye(3),
        R_right=np.eye(3),
        exact=True,
        coherent=False,
        source="pq",
    )
    assert _use_exact_grain_generation(embedding) is False


def test_use_exact_grain_generation_exact_coherent_true():
    embedding = BoundaryEmbedding(
        P=np.eye(3, dtype=int),
        Q=np.eye(3, dtype=int),
        R_left=np.eye(3),
        R_right=np.eye(3),
        exact=True,
        coherent=True,
        source="pq",
    )
    assert _use_exact_grain_generation(embedding) is True


def test_use_exact_grain_generation_exact_coherent_missing_pq_raises():
    # A real BoundaryEmbedding cannot itself be constructed exact=True with a missing
    # P/Q (BoundarySpec enforces that invariant at construction), so this defensive
    # guard -- preserved verbatim from GBMaker.py's __use_exact_grain_generation -- is
    # exercised with a minimal stand-in carrying just the attributes this function
    # reads.
    class _FakeEmbedding:
        exact = True
        coherent = True
        P = None
        Q = None

    with pytest.raises(GBMakerConstructionValueError):
        _use_exact_grain_generation(_FakeEmbedding())


# --------------------------------------------------------------------------------------
# _grain_strain_scales
# --------------------------------------------------------------------------------------


def test_grain_strain_scales_no_accommodation_defaults_to_one():
    assert _grain_strain_scales("left", {}) == (1.0, 1.0)
    assert _grain_strain_scales("right", {}) == (1.0, 1.0)


def test_grain_strain_scales_uses_axis_accommodation():
    accommodation = {"y": _accommodation(), "z": _accommodation(left_scale=1.5, right_scale=1.2)}
    assert _grain_strain_scales("left", accommodation) == (3.0, 1.5)
    assert _grain_strain_scales("right", accommodation) == (2.0, 1.2)


def test_grain_strain_scales_rejects_invalid_grain_side():
    with pytest.raises(GBMakerConstructionValueError):
        _grain_strain_scales("up", {})


# --------------------------------------------------------------------------------------
# _select_gb_region
# --------------------------------------------------------------------------------------


def test_select_gb_region_selects_atoms_within_window():
    left_atoms = _atoms([0.0, 3.0, 4.9])
    right_atoms = _atoms([5.1, 6.0, 9.0])
    region = _select_gb_region(
        left_atoms, right_atoms, vacuum_thickness=0.0, left_x=5.0, gb_thickness=1.0
    )
    assert sorted(region["x"].tolist()) == [4.9, 5.1]


# --------------------------------------------------------------------------------------
# _current_gap_metrics
# --------------------------------------------------------------------------------------


def test_current_gap_metrics_computes_central_and_periodic_gaps():
    left_atoms = _atoms([1.0, 2.0])
    right_atoms = _atoms([5.0, 8.0])
    left_bounds = np.array([0.0, 3.0])
    right_effective_bounds = np.array([4.0, 9.0])
    central_gap, periodic_gap, left_min_x, right_max_x = _current_gap_metrics(
        left_atoms, right_atoms, left_bounds, right_effective_bounds
    )
    assert central_gap == pytest.approx(3.0)  # 5.0 - 2.0
    assert periodic_gap == pytest.approx((9.0 - 8.0) + (1.0 - 0.0))
    assert left_min_x == 1.0
    assert right_max_x == 8.0


# --------------------------------------------------------------------------------------
# assemble_bicrystal -- exact-path central/periodic-gap validation
# --------------------------------------------------------------------------------------


def test_assemble_bicrystal_exact_path_rejects_negative_central_gap(monkeypatch):
    """Exact-path assembly must reject an overlapping central gap rather than delete
    atomic layers to fix it (issue #70's "exact paths ... accept nonnegative unequal
    projected central/periodic gaps" -- a negative gap is not accepted)."""
    from GBOpt.gbmaker import assembly as assembly_module

    def _fake_generate_exact_grains(**kwargs):
        # Overlapping grains: right grain starts before left grain ends.
        return _atoms([0.0, 1.0, 2.0]), _atoms([1.5, 2.5])

    monkeypatch.setattr(
        assembly_module, "_generate_exact_grains", _fake_generate_exact_grains
    )

    embedding = BoundaryEmbedding(
        P=np.eye(3, dtype=int),
        Q=np.eye(3, dtype=int),
        R_left=np.eye(3),
        R_right=np.eye(3),
        exact=True,
        coherent=True,
        source="pq",
    )

    with pytest.raises(GBMakerConstructionValueError):
        assemble_bicrystal(
            material=resolve_material_state(3.5, "fcc", "Ni"),
            embedding=embedding,
            R_left=np.eye(3),
            R_right=np.eye(3),
            left_periodic_miller_rows=np.eye(3, dtype=int),
            right_periodic_miller_rows=np.eye(3, dtype=int),
            left_x=2.0,
            right_x=2.0,
            x_dim=4.0,
            vacuum_thickness=0.0,
            inplane_periodic=(True, True),
            inplane_box_lengths=(10.0, 10.0),
            epsilon=1e-10,
            strain_accommodation={},
            gb_thickness=1.0,
            box_dims=np.array([[0.0, 4.0], [0.0, 10.0], [0.0, 10.0]]),
            normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
            gb_id=1,
        )


# --------------------------------------------------------------------------------------
# build_bicrystal / assemble_bicrystal -- end-to-end pipeline, cross-checked against a
# real GBMaker instance built from the same resolved construction knobs.
# --------------------------------------------------------------------------------------


def test_build_bicrystal_runs_without_a_gbmaker_instance():
    """Full construction executes without any GBMaker in sys.modules' live objects --
    covers issue #70's "full exact and approximate construction can be executed
    without instantiating GBMaker" acceptance criterion."""
    material = resolve_material_state(3.5, "fcc", "Ni")
    result = build_bicrystal(
        material=material,
        misorientation=np.array([0.0, 0.0, 0.0, 0.0, 45.0]),
        embedding=None,
        x_dim_min=20.0,
        vacuum_thickness=5.0,
        normal_topology=BoundaryNormalTopology.SINGLE_INTERFACE_SLAB,
        interaction_distance=10.0,
        gb_thickness=5.0,
        gb_id=1,
        epsilon=1e-10,
        repeat_factor=(2, 2),
        mismatch_tol=None,
        mismatch_max_cells=50,
        strain_grain="both",
    )
    assert result.atoms.shape[0] == result.left_atoms.shape[0] + result.right_atoms.shape[0]
    assert result.gb_id == 1
    assert result.normal_topology is BoundaryNormalTopology.SINGLE_INTERFACE_SLAB


def test_build_bicrystal_matches_legacy_gbmaker_construction_float_path():
    gb = GBMaker(
        3.5, "fcc", 5.0, np.array([0.0, 0.0, 0.0, 0.0, 45.0]), "Ni",
        vacuum=5.0, x_dim_min=20.0, interaction_distance=10.0,
    )
    result = _build_bicrystal_from_gbmaker(gb)

    assert np.array_equal(result.atoms, gb.whole_system)
    assert np.array_equal(result.left_atoms, gb.left_grain)
    assert np.array_equal(result.right_atoms, gb.right_grain)
    assert np.array_equal(result.gb_region_atoms, gb._GBMaker__gb_region)
    assert np.array_equal(result.box_dims, gb.box_dims)
    assert result.normal_topology is gb.normal_topology
    assert result.gb_id == gb.id


def test_build_bicrystal_matches_gbmaker_construction_exact_path():
    pq_spec = PQSpec(
        P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]],
        basis_mode="supplied",
    )
    gb = GBMaker.from_boundary_spec(
        3.5, "fcc", "Ni", pq_spec,
        mode="exact",
        gb_thickness=5.0,
        vacuum=5.0,
        x_dim_min=20.0,
        interaction_distance=10.0,
    )
    assert gb.uses_exact_construction

    result = _build_bicrystal_from_gbmaker(gb)

    assert np.array_equal(result.atoms, gb.whole_system)
    assert np.array_equal(result.left_atoms, gb.left_grain)
    assert np.array_equal(result.right_atoms, gb.right_grain)
    assert np.array_equal(result.gb_region_atoms, gb._GBMaker__gb_region)
    assert np.array_equal(result.box_dims, gb.box_dims)


# --------------------------------------------------------------------------------------
# Import-boundary regression: clean construction must not import optimization modules
# --------------------------------------------------------------------------------------


def test_gbmaker_construction_does_not_import_optimization_modules():
    """Issue #70's "import-boundary tests show clean construction does not import
    optimization modules" acceptance criterion. Uses a subprocess so this test's own
    (possibly already-populated) sys.modules cannot mask a real regression."""
    import subprocess

    code = (
        "import sys\n"
        "import GBOpt.gbmaker\n"
        "from GBOpt.gbmaker.assembly import build_bicrystal\n"
        "from GBOpt.GBMaker import GBMaker\n"
        "leaked = [m for m in sys.modules "
        "if m.startswith('GBOpt.optimization') or m.startswith('GBOpt.GBMinimizer')]\n"
        "assert leaked == [], leaked\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_gbmaker_package_does_not_import_optimizer_or_file_format_modules():
    """Issue #71's (R10) "architecture tests enforce that GBOpt.gbmaker does not import
    optimizer or file-format implementation modules" acceptance criterion.

    Unlike the sibling test above (which deliberately also imports GBOpt.GBMaker, the
    facade, to check *its* construction path), this isolates GBOpt.gbmaker the
    subpackage on its own. GBOpt/__init__.py unconditionally imports GBOpt.GBMaker
    (which itself imports GBOpt.io.lammps.data_writer for write_lammps), and Python
    always runs a package's __init__.py before any of its submodules -- so a plain
    "import GBOpt.gbmaker; assert 'GBOpt.io...' not in sys.modules" would fail
    regardless of what GBOpt.gbmaker itself imports. Stub out "GBOpt" with an empty
    module (its on-disk __path__ preserved via importlib.util.find_spec, which locates
    the package without executing it) before importing GBOpt.gbmaker, so
    GBOpt/__init__.py's body never runs in this subprocess and only GBOpt.gbmaker's own
    import graph is observed. Same technique as
    test_io_lammps_data_writer.py::test_data_writer_module_does_not_import_gbmaker
    (R12), per CLAUDE.md's "reuse the stub-parent-package technique for any 'not
    imported' target GBOpt/__init__.py eagerly imports" guidance.
    """
    import subprocess

    script = (
        "import importlib.util, sys, types\n"
        "spec = importlib.util.find_spec('GBOpt')\n"
        "stub = types.ModuleType('GBOpt')\n"
        "stub.__path__ = spec.submodule_search_locations\n"
        "sys.modules['GBOpt'] = stub\n"
        "import GBOpt.gbmaker\n"
        "leaked = [m for m in sys.modules if "
        "m.startswith('GBOpt.optimization') or m.startswith('GBOpt.GBMinimizer') "
        "or m.startswith('GBOpt.io')]\n"
        "assert leaked == [], leaked\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
