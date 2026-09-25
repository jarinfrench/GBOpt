# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for GBOpt.manipulation.crossover: crossover_slice_and_merge, SliceAndMerge."""

from dataclasses import dataclass, field

import numpy as np
import pytest

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
from GBOpt.interface.model import InterfaceCandidate
from GBOpt.manipulation.crossover import SliceAndMerge, crossover_slice_and_merge
from GBOpt.manipulation.types import (
    ManipulationArityError,
    ManipulationCapabilityError,
    ManipulationCompatibilityError,
    ManipulationConfigurationError,
    ManipulationContext,
)
from GBOpt.UnitCell import UnitCell

_BOX_DIMS = np.asarray([[0.0, 20.0], [0.0, 10.0], [0.0, 10.0]], dtype=float)
_GB_PLANE_X = 10.0
_LEFT_BOUNDS = (0.0, 10.0)
_RIGHT_BOUNDS = (10.0, 20.0)
_TOLERANCE = 1.0e-8
_GB_THICKNESS = 10.0


def _unit_cell() -> UnitCell:
    unit_cell = UnitCell()
    unit_cell.init_by_structure("fcc", 4.05, "Al")
    return unit_cell


def _atoms(offset: float = 0.0) -> np.ndarray:
    return np.asarray(
        [
            ("Al", 3.0 + offset, 1.0, 1.0),
            ("Al", 7.5 + offset, 2.0, 3.0),
            ("Al", 12.5 + offset, 4.0, 5.0),
            ("Al", 17.0 + offset, 6.0, 7.0),
        ],
        dtype=Atom.atom_dtype,
    )


def _labels() -> np.ndarray:
    return np.asarray(
        [LEFT_GRAIN_LABEL, LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL, RIGHT_GRAIN_LABEL],
        dtype=np.int8,
    )


class _FixedRandom:
    """Duck-typed rng exposing only ``.random()``, matching legacy test doubles."""

    def __init__(self, value: float = 0.5) -> None:
        self._value = value

    def random(self) -> float:
        return self._value


@dataclass
class _FakeParent:
    """Duck-typed ``CrossoverParent`` standing in for a legacy ``Parent``."""

    whole_system: np.ndarray
    box_dims: np.ndarray = field(default_factory=lambda: _BOX_DIMS.copy())
    gb_plane_x: float = _GB_PLANE_X
    gb_thickness: float = _GB_THICKNESS
    left_grain_x_bounds: np.ndarray = field(default_factory=lambda: np.asarray(_LEFT_BOUNDS))
    right_grain_x_bounds: np.ndarray = field(default_factory=lambda: np.asarray(_RIGHT_BOUNDS))
    grain_labels: np.ndarray | None = None
    inplane_periodic: tuple = (True, True)
    normal_topology: object = BoundaryNormalTopology.PERIODIC_BICRYSTAL
    coordinate_tolerance: float = _TOLERANCE
    unit_cell: object = field(default_factory=_unit_cell)


def _make_candidate(*, atoms=None, **overrides) -> InterfaceCandidate:
    kwargs = {
        "atoms": _atoms() if atoms is None else atoms,
        "box_dims": _BOX_DIMS,
        "gb_plane_x": _GB_PLANE_X,
        "left_grain_x_bounds": _LEFT_BOUNDS,
        "right_grain_x_bounds": _RIGHT_BOUNDS,
        "grain_labels": _labels(),
        "inplane_periodic": (True, True),
        "normal_topology": BoundaryNormalTopology.PERIODIC_BICRYSTAL,
        "coordinate_tolerance": _TOLERANCE,
        "interface_separation": 0.0,
    }
    kwargs.update(overrides)
    return InterfaceCandidate(**kwargs)


def _context(parent1, parent2, *, rng=None, **params) -> ManipulationContext:
    return ManipulationContext(
        parents=(parent1, parent2),
        rng=np.random.default_rng(0) if rng is None else rng,
        params=params,
    )


# ---------------------------------------------------------------------------
# crossover_slice_and_merge (pure function), duck-typed Parent-like inputs
# ---------------------------------------------------------------------------


def test_pure_function_unowned_parents_skip_topology_checks_but_validate_composition():
    first = _FakeParent(whole_system=_atoms(), grain_labels=None)
    second = _FakeParent(
        whole_system=_atoms(),
        # Mismatched topology would be rejected in owned mode; unowned mode never
        # inspects it.
        inplane_periodic=(True, False),
        grain_labels=None,
    )
    new_atoms, child_labels, provenance = crossover_slice_and_merge(
        first,
        second,
        surface_mode="normal_plane",
        max_tilt_degrees=5.0,
        rng=_FixedRandom(0.5),
    )
    assert child_labels is None
    assert provenance["surface_mode"] == "normal_plane"
    assert len(new_atoms) == len(first.whole_system)


def test_pure_function_owned_parents_produce_masked_labels():
    first = _FakeParent(whole_system=_atoms(), grain_labels=_labels())
    second = _FakeParent(whole_system=_atoms(2.0), grain_labels=_labels())
    new_atoms, child_labels, _provenance = crossover_slice_and_merge(
        first,
        second,
        surface_mode="normal_plane",
        max_tilt_degrees=5.0,
        rng=_FixedRandom(0.5),
    )
    assert child_labels is not None
    assert len(child_labels) == len(new_atoms)
    assert set(np.unique(child_labels).tolist()) <= {LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL}


def test_pure_function_rejects_mismatched_ownership_mode():
    first = _FakeParent(whole_system=_atoms(), grain_labels=_labels())
    second = _FakeParent(whole_system=_atoms(), grain_labels=None)
    with pytest.raises(ManipulationCompatibilityError, match="ownership mode"):
        crossover_slice_and_merge(
            first,
            second,
            surface_mode="normal_plane",
            max_tilt_degrees=5.0,
            rng=_FixedRandom(0.5),
        )


def test_pure_function_rejects_mismatched_topology_when_owned():
    first = _FakeParent(whole_system=_atoms(), grain_labels=_labels())
    second = _FakeParent(
        whole_system=_atoms(), grain_labels=_labels(), inplane_periodic=(True, False)
    )
    with pytest.raises(ManipulationCompatibilityError, match="matching boundary topology"):
        crossover_slice_and_merge(
            first,
            second,
            surface_mode="normal_plane",
            max_tilt_degrees=5.0,
            rng=_FixedRandom(0.5),
        )


def test_pure_function_rejects_non_affine_equivalent_geometry_when_owned():
    first = _FakeParent(whole_system=_atoms(), grain_labels=_labels())
    second = _FakeParent(
        whole_system=_atoms(), grain_labels=_labels(), gb_plane_x=11.0
    )
    with pytest.raises(
        ManipulationCompatibilityError, match="affine-equivalent physical grain"
    ):
        crossover_slice_and_merge(
            first,
            second,
            surface_mode="normal_plane",
            max_tilt_degrees=5.0,
            rng=_FixedRandom(0.5),
        )


def test_pure_function_rejects_mismatched_formula_vectors():
    other_unit_cell = UnitCell()
    other_unit_cell.init_by_structure("bcc", 3.3, "W")
    second_atoms = np.asarray(
        [
            ("W", 3.0, 1.0, 1.0),
            ("W", 7.5, 2.0, 3.0),
            ("W", 12.5, 4.0, 5.0),
            ("W", 17.0, 6.0, 7.0),
        ],
        dtype=Atom.atom_dtype,
    )
    first = _FakeParent(whole_system=_atoms(), grain_labels=_labels())
    second = _FakeParent(
        whole_system=second_atoms,
        grain_labels=_labels(),
        unit_cell=other_unit_cell,
    )
    with pytest.raises(ManipulationCapabilityError, match="different normalized formula"):
        crossover_slice_and_merge(
            first,
            second,
            surface_mode="normal_plane",
            max_tilt_degrees=5.0,
            rng=_FixedRandom(0.5),
        )


def test_pure_function_rejects_bad_surface_mode():
    first = _FakeParent(whole_system=_atoms(), grain_labels=None)
    second = _FakeParent(whole_system=_atoms(), grain_labels=None)
    with pytest.raises(ManipulationConfigurationError, match="surface_mode"):
        crossover_slice_and_merge(
            first,
            second,
            surface_mode="tilted_plane",
            max_tilt_degrees=5.0,
            rng=_FixedRandom(0.5),
        )


@pytest.mark.parametrize(
    "tilt",
    [-1.0, 90.0],
    ids=["negative", "right-angle"],
)
def test_pure_function_rejects_out_of_range_tilt(tilt):
    first = _FakeParent(whole_system=_atoms(), grain_labels=None)
    second = _FakeParent(whole_system=_atoms(), grain_labels=None)
    with pytest.raises(ManipulationConfigurationError, match="max_tilt_degrees"):
        crossover_slice_and_merge(
            first,
            second,
            surface_mode="periodic_wave",
            max_tilt_degrees=tilt,
            rng=_FixedRandom(0.5),
        )


def test_pure_function_rejects_non_real_tilt():
    first = _FakeParent(whole_system=_atoms(), grain_labels=None)
    second = _FakeParent(whole_system=_atoms(), grain_labels=None)
    with pytest.raises(TypeError, match="finite real"):
        crossover_slice_and_merge(
            first,
            second,
            surface_mode="normal_plane",
            max_tilt_degrees=True,
            rng=_FixedRandom(0.5),
        )


def test_pure_function_does_not_mutate_parent_atoms():
    first_atoms = _atoms()
    second_atoms = _atoms(2.0)
    first = _FakeParent(whole_system=first_atoms, grain_labels=None)
    second = _FakeParent(whole_system=second_atoms, grain_labels=None)
    crossover_slice_and_merge(
        first,
        second,
        surface_mode="normal_plane",
        max_tilt_degrees=5.0,
        rng=_FixedRandom(0.5),
    )
    np.testing.assert_array_equal(first.whole_system, first_atoms)
    np.testing.assert_array_equal(second.whole_system, second_atoms)


def test_pure_function_fixed_rng_deterministic_offset():
    first = _FakeParent(whole_system=_atoms(), grain_labels=None)
    second = _FakeParent(whole_system=_atoms(), grain_labels=None)
    _atoms1, _labels1, provenance1 = crossover_slice_and_merge(
        first,
        second,
        surface_mode="normal_plane",
        max_tilt_degrees=5.0,
        rng=_FixedRandom(0.5),
    )
    _atoms2, _labels2, provenance2 = crossover_slice_and_merge(
        first,
        second,
        surface_mode="normal_plane",
        max_tilt_degrees=5.0,
        rng=_FixedRandom(0.5),
    )
    assert provenance1 == provenance2


# ---------------------------------------------------------------------------
# SliceAndMerge (Manipulation operation), InterfaceCandidate-typed boundary
# ---------------------------------------------------------------------------


def test_name_and_arity():
    operation = SliceAndMerge()
    assert operation.name == "slice_and_merge"
    assert operation.arity == 2


def test_execute_requires_exactly_two_parents():
    parent = _make_candidate()
    single_parent_context = ManipulationContext(
        parents=(parent,),
        rng=np.random.default_rng(0),
        params={"unit_cell": (_unit_cell(), _unit_cell()), "gb_thickness": _GB_THICKNESS},
    )
    with pytest.raises(ManipulationArityError):
        SliceAndMerge().execute(single_parent_context)

    three_parent_context = ManipulationContext(
        parents=(parent, parent, parent),
        rng=np.random.default_rng(0),
        params={"unit_cell": (_unit_cell(), _unit_cell()), "gb_thickness": _GB_THICKNESS},
    )
    with pytest.raises(ManipulationArityError):
        SliceAndMerge().execute(three_parent_context)


def test_execute_requires_unit_cell_and_gb_thickness():
    parent1 = _make_candidate()
    parent2 = _make_candidate(atoms=_atoms(2.0))
    with pytest.raises(ManipulationConfigurationError, match="unit_cell"):
        SliceAndMerge().execute(_context(parent1, parent2, gb_thickness=_GB_THICKNESS))
    with pytest.raises(ManipulationConfigurationError, match="gb_thickness"):
        SliceAndMerge().execute(
            _context(parent1, parent2, unit_cell=(_unit_cell(), _unit_cell()))
        )


def test_execute_produces_labeled_child_with_preserved_geometry():
    parent1 = _make_candidate()
    parent2 = _make_candidate(atoms=_atoms(2.0))
    result = SliceAndMerge().execute(
        _context(
            parent1,
            parent2,
            rng=np.random.default_rng(0),
            unit_cell=(_unit_cell(), _unit_cell()),
            gb_thickness=_GB_THICKNESS,
        )
    )
    child = result.children[0]
    assert child.grain_labels is not None
    np.testing.assert_allclose(child.box_dims, parent1.box_dims)
    assert child.gb_plane_x == pytest.approx(parent1.gb_plane_x)
    assert child.normal_topology is parent1.normal_topology
    assert child.inplane_periodic == parent1.inplane_periodic


def test_execute_records_lineage_and_parameters():
    parent1 = _make_candidate()
    parent2 = _make_candidate(atoms=_atoms(2.0))
    result = SliceAndMerge().execute(
        _context(
            parent1,
            parent2,
            rng=np.random.default_rng(0),
            unit_cell=(_unit_cell(), _unit_cell()),
            gb_thickness=_GB_THICKNESS,
        )
    )
    assert dict(result.lineage) == {"operation": "slice_and_merge", "parent_count": 2}
    assert set(result.parameters) == {
        "surface_mode",
        "max_tilt_degrees",
        "amplitude_y",
        "amplitude_z",
        "phase_y",
        "phase_z",
        "offset",
    }


def test_execute_does_not_mutate_parents():
    parent1 = _make_candidate()
    parent2 = _make_candidate(atoms=_atoms(2.0))
    parent1_atoms_before = parent1.atoms.copy()
    parent2_atoms_before = parent2.atoms.copy()
    SliceAndMerge().execute(
        _context(
            parent1,
            parent2,
            rng=np.random.default_rng(0),
            unit_cell=(_unit_cell(), _unit_cell()),
            gb_thickness=_GB_THICKNESS,
        )
    )
    np.testing.assert_array_equal(parent1.atoms, parent1_atoms_before)
    np.testing.assert_array_equal(parent2.atoms, parent2_atoms_before)


def test_execute_preflight_validation_always_runs():
    """Unlike the legacy no-ownership path, InterfaceCandidate parents always carry
    real grain labels, so the owned-mode preflight checks always execute here."""
    parent1 = _make_candidate()
    parent2 = _make_candidate(atoms=_atoms(2.0), inplane_periodic=(True, False))
    with pytest.raises(ManipulationCompatibilityError, match="matching boundary topology"):
        SliceAndMerge().execute(
            _context(
                parent1,
                parent2,
                unit_cell=(_unit_cell(), _unit_cell()),
                gb_thickness=_GB_THICKNESS,
            )
        )
