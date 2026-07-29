# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for strict topology-aware bicrystal feasibility validation."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from GBOpt.BicrystalState import (
    BicrystalState,
    InterfaceDescriptor,
    RegionDescriptor,
    SurfaceDescriptor,
)
from GBOpt.UnitCell import UnitCell
from GBOpt.geometry_validation import (
    ContactPolicy,
    FeasibilityOverride,
    FeasibilityPolicy,
    GeometryValidationError,
    SlabPolicy,
    SpeciesPairThresholds,
    VoidPolicy,
    validate_bicrystal_state,
)

ATOM_DTYPE = np.dtype(
    [("name", "U2"), ("x", float), ("y", float), ("z", float)]
)


def _atoms() -> np.ndarray:
    return np.array(
        [
            ("U", 1.0, 1.0, 1.0),
            ("U", 4.0, 1.0, 1.0),
            ("O", 1.0, 3.0, 3.0),
            ("O", 4.0, 3.0, 3.0),
            ("U", 6.0, 1.0, 1.0),
            ("U", 9.0, 1.0, 1.0),
            ("O", 6.0, 3.0, 3.0),
            ("O", 9.0, 3.0, 3.0),
        ],
        dtype=ATOM_DTYPE,
    )


def _periodic_state(
    *,
    atoms: np.ndarray | None = None,
    boundary_conditions: tuple[str, str, str] = ("periodic", "periodic", "periodic"),
    interfaces: tuple[InterfaceDescriptor, ...] | None = None,
) -> BicrystalState:
    selected = _atoms() if atoms is None else atoms
    if interfaces is None:
        interfaces = (
            InterfaceDescriptor(
                "central_gb", 0, "interior", 5.0, 0, 1, (1.0, 0.0, 0.0)
            ),
            InterfaceDescriptor(
                "periodic_gb",
                0,
                "periodic_boundary",
                0.0,
                1,
                0,
                (1.0, 0.0, 0.0),
                periodic_partner_position=10.0,
            ),
        )
    return BicrystalState(
        atoms=selected,
        box_dims=np.array(((0.0, 10.0), (0.0, 4.0), (0.0, 4.0))),
        topology="periodic_bicrystal",
        boundary_conditions=boundary_conditions,
        atom_ids=np.arange(1, len(selected) + 1, dtype=np.int64),
        grain_ids=np.array((0, 0, 0, 0, 1, 1, 1, 1), dtype=np.int8),
        interfaces=interfaces,
    )


def _slab_state(*, atoms: np.ndarray | None = None, **overrides) -> BicrystalState:
    selected = _atoms() if atoms is None else atoms
    selected = selected.copy()
    if atoms is None:
        selected["x"] += 2.0
    kwargs = {
        "atoms": selected,
        "box_dims": np.array(((0.0, 14.0), (0.0, 4.0), (0.0, 4.0))),
        "topology": "single_interface_slab",
        "boundary_conditions": ("fixed", "periodic", "periodic"),
        "atom_ids": np.arange(1, len(selected) + 1, dtype=np.int64),
        "grain_ids": np.array((0, 0, 0, 0, 1, 1, 1, 1), dtype=np.int8),
        "interfaces": (
            InterfaceDescriptor(
                "central_gb", 0, "interior", 7.0, 0, 1, (1.0, 0.0, 0.0)
            ),
        ),
        "external_surfaces": (
            SurfaceDescriptor("left_surface", 0, 2.0, (-1.0, 0.0, 0.0), (0,)),
            SurfaceDescriptor("right_surface", 0, 12.0, (1.0, 0.0, 0.0), (1,)),
        ),
        "vacuum_regions": (
            RegionDescriptor("lower_vacuum", "vacuum", 0, 0.0, 2.0),
            RegionDescriptor("upper_vacuum", "vacuum", 0, 12.0, 14.0),
        ),
    }
    kwargs.update(overrides)
    return BicrystalState(**kwargs)


def _policy(
    *,
    hard: float = 0.5,
    warning: float = 1.0,
    void: VoidPolicy | None = None,
    slab: SlabPolicy | None = None,
) -> FeasibilityPolicy:
    thresholds = tuple(
        SpeciesPairThresholds(pair, 1.0e-6, hard, warning)
        for pair in (("O", "O"), ("O", "U"), ("U", "U"))
    )
    return FeasibilityPolicy(
        contact=ContactPolicy(pair_thresholds=thresholds),
        void=VoidPolicy() if void is None else void,
        slab=SlabPolicy() if slab is None else slab,
    )


def _reason_codes(report) -> set[str]:
    return {reason.code for reason in report.raw_reasons}


def test_known_control_is_feasible_and_unchanged() -> None:
    state = _periodic_state()
    before = state.structure_hash

    report = validate_bicrystal_state(state, policy=_policy())

    assert report.status == "feasible"
    assert report.raw_status == "feasible"
    assert len(report.interfaces) == 2
    assert report.slab is None
    assert state.structure_hash == before


def test_coincident_cross_interface_pair_is_invalid() -> None:
    atoms = _atoms()
    atoms[4]["x"] = 4.0

    report = validate_bicrystal_state(_periodic_state(atoms=atoms), policy=_policy())

    assert report.status == "invalid"
    assert "structure.periodic_duplicate_representatives" in _reason_codes(report)
    assert "interface.cross_contact_duplicate" in _reason_codes(report)


def test_severe_noncoincident_contact_is_infeasible() -> None:
    atoms = _atoms()
    atoms[4]["x"] = 4.2

    report = validate_bicrystal_state(_periodic_state(atoms=atoms), policy=_policy())

    assert report.status == "infeasible"
    assert "interface.cross_contact_below_hard_minimum" in _reason_codes(report)
    assert not report.duplicate_pairs


def test_warning_level_contact_is_warning() -> None:
    atoms = _atoms()
    atoms[4]["x"] = 4.8

    report = validate_bicrystal_state(_periodic_state(atoms=atoms), policy=_policy())

    assert report.status == "warning"
    assert "interface.cross_contact_below_warning_minimum" in _reason_codes(report)


def test_failure_at_only_periodic_interface_rejects_bicrystal() -> None:
    atoms = _atoms()
    atoms[1]["x"] = 0.2
    atoms[5]["x"] = 9.9

    report = validate_bicrystal_state(_periodic_state(atoms=atoms), policy=_policy())

    assert report.status == "infeasible"
    failing = {
        reason.descriptor_id
        for reason in report.raw_reasons
        if reason.code == "interface.cross_contact_below_hard_minimum"
    }
    assert failing == {"periodic_gb"}


def test_localized_channel_fails_local_void_policy() -> None:
    atoms = _atoms()
    atoms[4]["x"] = 8.5
    policy = _policy(
        void=VoidPolicy(
            hard_max_gap_range_bulk_factor=0.5,
            warning_max_gap_range_bulk_factor=0.25,
            hard_max_p95_bulk_factor=10.0,
            warning_max_p95_bulk_factor=9.0,
        )
    )

    report = validate_bicrystal_state(_periodic_state(atoms=atoms), policy=policy)

    assert report.status == "infeasible"
    assert "interface.local_gap_range_exceeded" in _reason_codes(report)


def test_periodic_duplicate_uses_only_declared_periodic_axes() -> None:
    atoms = _atoms()
    atoms[2]["y"] = 0.0
    atoms[3]["x"] = atoms[2]["x"]
    atoms[3]["y"] = 4.0
    atoms[3]["z"] = atoms[2]["z"]
    periodic = validate_bicrystal_state(_periodic_state(atoms=atoms), policy=_policy())
    fixed = validate_bicrystal_state(
        _periodic_state(atoms=atoms, boundary_conditions=("periodic", "fixed", "periodic")),
        policy=_policy(),
    )

    assert periodic.status == "invalid"
    assert fixed.status != "invalid"


def test_atom_order_does_not_change_metrics_or_decision() -> None:
    state = _periodic_state()
    order = np.array((6, 0, 5, 3, 7, 1, 4, 2))
    reordered = BicrystalState(
        atoms=state.atoms[order],
        box_dims=state.box_dims,
        topology=state.topology,
        boundary_conditions=state.boundary_conditions,
        atom_ids=state.atom_ids[order],
        grain_ids=state.grain_ids[order],
        interfaces=state.interfaces,
    )

    first = validate_bicrystal_state(state, policy=_policy())
    second = validate_bicrystal_state(reordered, policy=_policy())

    assert first.status == second.status
    assert [asdict(item) for item in first.interfaces] == [
        asdict(item) for item in second.interfaces
    ]


def test_periodic_wrapping_does_not_change_metrics_or_decision() -> None:
    state = _periodic_state()
    wrapped_atoms = state.atoms.copy()
    wrapped_atoms["y"] += 4.0
    wrapped_atoms["z"] -= 8.0
    wrapped = BicrystalState(
        atoms=wrapped_atoms,
        box_dims=np.array(((0.0, 10.0), (4.0, 8.0), (-8.0, -4.0))),
        topology=state.topology,
        boundary_conditions=state.boundary_conditions,
        atom_ids=state.atom_ids,
        grain_ids=state.grain_ids,
        interfaces=state.interfaces,
    )

    first = validate_bicrystal_state(state, policy=_policy())
    second = validate_bicrystal_state(wrapped, policy=_policy())

    assert first.status == second.status
    assert [item.gap_statistics for item in first.interfaces] == [
        item.gap_statistics for item in second.interfaces
    ]


def test_slab_vacuum_is_not_treated_as_gb_void() -> None:
    report = validate_bicrystal_state(_slab_state(), policy=_policy())

    assert report.status == "feasible"
    assert len(report.interfaces) == 1
    assert report.interfaces[0].interface_id == "central_gb"
    assert report.slab is not None
    assert [item.atom_count for item in report.slab.vacuum_regions] == [0, 0]


def test_atoms_in_declared_vacuum_fail_slab_rule_not_gb_void() -> None:
    atoms = _atoms()
    atoms["x"] += 2.0
    atoms[0]["x"] = 1.0

    report = validate_bicrystal_state(_slab_state(atoms=atoms), policy=_policy())

    assert report.status == "infeasible"
    assert "vacuum.contains_atoms" in _reason_codes(report)
    assert all(
        reason.code != "interface.local_gap_p95_exceeded"
        for reason in report.raw_reasons
        if reason.descriptor_id == "lower_vacuum"
    )


def test_fixed_and_buffer_regions_have_independent_rules() -> None:
    state = _slab_state(
        fixed_regions=(RegionDescriptor("fixed", "fixed", 0, 2.0, 2.2, (0,)),),
        buffer_regions=(RegionDescriptor("buffer", "buffer", 0, 11.8, 12.0, (1,)),),
    )
    policy = _policy(
        slab=SlabPolicy(
            minimum_fixed_region_thickness_angstrom=0.5,
            minimum_buffer_thickness_angstrom=0.5,
        )
    )

    report = validate_bicrystal_state(state, policy=policy)

    assert report.status == "infeasible"
    assert "fixed_region.thickness_below_minimum" in _reason_codes(report)
    assert "buffer_region.thickness_below_minimum" in _reason_codes(report)


def test_override_requires_reason_and_cannot_override_invalid() -> None:
    with pytest.raises(GeometryValidationError, match="non-empty reason"):
        FeasibilityOverride("feasible", "")

    atoms = _atoms()
    atoms[4]["x"] = 4.0
    with pytest.raises(GeometryValidationError, match="invalid result"):
        validate_bicrystal_state(
            _periodic_state(atoms=atoms),
            policy=_policy(),
            override=FeasibilityOverride("warning", "accepted for comparison"),
        )


def test_override_preserves_raw_metrics_and_status() -> None:
    atoms = _atoms()
    atoms[4]["x"] = 4.2
    state = _periodic_state(atoms=atoms)

    raw = validate_bicrystal_state(state, policy=_policy())
    overridden = validate_bicrystal_state(
        state,
        policy=_policy(),
        override=FeasibilityOverride("warning", "expert-reviewed seed"),
    )

    assert overridden.status == "warning"
    assert overridden.raw_status == "infeasible"
    assert overridden.interfaces == raw.interfaces
    assert overridden.raw_reasons == raw.raw_reasons
    assert "override.applied" in {reason.code for reason in overridden.reasons}


def test_policy_and_report_serialization_are_deterministic() -> None:
    policy = _policy()
    first = validate_bicrystal_state(_periodic_state(), policy=policy)
    second = validate_bicrystal_state(_periodic_state(), policy=policy)

    assert policy.to_json() == _policy().to_json()
    assert policy.policy_hash == _policy().policy_hash
    assert first.to_json() == second.to_json()
    assert first.report_hash == second.report_hash


def test_contact_threshold_equality_is_not_a_hard_failure() -> None:
    atoms = _atoms()
    atoms[4]["x"] = 4.5

    report = validate_bicrystal_state(
        _periodic_state(atoms=atoms), policy=_policy(hard=0.5, warning=0.5)
    )

    assert "interface.cross_contact_below_hard_minimum" not in _reason_codes(report)


def test_unit_cell_policy_builds_species_pair_thresholds() -> None:
    cell = UnitCell()
    cell.init_by_structure("fluorite", 5.454, ("U", "O"))

    policy = FeasibilityPolicy.from_unit_cell(cell)

    pairs = {item.species for item in policy.contact.pair_thresholds}
    assert pairs == {("O", "O"), ("O", "U"), ("U", "U")}
    assert all(item.hard_minimum_angstrom > 0.0 for item in policy.contact.pair_thresholds)


def test_zero_population_requirements_disable_region_population_failures() -> None:
    state = _slab_state(
        fixed_regions=(RegionDescriptor("fixed", "fixed", 0, 2.0, 2.1, (0,)),),
        buffer_regions=(RegionDescriptor("buffer", "buffer", 0, 11.9, 12.0, (1,)),),
    )

    report = validate_bicrystal_state(state, policy=_policy())

    assert "fixed_region.population_below_minimum" not in _reason_codes(report)
    assert "buffer_region.population_below_minimum" not in _reason_codes(report)


def test_mislabeled_fixed_region_is_invalid() -> None:
    state = _slab_state(
        fixed_regions=(RegionDescriptor("fixed", "buffer", 0, 2.0, 3.0, (0,)),)
    )

    report = validate_bicrystal_state(state, policy=_policy())

    assert report.status == "invalid"
    assert "region.fixed_kind_mismatch" in _reason_codes(report)


def test_inconsistent_periodic_interface_grain_order_is_invalid() -> None:
    interfaces = (
        InterfaceDescriptor("central_gb", 0, "interior", 5.0, 0, 1, (1.0, 0.0, 0.0)),
        InterfaceDescriptor(
            "periodic_gb",
            0,
            "periodic_boundary",
            0.0,
            0,
            1,
            (1.0, 0.0, 0.0),
            periodic_partner_position=10.0,
        ),
    )

    report = validate_bicrystal_state(
        _periodic_state(interfaces=interfaces), policy=_policy()
    )

    assert report.status == "invalid"
    assert "topology.periodic_interface_grain_order" in _reason_codes(report)


def test_vacuum_overlap_with_fixed_region_is_invalid() -> None:
    state = _slab_state(
        fixed_regions=(RegionDescriptor("fixed", "fixed", 0, 1.0, 3.0, (0,)),)
    )

    report = validate_bicrystal_state(state, policy=_policy())

    assert report.status == "invalid"
    assert "region.vacuum_solid_overlap" in _reason_codes(report)


def test_vacuum_surface_normal_inconsistency_is_invalid() -> None:
    state = _slab_state(
        external_surfaces=(
            SurfaceDescriptor("left_surface", 0, 2.0, (1.0, 0.0, 0.0), (0,)),
            SurfaceDescriptor("right_surface", 0, 12.0, (1.0, 0.0, 0.0), (1,)),
        )
    )

    report = validate_bicrystal_state(state, policy=_policy())

    assert report.status == "invalid"
    assert "vacuum.surface_normal_inconsistent" in _reason_codes(report)
