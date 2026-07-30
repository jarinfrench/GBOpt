# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for deterministic feasible rigid-translation seed generation."""

from __future__ import annotations

import json

import numpy as np
import pytest

import GBOpt.interface_initialization as initialization
from GBOpt.BicrystalState import (
    TRANSLATION_HISTORY_KEY,
    BicrystalState,
    InterfaceDescriptor,
    RegionDescriptor,
    SurfaceDescriptor,
)
from GBOpt.geometry_validation import (
    ContactPolicy,
    FeasibilityOverride,
    FeasibilityPolicy,
    SpeciesPairThresholds,
    validate_bicrystal_state,
)
from GBOpt.interface_initialization import (
    CartesianTranslationDomain,
    InterfaceInitializationError,
    InterfaceInitializer,
    generate_translation_seeds,
)

ATOM_DTYPE = np.dtype(
    [("name", "U2"), ("x", np.float64), ("y", np.float64), ("z", np.float64)]
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
    box_dims: np.ndarray | None = None,
) -> BicrystalState:
    selected = _atoms() if atoms is None else atoms
    box = (
        np.array(((0.0, 10.0), (0.0, 4.0), (0.0, 4.0)))
        if box_dims is None
        else box_dims
    )
    return BicrystalState(
        atoms=selected,
        box_dims=box,
        topology="periodic_bicrystal",
        boundary_conditions=("periodic", "periodic", "periodic"),
        atom_ids=np.arange(1, len(selected) + 1, dtype=np.int64),
        grain_ids=np.array((0, 0, 0, 0, 1, 1, 1, 1), dtype=np.int8),
        interfaces=(
            InterfaceDescriptor(
                "central_gb", 0, "interior", float(np.mean(box[0])), 0, 1, (1.0, 0.0, 0.0)
            ),
            InterfaceDescriptor(
                "periodic_gb",
                0,
                "periodic_boundary",
                float(box[0, 0]),
                1,
                0,
                (1.0, 0.0, 0.0),
                periodic_partner_position=float(box[0, 1]),
            ),
        ),
        termination_ids=(2, 4),
        metadata={"case_id": "synthetic_periodic", "source": {"row": 7}},
    )


def _slab_state() -> BicrystalState:
    atoms = _atoms().copy()
    atoms["x"] += 2.0
    return BicrystalState(
        atoms=atoms,
        box_dims=np.array(((0.0, 14.0), (0.0, 4.0), (0.0, 4.0))),
        topology="single_interface_slab",
        boundary_conditions=("fixed", "periodic", "periodic"),
        atom_ids=np.arange(1, len(atoms) + 1, dtype=np.int64),
        grain_ids=np.array((0, 0, 0, 0, 1, 1, 1, 1), dtype=np.int8),
        interfaces=(
            InterfaceDescriptor(
                "central_gb", 0, "interior", 7.0, 0, 1, (1.0, 0.0, 0.0)
            ),
        ),
        external_surfaces=(
            SurfaceDescriptor("left_surface", 0, 2.0, (-1.0, 0.0, 0.0), (0,)),
            SurfaceDescriptor("right_surface", 0, 12.0, (1.0, 0.0, 0.0), (1,)),
        ),
        vacuum_regions=(
            RegionDescriptor("lower_vacuum", "vacuum", 0, 0.0, 2.0),
            RegionDescriptor("upper_vacuum", "vacuum", 0, 12.0, 14.0),
        ),
        termination_ids=(3, 5),
        metadata={"case_id": "synthetic_slab", "source": {"row": 8}},
    )


def _policy(*, hard: float = 0.5, warning: float | None = None) -> FeasibilityPolicy:
    warning_value = hard if warning is None else warning
    thresholds = tuple(
        SpeciesPairThresholds(pair, 1.0e-6, hard, warning_value)
        for pair in (("O", "O"), ("O", "U"), ("U", "U"))
    )
    return FeasibilityPolicy(contact=ContactPolicy(pair_thresholds=thresholds))


def _domain(
    first=(0.0,),
    second=(0.0,),
    normal=(0.0,),
    **kwargs,
) -> CartesianTranslationDomain:
    return CartesianTranslationDomain(
        in_plane_components=(tuple(first), tuple(second)),
        normal_offsets=tuple(normal),
        **kwargs,
    )


def test_zero_translation_is_original_state_and_first_seed() -> None:
    state = _periodic_state()
    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(),
        max_seeds=1,
    )

    assert result.status == "seed_limit_reached"
    assert result.zero_translation_accepted
    assert result.seeds[0].state is state
    assert result.seeds[0].candidate.kind == "zero"
    assert result.attempts[0].candidate.order == 0
    assert result.attempts[0].disposition == "retained"


def test_reason_bearing_override_is_applied_and_hashed_in_phase6_result() -> None:
    override = FeasibilityOverride("feasible", "approved synthetic contact exception")
    result = InterfaceInitializer(
        _periodic_state(),
        _policy(hard=10.0),
        feasibility_override=override,
    ).generate_translation_seeds(
        translation_domain=_domain(),
        max_seeds=1,
    )

    assert result.seeds[0].report.raw_status == "infeasible"
    assert result.seeds[0].report.status == "feasible"
    assert result.feasibility_override == override
    assert result.to_dict()["feasibility_override"] == {
        "status": "feasible",
        "reason": "approved synthetic contact exception",
    }


def test_source_state_is_immutable_through_full_enumeration() -> None:
    state = _periodic_state()
    atoms_before = state.atoms.copy()
    manifest_before = state.manifest()

    InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0), second=(0.0, 1.0), normal=(0.0, 1.0)),
        max_seeds=20,
    )

    np.testing.assert_array_equal(state.atoms, atoms_before)
    assert state.manifest() == manifest_before


def test_every_nonzero_candidate_calls_phase5_primitive(monkeypatch) -> None:
    state = _periodic_state()
    domain = _domain(first=(0.0, 1.0), second=(0.0, 2.0), normal=(0.0, 0.5))
    expected_nonzero = len(domain.resolve_for(state).ordered_candidates()) - 1
    original = initialization.translate_grain
    calls = []

    def recording_translate(*args, **kwargs):
        calls.append(kwargs["displacement"])
        return original(*args, **kwargs)

    monkeypatch.setattr(initialization, "translate_grain", recording_translate)
    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=domain,
        max_seeds=100,
    )

    assert len(calls) == expected_nonzero
    assert calls == [
        candidate.candidate.displacement_lab for candidate in result.attempts[1:]
    ]


def test_cartesian_in_plane_order_is_norm_then_lexicographic() -> None:
    state = _periodic_state()
    candidates = _domain(
        first=(2.0, 0.0, -1.0),
        second=(1.0, 0.0),
    ).resolve_for(state).ordered_candidates()

    assert [candidate.displacement_lab for candidate in candidates] == [
        (0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 1.0),
        (0.0, 2.0, 0.0),
        (0.0, 2.0, 1.0),
    ]
    assert [candidate.kind for candidate in candidates] == [
        "zero", "in_plane", "in_plane", "in_plane", "in_plane", "in_plane"
    ]


def test_normal_offsets_order_by_absolute_value_negative_first() -> None:
    state = _periodic_state()
    candidates = _domain(normal=(2.0, 0.0, 1.0, -1.0, -2.0)).resolve_for(
        state
    ).ordered_candidates()

    assert [candidate.displacement_lab for candidate in candidates] == [
        (0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]


def test_combined_order_is_in_plane_major_normal_minor() -> None:
    state = _periodic_state()
    candidates = _domain(
        first=(0.0, 1.0),
        second=(0.0,),
        normal=(0.0, -1.0, 1.0),
    ).resolve_for(state).ordered_candidates()

    assert [(item.kind, item.displacement_lab) for item in candidates] == [
        ("zero", (0.0, 0.0, 0.0)),
        ("in_plane", (0.0, 1.0, 0.0)),
        ("normal", (-1.0, 0.0, 0.0)),
        ("normal", (1.0, 0.0, 0.0)),
        ("combined", (-1.0, 1.0, 0.0)),
        ("combined", (1.0, 1.0, 0.0)),
    ]


def test_periodic_equivalent_displacement_is_recorded_not_retained() -> None:
    state = _periodic_state()
    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0, 5.0)),
        max_seeds=20,
    )

    equivalent = next(
        attempt
        for attempt in result.attempts
        if attempt.candidate.displacement_lab == (0.0, 5.0, 0.0)
    )
    assert equivalent.canonical_displacement_lab == (0.0, 1.0, 0.0)
    assert equivalent.disposition == "periodic_equivalent"
    assert equivalent.validation_status == "feasible"
    assert equivalent.report is not None
    assert equivalent.rejection_reasons == (
        "initializer.periodic_equivalent_translation",
    )
    assert len({seed.canonical_displacement_lab for seed in result.seeds}) == len(
        result.seeds
    )


def test_duplicate_structure_hash_is_recorded_not_retained(monkeypatch) -> None:
    state = _periodic_state()

    def identity_translate(source, **kwargs):
        return source

    monkeypatch.setattr(initialization, "translate_grain", identity_translate)
    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0)),
        max_seeds=20,
    )

    assert result.attempts[1].disposition == "duplicate_structure"
    assert result.attempts[1].rejection_reasons == (
        "initializer.duplicate_structure",
    )
    assert len(result.seeds) == 1


def test_asymmetric_nonzero_bounds_are_handled_by_existing_primitive() -> None:
    atoms = _atoms().copy()
    atoms["x"] -= 2.0
    atoms["y"] += 10.0
    atoms["z"] -= 7.0
    state = _periodic_state(
        atoms=atoms,
        box_dims=np.array(((-2.0, 8.0), (10.0, 14.0), (-7.0, -3.0))),
    )
    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 5.0)),
        max_seeds=2,
    )

    translated = result.seeds[1].state
    np.testing.assert_allclose(
        translated.atoms["y"][translated.grain_ids == 1],
        (12.0, 12.0, 10.0, 10.0),
    )
    assert result.seeds[1].canonical_displacement_lab == (0.0, 1.0, 0.0)


def test_retained_periodic_seed_passes_both_interfaces() -> None:
    state = _periodic_state()
    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0)),
        max_seeds=2,
    )

    for seed in result.seeds:
        assert seed.report.status == "feasible"
        assert [metrics.interface_id for metrics in seed.report.interfaces] == [
            "central_gb",
            "periodic_gb",
        ]
        assert seed.report.slab is None


def test_slab_seed_preserves_physical_interface_surface_and_vacuum_checks() -> None:
    state = _slab_state()
    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0)),
        max_seeds=2,
    )

    for seed in result.seeds:
        assert seed.report.status == "feasible"
        assert len(seed.report.interfaces) == 1
        assert seed.report.interfaces[0].interface_id == "central_gb"
        assert seed.report.slab is not None
        assert len(seed.report.slab.surfaces) == 2
        assert len(seed.report.slab.vacuum_regions) == 2


def test_retained_seed_preserves_identity_topology_termination_and_provenance() -> None:
    state = _periodic_state()
    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0)),
        max_seeds=2,
    )
    seed = result.seeds[1]

    np.testing.assert_array_equal(seed.state.atom_ids, state.atom_ids)
    np.testing.assert_array_equal(seed.state.grain_ids, state.grain_ids)
    assert seed.state.topology == state.topology
    assert seed.state.interfaces == state.interfaces
    assert seed.state.termination_ids == state.termination_ids
    assert seed.state.manifest()["metadata"]["source"] == {"row": 7}
    history = seed.state.manifest()["metadata"][TRANSLATION_HISTORY_KEY]
    assert history[-1]["displacement_lab"] == [0.0, 1.0, 0.0]


def test_attempts_retain_reports_metrics_hashes_and_rejection_codes() -> None:
    state = _periodic_state()
    result = InterfaceInitializer(state, _policy(hard=2.1)).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0)),
        max_seeds=5,
    )

    rejected, retained = result.attempts
    assert rejected.disposition == "rejected_status"
    assert rejected.validation_status == "infeasible"
    assert rejected.structure_hash == state.structure_hash
    assert rejected.report is not None
    assert len(rejected.report.interfaces) == 2
    assert rejected.rejection_reasons == tuple(
        reason.code for reason in rejected.report.reasons
    )
    assert retained.disposition == "retained"
    assert retained.report is not None
    assert retained.report.to_dict()["interfaces"]


def test_default_rejects_warning_and_opt_in_retains_it() -> None:
    atoms = _atoms()
    atoms[4]["x"] = 4.8
    state = _periodic_state(atoms=atoms)
    policy = _policy(hard=0.5, warning=1.0)

    rejected = InterfaceInitializer(state, policy).generate_translation_seeds(
        translation_domain=_domain(), max_seeds=1
    )
    retained = InterfaceInitializer(
        state, policy, retain_warnings=True
    ).generate_translation_seeds(
        translation_domain=_domain(), max_seeds=1
    )

    assert rejected.status == "translation_domain_exhausted"
    assert rejected.attempts[0].rejection_reasons[0] == (
        "initializer.warning_not_retainable"
    )
    assert retained.status == "seed_limit_reached"
    assert retained.seeds[0].report.status == "warning"


def test_serialization_domain_and_result_hashes_are_deterministic() -> None:
    state = _periodic_state()
    first_domain = _domain(first=(1.0, 0.0), second=(0.0,), normal=(1.0, 0.0, -1.0))
    second_domain = _domain(first=(0.0, 1.0), second=(0.0,), normal=(-1.0, 1.0, 0.0))
    initializer = InterfaceInitializer(state, _policy())

    first = initializer.generate_translation_seeds(
        translation_domain=first_domain, max_seeds=20
    )
    second = initializer.generate_translation_seeds(
        translation_domain=second_domain, max_seeds=20
    )

    assert first.translation_domain is not None
    assert second.translation_domain is not None
    assert first.translation_domain.domain_hash == second.translation_domain.domain_hash
    assert first.result_hash == second.result_hash
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json())["result_hash"] == first.result_hash


def test_max_seeds_one_selects_first_acceptable_in_documented_order() -> None:
    state = _periodic_state()
    result = InterfaceInitializer(state, _policy(hard=2.1)).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 2.0, 1.0)),
        max_seeds=1,
    )

    assert result.status == "seed_limit_reached"
    assert [attempt.candidate.displacement_lab for attempt in result.attempts] == [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    assert result.seeds[0].candidate.displacement_lab == (0.0, 1.0, 0.0)


def test_multiple_seeds_have_deterministic_placement_diversity() -> None:
    state = _periodic_state()
    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0), second=(0.0, 1.0)),
        max_seeds=3,
    )

    assert result.status == "seed_limit_reached"
    assert [seed.candidate.displacement_lab for seed in result.seeds] == [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    ]
    assert len({seed.state.structure_hash for seed in result.seeds}) == 3
    assert len({seed.canonical_displacement_lab for seed in result.seeds}) == 3


def test_explicit_domain_exhaustion_carries_phase7_handoff() -> None:
    state = _periodic_state()
    result = InterfaceInitializer(state, _policy(hard=2.3)).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0)),
        max_seeds=2,
    )

    assert result.status == "translation_domain_exhausted"
    assert result.domain_exhausted
    assert not result.seed_limit_reached
    assert not result.seeds
    assert result.phase7_handoff == "termination_enumeration_required"
    assert len(result.attempts) == 2


def test_translation_error_is_recorded_and_search_exhausts_for_slab() -> None:
    state = _slab_state()
    result = InterfaceInitializer(state, _policy(hard=2.3)).generate_translation_seeds(
        translation_domain=_domain(normal=(0.0, 20.0)),
        max_seeds=2,
    )

    assert result.status == "translation_domain_exhausted"
    assert result.attempts[-1].disposition == "translation_error"
    assert result.attempts[-1].rejection_reasons == (
        "initializer.translation_error",
    )
    assert "outside box axis 0" in result.attempts[-1].error_message


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"in_plane_components": ((0.0, 0.0), (0.0,))}, "duplicates"),
        ({"in_plane_components": ((0.0,), ())}, "must not be empty"),
        ({"in_plane_components": ((0.0,), (0.0,)), "normal_offsets": (np.inf,)}, "finite"),
        ({"in_plane_components": ((0.0,), (0.0,)), "normal_axis": 0}, "both be omitted"),
        (
            {
                "in_plane_components": ((0.0,), (0.0,)),
                "normal_axis": 0,
                "in_plane_axes": (2, 1),
            },
            "in increasing lab-axis order",
        ),
    ],
)
def test_invalid_translation_domain_inputs_raise(kwargs, match) -> None:
    with pytest.raises(InterfaceInitializationError, match=match):
        CartesianTranslationDomain(**kwargs)


@pytest.mark.parametrize("max_seeds", [0, -1, 1.5, True, None])
def test_invalid_retained_seed_limits_return_invalid_input(max_seeds) -> None:
    result = InterfaceInitializer(_periodic_state(), _policy()).generate_translation_seeds(
        translation_domain=_domain(),
        max_seeds=max_seeds,
    )

    assert result.status == "invalid_input"
    assert result.invalid_reasons
    assert not result.attempts


def test_axis_definition_conflicting_with_state_returns_invalid_input() -> None:
    result = InterfaceInitializer(_periodic_state(), _policy()).generate_translation_seeds(
        translation_domain=_domain(normal_axis=1, in_plane_axes=(0, 2)),
        max_seeds=1,
    )

    assert result.status == "invalid_input"
    assert "conflicts with interface axis 0" in result.invalid_reasons[0]


@pytest.mark.parametrize(
    "state,policy,domain",
    [
        (object(), _policy(), _domain()),
        (_periodic_state(), object(), _domain()),
        (_periodic_state(), _policy(), object()),
    ],
)
def test_safe_entry_point_distinguishes_malformed_inputs(state, policy, domain) -> None:
    result = generate_translation_seeds(
        state=state,
        feasibility_policy=policy,
        translation_domain=domain,
        max_seeds=1,
    )

    assert result.status == "invalid_input"
    assert result.invalid_reasons


def test_validator_invalid_source_is_initializer_invalid_input() -> None:
    atoms = _atoms()
    atoms[4]["x"] = 4.0
    state = _periodic_state(atoms=atoms)
    assert validate_bicrystal_state(state, policy=_policy()).status == "invalid"

    result = InterfaceInitializer(state, _policy()).generate_translation_seeds(
        translation_domain=_domain(first=(0.0, 1.0)),
        max_seeds=2,
    )

    assert result.status == "invalid_input"
    assert result.invalid_reasons == ("initializer.source_state_invalid",)
    assert len(result.attempts) == 1
    assert result.attempts[0].validation_status == "invalid"


def test_public_api_is_compatible_with_phase4_and_phase5_types() -> None:
    from GBOpt import (
        CartesianTranslationDomain as ExportedDomain,
        InterfaceInitializer as ExportedInitializer,
        generate_translation_seeds as exported_generate,
    )

    assert ExportedDomain is CartesianTranslationDomain
    assert ExportedInitializer is InterfaceInitializer
    assert exported_generate is generate_translation_seeds

@pytest.fixture(scope="module")
def reduced_zhang_001_initializer_case():
    """Build the smallest supported exact production-path Zhang ST state."""
    from pathlib import Path

    from GBOpt import GBMaker
    from GBOpt.BoundarySpec import PQSpec
    from generate_structures import _load_cases

    case = next(
        item
        for item in _load_cases(Path("gb_data_gbopt.csv"), expected_cases=197)
        if item.case_id == "zhang_001_ST_100"
    )
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        gb = GBMaker.from_boundary_spec(
            a0=5.454,
            structure="fluorite",
            atom_types=("U", "O"),
            boundary=PQSpec(
                P=np.asarray(case.P, dtype=object),
                Q=np.asarray(case.Q, dtype=object),
                basis_mode="supplied",
            ),
            mode="exact",
            repeat_factor=(1, 1),
            x_dim_min=8.0,
            gb_thickness=0.0,
            vacuum=0.0,
            interaction_distance=0.0,
            mismatch_tol=1.0e-3,
            mismatch_max_cells=50,
            strain_grain="both",
            topology="periodic_bicrystal",
            boundary_conditions=("periodic", "periodic", "periodic"),
            termination_ids=(0, 0),
        )
    lengths = gb.bicrystal_state.box_dims[:, 1] - gb.bicrystal_state.box_dims[:, 0]
    domain = CartesianTranslationDomain(
        in_plane_components=(
            tuple(float(lengths[1] * index / 4) for index in range(4)),
            tuple(float(lengths[2] * index / 4) for index in range(4)),
        ),
        normal_offsets=(0.0,),
    )
    return gb, domain


def test_zhang_001_strict_seed_or_reproducible_exhaustion(
    reduced_zhang_001_initializer_case,
) -> None:
    gb, domain = reduced_zhang_001_initializer_case
    initializer = InterfaceInitializer(
        gb.bicrystal_state,
        FeasibilityPolicy.from_unit_cell(gb.unit_cell),
    )

    first = initializer.generate_translation_seeds(
        translation_domain=domain,
        max_seeds=1,
    )
    second = initializer.generate_translation_seeds(
        translation_domain=domain,
        max_seeds=1,
    )

    assert first.result_hash == second.result_hash
    assert first.to_json() == second.to_json()
    if first.seeds:
        assert all(seed.report.status == "feasible" for seed in first.seeds)
        assert all(len(seed.report.interfaces) == 2 for seed in first.seeds)
    else:
        assert first.status == "translation_domain_exhausted"
        assert first.phase7_handoff == "termination_enumeration_required"
        assert len(first.attempts) == 16
        assert all(attempt.report is not None for attempt in first.attempts)
        assert {
            code
            for attempt in first.attempts
            for code in attempt.rejection_reasons
        } >= {
            "interface.cross_contact_below_hard_minimum",
            "initializer.warning_not_retainable",
        }
