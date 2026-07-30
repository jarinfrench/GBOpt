# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for exact crystallographic termination construction and Phase 7 search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import GBOpt.termination_initialization as phase7
from GBOpt.BoundarySpec import PQSpec
from GBOpt.GBMaker import GBMaker, GBMakerValueError
from GBOpt.geometry_validation import (
    ContactPolicy,
    FeasibilityPolicy,
    SpeciesPairThresholds,
    VoidPolicy,
)
from GBOpt.interface_initialization import CartesianTranslationDomain
from GBOpt.termination import (
    GrainTermination,
    TerminationError,
    TerminationPair,
)
from GBOpt.termination_initialization import (
    ExactBoundaryReconstruction,
    TerminationDomain,
    TerminationInitializationError,
    TerminationInitializer,
    generate_termination_seeds,
)
from generate_structures import _load_cases


SIGMA5_PQ = PQSpec(
    P=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    Q=[[4, -3, 0], [3, 4, 0], [0, 0, 1]],
    basis_mode="supplied",
)


# Phase 7 tests intentionally use one in-plane repeat to keep exact
# reconstruction and termination enumeration small. GBMaker's warning
# behavior for repeat factors below two is covered separately.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:Recommended repeat factor is at least 2\.:UserWarning"
)


def _reconstruction(
    *,
    structure: str = "fcc",
    atom_types="Cu",
    a0: float = 3.615,
    vacuum: float = 0.0,
    topology: str = "periodic_bicrystal",
    provenance=None,
) -> ExactBoundaryReconstruction:
    return ExactBoundaryReconstruction(
        a0=a0,
        structure=structure,
        atom_types=atom_types,
        boundary=SIGMA5_PQ,
        gb_thickness=0.0,
        repeat_factor=(1, 1),
        x_dim_min=2.0 * a0,
        vacuum=vacuum,
        interaction_distance=1.0,
        topology=topology,
        boundary_conditions=(
            ("periodic", "periodic", "periodic")
            if topology == "periodic_bicrystal"
            else ("fixed", "periodic", "periodic")
        ),
        provenance=provenance,
    )


def _permissive_policy(species=("Cu",), *, hard: float = 0.1) -> FeasibilityPolicy:
    pairs = []
    for first_index, first in enumerate(species):
        for second in species[first_index:]:
            pairs.append(
                SpeciesPairThresholds(
                    tuple(sorted((first, second))),
                    1.0e-6,
                    hard,
                    hard,
                )
            )
    return FeasibilityPolicy(
        contact=ContactPolicy(pair_thresholds=tuple(pairs)),
        void=VoidPolicy(
            hard_max_empty_bin_fraction=1.0,
            warning_max_empty_bin_fraction=1.0,
            hard_max_gap_range_bulk_factor=100.0,
            warning_max_gap_range_bulk_factor=100.0,
            hard_max_p95_bulk_factor=100.0,
            warning_max_p95_bulk_factor=100.0,
        ),
    )


def _translation_domain(*, normal=(0.0,)) -> CartesianTranslationDomain:
    return CartesianTranslationDomain(
        in_plane_components=((0.0,), (0.0,)),
        normal_offsets=tuple(normal),
        normal_axis=0,
        in_plane_axes=(1, 2),
    )


def _domain_and_gb(reconstruction=None):
    reconstruction = _reconstruction() if reconstruction is None else reconstruction
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        gb = reconstruction.build(TerminationPair())
    return TerminationDomain.from_gbmaker(gb), gb


def _positions(atoms):
    return np.column_stack((atoms["x"], atoms["y"], atoms["z"]))


def test_grain_termination_canonicalizes_exact_phase_and_rejects_bad_values() -> None:
    assert GrainTermination("left", 6, 4).phase_numerator == 1
    assert GrainTermination("left", 6, 4).phase_denominator == 2
    assert GrainTermination("right", -1, 4).phase_numerator == 3
    assert GrainTermination("right", -1, 4).phase_denominator == 4

    with pytest.raises(TerminationError, match="positive"):
        GrainTermination("left", 0, 0)
    with pytest.raises(TerminationError, match="grain"):
        GrainTermination("center")  # type: ignore[arg-type]
    with pytest.raises(TerminationError, match="cut_convention"):
        GrainTermination("left", cut_convention="floating_cut")  # type: ignore[arg-type]


def test_default_phase_reconstructs_original_exact_geometry() -> None:
    reconstruction = _reconstruction()
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        legacy = GBMaker.from_boundary_spec(
            reconstruction.a0,
            reconstruction.structure,
            reconstruction.atom_types,
            reconstruction.boundary,
            mode="exact",
            gb_thickness=reconstruction.gb_thickness,
            repeat_factor=reconstruction.repeat_factor,
            x_dim_min=reconstruction.x_dim_min,
            vacuum=reconstruction.vacuum,
            interaction_distance=reconstruction.interaction_distance,
            topology=reconstruction.topology,
            boundary_conditions=reconstruction.boundary_conditions,
        )
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        phased = reconstruction.build(TerminationPair())

    np.testing.assert_array_equal(legacy.whole_system, phased.whole_system)
    np.testing.assert_array_equal(legacy.box_dims, phased.box_dims)
    assert legacy.bicrystal_state.structure_hash == phased.bicrystal_state.structure_hash
    assert phased.termination_ids == (0, 0)


def test_legacy_termination_ids_alone_remain_metadata_only() -> None:
    reconstruction = _reconstruction()
    common = dict(
        gb_thickness=reconstruction.gb_thickness,
        repeat_factor=reconstruction.repeat_factor,
        x_dim_min=reconstruction.x_dim_min,
        vacuum=reconstruction.vacuum,
        interaction_distance=reconstruction.interaction_distance,
        topology=reconstruction.topology,
        boundary_conditions=reconstruction.boundary_conditions,
    )
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        default = GBMaker.from_boundary_spec(
            reconstruction.a0,
            reconstruction.structure,
            reconstruction.atom_types,
            reconstruction.boundary,
            mode="exact",
            termination_ids=(0, 0),
            **common,
        )
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        metadata_only = GBMaker.from_boundary_spec(
            reconstruction.a0,
            reconstruction.structure,
            reconstruction.atom_types,
            reconstruction.boundary,
            mode="exact",
            termination_ids=(7, 11),
            **common,
        )

    np.testing.assert_array_equal(default.whole_system, metadata_only.whole_system)
    np.testing.assert_array_equal(default.left_grain, metadata_only.left_grain)
    np.testing.assert_array_equal(default.right_grain, metadata_only.right_grain)
    assert default.bicrystal_state.structure_hash != metadata_only.bicrystal_state.structure_hash
    assert default.bicrystal_state.state_hash != metadata_only.bicrystal_state.state_hash
    assert metadata_only.termination_pair is None


def test_nondefault_phase_changes_constructed_geometry_not_only_metadata() -> None:
    domain, default = _domain_and_gb()
    pair = TerminationPair(left=domain.left[1], right=domain.right[0])
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        changed = _reconstruction().build(pair)

    assert changed.termination_ids == (1, 0)
    assert changed.bicrystal_state.structure_hash != default.bicrystal_state.structure_hash
    assert not np.array_equal(_positions(changed.left_grain), _positions(default.left_grain))
    np.testing.assert_array_equal(changed.right_grain, default.right_grain)
    assert len(changed.left_grain) == len(default.left_grain)
    assert len(changed.right_grain) == len(default.right_grain)
    assert changed.topology == default.topology
    assert changed.boundary_conditions == default.boundary_conditions
    assert changed.bicrystal_state.metadata["termination_descriptors"] is not None


def test_finite_domain_is_exact_deterministic_and_uses_documented_order() -> None:
    domain, _ = _domain_and_gb()
    reduced = TerminationDomain(
        left=(domain.left[1], domain.left[0]),
        right=(domain.right[1], domain.right[0]),
    )
    candidates = reduced.candidates()

    assert [
        (item.canonical_pair.left.phase, item.canonical_pair.right.phase)
        for item in candidates
    ] == [
        (domain.left[0].phase, domain.right[0].phase),
        (domain.left[1].phase, domain.right[0].phase),
        (domain.left[0].phase, domain.right[1].phase),
        (domain.left[1].phase, domain.right[1].phase),
    ]
    assert reduced.domain_hash == TerminationDomain(
        left=tuple(reversed(reduced.left)),
        right=tuple(reversed(reduced.right)),
    ).domain_hash
    assert json.loads(reduced.to_json())["ordering"].startswith("default_pair")


def test_duplicate_and_unsupported_termination_descriptors_are_rejected() -> None:
    with pytest.raises(TerminationInitializationError, match="duplicate equivalent"):
        TerminationDomain(
            left=(
                GrainTermination("left", 1, 2),
                GrainTermination("left", 2, 4),
            ),
            right=(GrainTermination("right"),),
        )

    reconstruction = _reconstruction()
    unsupported = TerminationPair(
        left=GrainTermination("left", 1, 3),
        right=GrainTermination("right"),
    )
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        with pytest.raises(GBMakerValueError, match="not a supported exact"):
            reconstruction.build(unsupported)

    result = generate_termination_seeds(
        reconstruction=reconstruction,
        feasibility_policy=_permissive_policy(),
        termination_domain=TerminationDomain(
            left=(unsupported.left,), right=(unsupported.right,)
        ),
        translation_domain=_translation_domain(),
        max_seeds=1,
    )
    assert result.status == "invalid_input"
    assert "unsupported exact cut phases" in result.invalid_reasons[0]


def test_reconstruction_is_immutable_and_defensively_copies_provenance() -> None:
    provenance = {"case": "synthetic", "nested": {"rows": [1, 2]}}
    reconstruction = _reconstruction(provenance=provenance)
    original_json = reconstruction.to_json()
    provenance["nested"]["rows"].append(3)

    assert reconstruction.to_json() == original_json
    assert json.loads(original_json)["provenance"]["nested"]["rows"] == [1, 2]
    assert reconstruction.reconstruction_hash == _reconstruction(
        provenance={"case": "synthetic", "nested": {"rows": [1, 2]}}
    ).reconstruction_hash


def test_exact_nondefault_path_never_calls_float_plane_deletion_helper(monkeypatch) -> None:
    domain, _ = _domain_and_gb()

    def fail(*_args, **_kwargs):
        raise AssertionError("exact termination construction used float-path deletion")

    monkeypatch.setattr(GBMaker, "_GBMaker__equalize_float_periodic_gap", fail)
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        gb = _reconstruction().build(
            TerminationPair(left=domain.left[1], right=domain.right[1])
        )
    assert gb.uses_exact_construction


def test_fluorite_variants_preserve_complete_populations_and_stoichiometry() -> None:
    reconstruction = _reconstruction(
        structure="fluorite",
        atom_types=("U", "O"),
        a0=5.47,
    )
    domain, default = _domain_and_gb(reconstruction)
    pairs = (
        TerminationPair(),
        TerminationPair(left=domain.left[1], right=domain.right[0]),
        TerminationPair(left=domain.left[0], right=domain.right[1]),
    )
    expected = (len(default.left_grain), len(default.right_grain))

    for pair in pairs:
        with pytest.warns(UserWarning, match="Recommended repeat factor"):
            gb = reconstruction.build(pair)
        assert (len(gb.left_grain), len(gb.right_grain)) == expected
        for grain in (gb.left_grain, gb.right_grain):
            uranium = int(np.count_nonzero(grain["name"] == "U"))
            oxygen = int(np.count_nonzero(grain["name"] == "O"))
            assert uranium > 0
            assert oxygen == 2 * uranium


def test_zero_acceptable_candidate_skips_phase6_and_preserves_source(monkeypatch) -> None:
    reconstruction = _reconstruction()
    domain, source = _domain_and_gb(reconstruction)
    source_hashes = (
        source.bicrystal_state.structure_hash,
        source.bicrystal_state.state_hash,
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("Phase 6 must not run for an acceptable zero state")

    monkeypatch.setattr(phase7, "generate_translation_seeds", fail)
    result = TerminationInitializer(
        reconstruction=reconstruction,
        feasibility_policy=_permissive_policy(),
        termination_domain=TerminationDomain(
            left=(domain.left[0],), right=(domain.right[0],)
        ),
        translation_domain=_translation_domain(),
    ).generate_seeds(max_seeds=2)

    assert result.status == "default_termination_accepted"
    assert result.domain_exhausted
    assert len(result.seeds) == 1
    assert result.seeds[0].kind == "default_zero"
    assert result.attempts[0].nested_translation_result is None
    assert source_hashes == (
        source.bicrystal_state.structure_hash,
        source.bicrystal_state.state_hash,
    )


def test_infeasible_zero_variant_delegates_to_phase6_and_retains_nested_record() -> None:
    reconstruction = _reconstruction()
    domain, _ = _domain_and_gb(reconstruction)
    nondefault_only = TerminationDomain(
        left=(domain.left[1],), right=(domain.right[0],)
    )
    result = generate_termination_seeds(
        reconstruction=reconstruction,
        feasibility_policy=_permissive_policy(hard=1.0),
        termination_domain=nondefault_only,
        translation_domain=_translation_domain(
            normal=(0.0, -0.1, -0.2, -0.3, -0.4, -0.5)
        ),
        max_seeds=1,
    )

    assert result.status == "seed_limit_reached"
    assert len(result.seeds) == 1
    seed = result.seeds[0]
    attempt = result.attempts[0]
    assert seed.kind == "termination_plus_translation"
    assert seed.applied_translation_lab != (0.0, 0.0, 0.0)
    assert attempt.disposition == "retained_translated"
    assert attempt.zero_translation_report.status == "infeasible"
    assert attempt.nested_translation_result is not None
    assert attempt.nested_translation_result.result_hash == seed.nested_translation_result_hash
    assert attempt.nested_translation_result.attempts


def test_max_seeds_one_selects_first_acceptable_candidate_in_canonical_order() -> None:
    reconstruction = _reconstruction()
    domain, _ = _domain_and_gb(reconstruction)
    result = generate_termination_seeds(
        reconstruction=reconstruction,
        feasibility_policy=_permissive_policy(),
        termination_domain=TerminationDomain(
            left=(domain.left[0], domain.left[1]),
            right=(domain.right[0], domain.right[1]),
        ),
        translation_domain=_translation_domain(),
        max_seeds=1,
    )

    assert result.status == "seed_limit_reached"
    assert len(result.attempts) == 1
    assert result.seeds[0].kind == "default_zero"
    assert result.seeds[0].candidate.order == 0


def test_duplicate_reconstructed_structures_are_deduplicated(monkeypatch) -> None:
    reconstruction = _reconstruction()
    domain, source = _domain_and_gb(reconstruction)
    original_build = ExactBoundaryReconstruction.build

    def duplicate_build(self, termination_pair):
        if self is reconstruction:
            return source
        return original_build(self, termination_pair)

    monkeypatch.setattr(ExactBoundaryReconstruction, "build", duplicate_build)
    result = generate_termination_seeds(
        reconstruction=reconstruction,
        feasibility_policy=_permissive_policy(),
        termination_domain=TerminationDomain(
            left=(domain.left[0], domain.left[1]),
            right=(domain.right[0],),
        ),
        translation_domain=_translation_domain(),
        max_seeds=3,
    )

    assert len(result.seeds) == 1
    assert [attempt.disposition for attempt in result.attempts] == [
        "retained_zero",
        "duplicate_structure",
    ]
    assert result.attempts[1].rejection_reasons == (
        "termination.duplicate_reconstructed_structure",
    )


def test_exhaustion_records_every_attempt_and_nested_translation_domain() -> None:
    reconstruction = _reconstruction()
    domain, _ = _domain_and_gb(reconstruction)
    selected = TerminationDomain(
        left=(domain.left[1],), right=(domain.right[0], domain.right[1])
    )
    result = generate_termination_seeds(
        reconstruction=reconstruction,
        feasibility_policy=_permissive_policy(hard=3.0),
        termination_domain=selected,
        translation_domain=_translation_domain(),
        max_seeds=2,
    )

    assert result.status == "termination_translation_domain_exhausted"
    assert result.domain_exhausted
    assert not result.seeds
    assert len(result.attempts) == len(selected.candidates())
    assert all(attempt.zero_translation_report is not None for attempt in result.attempts)
    assert all(attempt.nested_translation_result is not None for attempt in result.attempts)
    assert all(
        attempt.nested_translation_result.status == "translation_domain_exhausted"
        for attempt in result.attempts
    )


def test_slab_search_preserves_interface_surface_and_vacuum_topology() -> None:
    reconstruction = _reconstruction(
        vacuum=3.0,
        topology="single_interface_slab",
    )
    domain, _ = _domain_and_gb(reconstruction)
    result = generate_termination_seeds(
        reconstruction=reconstruction,
        feasibility_policy=_permissive_policy(),
        termination_domain=TerminationDomain(
            left=(domain.left[0],), right=(domain.right[0],)
        ),
        translation_domain=_translation_domain(),
        max_seeds=1,
    )

    seed = result.seeds[0]
    assert seed.state.topology == "single_interface_slab"
    assert seed.state.boundary_conditions == ("fixed", "periodic", "periodic")
    assert len(seed.state.interfaces) == 1
    assert seed.report.slab is not None
    assert seed.state.external_surfaces
    assert seed.state.vacuum_regions


def test_repeated_search_serialization_and_hashes_are_deterministic() -> None:
    reconstruction = _reconstruction()
    domain, _ = _domain_and_gb(reconstruction)
    kwargs = dict(
        reconstruction=reconstruction,
        feasibility_policy=_permissive_policy(),
        termination_domain=TerminationDomain(
            left=(domain.left[0], domain.left[1]),
            right=(domain.right[0], domain.right[1]),
        ),
        translation_domain=_translation_domain(),
        max_seeds=3,
    )
    first = generate_termination_seeds(**kwargs)
    second = generate_termination_seeds(**kwargs)

    assert first.to_json() == second.to_json()
    assert first.result_hash == second.result_hash
    assert json.loads(first.to_json())["result_hash"] == first.result_hash
    assert [seed.state.structure_hash for seed in first.seeds] == list(
        dict.fromkeys(seed.state.structure_hash for seed in first.seeds)
    )


def test_safe_entry_point_reports_invalid_initializer_inputs() -> None:
    result = generate_termination_seeds(
        reconstruction="not reconstruction",
        feasibility_policy="not policy",
        termination_domain="not domain",
        translation_domain="not translation domain",
        max_seeds=0,
    )
    assert result.status == "invalid_input"
    assert result.invalid_reasons
    assert not result.attempts


def test_validation_exception_is_recorded_for_attempt(monkeypatch) -> None:
    reconstruction = _reconstruction()
    domain, _ = _domain_and_gb(reconstruction)

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic validator failure")

    monkeypatch.setattr(phase7, "validate_bicrystal_state", fail)
    result = generate_termination_seeds(
        reconstruction=reconstruction,
        feasibility_policy=_permissive_policy(),
        termination_domain=TerminationDomain(
            left=(domain.left[0],), right=(domain.right[0],)
        ),
        translation_domain=_translation_domain(),
        max_seeds=1,
    )

    assert result.status == "termination_translation_domain_exhausted"
    assert result.attempts[0].disposition == "validation_error"
    assert result.attempts[0].validation_error.endswith(
        "synthetic validator failure"
    )
    assert result.attempts[0].zero_translation_report is None


def test_public_phase7_symbols_are_exported() -> None:
    from GBOpt import (
        ExactBoundaryReconstruction as ExportedReconstruction,
        GrainTermination as ExportedTermination,
        TerminationDomain as ExportedDomain,
        TerminationInitializer as ExportedInitializer,
        generate_termination_seeds as exported_generate,
    )

    assert ExportedReconstruction is ExactBoundaryReconstruction
    assert ExportedTermination is GrainTermination
    assert ExportedDomain is TerminationDomain
    assert ExportedInitializer is TerminationInitializer
    assert exported_generate is generate_termination_seeds


def test_reduced_zhang_001_has_reproducible_seed_or_explained_exhaustion() -> None:
    cases = {
        case.case_id: case
        for case in _load_cases(Path("gb_data_gbopt.csv"), expected_cases=197)
    }
    case = cases["zhang_001_ST_100"]
    reconstruction = ExactBoundaryReconstruction(
        a0=5.454,
        structure="fluorite",
        atom_types=("U", "O"),
        boundary=PQSpec(P=case.P, Q=case.Q, basis_mode="supplied"),
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
        provenance={"case_id": case.case_id},
    )
    with pytest.warns(UserWarning, match="Recommended repeat factor"):
        source = reconstruction.build(TerminationPair())
    left, right = source.available_termination_descriptors
    domain = TerminationDomain(
        left=left[: min(2, len(left))],
        right=right[: min(2, len(right))],
    )
    policy = FeasibilityPolicy.from_unit_cell(source.unit_cell)
    result = generate_termination_seeds(
        reconstruction=reconstruction,
        feasibility_policy=policy,
        termination_domain=domain,
        translation_domain=_translation_domain(),
        max_seeds=1,
    )

    assert result.status in {
        "seed_limit_reached",
        "default_termination_accepted",
        "nondefault_termination_accepted",
        "termination_translated_seed_retained",
        "termination_translation_domain_exhausted",
    }
    if result.seeds:
        assert all(seed.report.status == "feasible" for seed in result.seeds)
        assert all(len(seed.report.interfaces) == 2 for seed in result.seeds)
    else:
        assert result.status == "termination_translation_domain_exhausted"
        assert len(result.attempts) == len(domain.candidates())
        assert all(attempt.rejection_reasons for attempt in result.attempts)
        assert all(
            attempt.nested_translation_result is not None
            for attempt in result.attempts
            if attempt.zero_translation_report is not None
            and attempt.zero_translation_report.status != "invalid"
        )
