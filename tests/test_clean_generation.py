# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for immutable Phase 8 campaign configuration."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from GBOpt.clean_generation import (
    CleanGenerationConfigError,
    CleanGenerationSettings,
    RationalPhase,
    TerminationDomainSelection,
    feasibility_policy_from_mapping,
    translation_domain_from_mapping,
)
from GBOpt.geometry_validation import FeasibilityOverride


def test_default_configuration_is_strict_periodic_and_deterministic() -> None:
    first = CleanGenerationSettings()
    second = CleanGenerationSettings.from_mapping({})

    assert first == second
    assert first.topology == "periodic_bicrystal"
    assert first.boundary_conditions == ("periodic", "periodic", "periodic")
    assert first.retain_warnings is False
    assert first.max_seeds == 1
    assert first.configuration_hash == second.configuration_hash
    assert first.to_dict()[
        "feasibility_policy_hash"] == first.feasibility_policy.policy_hash


def test_json_and_toml_configuration_are_semantically_equivalent(tmp_path) -> None:
    payload = {
        "topology": "single_interface_slab",
        "boundary_conditions": ["fixed", "periodic", "fixed"],
        "vacuum_angstrom": 8.0,
        "fixed_region_thickness_angstrom": 1.5,
        "surface_buffer_thickness_angstrom": 2.0,
        "retain_warnings": True,
        "max_seeds": 3,
        "translation_domain": {
            "in_plane_components": [[0, 1.0], [-1.0, 0]],
            "normal_offsets": [-0.25, 0, 0.25],
        },
        "termination_domain": {
            "mode": "explicit",
            "left": ["1/2", "0/1"],
            "right": ["0", {"numerator": 1, "denominator": 4}],
        },
        "override": {
            "status": "warning",
            "reason": "documented expert review",
        },
    }
    json_path = tmp_path / "clean.json"
    toml_path = tmp_path / "clean.toml"
    json_path.write_text(json.dumps({"clean_generation": payload}), encoding="utf-8")
    toml_path.write_text(
        """
[clean_generation]
topology = "single_interface_slab"
boundary_conditions = ["fixed", "periodic", "fixed"]
vacuum_angstrom = 8.0
fixed_region_thickness_angstrom = 1.5
surface_buffer_thickness_angstrom = 2.0
retain_warnings = true
max_seeds = 3

[clean_generation.translation_domain]
in_plane_components = [[0, 1.0], [-1.0, 0]]
normal_offsets = [-0.25, 0, 0.25]

[clean_generation.termination_domain]
mode = "explicit"
left = ["1/2", "0/1"]
right = ["0", "1/4"]

[clean_generation.override]
status = "warning"
reason = "documented expert review"
""".strip(),
        encoding="utf-8",
    )

    from_json = CleanGenerationSettings.from_file(json_path)
    from_toml = CleanGenerationSettings.from_file(toml_path)

    assert from_json == from_toml
    assert from_json.configuration_hash == from_toml.configuration_hash
    assert [phase.fraction for phase in from_json.termination_domain.left] == [0, 1 / 2]
    assert from_json.feasibility_override == FeasibilityOverride(
        "warning", "documented expert review"
    )


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        pytest.param({"unknown": 1}, "unsupported keys", id="unknown-key"),
        pytest.param(
            {
                "topology": "periodic_bicrystal",
                "boundary_conditions": ["fixed", "periodic", "periodic"],
            },
            "requires x boundary condition periodic",
            id="periodic-fixed-x",
        ),
        pytest.param(
            {"topology": "periodic_bicrystal", "vacuum_angstrom": 1.0},
            "requires zero vacuum",
            id="periodic-vacuum",
        ),
        pytest.param(
            {
                "termination_domain": {
                    "mode": "explicit",
                    "left": ["0", "1/2", "2/4"],
                    "right": ["0"],
                }
            },
            "duplicate equivalent",
            id="duplicate-phase",
        ),
        pytest.param(
            {
                "termination_domain": {
                    "mode": "explicit",
                    "left": ["1/2"],
                    "right": ["0"],
                }
            },
            "must include zero",
            id="missing-default",
        ),
    ],
)
def test_configuration_rejects_malformed_or_inconsistent_values(
    mapping: dict, match: str
) -> None:
    with pytest.raises(CleanGenerationConfigError, match=match):
        CleanGenerationSettings.from_mapping(mapping)


def test_exact_termination_selection_canonicalizes_and_hashes() -> None:
    selection = TerminationDomainSelection(
        mode="explicit",
        left=(RationalPhase(2, 4), RationalPhase(0, 7)),
        right=(RationalPhase(-1, 4), RationalPhase(0, 1)),
    )
    reordered = TerminationDomainSelection(
        mode="explicit",
        left=(RationalPhase(0, 1), RationalPhase(1, 2)),
        right=(RationalPhase(0, 1), RationalPhase(3, 4)),
    )

    assert selection == reordered
    assert selection.selection_hash == reordered.selection_hash
    assert [item.phase for item in selection.descriptors("left")] == [0, 1 / 2]
    assert [item.phase for item in selection.descriptors("right")] == [0, 3 / 4]


def test_translation_domain_parser_preserves_phase6_order_and_hash() -> None:
    domain = translation_domain_from_mapping(
        {
            "in_plane_components": [[1.0, 0.0, -1.0], [0.0, 2.0]],
            "normal_offsets": [0.5, -0.5, 0.0],
            "normal_axis": 0,
            "in_plane_axes": [1, 2],
        }
    )
    candidates = domain.ordered_candidates()

    assert candidates[0].displacement_lab == (0.0, 0.0, 0.0)
    assert candidates[1].displacement_lab == (0.0, -1.0, 0.0)
    assert candidates[-2].displacement_lab[0] == -0.5
    assert domain.domain_hash == translation_domain_from_mapping(
        domain.to_dict()).domain_hash


def test_complete_feasibility_policy_mapping_is_canonical() -> None:
    mapping = {
        "contact": {
            "duplicate_tolerance_angstrom": 1.0e-5,
            "hard_minimum_bulk_factor": 0.5,
            "warning_minimum_bulk_factor": 0.7,
            "pair_thresholds": [
                {
                    "species": ["O", "U"],
                    "duplicate_angstrom": 0.01,
                    "hard_minimum_angstrom": 1.2,
                    "warning_minimum_angstrom": 1.5,
                }
            ],
        },
        "void": {
            "hard_max_empty_bin_fraction": 0.8,
            "warning_max_empty_bin_fraction": 0.6,
            "min_bins_per_axis": 4,
            "max_bins_per_axis": 20,
        },
        "slab": {
            "minimum_vacuum_thickness_angstrom": 4.0,
            "minimum_fixed_region_atoms": 2,
            "minimum_buffer_region_atoms": 3,
        },
    }

    policy = feasibility_policy_from_mapping(mapping)
    reparsed = feasibility_policy_from_mapping(policy.to_dict())

    assert policy == reparsed
    assert policy.policy_hash == reparsed.policy_hash
    assert policy.contact.pair_thresholds[0].species == ("O", "U")


@pytest.mark.parametrize(
    "mutator",
    [
        pytest.param(lambda value: replace(value, retain_warnings=True), id="warnings"),
        pytest.param(lambda value: replace(value, max_seeds=2), id="seed-count"),
        pytest.param(
            lambda value: replace(
                value,
                translation_domain=translation_domain_from_mapping(
                    {
                        "in_plane_components": [[0.0, 0.5], [0.0]],
                        "normal_offsets": [0.0],
                    }
                ),
            ),
            id="translation-domain",
        ),
        pytest.param(
            lambda value: replace(
                value,
                termination_domain=TerminationDomainSelection(mode="default_only"),
            ),
            id="termination-domain",
        ),
        pytest.param(
            lambda value: replace(
                value,
                feasibility_override=FeasibilityOverride(
                    "feasible", "documented campaign exception"
                ),
            ),
            id="override",
        ),
    ],
)
def test_effective_setting_changes_configuration_hash(mutator) -> None:
    baseline = CleanGenerationSettings()
    changed = mutator(baseline)

    assert changed.configuration_hash != baseline.configuration_hash
