# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Focused orchestration and persistence tests for the Phase 8 campaign runner."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import GBOpt.geometry_validation as geometry_validation
import GBOpt.termination_initialization as termination_initialization
from GBOpt.BicrystalState import BicrystalState, translate_grain
from GBOpt.geometry_validation import FeasibilityOverride, validate_bicrystal_state
from GBOpt.termination import TerminationPair
from GBOpt.termination_initialization import check_decorated_population
from generate_structures import (
    _CONSTRUCTION_SCHEMA,
    _INITIALIZATION_SCHEMA,
    _REQUIRED_COLUMNS,
    _build_parser,
    _canonical_sha256,
    _execute_case,
    _existing_result,
    _generate_case,
    _hashed_document_matches,
    _load_cases,
    _make_config,
    _persist_process_failure,
    _run_parent,
    _save_state_atomic,
    _swap_case_directory,
    _write_campaign_outputs,
)


pytestmark = pytest.mark.filterwarnings(
    r"ignore:Recommended repeat factor is at least 2\.:UserWarning"
)


def _identity_row(*, boundary_type: str = "TW", axis_set: str = "100") -> dict:
    row = {name: "" for name in _REQUIRED_COLUMNS}
    identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for prefix, matrix in (("P", identity), ("Q", identity)):
        for axis_name, values in zip("xyz", matrix):
            for index, value in enumerate(values):
                row[f"{prefix}_{axis_name}{index}"] = str(value)
    row["UO2_Basak (J/m^2)"] = "0.0"
    row["Type"] = boundary_type
    row["Axis Set"] = axis_set
    row["CeO2_Gotte (J/m^2)"] = ""
    return row


def _write_csv(path: Path, rows: list[dict] | None = None) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows or [_identity_row()])
    return path


def _strict_contact_policy() -> dict:
    thresholds = []
    for species in (("O", "O"), ("O", "U"), ("U", "U")):
        thresholds.append(
            {
                "species": list(species),
                "duplicate_angstrom": 1.0e-6,
                "hard_minimum_angstrom": 10.0,
                "warning_minimum_angstrom": 10.0,
            }
        )
    return {"contact": {"pair_thresholds": thresholds}}


def _make_test_config(
    tmp_path: Path,
    *,
    clean_payload: dict | None = None,
    expected_ratio: tuple[int, int] = (1, 2),
    rows: list[dict] | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_file = _write_csv(tmp_path / "campaign.csv", rows)
    output_root = tmp_path / "out"
    args_list = [
        "--data-file",
        str(data_file),
        "--output-root",
        str(output_root),
        "--expected-cases",
        str(len(rows or [_identity_row()])),
        "--x-dim-min",
        "5",
        "--gb-thickness-periods",
        "0",
        "--repeat-factor",
        "1",
        "1",
        "--interaction-distance",
        "0",
        "--expected-ratio",
        str(expected_ratio[0]),
        str(expected_ratio[1]),
    ]
    if clean_payload is not None:
        clean_path = tmp_path / "clean.json"
        clean_path.write_text(json.dumps(clean_payload), encoding="utf-8")
        args_list.extend(("--clean-config", str(clean_path)))
    args = _build_parser().parse_args(args_list)
    output_root.mkdir()
    config = _make_config(args, data_file=data_file, output_root=output_root)
    cases = _load_cases(data_file, expected_cases=len(rows or [_identity_row()]))
    return cases, config


def _read_case(config, case):
    root = config.output_root / case.case_id
    return {
        name: json.loads((root / name).read_text(encoding="utf-8"))
        for name in ("case.json", "construction.json", "initialization.json")
    }


def _fake_phase7_factory(seed_kinds: tuple[str, ...], *, nested_marker: str = "nested"):
    def fake_generate(
        *,
        reconstruction,
        feasibility_policy,
        termination_domain,
        translation_domain,
        max_seeds,
        retain_warnings,
        feasibility_override=None,
    ):
        base_gb = reconstruction.build(TerminationPair())
        candidates = termination_domain.candidates()
        default = candidates[0]
        nondefault = next(candidate for candidate in candidates if not candidate.is_default)
        seeds = []
        attempt_payloads = []
        attempts = []
        for order, requested_kind in enumerate(seed_kinds):
            candidate = default if requested_kind == "default_translation" else nondefault
            gb = reconstruction.build(candidate.canonical_pair)
            state = gb.bicrystal_state
            phase7_kind = "nondefault_zero"
            displacement = (0.0, 0.0, 0.0)
            nested_hash = None
            if requested_kind in {"default_translation", "termination_translation"}:
                displacement = (0.0, 0.125 * (order + 1), 0.0)
                state = translate_grain(state, displacement=displacement)
                phase7_kind = "termination_plus_translation"
                nested_hash = f"{nested_marker}-{order}"
            report = validate_bicrystal_state(
                state,
                policy=feasibility_policy,
                override=FeasibilityOverride(
                    "feasible", f"test retained seed {order}"
                ),
            )
            population = check_decorated_population(gb)
            seed = SimpleNamespace(
                kind=phase7_kind,
                candidate=candidate,
                termination_pair=candidate.canonical_pair,
                applied_translation_lab=displacement,
                state=state,
                report=report,
                population_check=population,
                nested_translation_result_hash=nested_hash,
            )
            seed.seed_hash = _canonical_sha256(
                {
                    "order": order,
                    "kind": phase7_kind,
                    "structure_hash": state.structure_hash,
                    "report_hash": report.report_hash,
                }
            )
            seeds.append(seed)
            nested = None
            if nested_hash is not None:
                nested = {
                    "status": "seed_limit_reached",
                    "result_hash": nested_hash,
                    "attempts": [
                        {
                            "order": 0,
                            "displacement_lab": list(displacement),
                            "disposition": "retained",
                        }
                    ],
                }
            attempt_payloads.append(
                {
                    "candidate": candidate.to_dict(),
                    "disposition": (
                        "retained_translated" if nested is not None else "retained_zero"
                    ),
                    "nested_translation_result": nested,
                    "retained_seed_hashes": [seed.seed_hash],
                }
            )
            attempts.append(
                SimpleNamespace(
                    rejection_reasons=(), construction_error="", validation_error=""
                )
            )

        payload = {
            "schema_version": 1,
            "status": "seed_limit_reached",
            "reconstruction": reconstruction.to_dict(),
            "reconstruction_hash": reconstruction.reconstruction_hash,
            "termination_domain": termination_domain.to_dict(),
            "termination_domain_hash": termination_domain.domain_hash,
            "translation_domain": translation_domain.to_dict(),
            "translation_domain_hash": translation_domain.domain_hash,
            "attempts": attempt_payloads,
            "seeds": [
                {
                    "kind": seed.kind,
                    "candidate": seed.candidate.to_dict(),
                    "structure_hash": seed.state.structure_hash,
                    "state_hash": seed.state.state_hash,
                    "report_hash": seed.report.report_hash,
                    "seed_hash": seed.seed_hash,
                    "nested_translation_result_hash": seed.nested_translation_result_hash,
                }
                for seed in seeds
            ],
            "max_seeds": max_seeds,
            "retain_warnings": retain_warnings,
            "seed_limit_reached": True,
            "domain_exhausted": False,
            "source_structure_hash": base_gb.bicrystal_state.structure_hash,
            "source_state_hash": base_gb.bicrystal_state.state_hash,
            "invalid_reasons": [],
            "feasibility_override": None,
        }
        result_hash = _canonical_sha256(payload)

        class FakeResult:
            status = "seed_limit_reached"
            source_structure_hash = base_gb.bicrystal_state.structure_hash
            source_state_hash = base_gb.bicrystal_state.state_hash
            invalid_reasons = ()

            def __init__(self):
                self.seeds = tuple(seeds)
                self.attempts = tuple(attempts)
                self.result_hash = result_hash

            def to_dict(self):
                return {**payload, "result_hash": result_hash}

        return FakeResult()

    return fake_generate


def test_base_feasible_path_uses_exact_supplied_pq_and_skips_phase7(
    tmp_path, monkeypatch
) -> None:
    cases, config = _make_test_config(tmp_path)

    def forbidden(**kwargs):
        raise AssertionError("Phase 7 must not run for an accepted base state")

    monkeypatch.setattr(termination_initialization, "generate_termination_seeds", forbidden)
    result = _generate_case(cases[0], config)
    documents = _read_case(config, cases[0])
    case_root = config.output_root / cases[0].case_id

    assert result["status"] == "feasible_seed_ready"
    assert documents["construction.json"]["deterministic"]["construction_mode"] == "exact"
    assert documents["construction.json"]["deterministic"]["basis_mode"] == "supplied"
    assert documents["construction.json"]["deterministic"][
        "default_termination_descriptor"
    ] == TerminationPair().to_dict()
    assert documents["initialization.json"]["deterministic"]["search_invoked"] is False
    assert documents["initialization.json"]["deterministic"]["outcome"] == (
        "base_exact_state_accepted"
    )
    assert (case_root / "base_state").is_dir()
    assert (case_root / "seeds" / "seed_000" / "state").is_dir()
    assert (case_root / "base_state").resolve() != (
        case_root / "seeds" / "seed_000" / "state"
    ).resolve()
    assert documents["case.json"]["deterministic"]["base"]["structure_hash"] == (
        documents["case.json"]["deterministic"]["retained_seeds"][0][
            "structure_hash"
        ]
    )
    seed_metadata = json.loads(
        (case_root / "seeds" / "seed_000" / "seed.json").read_text(encoding="utf-8")
    )
    assert seed_metadata["deterministic"]["reconstruction"]["provenance"][
        "case_id"
    ] == cases[0].case_id


@pytest.mark.parametrize(
    ("seed_kinds", "expected_kinds"),
    [
        pytest.param(
            ("default_translation",),
            ("default_termination_translation",),
            id="default-translation",
        ),
        pytest.param(
            ("nondefault_zero",),
            ("nondefault_termination_zero",),
            id="nondefault-zero",
        ),
        pytest.param(
            ("termination_translation",),
            ("termination_plus_translation",),
            id="termination-translation",
        ),
        pytest.param(
            (
                "default_translation",
                "nondefault_zero",
                "termination_translation",
            ),
            (
                "default_termination_translation",
                "nondefault_termination_zero",
                "termination_plus_translation",
            ),
            id="multiple-retained-order",
        ),
    ],
)
def test_phase7_seed_kinds_and_order_are_persisted_without_reranking(
    tmp_path, monkeypatch, seed_kinds, expected_kinds
) -> None:
    clean = {
        "feasibility_policy": _strict_contact_policy(),
        "max_seeds": len(seed_kinds),
        "termination_domain": {"mode": "all"},
    }
    cases, config = _make_test_config(tmp_path, clean_payload=clean)
    monkeypatch.setattr(
        termination_initialization,
        "generate_termination_seeds",
        _fake_phase7_factory(seed_kinds),
    )

    result = _generate_case(cases[0], config)
    documents = _read_case(config, cases[0])
    retained = documents["case.json"]["deterministic"]["retained_seeds"]
    phase7 = documents["initialization.json"]["deterministic"]["phase7_result"]

    assert result["status"] == "feasible_seed_ready"
    assert tuple(item["kind"] for item in retained) == expected_kinds
    assert [item["order"] for item in retained] == list(range(len(seed_kinds)))
    assert len({item["structure_hash"] for item in retained}) == len(seed_kinds)
    assert phase7["attempts"][0]["nested_translation_result"] is not None or (
        seed_kinds[0] == "nondefault_zero"
    )
    assert documents["initialization.json"]["deterministic"][
        "phase7_result_hash"
    ] == phase7["result_hash"]


@pytest.mark.parametrize(
    ("clean_payload", "expected_status", "expected_stage"),
    [
        pytest.param(
            {
                "feasibility_policy": _strict_contact_policy(),
                "termination_domain": {"mode": "default_only"},
            },
            "translation_search_exhausted",
            "translation_domain",
            id="translation-only-exhaustion",
        ),
        pytest.param(
            {
                "feasibility_policy": _strict_contact_policy(),
                "termination_domain": {"mode": "all"},
            },
            "termination_search_exhausted",
            "termination_translation_domain",
            id="termination-exhaustion",
        ),
    ],
)
def test_real_phase7_exhaustion_status_mapping(
    tmp_path, clean_payload, expected_status, expected_stage
) -> None:
    cases, config = _make_test_config(tmp_path, clean_payload=clean_payload)

    result = _generate_case(cases[0], config)
    initialization = _read_case(config, cases[0])["initialization.json"]

    assert result["status"] == expected_status
    assert result["failure_stage"] == expected_stage
    assert initialization["deterministic"]["search_invoked"] is True
    assert initialization["deterministic"]["phase7_result"]["attempts"]
    assert not initialization["deterministic"]["retained_seeds"]


def test_unsupported_exact_termination_selection_is_actionable_initializer_failure(
    tmp_path,
) -> None:
    clean = {
        "feasibility_policy": _strict_contact_policy(),
        "termination_domain": {
            "mode": "explicit",
            "left": ["0", "1/3"],
            "right": ["0"],
        },
    }
    cases, config = _make_test_config(tmp_path, clean_payload=clean)

    result = _generate_case(cases[0], config)
    initialization = _read_case(config, cases[0])["initialization.json"]

    assert result["status"] == "seed_generation_failed"
    assert result["failure_stage"] == "initializer_input"
    assert "initialization.unsupported_termination_selection" in result["reason_codes"]
    assert initialization["deterministic"]["outcome"] == "invalid_configuration"


def test_unexpected_phase7_failure_preserves_base_and_actionable_reports(
    tmp_path, monkeypatch
) -> None:
    cases, config = _make_test_config(
        tmp_path,
        clean_payload={"feasibility_policy": _strict_contact_policy()},
    )

    def explode(**kwargs):
        raise RuntimeError("synthetic phase7 failure")

    monkeypatch.setattr(termination_initialization, "generate_termination_seeds", explode)
    result = _generate_case(cases[0], config)
    documents = _read_case(config, cases[0])

    assert result["status"] == "seed_generation_failed"
    assert result["failure_stage"] == "phase7_seed_generation"
    assert (config.output_root / cases[0].case_id / "base_state").is_dir()
    assert documents["case.json"]["deterministic"]["base"] is not None
    assert documents["initialization.json"]["deterministic"]["outcome"] == "error"


def test_campaign_stoichiometry_failure_persists_constructed_base(tmp_path) -> None:
    cases, config = _make_test_config(tmp_path, expected_ratio=(1, 3))

    result = _generate_case(cases[0], config)
    documents = _read_case(config, cases[0])
    root = config.output_root / cases[0].case_id

    assert result["status"] == "construction_failed"
    assert result["failure_stage"] == "decorated_population_or_stoichiometry"
    assert (root / "base_state").is_dir()
    assert documents["case.json"]["deterministic"]["base"] is not None
    assert documents["construction.json"]["deterministic"][
        "campaign_stoichiometry_check"
    ]["passed"] is False


def test_complete_case_resumes_and_tampered_seed_regenerates(tmp_path) -> None:
    cases, config = _make_test_config(tmp_path)
    generated = _generate_case(cases[0], config)

    resumed = _existing_result(cases[0], config)
    assert resumed is not None
    assert resumed["resumed"] is True
    assert resumed["status"] == generated["status"]

    report_path = (
        config.output_root / cases[0].case_id / "seeds" / "seed_000" / "report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "warning"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    assert _existing_result(cases[0], config) is None


def test_case_and_campaign_report_schemas_and_source_order_are_deterministic(tmp_path) -> None:
    rows = [_identity_row(boundary_type="TW"), _identity_row(boundary_type="ST")]
    cases, config = _make_test_config(tmp_path, rows=rows)
    results = [_generate_case(case, config) for case in reversed(cases)]
    _write_campaign_outputs(config.output_root, config=config, results=results)

    first_documents = _read_case(config, cases[0])
    construction_campaign = json.loads(
        (config.output_root / "construction_report.json").read_text(encoding="utf-8")
    )
    initialization_campaign = json.loads(
        (config.output_root / "initialization_report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((config.output_root / "manifest.json").read_text(encoding="utf-8"))

    assert _hashed_document_matches(
        first_documents["construction.json"], schema=_CONSTRUCTION_SCHEMA
    )
    assert _hashed_document_matches(
        first_documents["initialization.json"], schema=_INITIALIZATION_SCHEMA
    )
    assert [item["source_row"] for item in construction_campaign["cases"]] == [1, 2]
    assert [item["source_row"] for item in initialization_campaign["cases"]] == [1, 2]
    assert [item["source_row"] for item in manifest["cases"]] == [1, 2]
    assert manifest["effective_configuration_hash"] == (
        config.clean_settings.configuration_hash
    )
    assert manifest["software_identity"] == config.software_identity


def test_strict_warning_retention_rule_controls_base_acceptance(tmp_path, monkeypatch) -> None:
    original = geometry_validation.validate_bicrystal_state

    def force_warning(state, *, policy, override=None):
        return original(
            state,
            policy=policy,
            override=FeasibilityOverride("warning", "synthetic warning decision"),
        )

    monkeypatch.setattr(geometry_validation, "validate_bicrystal_state", force_warning)

    strict_cases, strict_config = _make_test_config(
        tmp_path / "strict",
        clean_payload={"initialization_enabled": False},
    )
    strict_result = _generate_case(strict_cases[0], strict_config)
    assert strict_result["status"] == "constructed_infeasible"

    retained_cases, retained_config = _make_test_config(
        tmp_path / "retained",
        clean_payload={"retain_warnings": True},
    )
    retained_result = _generate_case(retained_cases[0], retained_config)
    assert retained_result["status"] == "feasible_seed_ready"
    assert retained_result["base_feasibility_status"] == "warning"


@pytest.mark.parametrize(
    ("mode", "expected_process_state", "expected_stage"),
    [
        pytest.param("timeout", "timeout", "child_timeout", id="timeout"),
        pytest.param("launch", "launch_error", "child_launch", id="launch-error"),
        pytest.param("signal", "signaled", "child_process", id="signal"),
        pytest.param("no_result", "no_result", "child_process", id="no-result"),
    ],
)
def test_child_process_diagnostics_never_become_generic_campaign_statuses(
    tmp_path, monkeypatch, mode, expected_process_state, expected_stage
) -> None:
    cases, config = _make_test_config(tmp_path)

    def fake_run(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(args[0], config.timeout, output="partial")
        if mode == "launch":
            raise OSError("synthetic launch failure")
        if mode == "signal":
            return SimpleNamespace(returncode=-9, stdout="", stderr="terminated")
        return SimpleNamespace(returncode=3, stdout="diagnostic only", stderr="failure")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = _execute_case(cases[0], script_path=Path("generate_structures.py"), config=config)

    assert result["status"] == "seed_generation_failed"
    assert result["process_state"] == expected_process_state
    assert result["failure_stage"] == expected_stage
    assert result["status"] not in {
        "failed",
        "timeout",
        "signaled",
        "launch_error",
        "no_result",
        "internal_error",
    }


def test_list_selection_does_not_construct_cases(tmp_path, monkeypatch, capsys) -> None:
    data_file = _write_csv(tmp_path / "campaign.csv")
    output_root = tmp_path / "out"
    args = _build_parser().parse_args(
        [
            "--data-file",
            str(data_file),
            "--output-root",
            str(output_root),
            "--expected-cases",
            "1",
            "--list",
        ]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("--list must not execute a case")

    monkeypatch.setattr("generate_structures._execute_case", forbidden)
    assert _run_parent(args) == 0
    output = capsys.readouterr().out
    assert "zhang_001_TW_100" in output
    assert not any(output_root.glob("zhang_*"))


def test_generation_signature_changes_with_effective_clean_setting(tmp_path) -> None:
    data_file = _write_csv(tmp_path / "campaign.csv")
    output_root = tmp_path / "out"
    output_root.mkdir()

    def make(extra: list[str]):
        args = _build_parser().parse_args(
            [
                "--data-file",
                str(data_file),
                "--output-root",
                str(output_root),
                "--expected-cases",
                "1",
                *extra,
            ]
        )
        return _make_config(args, data_file=data_file, output_root=output_root)

    baseline = make([])
    variants = [
        make(["--retain-warnings"]),
        make(["--max-seeds", "2"]),
        make(["--normal-offsets", "0", "0.25"]),
        make(["--disable-termination-search"]),
    ]
    signatures = {
        baseline.generation_signature,
        *(item.generation_signature for item in variants),
    }
    assert len(signatures) == 5


def test_state_atomic_write_removes_partial_directory_on_interruption(
    tmp_path,
) -> None:
    target = tmp_path / "state"

    class InterruptedState:
        structure_hash = "structure"
        state_hash = "state"

        def save(self, directory):
            directory.mkdir(parents=True)
            (directory / "partial").write_text("partial", encoding="utf-8")
            raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _save_state_atomic(InterruptedState(), target)

    assert not target.exists()
    assert not list(tmp_path.glob(".state.tmp-*"))


def test_case_directory_swap_restores_previous_complete_case_on_interruption(
    tmp_path, monkeypatch
) -> None:
    target = tmp_path / "case"
    staging = tmp_path / ".case.tmp"
    target.mkdir()
    staging.mkdir()
    (target / "marker").write_text("old", encoding="utf-8")
    (staging / "marker").write_text("new", encoding="utf-8")
    real_replace = os.replace
    failed = False

    def interrupted_replace(source, destination):
        nonlocal failed
        if Path(source) == staging and not failed:
            failed = True
            raise OSError("synthetic rename interruption")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="synthetic rename interruption"):
        _swap_case_directory(staging, target)

    assert (target / "marker").read_text(encoding="utf-8") == "old"
    assert not staging.exists()


def test_single_interface_slab_campaign_persists_real_surface_fixed_and_buffer_regions(
    tmp_path,
) -> None:
    clean = {
        "topology": "single_interface_slab",
        "boundary_conditions": ["fixed", "periodic", "periodic"],
        "vacuum_angstrom": 4.0,
        "fixed_region_thickness_angstrom": 0.5,
        "surface_buffer_thickness_angstrom": 0.5,
        "override": {
            "status": "feasible",
            "reason": "test slab topology contract",
        },
    }
    cases, config = _make_test_config(tmp_path, clean_payload=clean)

    result = _generate_case(cases[0], config)
    case_root = config.output_root / cases[0].case_id
    state = BicrystalState.load(case_root / "base_state")
    construction = json.loads(
        (case_root / "construction.json").read_text(encoding="utf-8")
    )

    assert result["status"] == "feasible_seed_ready"
    assert state.topology == "single_interface_slab"
    assert state.boundary_conditions == ("fixed", "periodic", "periodic")
    assert len(state.external_surfaces) == 2
    assert len(state.vacuum_regions) == 2
    assert len(state.fixed_regions) == 2
    assert len(state.buffer_regions) == 2
    assert construction["deterministic"]["base"]["topology"] == "single_interface_slab"
    assert construction["deterministic"]["base"]["feasibility_report"]["override"]["reason"] == (
        "test slab topology contract"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_zhang_001_reduced_exact_clean_generation_is_actionable(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[1]
    data_file = repository / "gb_data_gbopt.csv"
    output_root = tmp_path / "out"
    output_root.mkdir()
    args = _build_parser().parse_args(
        [
            "--data-file",
            str(data_file),
            "--output-root",
            str(output_root),
            "--expected-cases",
            "197",
            "--x-dim-min",
            "5",
            "--gb-thickness-periods",
            "0",
            "--interaction-distance",
            "0",
            "--repeat-factor",
            "1",
            "1",
            "--disable-termination-search",
        ]
    )
    config = _make_config(args, data_file=data_file, output_root=output_root)
    case = _load_cases(data_file, expected_cases=197)[0]

    result = _generate_case(case, config)
    case_root = output_root / case.case_id
    initialization = json.loads(
        (case_root / "initialization.json").read_text(encoding="utf-8")
    )

    assert case.case_id == "zhang_001_ST_100"
    assert result["status"] in {
        "feasible_seed_ready",
        "translation_search_exhausted",
        "seed_generation_failed",
    }
    assert (case_root / "base_state").is_dir()
    assert (case_root / "construction.json").is_file()
    assert (
        initialization["deterministic"]["reason_codes"]
        or result["status"] == "feasible_seed_ready"
    )
    if result["status"] != "feasible_seed_ready":
        assert initialization["deterministic"]["phase7_result"]["attempts"]


def test_parent_child_end_to_end_updates_reports_and_resumes(tmp_path) -> None:
    data_file = _write_csv(tmp_path / "campaign.csv")
    output_root = tmp_path / "out"
    argv = [
        "--data-file",
        str(data_file),
        "--output-root",
        str(output_root),
        "--expected-cases",
        "1",
        "--workers",
        "1",
        "--timeout",
        "30",
        "--x-dim-min",
        "5",
        "--gb-thickness-periods",
        "0",
        "--repeat-factor",
        "1",
        "1",
        "--interaction-distance",
        "0",
    ]

    assert _run_parent(_build_parser().parse_args(argv)) == 0
    for filename in (
        "clean_generation_results.tsv",
        "manifest.json",
        "construction_report.json",
        "initialization_report.json",
    ):
        assert (output_root / filename).is_file()

    assert _run_parent(_build_parser().parse_args(argv)) == 0
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cases"][0]["status"] == "feasible_seed_ready"
    assert manifest["cases"][0]["resumed"] is True


def test_process_failure_does_not_overwrite_a_complete_verified_case(tmp_path) -> None:
    cases, config = _make_test_config(tmp_path)
    generated = _generate_case(cases[0], config)
    case_json = config.output_root / cases[0].case_id / "case.json"
    original = case_json.read_bytes()
    result = {
        "case_id": cases[0].case_id,
        "source_row": cases[0].source_row,
        "process_state": "no_result",
        "returncode": 1,
        "signal": "",
        "stdout_tail": "",
        "stderr_tail": "synthetic",
    }

    _persist_process_failure(cases[0], config, result)

    assert result["status"] == generated["status"]
    assert result["process_state"] == "recovered_after_no_result"
    assert case_json.read_bytes() == original
