# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Tests for persisted Phase 4 feasibility report integrity checks."""

from __future__ import annotations

from copy import deepcopy

import pytest

from generate_structures import _canonical_sha256, _feasibility_report_matches


def _report() -> dict:
    payload = {
        "raw_status": "feasible",
        "status": "feasible",
        "reasons": [],
        "raw_reasons": [],
        "duplicate_pairs": [],
        "interfaces": [],
        "slab": None,
        "structure_hash": "structure",
        "state_hash": "state",
        "policy": {"contact": {}, "void": {}, "slab": {}},
        "override": None,
    }
    return {
        **payload,
        "policy_hash": _canonical_sha256(payload["policy"]),
        "report_hash": _canonical_sha256(payload),
    }


def test_persisted_feasibility_report_hashes_are_accepted() -> None:
    assert _feasibility_report_matches(_report())


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        pytest.param("status", "warning", id="report-payload"),
        pytest.param("report_hash", "bad", id="report-hash"),
        pytest.param("policy_hash", "bad", id="policy-hash"),
    ],
)
def test_persisted_feasibility_report_tampering_is_rejected(
    mutation: str, value: object
) -> None:
    report = deepcopy(_report())
    report[mutation] = value

    assert not _feasibility_report_matches(report)


def test_missing_feasibility_hash_is_rejected() -> None:
    report = _report()
    del report["report_hash"]

    assert not _feasibility_report_matches(report)
