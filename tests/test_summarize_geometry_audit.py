"""Tests for the campaign geometry-audit summary report."""

from pathlib import Path

from summarize_geometry_audit import build_report


def _row(
    *,
    case_id: str,
    audit_status: str,
    boundary_type: str,
    axis_set: str,
    p_det_abs: str,
    max_miller_row_norm: str,
    natoms: str,
    reasons: str = "[]",
    central_range: str = "2.0",
    periodic_range: str = "2.0",
    central_median: str = "1.0",
    central_p95: str = "2.0",
    periodic_median: str = "1.0",
    periodic_p95: str = "2.0",
    central_cross: str = "1.0",
    periodic_cross: str = "1.0",
) -> dict[str, str]:
    return {
        "case_id": case_id,
        "status": "generated",
        "audit_status": audit_status,
        "audit_reasons": reasons,
        "boundary_type": boundary_type,
        "axis_set": axis_set,
        "p_det_abs": p_det_abs,
        "q_det_abs": p_det_abs,
        "max_miller_row_norm": max_miller_row_norm,
        "natoms": natoms,
        "box_x_angstrom": "100",
        "box_y_angstrom": "40",
        "box_z_angstrom": "30",
        "central_gap_median_angstrom": central_median,
        "central_gap_p95_angstrom": central_p95,
        "central_gap_range_angstrom": central_range,
        "periodic_gap_median_angstrom": periodic_median,
        "periodic_gap_p95_angstrom": periodic_p95,
        "periodic_gap_range_angstrom": periodic_range,
        "central_empty_left_fraction": "0",
        "central_empty_right_fraction": "0",
        "periodic_empty_left_fraction": "0",
        "periodic_empty_right_fraction": "0",
        "left_internal_min_angstrom": "1.0",
        "right_internal_min_angstrom": "1.0",
        "central_cross_min_angstrom": central_cross,
        "periodic_cross_min_angstrom": periodic_cross,
        "periodic_duplicate_count": "0",
    }


def test_build_report_counts_statuses_and_breakdowns():
    rows = [
        _row(
            case_id="a",
            audit_status="ok",
            boundary_type="TW",
            axis_set="100",
            p_det_abs="5",
            max_miller_row_norm="3",
            natoms="1000",
        ),
        _row(
            case_id="b",
            audit_status="suspicious",
            boundary_type="ST",
            axis_set="110",
            p_det_abs="125",
            max_miller_row_norm="12",
            natoms="50000",
            reasons='["central_interface_large_gap_range"]',
            central_range="3.0",
        ),
    ]

    report = build_report(rows, Path("generation_results.tsv"))

    assert "warning-only" in report
    assert "No threshold in this report rejects generation" in report
    assert "| ok | 1 |" in report
    assert "| suspicious | 1 |" in report
    assert "| ST | 1 | 1 | 100.0% |" in report
    assert "| TW | 1 | 0 | 0.0% |" in report
    assert "| 101-1000 | 1 | 1 | 100.0% |" in report
    assert "| 25k-100k | 1 | 1 | 100.0% |" in report


def test_build_report_includes_reason_frequency_and_combinations():
    rows = [
        _row(
            case_id="a",
            audit_status="suspicious",
            boundary_type="ST",
            axis_set="110",
            p_det_abs="20",
            max_miller_row_norm="6",
            natoms="10000",
            reasons=(
                '["central_interface_large_gap_range",'
                '"periodic_interface_large_gap_range"]'
            ),
        ),
        _row(
            case_id="b",
            audit_status="suspicious",
            boundary_type="AT",
            axis_set="100",
            p_det_abs="20",
            max_miller_row_norm="6",
            natoms="10000",
            reasons='["central_interface_large_gap_range"]',
        ),
    ]

    report = build_report(rows, Path("generation_results.tsv"))

    assert "## Classification reasons" in report
    assert "| central_interface_large_gap_range | 2 |" in report
    assert "| periodic_interface_large_gap_range | 1 |" in report
    assert "#### AT" in report
    assert "#### ST" in report
    assert (
        "| central_interface_large_gap_range; "
        "periodic_interface_large_gap_range | 1 |"
    ) in report


def test_build_report_summarizes_normalized_severity():
    rows = [
        _row(
            case_id="a",
            audit_status="suspicious",
            boundary_type="ST",
            axis_set="110",
            p_det_abs="20",
            max_miller_row_norm="6",
            natoms="10000",
            reasons='["central_interface_large_gap_range"]',
            central_range="3.0",
            periodic_range="4.0",
            central_median="1.0",
            central_p95="2.5",
            periodic_median="1.0",
            periodic_p95="2.0",
            central_cross="0.4",
            periodic_cross="0.8",
        ),
        _row(
            case_id="b",
            audit_status="ok",
            boundary_type="TW",
            axis_set="100",
            p_det_abs="5",
            max_miller_row_norm="3",
            natoms="1000",
            central_range="100.0",
        ),
    ]

    report = build_report(rows, Path("generation_results.tsv"))

    assert "## Normalized severity" in report
    assert "Only `suspicious` and `invalid` cases are included" in report
    assert (
        "| Central gap range / bulk | 1 | 3.0000 | 3.0000 | "
        "3.0000 | 3.0000 | > 2.0 |"
    ) in report
    assert (
        "| Central cross distance / bulk | 1 | 0.4000 | 0.4000 | "
        "0.4000 | 0.4000 | < 0.45 |"
    ) in report
    assert "100.0000" not in report
