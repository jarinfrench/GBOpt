"""Tests for the campaign geometry-audit summary report."""

from pathlib import Path

from summarize_geometry_audit import build_report


def test_build_report_counts_statuses_and_breakdowns():
    rows = [
        {
            "case_id": "a",
            "status": "generated",
            "audit_status": "ok",
            "boundary_type": "TW",
            "axis_set": "100",
            "p_det_abs": "5",
            "q_det_abs": "5",
            "max_miller_row_norm": "3",
            "natoms": "1000",
            "box_x_angstrom": "50",
            "box_y_angstrom": "20",
            "box_z_angstrom": "20",
        },
        {
            "case_id": "b",
            "status": "generated",
            "audit_status": "suspicious",
            "boundary_type": "ST",
            "axis_set": "110",
            "p_det_abs": "125",
            "q_det_abs": "125",
            "max_miller_row_norm": "12",
            "natoms": "50000",
            "box_x_angstrom": "100",
            "box_y_angstrom": "40",
            "box_z_angstrom": "30",
        },
    ]

    report = build_report(rows, Path("generation_results.tsv"))

    assert "| ok | 1 |" in report
    assert "| suspicious | 1 |" in report
    assert "| ST | 1 | 1 | 100.0% |" in report
    assert "| TW | 1 | 0 | 0.0% |" in report
    assert "| 101-1000 | 1 | 1 | 100.0% |" in report
    assert "| 25k-100k | 1 | 1 | 100.0% |" in report
