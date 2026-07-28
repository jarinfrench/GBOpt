#!/usr/bin/env python3
"""Summarize geometry-audit results from a Zhang generation-results TSV.

The report is descriptive only. It counts audit statuses, breaks suspicious frequency
out by campaign descriptors, reports classification reasons, and summarizes normalized
severity metrics. No threshold in this report rejects generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, Sequence


_REQUIRED_FIELDS = (
    "case_id",
    "status",
    "audit_status",
    "audit_reasons",
    "boundary_type",
    "axis_set",
    "p_det_abs",
    "q_det_abs",
    "max_miller_row_norm",
    "natoms",
    "box_x_angstrom",
    "box_y_angstrom",
    "box_z_angstrom",
    "central_gap_median_angstrom",
    "central_gap_p95_angstrom",
    "central_gap_range_angstrom",
    "periodic_gap_median_angstrom",
    "periodic_gap_p95_angstrom",
    "periodic_gap_range_angstrom",
    "central_empty_left_fraction",
    "central_empty_right_fraction",
    "periodic_empty_left_fraction",
    "periodic_empty_right_fraction",
    "left_internal_min_angstrom",
    "right_internal_min_angstrom",
    "central_cross_min_angstrom",
    "periodic_cross_min_angstrom",
    "periodic_duplicate_count",
)

_WARNING_THRESHOLDS = (
    (
        "Empty-bin fraction",
        "> 0.25",
        "Either side of either interface",
    ),
    (
        "Gap range / bulk nearest-neighbor distance",
        "> 2.0",
        "Central or periodic interface",
    ),
    (
        "(p95 - median) gap / bulk nearest-neighbor distance",
        "> 1.0",
        "Central or periodic interface",
    ),
    (
        "Cross-interface minimum / bulk nearest-neighbor distance",
        "< 0.45",
        "Central or periodic interface",
    ),
    (
        "Periodic duplicate separation",
        "<= 1.0e-6 A",
        "Any fully periodic atom pair",
    ),
)

_BUCKET_ORDER = {
    "<=10": 0,
    "11-100": 1,
    "101-1000": 2,
    ">1000": 3,
    "<=5": 0,
    "5-10": 1,
    "10-25": 2,
    ">25": 3,
    "<=25k": 0,
    "25k-100k": 1,
    "100k-250k": 2,
    ">250k": 3,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="generation_results.tsv path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown output path; defaults to stdout.",
    )
    return parser


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"TSV has no header: {path}")
        missing = [
            field for field in _REQUIRED_FIELDS if field not in reader.fieldnames
        ]
        if missing:
            raise ValueError("TSV is missing Phase-A fields: " + ", ".join(missing))
        return list(reader)


def _float(row: dict[str, str], field: str) -> float | None:
    text = row.get(field, "").strip()
    if not text:
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def _status(row: dict[str, str]) -> str:
    audit = row.get("audit_status", "").strip()
    if audit:
        return audit
    generation = row.get("status", "unknown").strip() or "unknown"
    return f"generation_{generation}"


def _reasons(row: dict[str, str]) -> tuple[str, ...]:
    text = row.get("audit_reasons", "").strip()
    if not text:
        return ()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid audit_reasons JSON for {row.get('case_id', '<unknown>')}: "
            f"{text!r}"
        ) from exc

    if not isinstance(payload, list) or not all(
        isinstance(reason, str) for reason in payload
    ):
        raise ValueError(
            "audit_reasons must be a JSON list of strings for "
            f"{row.get('case_id', '<unknown>')}"
        )
    return tuple(payload)


def _table(headers: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> list[str]:
    materialized = list(rows)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    if materialized:
        lines.extend(
            "| " + " | ".join(str(value) for value in row) + " |"
            for row in materialized
        )
    else:
        empty_row = (
            "No data" if index == 0 else ""
            for index in range(len(headers))
        )
        lines.append("| " + " | ".join(empty_row) + " |")
    return lines


def _categorical_breakdown(
    rows: list[dict[str, str]], field: str
) -> list[tuple[object, ...]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get(field, "") or "<blank>"].append(row)

    result = []
    for key in sorted(grouped):
        values = grouped[key]
        suspicious = sum(_status(row) in {"suspicious", "invalid"} for row in values)
        result.append((key, len(values), suspicious, f"{suspicious / len(values):.1%}"))
    return result


def _bucket_breakdown(
    rows: list[dict[str, str]],
    field: str,
    bucket: Callable[[float], str],
) -> list[tuple[object, ...]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        value = _float(row, field)
        if value is not None:
            grouped[bucket(value)].append(row)

    result = []
    for key in sorted(
        grouped,
        key=lambda value: (_BUCKET_ORDER.get(value, 999), value),
    ):
        values = grouped[key]
        suspicious = sum(_status(row) in {"suspicious", "invalid"} for row in values)
        result.append((key, len(values), suspicious, f"{suspicious / len(values):.1%}"))
    return result


def _numeric_summary(
    rows: list[dict[str, str]], field: str
) -> list[tuple[object, ...]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row, field)
        if value is not None:
            grouped[_status(row)].append(value)

    result = []
    for status in sorted(grouped):
        values = grouped[status]
        result.append(
            (
                status,
                len(values),
                f"{min(values):.6g}",
                f"{median(values):.6g}",
                f"{max(values):.6g}",
            )
        )
    return result


def _reason_frequency(
    rows: Sequence[dict[str, str]],
) -> list[tuple[str, int]]:
    counts = Counter(reason for row in rows for reason in _reasons(row))
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _reason_combinations(
    rows: Sequence[dict[str, str]], limit: int = 15
) -> list[tuple[str, int]]:
    counts = Counter(_reasons(row) for row in rows if _reasons(row))
    ordered = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [("; ".join(reasons), count) for reasons, count in ordered[:limit]]


def _bulk_reference(row: dict[str, str]) -> float | None:
    values = [
        _float(row, "left_internal_min_angstrom"),
        _float(row, "right_internal_min_angstrom"),
    ]
    finite = [value for value in values if value is not None and value > 0.0]
    return min(finite) if finite else None


def _nearest_rank_percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")

    ordered = sorted(values)
    rank = max(1, math.ceil(probability * len(ordered)))
    return ordered[rank - 1]


def _normalized_severity_values(
    rows: Sequence[dict[str, str]],
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        if _status(row) not in {"suspicious", "invalid"}:
            continue

        bulk = _bulk_reference(row)
        if bulk is None:
            continue

        direct_metrics = (
            ("Central gap range / bulk", "central_gap_range_angstrom"),
            ("Periodic gap range / bulk", "periodic_gap_range_angstrom"),
            ("Central cross distance / bulk", "central_cross_min_angstrom"),
            ("Periodic cross distance / bulk", "periodic_cross_min_angstrom"),
        )
        for label, field in direct_metrics:
            value = _float(row, field)
            if value is not None:
                result[label].append(value / bulk)

        tail_metrics = (
            (
                "Central gap tail / bulk",
                "central_gap_p95_angstrom",
                "central_gap_median_angstrom",
            ),
            (
                "Periodic gap tail / bulk",
                "periodic_gap_p95_angstrom",
                "periodic_gap_median_angstrom",
            ),
        )
        for label, p95_field, median_field in tail_metrics:
            p95 = _float(row, p95_field)
            middle = _float(row, median_field)
            if p95 is not None and middle is not None:
                result[label].append((p95 - middle) / bulk)

    return result


def _normalized_severity_summary(
    rows: Sequence[dict[str, str]],
) -> list[tuple[object, ...]]:
    values_by_metric = _normalized_severity_values(rows)
    triggers = {
        "Central gap range / bulk": "> 2.0",
        "Periodic gap range / bulk": "> 2.0",
        "Central gap tail / bulk": "> 1.0",
        "Periodic gap tail / bulk": "> 1.0",
        "Central cross distance / bulk": "< 0.45",
        "Periodic cross distance / bulk": "< 0.45",
    }
    order = (
        "Central gap range / bulk",
        "Periodic gap range / bulk",
        "Central gap tail / bulk",
        "Periodic gap tail / bulk",
        "Central cross distance / bulk",
        "Periodic cross distance / bulk",
    )

    result = []
    for label in order:
        values = values_by_metric.get(label, [])
        if not values:
            continue
        result.append(
            (
                label,
                len(values),
                f"{min(values):.4f}",
                f"{median(values):.4f}",
                f"{_nearest_rank_percentile(values, 0.95):.4f}",
                f"{max(values):.4f}",
                triggers[label],
            )
        )
    return result


def _empty_and_duplicate_summary(
    rows: Sequence[dict[str, str]],
) -> list[tuple[object, ...]]:
    metrics = (
        ("Central empty-left fraction", "central_empty_left_fraction"),
        ("Central empty-right fraction", "central_empty_right_fraction"),
        ("Periodic empty-left fraction", "periodic_empty_left_fraction"),
        ("Periodic empty-right fraction", "periodic_empty_right_fraction"),
    )
    result: list[tuple[object, ...]] = []
    for label, field in metrics:
        values = [
            value
            for row in rows
            if (value := _float(row, field)) is not None
        ]
        if values:
            result.append((label, f"{max(values):.6g}"))

    duplicate_cases = 0
    maximum_duplicates = 0
    for row in rows:
        count = _float(row, "periodic_duplicate_count")
        if count is None:
            continue
        integer_count = int(count)
        maximum_duplicates = max(maximum_duplicates, integer_count)
        if integer_count > 0:
            duplicate_cases += 1
    result.extend(
        (
            ("Cases with periodic duplicates", duplicate_cases),
            ("Maximum duplicate-pair count", maximum_duplicates),
        )
    )
    return result


def _determinant_bucket(value: float) -> str:
    if value <= 10:
        return "<=10"
    if value <= 100:
        return "11-100"
    if value <= 1000:
        return "101-1000"
    return ">1000"


def _miller_norm_bucket(value: float) -> str:
    if value <= 5:
        return "<=5"
    if value <= 10:
        return "5-10"
    if value <= 25:
        return "10-25"
    return ">25"


def _atom_count_bucket(value: float) -> str:
    if value <= 25_000:
        return "<=25k"
    if value <= 100_000:
        return "25k-100k"
    if value <= 250_000:
        return "100k-250k"
    return ">250k"


def build_report(rows: list[dict[str, str]], source: Path) -> str:
    """Return a Markdown geometry-audit summary."""
    status_counts = Counter(_status(row) for row in rows)
    lines = [
        "# Zhang Geometry-Audit Summary",
        "",
        f"Source: `{source}`",
        "",
        "> **Classification policy:** Phase 1 audit statuses are descriptive and "
        "warning-only. No threshold in this report rejects generation.",
        "",
        "## Classification thresholds",
        "",
    ]
    lines.extend(_table(("Diagnostic", "Trigger", "Scope"), _WARNING_THRESHOLDS))

    lines.extend(["", "## Audit status", ""])
    lines.extend(_table(("Status", "Cases"), sorted(status_counts.items())))

    for heading, field in (
        ("Boundary type", "boundary_type"),
        ("Axis set", "axis_set"),
    ):
        lines.extend(["", f"## Suspicious frequency by {heading.lower()}", ""])
        lines.extend(
            _table(
                (heading, "Cases", "Suspicious/invalid", "Rate"),
                _categorical_breakdown(rows, field),
            )
        )

    bucket_specs = (
        (
            "maximum P/Q determinant",
            "max_det_abs",
            _determinant_bucket,
        ),
        (
            "maximum Miller-row norm",
            "max_miller_row_norm",
            _miller_norm_bucket,
        ),
        (
            "atom count",
            "natoms",
            _atom_count_bucket,
        ),
    )

    normalized_rows = []
    for row in rows:
        copied = dict(row)
        p_det = _float(row, "p_det_abs")
        q_det = _float(row, "q_det_abs")
        copied["max_det_abs"] = (
            "" if p_det is None or q_det is None else str(max(p_det, q_det))
        )
        normalized_rows.append(copied)

    for heading, field, bucket in bucket_specs:
        lines.extend(["", f"## Suspicious frequency by {heading}", ""])
        lines.extend(
            _table(
                ("Bucket", "Cases", "Suspicious/invalid", "Rate"),
                _bucket_breakdown(normalized_rows, field, bucket),
            )
        )

    lines.extend(["", "## Classification reasons", ""])
    lines.extend(_table(("Reason", "Cases"), _reason_frequency(rows)))

    lines.extend(["", "### Reasons by boundary type", ""])
    boundary_types = sorted(
        {row.get("boundary_type", "") or "<blank>" for row in rows}
    )
    for boundary_type in boundary_types:
        subset = [
            row
            for row in rows
            if (row.get("boundary_type", "") or "<blank>") == boundary_type
        ]
        lines.extend([f"#### {boundary_type}", ""])
        lines.extend(_table(("Reason", "Cases"), _reason_frequency(subset)))
        lines.append("")

    lines.extend(["## Most common reason combinations", ""])
    lines.extend(
        _table(
            ("Reasons", "Cases"),
            _reason_combinations(rows),
        )
    )

    lines.extend(["", "## Normalized severity", ""])
    lines.extend(
        [
            "Only `suspicious` and `invalid` cases are included. The bulk "
            "reference is the smaller of the left- and right-grain internal "
            "minimum distances.",
            "",
        ]
    )
    lines.extend(
        _table(
            ("Metric", "Cases", "Minimum", "Median", "P95", "Maximum", "Trigger"),
            _normalized_severity_summary(rows),
        )
    )

    lines.extend(["", "## Empty-bin and duplicate diagnostics", ""])
    lines.extend(
        _table(
            ("Diagnostic", "Campaign value"),
            _empty_and_duplicate_summary(rows),
        )
    )

    lines.extend(["", "## Numeric ranges by audit status", ""])
    for label, field in (
        ("Maximum Miller-row norm", "max_miller_row_norm"),
        ("Atom count", "natoms"),
        ("Box x (A)", "box_x_angstrom"),
        ("Box y (A)", "box_y_angstrom"),
        ("Box z (A)", "box_z_angstrom"),
    ):
        lines.extend([f"### {label}", ""])
        lines.extend(
            _table(
                ("Audit status", "Cases", "Minimum", "Median", "Maximum"),
                _numeric_summary(normalized_rows, field),
            )
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = _parser().parse_args()
    results = args.results.expanduser().resolve()
    rows = _read_rows(results)
    report = build_report(rows, results)
    if args.output is None:
        print(report, end="")
    else:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
