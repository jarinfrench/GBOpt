#!/usr/bin/env python3
"""Generate and audit the Zhang UO2 Phase 2 campaign.

Each selected case runs in an isolated child process. The parent writes a metrics table,
manifest, audit/performance report, outlier table, complete LAMMPS data files, two
lightweight visualization GIFs, and closeout documentation for the exact decorated-site
implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

RESULT_MARKER = "__PHASE2_CLOSEOUT_RESULT__="
SCHEMA = 2
EXPECTED_CASES = 197
MATRIX_ROWS = ("x", "y", "z")
RESULT_FIELDS = (
    "case_id", "source_row", "status", "boundary_type", "axis_set",
    "audit_status", "audit_reasons", "natoms", "left_atoms", "right_atoms",
    "u_atoms", "o_atoms", "left_origin_representatives", "right_origin_representatives",
    "projected_central_gap_a", "projected_periodic_gap_a",
    "central_gap_min_a", "central_gap_median_a", "central_gap_p95_a",
    "central_gap_max_a", "central_gap_range_a", "central_empty_left_fraction",
    "central_empty_right_fraction", "periodic_gap_min_a", "periodic_gap_median_a",
    "periodic_gap_p95_a", "periodic_gap_max_a", "periodic_gap_range_a",
    "periodic_empty_left_fraction", "periodic_empty_right_fraction",
    "bulk_reference_a", "left_internal_min_a", "right_internal_min_a",
    "central_cross_min_a", "periodic_cross_min_a", "periodic_duplicate_count",
    "bins_y", "bins_z", "probe_s", "build_s", "audit_s", "total_s",
    "peak_rss_mib", "warning_count", "warnings", "box_x_a", "box_y_a", "box_z_a",
    "data_file", "data_sha256", "preview_file", "error_type", "message",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--geometry-audit-file", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preview-points", type=int, default=5000)
    parser.add_argument("--no-gifs", action="store_true")
    parser.add_argument(
        "--charges",
        type=float,
        nargs=2,
        metavar=("U_CHARGE", "O_CHARGE"),
        default=(2.4, -1.2),
        help="LAMMPS charges for U and O atoms (default: 2.4 -1.2).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Decimal precision in LAMMPS data files (default: 12).",
    )
    parser.add_argument("--baseline-ok", type=int, default=36)
    parser.add_argument("--baseline-suspicious", type=int, default=161)
    parser.add_argument("--baseline-invalid", type=int, default=0)
    parser.add_argument("--lattice-constant", type=float, default=5.454)
    parser.add_argument("--x-dim-min", type=float, default=60.0)
    parser.add_argument("--gb-thickness-periods", type=float, default=2.0)
    parser.add_argument("--interaction-distance", type=float, default=11.0)
    parser.add_argument("--mismatch-tol", type=float, default=0.005)
    parser.add_argument("--mismatch-max-cells", type=int, default=50)
    parser.add_argument("--strain-grain", choices=("left", "right", "both"), default="both")
    parser.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    cases: list[dict[str, Any]] = []
    for source_row, row in enumerate(rows, start=1):
        boundary_type = row["Type"].strip()
        axis_set = row["Axis Set"].strip()
        case_id = f"zhang_{source_row:03d}_{boundary_type}_{axis_set}"
        def matrix(prefix: str) -> list[list[int]]:
            return [
                [int(row[f"{prefix}_{axis}{index}"]) for index in range(3)]
                for axis in MATRIX_ROWS
            ]
        cases.append({
            "case_id": case_id,
            "source_row": source_row,
            "boundary_type": boundary_type,
            "axis_set": axis_set,
            "P": matrix("P"),
            "Q": matrix("Q"),
        })
    return cases


def _load_audit_module(path: Path | None):
    if path is None:
        try:
            from GBOpt import geometry_audit as module  # type: ignore
        except ImportError:
            import geometry_audit as module  # type: ignore
        return module
    spec = importlib.util.spec_from_file_location("phase2_geometry_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import geometry-audit module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _peak_rss_mib() -> float | None:
    if resource is None:
        return None
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _count_species(atoms: Any) -> dict[str, int]:
    import numpy as np
    names, counts = np.unique(atoms["name"], return_counts=True)
    return {str(name): int(count) for name, count in zip(names, counts, strict=True)}


def _validate_atoms(atoms: Any, box: Any, *, label: str) -> None:
    import numpy as np
    if len(atoms) == 0:
        raise RuntimeError(f"{label} is empty")
    coords = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
    if not np.all(np.isfinite(coords)):
        raise RuntimeError(f"{label} contains non-finite coordinates")
    bounds = np.asarray(box, dtype=float)
    tol = 2.0e-8
    lower = bounds[:, 0] - tol
    upper = bounds[:, 1] + tol
    if np.any(coords < lower) or np.any(coords > upper):
        raise RuntimeError(f"{label} contains atoms outside the simulation box")


def _sample_positions(atoms: Any, count: int):
    import numpy as np
    n = len(atoms)
    if n <= count:
        indices = np.arange(n, dtype=int)
    else:
        indices = np.linspace(0, n - 1, count, dtype=int)
    return np.column_stack((atoms["x"][indices], atoms["y"][indices], atoms["z"][indices]))


def _flatten_gap(prefix: str, stats: Any, result: dict[str, Any]) -> None:
    result[f"{prefix}_gap_min_a"] = stats.minimum_angstrom
    result[f"{prefix}_gap_median_a"] = stats.median_angstrom
    result[f"{prefix}_gap_p95_a"] = stats.percentile_95_angstrom
    result[f"{prefix}_gap_max_a"] = stats.maximum_angstrom
    result[f"{prefix}_gap_range_a"] = stats.range_angstrom
    result[f"{prefix}_empty_left_fraction"] = stats.empty_left_bin_fraction
    result[f"{prefix}_empty_right_fraction"] = stats.empty_right_bin_fraction


def _run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    temporary_data: Path | None = None
    if args.project_root is not None:
        sys.path.insert(0, str(args.project_root.resolve()))
    try:
        structure_dir = args.output_dir.resolve() / "structures" / case["case_id"]
        structure_dir.mkdir(parents=True, exist_ok=True)
        data_path = structure_dir / "structure.data"
        data_path.unlink(missing_ok=True)

        import numpy as np
        from GBOpt import GBMaker
        from GBOpt.BoundarySpec import PQSpec
        audit_module = _load_audit_module(
            None if args.geometry_audit_file is None else args.geometry_audit_file.resolve()
        )
        P = np.asarray(case["P"], dtype=object)
        Q = np.asarray(case["Q"], dtype=object)
        boundary = PQSpec(P=P, Q=Q, basis_mode="supplied")
        common = dict(
            a0=args.lattice_constant,
            structure="fluorite",
            atom_types=("U", "O"),
            boundary=boundary,
            mode="exact",
            repeat_factor=(1, 1),
            x_dim_min=args.x_dim_min,
            vacuum=0.0,
            interaction_distance=args.interaction_distance,
            mismatch_tol=args.mismatch_tol,
            mismatch_max_cells=args.mismatch_max_cells,
            strain_grain=args.strain_grain,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            probe = GBMaker.from_boundary_spec(
                gb_thickness=args.lattice_constant,
                **common,
            )
            t1 = time.perf_counter()
            thickness = args.gb_thickness_periods * max(
                float(probe.spacing["x"]["left"]),
                float(probe.spacing["x"]["right"]),
            )
            del probe
            gb = GBMaker.from_boundary_spec(gb_thickness=thickness, **common)
            t2 = time.perf_counter()

        if not bool(gb.uses_exact_construction):
            raise RuntimeError("construction did not use the exact path")
        if tuple(bool(value) for value in gb.inplane_periodic) != (True, True):
            raise RuntimeError("construction is not periodic in both in-plane axes")
        _validate_atoms(gb.left_grain, gb.box_dims, label="left grain")
        _validate_atoms(gb.right_grain, gb.box_dims, label="right grain")
        _validate_atoms(gb.whole_system, gb.box_dims, label="whole system")

        left_counts = _count_species(gb.left_grain)
        right_counts = _count_species(gb.right_grain)
        whole_counts = _count_species(gb.whole_system)
        for label, counts in (("left", left_counts), ("right", right_counts), ("whole", whole_counts)):
            if counts.get("O", 0) != 2 * counts.get("U", 0):
                raise RuntimeError(f"{label} grain/system is not U:O=1:2: {counts}")
        if len(gb.left_grain) % 12:
            raise RuntimeError("fluorite grain population is not divisible by 12")

        audit = audit_module.audit_bicrystal_geometry(
            gb.left_grain,
            gb.right_grain,
            gb.box_dims,
            central_plane_x=float(gb.gb_plane_x),
        )
        t3 = time.perf_counter()
        nn = audit.nearest_neighbors
        if nn.periodic_duplicate_count:
            raise RuntimeError(
                f"periodic duplicate representatives: {nn.periodic_duplicate_count}"
            )

        xlo, xhi = map(float, gb.box_dims[0])
        projected_central = float(np.min(gb.right_grain["x"]) - np.max(gb.left_grain["x"]))
        projected_periodic = float(
            (xhi - np.max(gb.right_grain["x"]))
            + (np.min(gb.left_grain["x"]) - xlo)
        )
        if projected_central < -2.0e-8 or projected_periodic < -2.0e-8:
            raise RuntimeError(
                "negative projected interface gap: "
                f"central={projected_central}, periodic={projected_periodic}"
            )

        temporary_data = structure_dir / f".structure.data.tmp-{os.getpid()}"
        temporary_data.unlink(missing_ok=True)
        gb.write_lammps(
            str(temporary_data),
            type_as_int=True,
            charges={"U": float(args.charges[0]), "O": float(args.charges[1])},
            precision=args.precision,
        )
        if not temporary_data.is_file() or temporary_data.stat().st_size == 0:
            raise RuntimeError(
                f"GBOpt did not write a valid LAMMPS data file for {case['case_id']}"
            )
        os.replace(temporary_data, data_path)
        temporary_data = None
        data_sha256 = _sha256(data_path)

        preview_path = ""
        if args.preview_points > 0:
            preview_dir = args.output_dir.resolve() / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            total = args.preview_points
            left_n = max(1, round(total * len(gb.left_grain) / len(gb.whole_system)))
            right_n = max(1, total - left_n)
            preview = preview_dir / f"{case['case_id']}.npz"
            np.savez_compressed(
                preview,
                left=_sample_positions(gb.left_grain, left_n),
                right=_sample_positions(gb.right_grain, right_n),
                box=np.asarray(gb.box_dims, dtype=float),
                plane=np.asarray([float(gb.gb_plane_x)], dtype=float),
            )
            preview_path = str(preview)

        result: dict[str, Any] = {
            "case_id": case["case_id"],
            "source_row": case["source_row"],
            "status": "generated",
            "boundary_type": case["boundary_type"],
            "axis_set": case["axis_set"],
            "audit_status": audit.status,
            "audit_reasons": ",".join(audit.reasons),
            "natoms": int(len(gb.whole_system)),
            "left_atoms": int(len(gb.left_grain)),
            "right_atoms": int(len(gb.right_grain)),
            "u_atoms": whole_counts.get("U", 0),
            "o_atoms": whole_counts.get("O", 0),
            "left_origin_representatives": int(len(gb.left_grain) // 12),
            "right_origin_representatives": int(len(gb.right_grain) // 12),
            "projected_central_gap_a": projected_central,
            "projected_periodic_gap_a": projected_periodic,
            "bulk_reference_a": audit.bulk_reference_distance_angstrom,
            "left_internal_min_a": nn.left_internal_min_angstrom,
            "right_internal_min_a": nn.right_internal_min_angstrom,
            "central_cross_min_a": nn.central_cross_min_angstrom,
            "periodic_cross_min_a": nn.periodic_cross_min_angstrom,
            "periodic_duplicate_count": nn.periodic_duplicate_count,
            "bins_y": audit.bins_y,
            "bins_z": audit.bins_z,
            "probe_s": t1 - t0,
            "build_s": t2 - t1,
            "audit_s": t3 - t2,
            "total_s": time.perf_counter() - started,
            "peak_rss_mib": _peak_rss_mib(),
            "warning_count": len(caught),
            "warnings": json.dumps([f"{w.category.__name__}: {w.message}" for w in caught]),
            "box_x_a": float(gb.box_dims[0][1] - gb.box_dims[0][0]),
            "box_y_a": float(gb.box_dims[1][1] - gb.box_dims[1][0]),
            "box_z_a": float(gb.box_dims[2][1] - gb.box_dims[2][0]),
            "data_file": str(data_path),
            "data_sha256": data_sha256,
            "preview_file": preview_path,
            "error_type": "",
            "message": "",
        }
        _flatten_gap("central", audit.central_interface, result)
        _flatten_gap("periodic", audit.periodic_interface, result)
        return result
    except Exception as exc:
        return {
            "case_id": case["case_id"],
            "source_row": case["source_row"],
            "status": "failed",
            "boundary_type": case["boundary_type"],
            "axis_set": case["axis_set"],
            "total_s": time.perf_counter() - started,
            "peak_rss_mib": _peak_rss_mib(),
            "error_type": type(exc).__name__,
            "message": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
    finally:
        if temporary_data is not None:
            temporary_data.unlink(missing_ok=True)
        if args.project_root is not None:
            root = str(args.project_root.resolve())
            if sys.path and sys.path[0] == root:
                sys.path.pop(0)


def _write_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: int(item["source_row"])):
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _has_reusable_data_file(row: dict[str, Any]) -> bool:
    """Return whether a completed row still has its verified full structure file."""
    if row.get("status") != "generated":
        return False
    data_file = str(row.get("data_file", "")).strip()
    expected_sha256 = str(row.get("data_sha256", "")).strip()
    if not data_file or not expected_sha256:
        return False
    path = Path(data_file)
    return path.is_file() and _sha256(path) == expected_sha256


def _execute_child(case: dict[str, Any], args: argparse.Namespace, script: Path) -> dict[str, Any]:
    command = [
        sys.executable, str(script),
        "--data-file", str(args.data_file.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--run-one", case["case_id"],
        "--lattice-constant", repr(args.lattice_constant),
        "--x-dim-min", repr(args.x_dim_min),
        "--gb-thickness-periods", repr(args.gb_thickness_periods),
        "--interaction-distance", repr(args.interaction_distance),
        "--mismatch-tol", repr(args.mismatch_tol),
        "--mismatch-max-cells", str(args.mismatch_max_cells),
        "--strain-grain", args.strain_grain,
        "--preview-points", str(args.preview_points),
        "--charges", *(repr(value) for value in args.charges),
        "--precision", str(args.precision),
    ]
    if args.project_root is not None:
        command += ["--project-root", str(args.project_root.resolve())]
    if args.geometry_audit_file is not None:
        command += ["--geometry-audit-file", str(args.geometry_audit_file.resolve())]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "case_id": case["case_id"], "source_row": case["source_row"],
            "status": "timeout", "boundary_type": case["boundary_type"],
            "axis_set": case["axis_set"], "total_s": time.perf_counter() - started,
            "error_type": "TimeoutExpired", "message": str(exc),
        }
    marker_line = None
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            marker_line = line[len(RESULT_MARKER):]
    if marker_line is None:
        return {
            "case_id": case["case_id"], "source_row": case["source_row"],
            "status": "failed", "boundary_type": case["boundary_type"],
            "axis_set": case["axis_set"], "total_s": time.perf_counter() - started,
            "error_type": "ChildProtocolError",
            "message": f"returncode={completed.returncode}\nstdout={completed.stdout[-4000:]}\nstderr={completed.stderr[-4000:]}",
        }
    return json.loads(marker_line)


def _numbers(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value in (None, ""):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    lo = math.floor(index)
    hi = math.ceil(index)
    if lo == hi:
        return ordered[lo]
    weight = index - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _top(rows: Sequence[dict[str, Any]], key: str, *, reverse: bool, count: int = 10) -> list[dict[str, Any]]:
    valid = [row for row in rows if row.get(key, "") not in (None, "")]
    return sorted(valid, key=lambda row: float(row[key]), reverse=reverse)[:count]


def _write_outliers(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = (
        "case_id", "boundary_type", "axis_set", "audit_status", "audit_reasons",
        "natoms", "central_gap_range_a", "periodic_gap_range_a",
        "central_cross_min_a", "periodic_cross_min_a", "total_s", "peak_rss_mib",
    )
    selected = [row for row in rows if row.get("audit_status") != "ok" or row.get("status") != "generated"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in sorted(selected, key=lambda item: int(item["source_row"])):
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_report(path: Path, rows: Sequence[dict[str, Any]], args: argparse.Namespace) -> None:
    generated = [row for row in rows if row.get("status") == "generated"]
    status_counts = Counter(str(row.get("status")) for row in rows)
    audit_counts = Counter(str(row.get("audit_status")) for row in generated)
    reason_counts: Counter[str] = Counter()
    for row in generated:
        reason_counts.update(filter(None, str(row.get("audit_reasons", "")).split(",")))

    total_times = _numbers(generated, "total_s")
    build_times = _numbers(generated, "build_s")
    audit_times = _numbers(generated, "audit_s")
    atoms = _numbers(generated, "natoms")
    memory = _numbers(generated, "peak_rss_mib")

    by_group: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in generated:
        by_group[(str(row["boundary_type"]), str(row["axis_set"]))][str(row["audit_status"])] += 1

    compact = min(generated, key=lambda row: int(row["natoms"])) if generated else None
    largest = max(generated, key=lambda row: int(row["natoms"])) if generated else None
    median_case = None
    if generated:
        target = statistics.median(int(row["natoms"]) for row in generated)
        median_case = min(generated, key=lambda row: abs(int(row["natoms"]) - target))

    lines = [
        "# Phase 2 Zhang UO2 Campaign Closeout Report", "",
        "## Scope", "",
        "This report closes the exact rational-basis, decorated-site enumeration, and exact GBMaker integration work. The geometry audit remains observational: suspicious zero-translation interfaces are retained for the later termination and relative-translation phase.", "",
        "## Campaign completion", "",
        f"- Selected cases: **{len(rows)}**",
        f"- Generated: **{status_counts.get('generated', 0)}**",
        f"- Failed: **{status_counts.get('failed', 0)}**",
        f"- Timed out: **{status_counts.get('timeout', 0)}**",
        f"- Exact stoichiometry, box membership, and periodic uniqueness were enforced for every generated case.",
        "- Every generated case includes a complete species-preserving LAMMPS data file under `structures/<case_id>/structure.data`.", "",
        "## Geometry audit", "",
        f"Phase 1 baseline classification: **{args.baseline_ok} ok / {args.baseline_suspicious} suspicious / {args.baseline_invalid} invalid**.",
        f"Phase 2 classification: **{audit_counts.get('ok', 0)} ok / {audit_counts.get('suspicious', 0)} suspicious / {audit_counts.get('invalid', 0)} invalid**.", "",
        "The Phase 2 classification should not be interpreted as a direct pass-rate target. Complete decorated slabs expose close zero-translation contacts that the former layer-deletion path could hide. The relevant Phase 2 structural invariants are exact population, no clipping-induced channels, no periodic duplicates, and no negative projected x overlap.", "",
        "### Classification by boundary type and axis", "",
        "| Type | Axis | OK | Suspicious | Invalid |", "|---|---:|---:|---:|---:|",
    ]
    for (kind, axis), counts in sorted(by_group.items()):
        lines.append(f"| {kind} | {axis} | {counts.get('ok', 0)} | {counts.get('suspicious', 0)} | {counts.get('invalid', 0)} |")
    lines += ["", "### Most frequent audit reasons", ""]
    for reason, count in reason_counts.most_common():
        lines.append(f"- `{reason}`: {count}")
    if not reason_counts:
        lines.append("- None")

    lines += [
        "", "## Performance", "",
        "| Metric | Minimum | Median | 95th percentile | Maximum |",
        "|---|---:|---:|---:|---:|",
        f"| Total case time (s) | {_fmt(min(total_times) if total_times else None)} | {_fmt(_percentile(total_times, 0.5))} | {_fmt(_percentile(total_times, 0.95))} | {_fmt(max(total_times) if total_times else None)} |",
        f"| Final build time (s) | {_fmt(min(build_times) if build_times else None)} | {_fmt(_percentile(build_times, 0.5))} | {_fmt(_percentile(build_times, 0.95))} | {_fmt(max(build_times) if build_times else None)} |",
        f"| Geometry audit time (s) | {_fmt(min(audit_times) if audit_times else None)} | {_fmt(_percentile(audit_times, 0.5))} | {_fmt(_percentile(audit_times, 0.95))} | {_fmt(max(audit_times) if audit_times else None)} |",
        f"| Atom count | {_fmt(min(atoms) if atoms else None, 0)} | {_fmt(_percentile(atoms, 0.5), 0)} | {_fmt(_percentile(atoms, 0.95), 0)} | {_fmt(max(atoms) if atoms else None, 0)} |",
        f"| Peak child RSS (MiB) | {_fmt(min(memory) if memory else None)} | {_fmt(_percentile(memory, 0.5))} | {_fmt(_percentile(memory, 0.95))} | {_fmt(max(memory) if memory else None)} |",
        "", "### Representative sizes", "",
        "| Selection | Case | Atoms | Total time (s) | Peak RSS (MiB) | Audit status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for label, row in (("Compact", compact), ("Median", median_case), ("Largest", largest)):
        if row is not None:
            lines.append(f"| {label} | `{row['case_id']}` | {int(row['natoms']):,} | {float(row['total_s']):.3f} | {_fmt(float(row['peak_rss_mib']) if row.get('peak_rss_mib') not in ('', None) else None)} | {row['audit_status']} |")

    lines += [
        "", "## Highest-priority remaining interfaces", "",
        "These are translation/termination outliers, not decorated-site enumeration failures.", "",
        "### Smallest central cross-interface distances", "",
        "| Case | Distance (A) | Reasons |", "|---|---:|---|",
    ]
    for row in _top(generated, "central_cross_min_a", reverse=False):
        lines.append(f"| `{row['case_id']}` | {float(row['central_cross_min_a']):.6f} | {row['audit_reasons']} |")
    lines += ["", "### Largest local-gap ranges", "", "| Case | Interface | Range (A) | Reasons |", "|---|---|---:|---|"]
    combined: list[tuple[float, str, dict[str, Any]]] = []
    for row in generated:
        for interface, key in (("central", "central_gap_range_a"), ("periodic", "periodic_gap_range_a")):
            if row.get(key, "") not in (None, ""):
                combined.append((float(row[key]), interface, row))
    for value, interface, row in sorted(combined, key=lambda item: item[0], reverse=True)[:10]:
        lines.append(f"| `{row['case_id']}` | {interface} | {value:.6f} | {row['audit_reasons']} |")

    lines += [
        "", "## Phase 2 gap-handling contract", "",
        "- Exact construction enumerates every rational basis site in the periodic supercell.",
        "- It does not clip complete conventional-cell origins after basis decoration.",
        "- It does not delete basis-resolved atomic planes to force global projected x gaps to match.",
        "- Different central and periodic projected gaps are permitted when both are nonnegative.",
        "- Geometry-audit status is diagnostic at this stage; local overlaps and open channels are inputs to explicit termination and relative-translation selection.",
        "- The floating construction path retains its existing complete-origin gap handling.",
        "", "## API and behavior recorded for release notes", "",
        "- Added immutable `RationalBasis` and `UnitCell.rational_basis` for all built-in structures.",
        "- Added immutable `SupercellSites` and `enumerate_supercell_sites()` with exact integer wrapping and deterministic basis-index ordering.",
        "- `SupercellSites.supercell_coordinate_denominator` is derived from `denominator * supercell_index`.",
        "- Exact `GBMaker` construction now consumes decorated sites directly and requires a rational basis.",
        "- Custom unit cells without an exact rational basis are rejected by exact construction.",
        "- Exact construction no longer performs complete-origin filtering or exact layer deletion.",
        "", "## Closeout decision", "",
    ]
    if status_counts.get("generated", 0) == len(rows):
        lines.append("The exact decorated-site implementation satisfies the Phase 2 construction invariants for the selected campaign. Remaining suspicious geometry is assigned to explicit termination and relative-translation work, not to further clipping or layer deletion.")
    else:
        lines.append("Phase 2 cannot be closed until the failed or timed-out campaign cases are resolved. See `phase2_campaign_outliers.tsv` and the per-case error messages.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gap_contract(path: Path) -> None:
    path.write_text(
        "# Exact decorated-site construction and interface-gap contract\n\n"
        "Exact construction populates a complete periodic decorated supercell from "
        "an exact rational `UnitCell` basis. It preserves every basis coset and does "
        "not recover slab shape by clipping decorated atoms or by promoting atom "
        "masks to conventional-cell origins.\n\n"
        "The central and periodic interfaces may expose different crystallographic "
        "terminations. Consequently, their global projected x gaps are not required "
        "to be equal. Both projected gaps must remain nonnegative within numerical "
        "tolerance, but equality is not a construction invariant.\n\n"
        "The exact path does not delete atomic planes to equalize those gaps. Plane "
        "deletion is unsafe for basis-resolved structures because one x plane need "
        "not contain a stoichiometric basis population. Local gap ranges, "
        "cross-interface minimum distances, empty-bin fractions, and periodic "
        "duplicates are reported by the geometry audit. Translation and termination "
        "selection must act on those local metrics explicitly in the next phase.\n\n"
        "The floating construction path is unchanged and may continue to use "
        "complete-origin labels where its existing contract requires them.\n",
        encoding="utf-8",
    )


def _write_changelog(path: Path) -> None:
    path.write_text(
        "## Changed\n\n"
        "- Added exact rational basis metadata for all built-in `UnitCell` "
        "structures.\n"
        "- Added exact decorated-supercell enumeration with deterministic "
        "basis-index ordering, negative-determinant support, and exact periodic "
        "wrapping.\n"
        "- Updated exact `GBMaker` generation to populate complete decorated sites "
        "directly.\n"
        "- Removed complete-origin filtering and atomic-plane gap equalization from "
        "the exact decorated-site path.\n"
        "- Exact custom-cell construction now requires an explicit rational basis.\n"
        "- Updated exact-path tests to treat unequal nonnegative projected interface "
        "gaps as valid and to defer local overlap remediation to explicit termination "
        "and relative-translation selection.\n",
        encoding="utf-8",
    )


def _render_gifs(rows: Sequence[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        print(f"Skipping GIFs: {exc}", file=sys.stderr)
        return
    valid = [row for row in rows if row.get("preview_file") and Path(str(row["preview_file"])).is_file()]
    if not valid:
        return
    max_x = max(float(row["box_x_a"]) for row in valid)
    max_y = max(float(row["box_y_a"]) for row in valid)
    max_z = max(float(row["box_z_a"]) for row in valid)

    def frames(normalized: bool) -> list[Image.Image]:
        images: list[Image.Image] = []
        for row in sorted(valid, key=lambda item: int(item["source_row"])):
            data = np.load(str(row["preview_file"]))
            left = data["left"]
            right = data["right"]
            box = data["box"]
            lengths = box[:, 1] - box[:, 0]
            if normalized:
                left = (left - box[:, 0]) / lengths
                right = (right - box[:, 0]) / lengths
                xlim, ylim, zlim = (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
            else:
                left = left - box[:, 0]
                right = right - box[:, 0]
                xlim, ylim, zlim = (0.0, max_x), (0.0, max_y), (0.0, max_z)
            fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=100)
            for ax, second, limits, label in (
                (axes[0], 1, ylim, "y"),
                (axes[1], 2, zlim, "z"),
            ):
                ax.scatter(left[:, 0], left[:, second], s=0.35, alpha=0.55, rasterized=True)
                ax.scatter(right[:, 0], right[:, second], s=0.35, alpha=0.55, rasterized=True)
                ax.set_xlim(*xlim)
                ax.set_ylim(*limits)
                ax.set_xlabel("x / Lx" if normalized else "x (A)")
                ax.set_ylabel(f"{label} / L{label}" if normalized else f"{label} (A)")
                ax.set_aspect("auto")
            fig.suptitle(f"{row['case_id']}  audit={row['audit_status']}  atoms={int(row['natoms']):,}")
            fig.tight_layout()
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            images.append(Image.fromarray(rgba[:, :, :3].copy()))
            plt.close(fig)
        return images

    for name, normalized in (("phase2_full_scale.gif", False), ("phase2_normalized_scale.gif", True)):
        image_frames = frames(normalized)
        if image_frames:
            image_frames[0].save(
                output_dir / name,
                save_all=True,
                append_images=image_frames[1:],
                duration=180,
                loop=0,
                optimize=False,
            )


def _run_parent(args: argparse.Namespace) -> int:
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(args.data_file.resolve())
    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(f"Expected {EXPECTED_CASES} cases; found {len(cases)}")
    selected = cases
    if args.case:
        requested = set(args.case)
        selected = [case for case in selected if case["case_id"] in requested]
        missing = requested - {case["case_id"] for case in selected}
        if missing:
            raise RuntimeError(f"Unknown case IDs: {sorted(missing)}")
    if args.limit is not None:
        selected = selected[: args.limit]
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    metrics_path = args.output_dir / "phase2_campaign_metrics.tsv"
    existing = [] if args.force else _read_existing(metrics_path)
    existing_by_id = {
        str(row["case_id"]): row
        for row in existing
        if _has_reusable_data_file(row)
    }
    results = [existing_by_id[case["case_id"]] for case in selected if case["case_id"] in existing_by_id]
    pending = [case for case in selected if case["case_id"] not in existing_by_id]

    script = Path(__file__).resolve()
    done = len(results)
    total = len(selected)
    for row in results:
        print(f"[{done:03d}/{total:03d}] reused {row['case_id']}")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_execute_child, case, args, script): case for case in pending}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            done += 1
            print(
                f"[{done:03d}/{total:03d}] {result.get('status', 'unknown'):>9} "
                f"{result['case_id']} atoms={result.get('natoms', '')} "
                f"audit={result.get('audit_status', '')} time={result.get('total_s', '')}",
                flush=True,
            )
            _write_tsv(metrics_path, results)

    results.sort(key=lambda row: int(row["source_row"]))
    _write_tsv(metrics_path, results)
    manifest = {
        "schema": SCHEMA,
        "source_csv": str(args.data_file.resolve()),
        "source_csv_sha256": _sha256(args.data_file.resolve()),
        "project_root": None if args.project_root is None else str(args.project_root.resolve()),
        "geometry_audit_file": None if args.geometry_audit_file is None else str(args.geometry_audit_file.resolve()),
        "configuration": {
            "lattice_constant": args.lattice_constant,
            "x_dim_min": args.x_dim_min,
            "gb_thickness_periods": args.gb_thickness_periods,
            "interaction_distance": args.interaction_distance,
            "mismatch_tol": args.mismatch_tol,
            "mismatch_max_cells": args.mismatch_max_cells,
            "strain_grain": args.strain_grain,
            "charges": [float(value) for value in args.charges],
            "precision": args.precision,
            "lammps_structure_dir": str(args.output_dir / "structures"),
        },
        "status_counts": dict(Counter(str(row.get("status")) for row in results)),
        "audit_status_counts": dict(Counter(str(row.get("audit_status")) for row in results if row.get("status") == "generated")),
        "metrics_file": str(metrics_path),
    }
    _atomic_json(args.output_dir / "phase2_campaign_manifest.json", manifest)
    _write_outliers(args.output_dir / "phase2_campaign_outliers.tsv", results)
    _write_report(args.output_dir / "phase2_closeout_report.md", results, args)
    _write_gap_contract(args.output_dir / "exact_decorated_site_gap_contract.md")
    _write_changelog(args.output_dir / "CHANGELOG_phase2_fragment.md")
    if not args.no_gifs:
        _render_gifs(results, args.output_dir)
    return 0 if all(row.get("status") == "generated" for row in results) else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run_one is not None:
        cases = _load_cases(args.data_file.resolve())
        case = next((item for item in cases if item["case_id"] == args.run_one), None)
        if case is None:
            raise RuntimeError(f"Unknown case ID: {args.run_one}")
        result = _run_case(case, args)
        print(RESULT_MARKER + json.dumps(result, separators=(",", ":")))
        return 0 if result.get("status") == "generated" else 1
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
