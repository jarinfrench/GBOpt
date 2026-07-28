#!/usr/bin/env python3
"""Generate charged UO2 grain-boundary structures for the Zhang campaign.

The input is the normalized ``gb_data_gbopt.csv`` table.  Each nonblank row is
constructed independently with GBOpt's exact supplied-P/Q path and written as:

    OUTPUT_ROOT/
      generation_results.tsv
      manifest.json
      zhang_001_ST_100/
        structure.data
        metadata.json
      ...

Each case runs in a child process so memory is released after large structures and a
single failure does not terminate the campaign-wide generation pass.

Examples
--------
Generate all 197 structures serially::

    uv run python generate_campaign_structures.py \
        --data-file gb_data_gbopt.csv \
        --output-root structures/campaign

Generate a small validation subset::

    uv run python generate_campaign_structures.py \
        --data-file gb_data_gbopt.csv \
        --output-root structures/campaign \
        --limit 3

Regenerate selected cases::

    uv run python generate_campaign_structures.py \
        --data-file gb_data_gbopt.csv \
        --output-root structures/campaign \
        --case zhang_009_ST_100 \
        --row 181 \
        --force
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    import resource
except ImportError:  # pragma: no cover - available on Linux/macOS.
    resource = None  # type: ignore[assignment]


_RESULT_MARKER = "__GBOPT_STRUCTURE_RESULT__="
_EXPECTED_CASES = 197
_GENERATOR_SCHEMA = 2
_ALLOWED_TYPES = ("ST", "AT", "TW", "MX")
_ALLOWED_AXIS_SETS = ("100", "110", "111")
_REQUIRED_COLUMNS = (
    "P_x0",
    "P_x1",
    "P_x2",
    "P_y0",
    "P_y1",
    "P_y2",
    "P_z0",
    "P_z1",
    "P_z2",
    "Q_x0",
    "Q_x1",
    "Q_x2",
    "Q_y0",
    "Q_y1",
    "Q_y2",
    "Q_z0",
    "Q_z1",
    "Q_z2",
    "UO2_Basak (J/m^2)",
    "Type",
    "Axis Set",
    "CeO2_Gotte (J/m^2)",
)
_RESULT_FIELDS = (
    "case_id",
    "source_row",
    "status",
    "boundary_type",
    "axis_set",
    "natoms",
    "u_atoms",
    "o_atoms",
    "elapsed_s",
    "peak_rss_mib",
    "returncode",
    "signal",
    "data_file",
    "metadata_file",
    "data_sha256",
    "audit_status",
    "audit_reasons",
    "audit_bins_y",
    "audit_bins_z",
    "central_gap_min_angstrom",
    "central_gap_median_angstrom",
    "central_gap_p95_angstrom",
    "central_gap_max_angstrom",
    "central_gap_range_angstrom",
    "central_empty_left_fraction",
    "central_empty_right_fraction",
    "periodic_gap_min_angstrom",
    "periodic_gap_median_angstrom",
    "periodic_gap_p95_angstrom",
    "periodic_gap_max_angstrom",
    "periodic_gap_range_angstrom",
    "periodic_empty_left_fraction",
    "periodic_empty_right_fraction",
    "left_internal_min_angstrom",
    "right_internal_min_angstrom",
    "central_cross_min_angstrom",
    "periodic_cross_min_angstrom",
    "periodic_duplicate_count",
    "p_det_abs",
    "q_det_abs",
    "max_miller_row_norm",
    "box_x_angstrom",
    "box_y_angstrom",
    "box_z_angstrom",
    "warning_count",
    "warnings",
    "error_type",
    "message",
    "stdout_tail",
    "stderr_tail",
)

Matrix3 = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]


@dataclass(frozen=True)
class BoundaryCase:
    """One row in the normalized Zhang campaign table."""

    source_row: int
    case_id: str
    P: Matrix3
    Q: Matrix3
    uo2_reference_j_m2: float
    boundary_type: str
    axis_set: str
    ceo2_reference_j_m2: float | None


@dataclass(frozen=True)
class GenerationConfig:
    """Geometry and output parameters shared by all generated cases."""

    data_file: Path
    output_root: Path
    project_root: Path | None
    expected_cases: int
    lattice_constant: float
    structure: str
    atom_types: tuple[str, str]
    charges: tuple[float, float]
    expected_ratio: tuple[int, int]
    x_dim_min: float
    gb_thickness_periods: float
    repeat_factor: tuple[int, int]
    interaction_distance: float
    vacuum: float
    mismatch_tol: float
    mismatch_max_cells: int
    strain_grain: str
    precision: int
    timeout: float
    diagnostic_chars: int
    source_sha256: str
    generation_signature: str


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return parsed


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("value must be finite")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("gb_data_gbopt.csv"),
        help="Normalized Zhang/GBOpt CSV (default: gb_data_gbopt.csv).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("structures/campaign"),
        help="Root directory for generated structures (default: structures/campaign).",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Optional GBOpt source checkout to prepend to sys.path. Normally GBOpt "
            "should be installed in the active environment."
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="Concurrent child processes. Keep at 1 unless memory permits (default: 1).",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=3600.0,
        help="Wall-clock timeout per structure in seconds (default: 3600).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even when a matching, hash-valid output already exists.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="CASE_ID",
        help="Generate one case ID; repeat as needed.",
    )
    parser.add_argument(
        "--row",
        action="append",
        default=[],
        type=_positive_int,
        metavar="N",
        help="Generate one one-based source row; repeat as needed.",
    )
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        metavar="GLOB",
        help="Generate case IDs matching a shell-style glob; repeat as needed.",
    )
    parser.add_argument(
        "--type",
        action="append",
        choices=_ALLOWED_TYPES,
        default=[],
        dest="boundary_types",
        help="Restrict to a boundary type; repeat as needed.",
    )
    parser.add_argument(
        "--axis-set",
        action="append",
        choices=_ALLOWED_AXIS_SETS,
        default=[],
        help="Restrict to an axis set; repeat as needed.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Generate only the first N selected cases.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Validate and list selected cases without constructing structures.",
    )
    parser.add_argument(
        "--expected-cases",
        type=_nonnegative_int,
        default=_EXPECTED_CASES,
        help="Expected nonblank CSV rows; 0 disables the check (default: 197).",
    )

    geometry = parser.add_argument_group("structure geometry")
    geometry.add_argument(
        "--lattice-constant",
        type=_positive_float,
        default=5.454,
        help="UO2 lattice constant in Angstroms (default: 5.454).",
    )
    geometry.add_argument(
        "--structure",
        default="fluorite",
        help="GBOpt structure name (default: fluorite).",
    )
    geometry.add_argument(
        "--atom-types",
        nargs=2,
        default=("U", "O"),
        metavar=("CATION", "ANION"),
        help="Species names (default: U O).",
    )
    geometry.add_argument(
        "--charges",
        nargs=2,
        type=_finite_float,
        default=(2.4, -1.2),
        metavar=("CATION", "ANION"),
        help="LAMMPS charges corresponding to --atom-types (default: 2.4 -1.2).",
    )
    geometry.add_argument(
        "--expected-ratio",
        nargs=2,
        type=_positive_int,
        default=(1, 2),
        metavar=("CATION", "ANION"),
        help="Required stoichiometric ratio (default: 1 2).",
    )
    geometry.add_argument(
        "--x-dim-min",
        type=_positive_float,
        default=60.0,
        help="Minimum thickness of each grain along x in Angstroms (default: 60).",
    )
    geometry.add_argument(
        "--gb-thickness-periods",
        type=_nonnegative_float,
        default=2.0,
        help=(
            "GB-region thickness in maximum x-plane spacings. Zero stores a zero-width "
            "GB region and avoids the probe build (default: 2)."
        ),
    )
    geometry.add_argument(
        "--repeat-factor",
        nargs=2,
        type=_positive_int,
        default=(1, 1),
        metavar=("Y", "Z"),
        help="Nominal in-plane repeat factors (default: 1 1).",
    )
    geometry.add_argument(
        "--interaction-distance",
        type=_nonnegative_float,
        default=11.0,
        help=(
            "Interaction distance used by GBOpt's minimum in-plane-size check "
            "(default: 11)."
        ),
    )
    geometry.add_argument(
        "--vacuum",
        type=_nonnegative_float,
        default=0.0,
        help="Vacuum thickness along x in Angstroms (default: 0).",
    )
    geometry.add_argument(
        "--mismatch-tol",
        type=_nonnegative_float,
        default=0.005,
        help="Maximum relative in-plane mismatch (default: 0.005 = 0.5%%).",
    )
    geometry.add_argument(
        "--mismatch-max-cells",
        type=_positive_int,
        default=50,
        help="Maximum repeat count searched per grain (default: 50).",
    )
    geometry.add_argument(
        "--strain-grain",
        choices=("both", "left", "right"),
        default="both",
        help="In-plane mismatch strain policy (default: both).",
    )
    geometry.add_argument(
        "--precision",
        type=_positive_int,
        default=12,
        help="Decimal precision in LAMMPS data files (default: 12).",
    )
    parser.add_argument(
        "--diagnostic-chars",
        type=_positive_int,
        default=6000,
        help="Maximum captured stdout/stderr characters per case (default: 6000).",
    )

    # Internal child-process mode.
    parser.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    return parser


def _resolve(path: Path, *, base: Path | None = None) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (base or Path.cwd()) / resolved
    return resolved.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # Parent list mode does not require NumPy.
        np = None  # type: ignore[assignment]

    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_float(
    value: str,
    *,
    field: str,
    source_row: int,
    optional: bool = False,
) -> float | None:
    text = value.strip()
    if not text and optional:
        return None
    if not text:
        raise ValueError(f"Row {source_row}: required field {field!r} is blank")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Row {source_row}: field {field!r} must be numeric; got {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Row {source_row}: field {field!r} must be finite; got {value!r}"
        )
    return parsed


def _matrix_from_record(
    record: dict[str, str], *, prefix: str, source_row: int
) -> Matrix3:
    rows: list[tuple[int, int, int]] = []
    for axis in "xyz":
        values: list[int] = []
        for column in range(3):
            field = f"{prefix}_{axis}{column}"
            text = record[field].strip()
            if not text:
                raise ValueError(f"Row {source_row}: field {field!r} is blank")
            try:
                values.append(int(text))
            except ValueError as exc:
                raise ValueError(
                    f"Row {source_row}: field {field!r} must be an integer; "
                    f"got {record[field]!r}"
                ) from exc
        rows.append(tuple(values))  # type: ignore[arg-type]
    return tuple(rows)  # type: ignore[return-value]


def _load_cases(path: Path, *, expected_cases: int) -> list[BoundaryCase]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = [
            field for field in _REQUIRED_COLUMNS if field not in reader.fieldnames
        ]
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))

        cases: list[BoundaryCase] = []
        for record in reader:
            if not any((value or "").strip() for value in record.values()):
                continue
            source_row = len(cases) + 1
            boundary_type = record["Type"].strip()
            axis_set = record["Axis Set"].strip()
            if boundary_type not in _ALLOWED_TYPES:
                raise ValueError(
                    f"Row {source_row}: Type must be one of {_ALLOWED_TYPES}; "
                    f"got {boundary_type!r}"
                )
            if axis_set not in _ALLOWED_AXIS_SETS:
                raise ValueError(
                    f"Row {source_row}: Axis Set must be one of {_ALLOWED_AXIS_SETS}; "
                    f"got {axis_set!r}"
                )
            case_id = f"zhang_{source_row:03d}_{boundary_type}_{axis_set}"
            cases.append(
                BoundaryCase(
                    source_row=source_row,
                    case_id=case_id,
                    P=_matrix_from_record(record, prefix="P", source_row=source_row),
                    Q=_matrix_from_record(record, prefix="Q", source_row=source_row),
                    uo2_reference_j_m2=float(
                        _parse_float(
                            record["UO2_Basak (J/m^2)"],
                            field="UO2_Basak (J/m^2)",
                            source_row=source_row,
                        )
                    ),
                    boundary_type=boundary_type,
                    axis_set=axis_set,
                    ceo2_reference_j_m2=_parse_float(
                        record["CeO2_Gotte (J/m^2)"],
                        field="CeO2_Gotte (J/m^2)",
                        source_row=source_row,
                        optional=True,
                    ),
                )
            )

    if expected_cases and len(cases) != expected_cases:
        raise ValueError(
            f"Expected {expected_cases} nonblank rows, found {len(cases)} in {path}"
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Generated case IDs are not unique")
    return cases


def _select_cases(
    cases: Sequence[BoundaryCase],
    *,
    case_ids: Sequence[str],
    rows: Sequence[int],
    patterns: Sequence[str],
    boundary_types: Sequence[str],
    axis_sets: Sequence[str],
    limit: int | None,
) -> list[BoundaryCase]:
    by_id = {case.case_id: case for case in cases}
    by_row = {case.source_row: case for case in cases}
    unknown_ids = [case_id for case_id in case_ids if case_id not in by_id]
    unknown_rows = [row for row in rows if row not in by_row]
    if unknown_ids:
        raise KeyError("Unknown case IDs: " + ", ".join(unknown_ids))
    if unknown_rows:
        raise KeyError("Unknown source rows: " + ", ".join(map(str, unknown_rows)))

    explicit = set(case_ids)
    explicit.update(by_row[row].case_id for row in rows)
    selected: list[BoundaryCase] = []
    for case in cases:
        if explicit or patterns:
            if case.case_id not in explicit and not any(
                fnmatch.fnmatchcase(case.case_id, pattern) for pattern in patterns
            ):
                continue
        if boundary_types and case.boundary_type not in boundary_types:
            continue
        if axis_sets and case.axis_set not in axis_sets:
            continue
        selected.append(case)
    return selected if limit is None else selected[:limit]


def _configuration_payload(
    args: argparse.Namespace, source_sha256: str
) -> dict[str, Any]:
    return {
        "generator_schema": _GENERATOR_SCHEMA,
        "source_sha256": source_sha256,
        "lattice_constant": args.lattice_constant,
        "structure": args.structure,
        "atom_types": list(args.atom_types),
        "charges": list(args.charges),
        "expected_ratio": list(args.expected_ratio),
        "x_dim_min": args.x_dim_min,
        "gb_thickness_periods": args.gb_thickness_periods,
        "repeat_factor": list(args.repeat_factor),
        "interaction_distance": args.interaction_distance,
        "vacuum": args.vacuum,
        "mismatch_tol": args.mismatch_tol,
        "mismatch_max_cells": args.mismatch_max_cells,
        "strain_grain": args.strain_grain,
        "precision": args.precision,
    }


def _generation_signature(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _make_config(
    args: argparse.Namespace, *, data_file: Path, output_root: Path
) -> GenerationConfig:
    source_sha256 = _sha256(data_file)
    payload = _configuration_payload(args, source_sha256)
    project_root = None
    if args.project_root is not None:
        project_root = _resolve(args.project_root)
        if not project_root.is_dir():
            raise FileNotFoundError(f"Project root not found: {project_root}")
    return GenerationConfig(
        data_file=data_file,
        output_root=output_root,
        project_root=project_root,
        expected_cases=args.expected_cases,
        lattice_constant=args.lattice_constant,
        structure=args.structure,
        atom_types=tuple(args.atom_types),
        charges=tuple(args.charges),
        expected_ratio=tuple(args.expected_ratio),
        x_dim_min=args.x_dim_min,
        gb_thickness_periods=args.gb_thickness_periods,
        repeat_factor=tuple(args.repeat_factor),
        interaction_distance=args.interaction_distance,
        vacuum=args.vacuum,
        mismatch_tol=args.mismatch_tol,
        mismatch_max_cells=args.mismatch_max_cells,
        strain_grain=args.strain_grain,
        precision=args.precision,
        timeout=args.timeout,
        diagnostic_chars=args.diagnostic_chars,
        source_sha256=source_sha256,
        generation_signature=_generation_signature(payload),
    )


def _counts(atoms: Any) -> dict[str, int]:
    import numpy as np

    names, values = np.unique(atoms["name"], return_counts=True)
    return {str(name): int(count) for name, count in zip(names, values)}


def _validate_stoichiometry(
    atoms: Any,
    *,
    atom_types: tuple[str, str],
    expected_ratio: tuple[int, int],
    label: str,
) -> dict[str, int]:
    if atoms is None or getattr(atoms, "size", 0) == 0:
        raise RuntimeError(f"{label} is empty")
    counts = _counts(atoms)
    if set(counts) != set(atom_types):
        raise RuntimeError(
            f"{label} has species {counts}; expected exactly {sorted(atom_types)}"
        )
    first, second = atom_types
    first_ratio, second_ratio = expected_ratio
    if counts[first] * second_ratio != counts[second] * first_ratio:
        raise RuntimeError(
            f"{label} is not stoichiometric {first}:{second}="
            f"{first_ratio}:{second_ratio}: {counts}"
        )
    return counts


def _strain_metadata(gb: Any) -> dict[str, Any]:
    data = getattr(gb, "_GBMaker__strain_accommodation", {})
    return {
        axis: {
            "left_repeats": value.left_repeats,
            "right_repeats": value.right_repeats,
            "left_unstrained_length": value.left_unstrained_length,
            "right_unstrained_length": value.right_unstrained_length,
            "box_length": value.box_length,
            "left_scale": value.left_scale,
            "right_scale": value.right_scale,
            "mismatch": value.mismatch,
        }
        for axis, value in data.items()
    }


def _det3_abs(matrix: Matrix3) -> int:
    """Return the absolute exact determinant of a three-by-three integer matrix."""
    a, b, c = matrix
    determinant = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    return abs(int(determinant))


def _max_miller_row_norm(P: Matrix3, Q: Matrix3) -> float:
    """Return the maximum Euclidean row norm across one P/Q pair."""
    return max(
        math.sqrt(sum(component * component for component in row))
        for matrix in (P, Q)
        for row in matrix
    )


def _audit_result_fields(audit: dict[str, Any]) -> dict[str, Any]:
    """Flatten selected geometry-audit scalars for TSV and manifest output."""
    central = audit["central_interface"]
    periodic = audit["periodic_interface"]
    nearest = audit["nearest_neighbors"]
    return {
        "audit_status": audit["status"],
        "audit_reasons": json.dumps(audit["reasons"], separators=(",", ":")),
        "audit_bins_y": audit["bins_y"],
        "audit_bins_z": audit["bins_z"],
        "central_gap_min_angstrom": central["minimum_angstrom"],
        "central_gap_median_angstrom": central["median_angstrom"],
        "central_gap_p95_angstrom": central["percentile_95_angstrom"],
        "central_gap_max_angstrom": central["maximum_angstrom"],
        "central_gap_range_angstrom": central["range_angstrom"],
        "central_empty_left_fraction": central["empty_left_bin_fraction"],
        "central_empty_right_fraction": central["empty_right_bin_fraction"],
        "periodic_gap_min_angstrom": periodic["minimum_angstrom"],
        "periodic_gap_median_angstrom": periodic["median_angstrom"],
        "periodic_gap_p95_angstrom": periodic["percentile_95_angstrom"],
        "periodic_gap_max_angstrom": periodic["maximum_angstrom"],
        "periodic_gap_range_angstrom": periodic["range_angstrom"],
        "periodic_empty_left_fraction": periodic["empty_left_bin_fraction"],
        "periodic_empty_right_fraction": periodic["empty_right_bin_fraction"],
        "left_internal_min_angstrom": nearest["left_internal_min_angstrom"],
        "right_internal_min_angstrom": nearest["right_internal_min_angstrom"],
        "central_cross_min_angstrom": nearest["central_cross_min_angstrom"],
        "periodic_cross_min_angstrom": nearest["periodic_cross_min_angstrom"],
        "periodic_duplicate_count": nearest["periodic_duplicate_count"],
    }


def _peak_rss_mib() -> float | None:
    if resource is None:
        return None
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return rss / divisor


def _gbopt_provenance() -> dict[str, Any]:
    import GBOpt

    try:
        version = importlib.metadata.version("GBOpt")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(GBOpt, "__version__", "unknown")
    return {
        "version": version,
        "module_path": str(Path(GBOpt.__file__).resolve()) if GBOpt.__file__ else None,
        "python_executable": sys.executable,
        "python_version": sys.version,
    }


def _generate_case(case: BoundaryCase, config: GenerationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    target_dir = config.output_root / case.case_id
    target_dir.mkdir(parents=True, exist_ok=True)
    data_path = target_dir / "structure.data"
    metadata_path = target_dir / "metadata.json"
    temporary_data = target_dir / f".structure.data.tmp-{os.getpid()}"

    if config.project_root is not None:
        sys.path.insert(0, str(config.project_root))
    try:
        import numpy as np
        from GBOpt import GBMaker
        from GBOpt.BoundarySpec import PQSpec
        from GBOpt.geometry_audit import audit_bicrystal_geometry

        P = np.asarray(case.P, dtype=object)
        Q = np.asarray(case.Q, dtype=object)
        boundary = PQSpec(P=P, Q=Q, basis_mode="supplied")
        common = dict(
            a0=config.lattice_constant,
            structure=config.structure,
            atom_types=config.atom_types,
            boundary=boundary,
            mode="exact",
            repeat_factor=config.repeat_factor,
            x_dim_min=config.x_dim_min,
            vacuum=config.vacuum,
            interaction_distance=config.interaction_distance,
            mismatch_tol=config.mismatch_tol,
            mismatch_max_cells=config.mismatch_max_cells,
            strain_grain=config.strain_grain,
        )

        if config.gb_thickness_periods > 0.0:
            probe = GBMaker.from_boundary_spec(
                gb_thickness=config.lattice_constant,
                **common,
            )
            gb_thickness = config.gb_thickness_periods * max(
                float(probe.spacing["x"]["left"]),
                float(probe.spacing["x"]["right"]),
            )
            del probe
        else:
            gb_thickness = 0.0

        gb = GBMaker.from_boundary_spec(gb_thickness=gb_thickness, **common)
        if not bool(gb.uses_exact_construction):
            raise RuntimeError(f"{case.case_id} did not use exact P/Q construction")
        periodic = tuple(bool(value) for value in gb.inplane_periodic)
        if periodic != (True, True):
            raise RuntimeError(
                f"{case.case_id} is not periodic in both in-plane axes: {periodic}"
            )

        left_counts = _validate_stoichiometry(
            gb.left_grain,
            atom_types=config.atom_types,
            expected_ratio=config.expected_ratio,
            label=f"{case.case_id} left grain",
        )
        right_counts = _validate_stoichiometry(
            gb.right_grain,
            atom_types=config.atom_types,
            expected_ratio=config.expected_ratio,
            label=f"{case.case_id} right grain",
        )
        whole_counts = _validate_stoichiometry(
            gb.whole_system,
            atom_types=config.atom_types,
            expected_ratio=config.expected_ratio,
            label=f"{case.case_id} whole system",
        )

        geometry_audit = audit_bicrystal_geometry(
            gb.left_grain,
            gb.right_grain,
            gb.box_dims,
            central_plane_x=float(gb.gb_plane_x),
        )
        geometry_audit_payload = geometry_audit.to_dict()
        audit_fields = _audit_result_fields(geometry_audit_payload)
        box_lengths = np.asarray(gb.box_dims, dtype=float)[:, 1] - np.asarray(
            gb.box_dims,
            dtype=float,
        )[:, 0]

        charge_map = dict(zip(config.atom_types, config.charges))
        temporary_data.unlink(missing_ok=True)
        gb.write_lammps(
            str(temporary_data),
            type_as_int=True,
            charges=dict(charge_map),
            precision=config.precision,
        )
        if not temporary_data.is_file() or temporary_data.stat().st_size == 0:
            raise RuntimeError(
                f"GBOpt did not write a valid data file for {case.case_id}"
            )
        os.replace(temporary_data, data_path)
        data_sha256 = _sha256(data_path)

        metadata = {
            "generator_schema": _GENERATOR_SCHEMA,
            "generation_signature": config.generation_signature,
            "case_id": case.case_id,
            "source_row": case.source_row,
            "source_csv": str(config.data_file),
            "source_csv_sha256": config.source_sha256,
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "reference_energies_j_m2": {
                "uo2_basak": case.uo2_reference_j_m2,
                "ceo2_gotte": case.ceo2_reference_j_m2,
            },
            "P": case.P,
            "Q": case.Q,
            "basis_mode": "supplied",
            "construction_mode": "exact",
            "uses_exact_construction": bool(gb.uses_exact_construction),
            "inplane_periodic": periodic,
            "material": {
                "structure": config.structure,
                "atom_types": config.atom_types,
                "charges": charge_map,
                "expected_ratio": config.expected_ratio,
                "lattice_constant_angstrom": config.lattice_constant,
            },
            "geometry": {
                "x_dim_min_angstrom": config.x_dim_min,
                "gb_thickness_periods": config.gb_thickness_periods,
                "gb_thickness_angstrom": float(gb.gb_thickness),
                "gb_plane_x_angstrom": float(gb.gb_plane_x),
                "repeat_factor": gb.repeat_factor,
                "interaction_distance_angstrom": config.interaction_distance,
                "vacuum_angstrom": config.vacuum,
                "mismatch_tol": config.mismatch_tol,
                "mismatch_max_cells": config.mismatch_max_cells,
                "strain_grain": config.strain_grain,
                "spacing": gb.spacing,
                "strain_accommodation": _strain_metadata(gb),
                "box_dims_angstrom": gb.box_dims,
            },
            "atoms": {
                "total": int(gb.whole_system.size),
                "whole_counts": whole_counts,
                "left_total": int(gb.left_grain.size),
                "left_counts": left_counts,
                "right_total": int(gb.right_grain.size),
                "right_counts": right_counts,
            },
            "geometry_audit": geometry_audit_payload,
            "files": {
                "data_file": data_path.name,
                "data_sha256": data_sha256,
            },
            "gbopt": _gbopt_provenance(),
        }
        _atomic_json(metadata_path, metadata)
        elapsed = time.perf_counter() - started
        peak = _peak_rss_mib()
        first, second = config.atom_types
        result = {
            "case_id": case.case_id,
            "source_row": case.source_row,
            "status": "generated",
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "natoms": int(gb.whole_system.size),
            "u_atoms": whole_counts.get(first, 0),
            "o_atoms": whole_counts.get(second, 0),
            "elapsed_s": round(elapsed, 6),
            "peak_rss_mib": "" if peak is None else round(peak, 3),
            "data_file": str(data_path),
            "metadata_file": str(metadata_path),
            "data_sha256": data_sha256,
            "p_det_abs": _det3_abs(case.P),
            "q_det_abs": _det3_abs(case.Q),
            "max_miller_row_norm": _max_miller_row_norm(case.P, case.Q),
            "box_x_angstrom": float(box_lengths[0]),
            "box_y_angstrom": float(box_lengths[1]),
            "box_z_angstrom": float(box_lengths[2]),
            "warning_count": 0,
            "warnings": "[]",
            "error_type": "",
            "message": "",
        }
        result.update(audit_fields)
        return result
    finally:
        temporary_data.unlink(missing_ok=True)
        if config.project_root is not None:
            project_root_text = str(config.project_root)
            if sys.path and sys.path[0] == project_root_text:
                sys.path.pop(0)


def _child_run(case: BoundaryCase, config: GenerationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = _generate_case(case, config)
        except Exception as exc:
            peak = _peak_rss_mib()
            result = {
                "case_id": case.case_id,
                "source_row": case.source_row,
                "status": "failed",
                "boundary_type": case.boundary_type,
                "axis_set": case.axis_set,
                "natoms": "",
                "u_atoms": "",
                "o_atoms": "",
                "elapsed_s": round(time.perf_counter() - started, 6),
                "peak_rss_mib": "" if peak is None else round(peak, 3),
                "data_file": "",
                "metadata_file": "",
                "data_sha256": "",
                "error_type": type(exc).__name__,
                "message": "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            }
        warning_messages = [
            f"{item.category.__name__}: {item.message}" for item in caught
        ]
        result["warning_count"] = len(warning_messages)
        result["warnings"] = json.dumps(warning_messages, separators=(",", ":"))
    return result


def _existing_result(
    case: BoundaryCase, config: GenerationConfig
) -> dict[str, Any] | None:
    target_dir = config.output_root / case.case_id
    data_path = target_dir / "structure.data"
    metadata_path = target_dir / "metadata.json"
    if not data_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_hash = metadata["files"]["data_sha256"]
        if metadata.get("case_id") != case.case_id:
            return None
        if metadata.get("generation_signature") != config.generation_signature:
            return None
        if _sha256(data_path) != expected_hash:
            return None
        atoms = metadata["atoms"]
        audit = metadata["geometry_audit"]
        geometry = metadata["geometry"]
        box_lengths = [
            float(bounds[1]) - float(bounds[0])
            for bounds in geometry["box_dims_angstrom"]
        ]
        counts = atoms["whole_counts"]
        first, second = config.atom_types
        result = {
            "case_id": case.case_id,
            "source_row": case.source_row,
            "status": "skipped",
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "natoms": atoms["total"],
            "u_atoms": counts.get(first, 0),
            "o_atoms": counts.get(second, 0),
            "elapsed_s": 0.0,
            "peak_rss_mib": "",
            "returncode": 0,
            "signal": "",
            "data_file": str(data_path),
            "metadata_file": str(metadata_path),
            "data_sha256": expected_hash,
            "p_det_abs": _det3_abs(case.P),
            "q_det_abs": _det3_abs(case.Q),
            "max_miller_row_norm": _max_miller_row_norm(case.P, case.Q),
            "box_x_angstrom": float(box_lengths[0]),
            "box_y_angstrom": float(box_lengths[1]),
            "box_z_angstrom": float(box_lengths[2]),
            "warning_count": 0,
            "warnings": "[]",
            "error_type": "",
            "message": "Existing output matches the current generation signature.",
            "stdout_tail": "",
            "stderr_tail": "",
        }
        result.update(_audit_result_fields(audit))
        return result
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _common_child_args(config: GenerationConfig) -> list[str]:
    arguments = [
        "--data-file",
        str(config.data_file),
        "--output-root",
        str(config.output_root),
        "--expected-cases",
        str(config.expected_cases),
        "--lattice-constant",
        repr(config.lattice_constant),
        "--structure",
        config.structure,
        "--atom-types",
        *config.atom_types,
        "--charges",
        *(repr(value) for value in config.charges),
        "--expected-ratio",
        *(str(value) for value in config.expected_ratio),
        "--x-dim-min",
        repr(config.x_dim_min),
        "--gb-thickness-periods",
        repr(config.gb_thickness_periods),
        "--repeat-factor",
        *(str(value) for value in config.repeat_factor),
        "--interaction-distance",
        repr(config.interaction_distance),
        "--vacuum",
        repr(config.vacuum),
        "--mismatch-tol",
        repr(config.mismatch_tol),
        "--mismatch-max-cells",
        str(config.mismatch_max_cells),
        "--strain-grain",
        config.strain_grain,
        "--precision",
        str(config.precision),
        "--timeout",
        repr(config.timeout),
        "--diagnostic-chars",
        str(config.diagnostic_chars),
    ]
    if config.project_root is not None:
        arguments.extend(("--project-root", str(config.project_root)))
    return arguments


def _tail(text: str, limit: int) -> str:
    return text if len(text) <= limit else "...[truncated]...\n" + text[-limit:]


def _signal_name(returncode: int) -> str:
    if returncode >= 0:
        return ""
    number = -returncode
    try:
        return signal.Signals(number).name
    except ValueError:
        return f"SIG{number}"


def _parse_child_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            return json.loads(line[len(_RESULT_MARKER) :])
    return None


def _diagnostic_stdout(stdout: str) -> str:
    return "\n".join(
        line for line in stdout.splitlines() if not line.startswith(_RESULT_MARKER)
    )


def _execute_case(
    case: BoundaryCase,
    *,
    script_path: Path,
    config: GenerationConfig,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script_path),
        "--run-one",
        case.case_id,
        *_common_child_args(config),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=config.output_root.parent,
            capture_output=True,
            text=True,
            timeout=config.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        return {
            "case_id": case.case_id,
            "source_row": case.source_row,
            "status": "timeout",
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "elapsed_s": round(time.perf_counter() - started, 6),
            "returncode": "",
            "signal": "",
            "warning_count": 0,
            "warnings": "[]",
            "error_type": "TimeoutExpired",
            "message": f"Case exceeded {config.timeout:g} seconds.",
            "stdout_tail": _tail(stdout, config.diagnostic_chars),
            "stderr_tail": _tail(stderr, config.diagnostic_chars),
        }
    except OSError as exc:
        return {
            "case_id": case.case_id,
            "source_row": case.source_row,
            "status": "launch_error",
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "elapsed_s": round(time.perf_counter() - started, 6),
            "returncode": "",
            "signal": "",
            "warning_count": 0,
            "warnings": "[]",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "stdout_tail": "",
            "stderr_tail": "",
        }

    result = _parse_child_result(completed.stdout)
    signal_name = _signal_name(completed.returncode)
    if result is None:
        status = "signaled" if completed.returncode < 0 else "no_result"
        message = (
            f"Child terminated by {signal_name}."
            if completed.returncode < 0
            else "Child exited without a structured result."
        )
        if signal_name == "SIGKILL":
            message += " SIGKILL may indicate an out-of-memory kill."
        result = {
            "case_id": case.case_id,
            "source_row": case.source_row,
            "status": status,
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "elapsed_s": round(time.perf_counter() - started, 6),
            "warning_count": 0,
            "warnings": "[]",
            "error_type": signal_name or "MissingChildResult",
            "message": message,
        }
    result["returncode"] = completed.returncode
    result["signal"] = signal_name
    result["stdout_tail"] = _tail(
        _diagnostic_stdout(completed.stdout), config.diagnostic_chars
    )
    result["stderr_tail"] = _tail(completed.stderr, config.diagnostic_chars)
    return result


def _write_results(path: Path, results: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=_RESULT_FIELDS,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for result in sorted(results, key=lambda item: int(item["source_row"])):
            writer.writerow({field: result.get(field, "") for field in _RESULT_FIELDS})


def _write_manifest(
    path: Path,
    *,
    config: GenerationConfig,
    results: Sequence[dict[str, Any]],
) -> None:
    entries = []
    for result in sorted(results, key=lambda item: int(item["source_row"])):
        entries.append(
            {
                "case_id": result.get("case_id"),
                "source_row": result.get("source_row"),
                "status": result.get("status"),
                "boundary_type": result.get("boundary_type"),
                "axis_set": result.get("axis_set"),
                "natoms": result.get("natoms"),
                "data_file": result.get("data_file"),
                "metadata_file": result.get("metadata_file"),
                "data_sha256": result.get("data_sha256"),
                "audit_status": result.get("audit_status"),
                "audit_reasons": result.get("audit_reasons"),
                "central_gap_max_angstrom": result.get(
                    "central_gap_max_angstrom"
                ),
                "central_gap_range_angstrom": result.get(
                    "central_gap_range_angstrom"
                ),
                "periodic_gap_max_angstrom": result.get(
                    "periodic_gap_max_angstrom"
                ),
                "periodic_gap_range_angstrom": result.get(
                    "periodic_gap_range_angstrom"
                ),
                "central_cross_min_angstrom": result.get(
                    "central_cross_min_angstrom"
                ),
                "periodic_cross_min_angstrom": result.get(
                    "periodic_cross_min_angstrom"
                ),
                "periodic_duplicate_count": result.get(
                    "periodic_duplicate_count"
                ),
            }
        )
    _atomic_json(
        path,
        {
            "generator_schema": _GENERATOR_SCHEMA,
            "generation_signature": config.generation_signature,
            "source_csv": str(config.data_file),
            "source_csv_sha256": config.source_sha256,
            "output_root": str(config.output_root),
            "configuration": _jsonable(asdict(config)),
            "status_counts": dict(Counter(str(item["status"]) for item in results)),
            "structures": entries,
        },
    )


def _display(index: int, total: int, result: dict[str, Any]) -> None:
    status = str(result.get("status", "internal_error"))
    case_id = str(result.get("case_id", "<unknown>"))
    elapsed = result.get("elapsed_s", "")
    natoms = result.get("natoms", "")
    suffix = f" {elapsed}s" if elapsed != "" else ""
    if natoms != "":
        suffix += f" {natoms} atoms"
    audit_status = result.get("audit_status", "")
    if audit_status:
        suffix += f" audit={audit_status}"
    print(f"[{index:03d}/{total:03d}] {status:>10} {case_id}{suffix}", flush=True)
    if status not in {"generated", "skipped"}:
        lines = str(result.get("message", "")).strip().splitlines()
        if lines:
            print(f"    {lines[-1]}", flush=True)


def _run_parent(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    data_file = _resolve(args.data_file)
    if not data_file.is_file():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _make_config(args, data_file=data_file, output_root=output_root)
    cases = _load_cases(data_file, expected_cases=args.expected_cases)
    selected = _select_cases(
        cases,
        case_ids=args.case,
        rows=args.row,
        patterns=args.match,
        boundary_types=args.boundary_types,
        axis_sets=args.axis_set,
        limit=args.limit,
    )
    if not selected:
        raise ValueError("No cases matched the requested selection")

    print(f"Data file      : {data_file}")
    print(f"SHA-256        : {config.source_sha256}")
    print(f"Boundary rows  : {len(cases)}")
    print(f"Selected       : {len(selected)}")
    print(f"Output root    : {output_root}")
    print(
        "Construction   : exact supplied P/Q; "
        f"x_dim_min={config.x_dim_min:g} A; "
        f"interaction_distance={config.interaction_distance:g} A; "
        f"mismatch_tol={config.mismatch_tol:g}; "
        f"strain={config.strain_grain}"
    )

    if args.list:
        for case in selected:
            print(
                f"{case.case_id}\trow={case.source_row}\ttype={case.boundary_type}"
                f"\taxis={case.axis_set}\tUO2={case.uo2_reference_j_m2:g}"
            )
        return 0

    results: list[dict[str, Any]] = []
    pending: list[BoundaryCase] = []
    if args.force:
        pending = list(selected)
    else:
        for case in selected:
            existing = _existing_result(case, config)
            if existing is None:
                pending.append(case)
            else:
                results.append(existing)

    for index, result in enumerate(results, start=1):
        _display(index, len(selected), result)

    if pending:
        completed_count = len(results)
        future_to_case: dict[Future[dict[str, Any]], BoundaryCase] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for case in pending:
                future_to_case[
                    executor.submit(
                        _execute_case,
                        case,
                        script_path=script_path,
                        config=config,
                    )
                ] = case
            try:
                for future in as_completed(future_to_case):
                    case = future_to_case[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # Defensive parent-side isolation.
                        result = {
                            "case_id": case.case_id,
                            "source_row": case.source_row,
                            "status": "internal_error",
                            "boundary_type": case.boundary_type,
                            "axis_set": case.axis_set,
                            "elapsed_s": "",
                            "returncode": "",
                            "signal": "",
                            "warning_count": 0,
                            "warnings": "[]",
                            "error_type": type(exc).__name__,
                            "message": "".join(
                                traceback.format_exception(
                                    type(exc), exc, exc.__traceback__
                                )
                            ),
                            "stdout_tail": "",
                            "stderr_tail": "",
                        }
                    results.append(result)
                    completed_count += 1
                    _display(completed_count, len(selected), result)
                    _write_results(output_root / "generation_results.tsv", results)
                    _write_manifest(
                        output_root / "manifest.json",
                        config=config,
                        results=results,
                    )
            except KeyboardInterrupt:
                for future in future_to_case:
                    future.cancel()
                print("\nInterrupted; completed results were written.", file=sys.stderr)
                return 130

    _write_results(output_root / "generation_results.tsv", results)
    _write_manifest(output_root / "manifest.json", config=config, results=results)
    counts = Counter(str(result["status"]) for result in results)
    print("\nSummary")
    print("-------")
    for status, count in sorted(counts.items()):
        print(f"{status:>10}: {count}")
    print(f"{'total':>10}: {len(results)}")
    print(f"Results   : {output_root / 'generation_results.tsv'}")
    print(f"Manifest  : {output_root / 'manifest.json'}")
    successful = {"generated", "skipped"}
    return 0 if all(result["status"] in successful for result in results) else 1


def _run_child(args: argparse.Namespace) -> int:
    data_file = _resolve(args.data_file)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _make_config(args, data_file=data_file, output_root=output_root)
    cases = _load_cases(data_file, expected_cases=args.expected_cases)
    by_id = {case.case_id: case for case in cases}
    case = by_id.get(args.run_one)
    if case is None:
        raise KeyError(f"Unknown case ID: {args.run_one}")
    result = _child_run(case, config)
    print(_RESULT_MARKER + json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result.get("status") == "generated" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_child(args) if args.run_one else _run_parent(args)
    except (FileNotFoundError, ImportError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
