#!/usr/bin/env python3
"""Generate auditable clean UO2 grain-boundary seeds for the Zhang campaign.

Each selected CSV row is constructed through GBOpt's exact supplied-P/Q path in an
isolated child process.  The unmodified zero-translation construction is persisted
first.  A strict Phase 4 feasibility decision is then composed with the existing
Phase 6/7 translation and exact-termination initializers.  The workflow ends at
retained clean seeds or an actionable clean-generation failure; it never evaluates a
target property or enters optimization.
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
import shutil
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
from typing import Any, Mapping, Sequence

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


_RESULT_MARKER = "__GBOPT_CLEAN_GENERATION_RESULT__="
_EXPECTED_CASES = 197
_GENERATOR_SCHEMA = 4
_CASE_SCHEMA = "gbopt-clean-generation-case-v1"
_CONSTRUCTION_SCHEMA = "gbopt-clean-generation-construction-v1"
_INITIALIZATION_SCHEMA = "gbopt-clean-generation-initialization-v1"
_MANIFEST_SCHEMA = "gbopt-clean-generation-manifest-v1"
_CAMPAIGN_REPORT_SCHEMA = "gbopt-clean-generation-campaign-report-v1"
_ALLOWED_TYPES = ("ST", "AT", "TW", "MX")
_ALLOWED_AXIS_SETS = ("100", "110", "111")
_CLEAN_STATUSES = frozenset(
    {
        "construction_failed",
        "constructed_infeasible",
        "translation_search_exhausted",
        "termination_search_exhausted",
        "seed_generation_failed",
        "feasible_seed_ready",
    }
)
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
    "failure_stage",
    "reason_codes",
    "resumed",
    "boundary_type",
    "axis_set",
    "topology",
    "boundary_conditions",
    "base_natoms",
    "base_feasibility_status",
    "base_feasibility_raw_status",
    "base_structure_hash",
    "base_state_hash",
    "retained_seed_count",
    "retained_seed_hashes",
    "phase7_status",
    "phase7_result_hash",
    "elapsed_s",
    "peak_rss_mib",
    "process_state",
    "returncode",
    "signal",
    "case_directory",
    "case_metadata_file",
    "construction_report_file",
    "initialization_report_file",
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
    """Immutable campaign construction and clean-initialization configuration."""

    data_file: Path
    output_root: Path
    project_root: Path | None
    clean_config_file: Path | None
    clean_config_sha256: str | None
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
    mismatch_tol: float | None
    mismatch_max_cells: int
    strain_grain: str
    precision: int
    timeout: float
    diagnostic_chars: int
    source_sha256: str
    software_identity: Mapping[str, Any]
    clean_settings: Any
    generation_signature: str


class CampaignConfigurationError(ValueError):
    """Raised when CLI and clean-generation configuration are inconsistent."""


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
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-file", type=Path, default=Path("gb_data_gbopt.csv"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("structures/clean_campaign")
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--clean-config",
        type=Path,
        default=None,
        help="JSON or TOML clean-generation policy/domain file.",
    )
    parser.add_argument("--workers", type=_positive_int, default=1)
    parser.add_argument("--timeout", type=_positive_float, default=3600.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--case", action="append", default=[], metavar="CASE_ID")
    parser.add_argument(
        "--row", action="append", default=[], type=_positive_int, metavar="N"
    )
    parser.add_argument("--match", action="append", default=[], metavar="GLOB")
    parser.add_argument(
        "--type", action="append", choices=_ALLOWED_TYPES, default=[], dest="boundary_types"
    )
    parser.add_argument(
        "--axis-set", action="append", choices=_ALLOWED_AXIS_SETS, default=[]
    )
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--expected-cases", type=_nonnegative_int, default=_EXPECTED_CASES)

    geometry = parser.add_argument_group("exact construction")
    geometry.add_argument("--lattice-constant", type=_positive_float, default=5.454)
    geometry.add_argument("--structure", default="fluorite")
    geometry.add_argument("--atom-types", nargs=2, default=("U", "O"))
    geometry.add_argument(
        "--charges", nargs=2, type=_finite_float, default=(2.4, -1.2)
    )
    geometry.add_argument(
        "--expected-ratio", nargs=2, type=_positive_int, default=(1, 2)
    )
    geometry.add_argument("--x-dim-min", type=_positive_float, default=60.0)
    geometry.add_argument(
        "--gb-thickness-periods", type=_nonnegative_float, default=2.0
    )
    geometry.add_argument(
        "--repeat-factor", nargs=2, type=_positive_int, default=(1, 1)
    )
    geometry.add_argument(
        "--interaction-distance", type=_nonnegative_float, default=11.0
    )
    geometry.add_argument("--mismatch-tol", type=_nonnegative_float, default=0.005)
    geometry.add_argument("--mismatch-max-cells", type=_positive_int, default=50)
    geometry.add_argument(
        "--strain-grain", choices=("both", "left", "right"), default="both"
    )
    geometry.add_argument("--precision", type=_positive_int, default=12)

    clean = parser.add_argument_group("clean-generation overrides")
    clean.add_argument(
        "--topology",
        choices=("periodic_bicrystal", "single_interface_slab"),
        default=None,
    )
    clean.add_argument(
        "--boundary-conditions",
        nargs=3,
        choices=("periodic", "fixed"),
        default=None,
        metavar=("X", "Y", "Z"),
    )
    clean.add_argument("--vacuum", type=_nonnegative_float, default=None)
    clean.add_argument(
        "--fixed-region-thickness", type=_nonnegative_float, default=None
    )
    clean.add_argument(
        "--surface-buffer-thickness", type=_nonnegative_float, default=None
    )
    clean.add_argument(
        "--retain-warnings",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    clean.add_argument("--max-seeds", type=_positive_int, default=None)
    clean.add_argument(
        "--initialization",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    clean.add_argument(
        "--in-plane-components-y", nargs="+", type=_finite_float, default=None
    )
    clean.add_argument(
        "--in-plane-components-z", nargs="+", type=_finite_float, default=None
    )
    clean.add_argument("--normal-offsets", nargs="+", type=_finite_float, default=None)
    clean.add_argument(
        "--termination-mode",
        choices=("all", "default_only", "explicit"),
        default=None,
    )
    clean.add_argument("--left-termination-phase", action="append", default=[])
    clean.add_argument("--right-termination-phase", action="append", default=[])
    clean.add_argument(
        "--disable-termination-search",
        action="store_true",
        help="Alias for --termination-mode default_only.",
    )
    clean.add_argument(
        "--override-status", choices=("infeasible", "warning", "feasible"), default=None
    )
    clean.add_argument("--override-reason", default=None)

    parser.add_argument("--diagnostic-chars", type=_positive_int, default=6000)
    parser.add_argument("--run-one", default=None, help=argparse.SUPPRESS)
    return parser


def _resolve(path: Path, *, base: Path | None = None) -> Path:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() if base is None else base) / resolved
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
    except ImportError:  # pragma: no cover
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _hashed_document(
    *,
    schema: str,
    generation_signature: str,
    deterministic: Mapping[str, Any],
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    hash_payload = {
        "schema": schema,
        "generation_signature": generation_signature,
        "deterministic": _jsonable(deterministic),
    }
    return {
        **hash_payload,
        "diagnostics": _jsonable(diagnostics or {}),
        "report_hash": _canonical_sha256(hash_payload),
    }


def _hashed_document_matches(
    value: Any, *, schema: str | None = None, signature: str | None = None
) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        if schema is not None and value["schema"] != schema:
            return False
        if signature is not None and value["generation_signature"] != signature:
            return False
        payload = {
            "schema": value["schema"],
            "generation_signature": value["generation_signature"],
            "deterministic": value["deterministic"],
        }
        return value["report_hash"] == _canonical_sha256(payload)
    except (KeyError, TypeError, ValueError):
        return False


def _feasibility_report_matches(report: Any) -> bool:
    """Verify a serialized Phase 4 report's policy and report hashes."""
    if not isinstance(report, dict):
        return False
    try:
        report_payload = {
            key: value
            for key, value in report.items()
            if key not in {"report_hash", "policy_hash"}
        }
        return (
            isinstance(report["report_hash"], str)
            and isinstance(report["policy_hash"], str)
            and _canonical_sha256(report["policy"]) == report["policy_hash"]
            and _canonical_sha256(report_payload) == report["report_hash"]
        )
    except (KeyError, TypeError, ValueError):
        return False


def _phase7_result_matches(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    try:
        payload = {key: value for key, value in result.items() if key != "result_hash"}
        return result["result_hash"] == _canonical_sha256(payload)
    except (KeyError, TypeError, ValueError):
        return False


def _state_directory_matches(
    state_path: Path, *, structure_hash: str, state_hash: str
) -> bool:
    manifest_path = state_path / "state.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("structure_hash") != structure_hash:
            return False
        if manifest.get("state_hash") != state_hash:
            return False
        arrays = manifest["arrays"]
        array_hashes = manifest["array_hashes"]
        if set(arrays) != set(array_hashes):
            return False
        for name, filename in arrays.items():
            path = state_path / filename
            if not path.is_file() or _sha256(path) != array_hashes[name]:
                return False
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _save_state_atomic(state: Any, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
    shutil.rmtree(temporary, ignore_errors=True)
    try:
        state.save(temporary)
        if not _state_directory_matches(
            temporary,
            structure_hash=state.structure_hash,
            state_hash=state.state_hash,
        ):
            raise RuntimeError("Persisted BicrystalState failed immediate hash verification")
        os.replace(temporary, target)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _swap_case_directory(staging: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.backup-{os.getpid()}-{time.time_ns()}")
    try:
        if target.exists():
            os.replace(target, backup)
        os.replace(staging, target)
    except Exception:
        if not target.exists() and backup.exists():
            os.replace(backup, target)
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)
        shutil.rmtree(staging, ignore_errors=True)


def _parse_float(
    value: str, *, field: str, source_row: int, optional: bool = False
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
        raise ValueError(f"Row {source_row}: field {field!r} must be finite")
    return parsed


def _matrix_from_record(
    record: dict[str, str], *, prefix: str, source_row: int
) -> Matrix3:
    rows: list[tuple[int, int, int]] = []
    for axis in "xyz":
        values = []
        for column in range(3):
            field = f"{prefix}_{axis}{column}"
            try:
                values.append(int(record[field].strip()))
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Row {source_row}: field {field!r} must be an integer"
                ) from exc
        rows.append(tuple(values))  # type: ignore[arg-type]
    return tuple(rows)  # type: ignore[return-value]


def _load_cases(path: Path, *, expected_cases: int) -> list[BoundaryCase]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = [field for field in _REQUIRED_COLUMNS if field not in reader.fieldnames]
        if missing:
            raise ValueError("CSV is missing required columns: " + ", ".join(missing))
        cases = []
        for record in reader:
            if not any((value or "").strip() for value in record.values()):
                continue
            source_row = len(cases) + 1
            boundary_type = record["Type"].strip()
            axis_set = record["Axis Set"].strip()
            if boundary_type not in _ALLOWED_TYPES:
                raise ValueError(f"Row {source_row}: unsupported boundary Type")
            if axis_set not in _ALLOWED_AXIS_SETS:
                raise ValueError(f"Row {source_row}: unsupported Axis Set")
            cases.append(
                BoundaryCase(
                    source_row=source_row,
                    case_id=f"zhang_{source_row:03d}_{boundary_type}_{axis_set}",
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
    unknown_ids = [value for value in case_ids if value not in by_id]
    unknown_rows = [value for value in rows if value not in by_row]
    if unknown_ids:
        raise KeyError("Unknown case IDs: " + ", ".join(unknown_ids))
    if unknown_rows:
        raise KeyError("Unknown source rows: " + ", ".join(map(str, unknown_rows)))
    explicitly_selected = bool(case_ids or rows or patterns)
    selected = []
    for case in cases:
        if explicitly_selected and not (
            case.case_id in case_ids
            or case.source_row in rows
            or any(fnmatch.fnmatchcase(case.case_id, pattern) for pattern in patterns)
        ):
            continue
        if boundary_types and case.boundary_type not in boundary_types:
            continue
        if axis_sets and case.axis_set not in axis_sets:
            continue
        selected.append(case)
    return selected if limit is None else selected[:limit]


def _load_clean_settings(args: argparse.Namespace) -> tuple[Any, Path | None, str | None]:
    from GBOpt.clean_generation import (
        CleanGenerationSettings,
        RationalPhase,
        TerminationDomainSelection,
    )
    from GBOpt.geometry_validation import FeasibilityOverride
    from GBOpt.interface_initialization import CartesianTranslationDomain

    config_file = None
    config_sha = None
    if args.clean_config is None:
        settings = CleanGenerationSettings()
    else:
        config_file = _resolve(args.clean_config)
        if not config_file.is_file():
            raise FileNotFoundError(f"Clean configuration not found: {config_file}")
        config_sha = _sha256(config_file)
        settings = CleanGenerationSettings.from_file(config_file)

    if (args.in_plane_components_y is None) != (args.in_plane_components_z is None):
        raise CampaignConfigurationError(
            "Both --in-plane-components-y and --in-plane-components-z are required together."
        )
    translation_domain = settings.translation_domain
    if args.in_plane_components_y is not None or args.normal_offsets is not None:
        translation_domain = CartesianTranslationDomain(
            in_plane_components=(
                tuple(
                    settings.translation_domain.in_plane_components[0]
                    if args.in_plane_components_y is None
                    else args.in_plane_components_y
                ),
                tuple(
                    settings.translation_domain.in_plane_components[1]
                    if args.in_plane_components_z is None
                    else args.in_plane_components_z
                ),
            ),
            normal_offsets=tuple(
                settings.translation_domain.normal_offsets
                if args.normal_offsets is None
                else args.normal_offsets
            ),
        )

    mode = args.termination_mode
    if args.disable_termination_search:
        if mode not in (None, "default_only"):
            raise CampaignConfigurationError(
                "--disable-termination-search conflicts with --termination-mode"
            )
        mode = "default_only"
    phase_arguments = bool(args.left_termination_phase or args.right_termination_phase)
    termination = settings.termination_domain
    if mode is not None or phase_arguments:
        selected_mode = mode or "explicit"
        left = tuple(
            RationalPhase.parse(value, f"left termination phase {index}")
            for index, value in enumerate(args.left_termination_phase)
        )
        right = tuple(
            RationalPhase.parse(value, f"right termination phase {index}")
            for index, value in enumerate(args.right_termination_phase)
        )
        termination = TerminationDomainSelection(
            mode=selected_mode,
            left=left if selected_mode == "explicit" else (),
            right=right if selected_mode == "explicit" else (),
        )

    if (args.override_status is None) != (args.override_reason is None):
        raise CampaignConfigurationError(
            "--override-status and --override-reason must be supplied together"
        )
    override = settings.feasibility_override
    if args.override_status is not None:
        override = FeasibilityOverride(
            status=args.override_status, reason=args.override_reason
        )

    settings = settings.with_overrides(
        topology=args.topology,
        boundary_conditions=args.boundary_conditions,
        vacuum_angstrom=args.vacuum,
        fixed_region_thickness_angstrom=args.fixed_region_thickness,
        surface_buffer_thickness_angstrom=args.surface_buffer_thickness,
        retain_warnings=args.retain_warnings,
        max_seeds=args.max_seeds,
        initialization_enabled=args.initialization,
    )
    settings = CleanGenerationSettings(
        topology=settings.topology,
        boundary_conditions=settings.boundary_conditions,
        vacuum_angstrom=settings.vacuum_angstrom,
        fixed_region_thickness_angstrom=settings.fixed_region_thickness_angstrom,
        surface_buffer_thickness_angstrom=settings.surface_buffer_thickness_angstrom,
        feasibility_policy=settings.feasibility_policy,
        feasibility_override=override,
        translation_domain=translation_domain,
        termination_domain=termination,
        retain_warnings=settings.retain_warnings,
        max_seeds=settings.max_seeds,
        initialization_enabled=settings.initialization_enabled,
    )
    return settings, config_file, config_sha


def _configuration_payload(
    args: argparse.Namespace,
    source_sha256: str,
    clean_settings: Any,
    clean_sha: str | None,
    software_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "generator_schema": _GENERATOR_SCHEMA,
        "source_sha256": source_sha256,
        "clean_config_sha256": clean_sha,
        "software_identity": _jsonable(software_identity),
        "lattice_constant": args.lattice_constant,
        "structure": args.structure,
        "atom_types": list(args.atom_types),
        "charges": list(args.charges),
        "expected_ratio": list(args.expected_ratio),
        "x_dim_min": args.x_dim_min,
        "gb_thickness_periods": args.gb_thickness_periods,
        "repeat_factor": list(args.repeat_factor),
        "interaction_distance": args.interaction_distance,
        "mismatch_tol": args.mismatch_tol,
        "mismatch_max_cells": args.mismatch_max_cells,
        "strain_grain": args.strain_grain,
        "precision": args.precision,
        "clean_generation": clean_settings.to_dict(),
    }


def _make_config(
    args: argparse.Namespace, *, data_file: Path, output_root: Path
) -> GenerationConfig:
    source_sha256 = _sha256(data_file)
    project_root = None
    if args.project_root is not None:
        project_root = _resolve(args.project_root)
        if not project_root.is_dir():
            raise FileNotFoundError(f"Project root not found: {project_root}")
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
    clean_settings, clean_file, clean_sha = _load_clean_settings(args)
    software_identity = _software_identity()
    payload = _configuration_payload(
        args,
        source_sha256,
        clean_settings,
        clean_sha,
        software_identity,
    )
    return GenerationConfig(
        data_file=data_file,
        output_root=output_root,
        project_root=project_root,
        clean_config_file=clean_file,
        clean_config_sha256=clean_sha,
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
        mismatch_tol=args.mismatch_tol,
        mismatch_max_cells=args.mismatch_max_cells,
        strain_grain=args.strain_grain,
        precision=args.precision,
        timeout=args.timeout,
        diagnostic_chars=args.diagnostic_chars,
        source_sha256=source_sha256,
        software_identity=software_identity,
        clean_settings=clean_settings,
        generation_signature=_canonical_sha256(payload),
    )


def _counts(atoms: Any) -> dict[str, int]:
    import numpy as np

    names, values = np.unique(atoms["name"], return_counts=True)
    return {str(name): int(count) for name, count in zip(names, values)}


def _campaign_stoichiometry_check(
    atoms: Any,
    *,
    atom_types: tuple[str, str],
    expected_ratio: tuple[int, int],
    label: str,
) -> dict[str, Any]:
    """Return a non-throwing campaign-specific species-ratio check.

    Exact construction diagnostics must remain persistable even when a Zhang UO2
    species-ratio invariant fails.  Structural construction errors still raise before
    this helper is reached; species/count failures are represented as reason-bearing
    deterministic data instead of discarding the constructed state.
    """
    reasons: list[str] = []
    if atoms is None or getattr(atoms, "size", 0) == 0:
        return {
            "label": label,
            "passed": False,
            "counts": {},
            "expected_species": list(atom_types),
            "expected_ratio": list(expected_ratio),
            "reasons": ["campaign.stoichiometry.empty_population"],
        }
    counts = _counts(atoms)
    if set(counts) != set(atom_types):
        reasons.append("campaign.stoichiometry.species_set_mismatch")
    else:
        first, second = atom_types
        first_ratio, second_ratio = expected_ratio
        if counts[first] * second_ratio != counts[second] * first_ratio:
            reasons.append("campaign.stoichiometry.ratio_mismatch")
    return {
        "label": label,
        "passed": not reasons,
        "counts": counts,
        "expected_species": list(atom_types),
        "expected_ratio": list(expected_ratio),
        "reasons": reasons,
    }


def _peak_rss_mib() -> float | None:
    if resource is None:
        return None
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)


def _git_provenance(source: Path) -> dict[str, Any]:
    root = source if source.is_dir() else source.parent
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", top, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", top, "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
        return {"repository_root": top, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"repository_root": None, "commit": None, "dirty": None}


def _gbopt_provenance() -> dict[str, Any]:
    import GBOpt

    try:
        version = importlib.metadata.version("GBOpt")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(GBOpt, "__version__", "unknown")
    module_path = Path(GBOpt.__file__).resolve() if GBOpt.__file__ else None
    return {
        "version": version,
        "module_path": None if module_path is None else str(module_path),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "source_control": (
            {"repository_root": None, "commit": None, "dirty": None}
            if module_path is None
            else _git_provenance(module_path)
        ),
    }


def _software_identity() -> dict[str, Any]:
    """Return the source identity that participates in resumability signatures."""
    provenance = _gbopt_provenance()
    module_text = provenance.get("module_path")
    package_root = None if module_text is None else Path(str(module_text)).parent
    source_names = (
        "__init__.py",
        "BicrystalState.py",
        "BoundarySpec.py",
        "GBMaker.py",
        "UnitCell.py",
        "clean_generation.py",
        "gbmaker_supercell.py",
        "geometry_audit.py",
        "geometry_validation.py",
        "interface_initialization.py",
        "termination.py",
        "termination_initialization.py",
    )
    source_hashes: dict[str, str | None] = {}
    if package_root is not None:
        for name in source_names:
            path = package_root / name
            source_hashes[name] = _sha256(path) if path.is_file() else None
    generator_path = Path(__file__).resolve()
    return {
        **provenance,
        "generator_path": str(generator_path),
        "generator_sha256": _sha256(generator_path),
        "source_file_sha256": source_hashes,
    }


def _write_lammps_state(
    gb: Any,
    state: Any,
    path: Path,
    *,
    charges: Mapping[str, float],
    precision: int,
) -> str:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        gb.write_lammps(
            str(temporary),
            atoms=state.atoms,
            box_sizes=state.box_dims,
            type_as_int=True,
            charges=dict(charges),
            precision=precision,
        )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"GBOpt did not write a valid data file: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _resolve_termination_domain(selection: Any, gb: Any) -> Any:
    from GBOpt.termination_initialization import TerminationDomain

    available_left, available_right = gb.available_termination_descriptors
    if selection.mode == "all":
        return TerminationDomain(left=available_left, right=available_right)
    if selection.mode == "default_only":
        left = next(item for item in available_left if item.is_default)
        right = next(item for item in available_right if item.is_default)
        return TerminationDomain(left=(left,), right=(right,))
    requested_left = selection.descriptors("left")
    requested_right = selection.descriptors("right")
    unsupported = [
        item.to_dict()
        for item in (*requested_left, *requested_right)
        if item not in (available_left if item.grain == "left" else available_right)
    ]
    if unsupported:
        raise CampaignConfigurationError(
            "Explicit termination selection contains unsupported exact phases: "
            + json.dumps(unsupported, sort_keys=True, separators=(",", ":"))
        )
    return TerminationDomain(left=requested_left, right=requested_right)


def _reason_codes_from_phase7(result: Any) -> list[str]:
    codes = list(result.invalid_reasons)
    for attempt in result.attempts:
        codes.extend(attempt.rejection_reasons)
        if attempt.construction_error:
            codes.append("termination.construction_error")
        if attempt.validation_error:
            codes.append("termination.validation_error")
    return sorted(set(codes))


def _campaign_seed_kind(seed: Any) -> str:
    if seed.kind == "default_zero":
        return "base_zero"
    if seed.kind == "nondefault_zero":
        return "nondefault_termination_zero"
    return (
        "default_termination_translation"
        if seed.candidate.is_default
        else "termination_plus_translation"
    )


def _persist_seed(
    *,
    seed_root: Path,
    order: int,
    state: Any,
    report: Any,
    gb_writer: Any,
    charges: Mapping[str, float],
    precision: int,
    descriptor: Mapping[str, Any],
    generation_signature: str,
) -> dict[str, Any]:
    seed_dir = seed_root / f"seed_{order:03d}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    state_dir = seed_dir / "state"
    _save_state_atomic(state, state_dir)
    report_payload = report.to_dict()
    _atomic_json(seed_dir / "report.json", report_payload)
    data_sha = _write_lammps_state(
        gb_writer,
        state,
        seed_dir / "structure.data",
        charges=charges,
        precision=precision,
    )
    deterministic = {
        "order": order,
        **_jsonable(descriptor),
        "state": {
            "directory": "state",
            "structure_hash": state.structure_hash,
            "state_hash": state.state_hash,
        },
        "feasibility_report": {
            "path": "report.json",
            "report_hash": report.report_hash,
            "policy_hash": report.policy.policy_hash,
            "status": report.status,
            "raw_status": report.raw_status,
        },
        "structure_file": {"path": "structure.data", "sha256": data_sha},
    }
    document = _hashed_document(
        schema="gbopt-clean-generation-seed-v1",
        generation_signature=generation_signature,
        deterministic=deterministic,
    )
    _atomic_json(seed_dir / "seed.json", document)
    return {
        "order": order,
        "path": str(Path("seeds") / seed_dir.name),
        "seed_metadata": str(Path("seeds") / seed_dir.name / "seed.json"),
        "seed_hash": document["report_hash"],
        "kind": deterministic["kind"],
        "structure_hash": state.structure_hash,
        "state_hash": state.state_hash,
        "report_hash": report.report_hash,
        "data_sha256": data_sha,
    }


def _base_descriptor(
    *,
    state: Any,
    data_sha: str,
    report: Any,
    population: Any,
    campaign_stoichiometry: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "state_directory": "base_state",
        "structure_file": "base_structure.data",
        "structure_file_sha256": data_sha,
        "structure_hash": state.structure_hash,
        "state_hash": state.state_hash,
        "natoms": int(state.atoms.size),
        "topology": state.topology,
        "boundary_conditions": list(state.boundary_conditions),
        "termination_ids": (
            None if state.termination_ids is None else list(state.termination_ids)
        ),
        "decorated_population_check": population.to_dict(),
        "campaign_stoichiometry_check": _jsonable(campaign_stoichiometry),
        "feasibility_report": report.to_dict(),
    }


def _case_source(case: BoundaryCase, config: GenerationConfig) -> dict[str, Any]:
    return {
        "source_csv": str(config.data_file),
        "source_csv_sha256": config.source_sha256,
        "source_row": case.source_row,
        "case_id": case.case_id,
        "P": case.P,
        "Q": case.Q,
        "boundary_type": case.boundary_type,
        "axis_set": case.axis_set,
        "published_reference_metadata": {
            "uo2_basak_j_m2": case.uo2_reference_j_m2,
            "ceo2_gotte_j_m2": case.ceo2_reference_j_m2,
            "used_for_seed_ordering": False,
        },
    }


def _finalize_case(
    *,
    staging: Path,
    target: Path,
    config: GenerationConfig,
    case: BoundaryCase,
    status: str,
    failure_stage: str,
    reason_codes: Sequence[str],
    construction_document: dict[str, Any],
    initialization_document: dict[str, Any],
    retained_seeds: Sequence[dict[str, Any]],
    base: Mapping[str, Any] | None,
    phase7_status: str | None,
    phase7_result_hash: str | None,
    error_type: str = "",
    message: str = "",
) -> dict[str, Any]:
    if status not in _CLEAN_STATUSES:
        raise RuntimeError(f"Unsupported clean-generation status {status!r}")
    _atomic_json(staging / "construction.json", construction_document)
    _atomic_json(staging / "initialization.json", initialization_document)
    deterministic = {
        "generator_schema": _GENERATOR_SCHEMA,
        "source": _case_source(case, config),
        "status": status,
        "failure_stage": failure_stage,
        "reason_codes": sorted(set(reason_codes)),
        "effective_configuration": config.clean_settings.to_dict(),
        "effective_configuration_hash": config.clean_settings.configuration_hash,
        "construction_report": {
            "path": "construction.json",
            "report_hash": construction_document["report_hash"],
        },
        "initialization_report": {
            "path": "initialization.json",
            "report_hash": initialization_document["report_hash"],
        },
        "base": None if base is None else _jsonable(base),
        "retained_seeds": list(retained_seeds),
        "phase7_status": phase7_status,
        "phase7_result_hash": phase7_result_hash,
        "software": _jsonable(config.software_identity),
    }
    case_document = _hashed_document(
        schema=_CASE_SCHEMA,
        generation_signature=config.generation_signature,
        deterministic=deterministic,
        diagnostics={
            "error_type": error_type,
            "message": message,
            "warnings": [],
            "elapsed_s": None,
            "peak_rss_mib": None,
        },
    )
    _atomic_json(staging / "case.json", case_document)
    _swap_case_directory(staging, target)
    base_report = None if base is None else base.get("feasibility_report")
    return {
        "case_id": case.case_id,
        "source_row": case.source_row,
        "status": status,
        "failure_stage": failure_stage,
        "reason_codes": json.dumps(sorted(set(reason_codes)), separators=(",", ":")),
        "resumed": False,
        "boundary_type": case.boundary_type,
        "axis_set": case.axis_set,
        "topology": config.clean_settings.topology,
        "boundary_conditions": json.dumps(
            list(config.clean_settings.boundary_conditions), separators=(",", ":")
        ),
        "base_natoms": "" if base is None else base.get("natoms", ""),
        "base_feasibility_status": (
            "" if base_report is None else base_report.get("status", "")
        ),
        "base_feasibility_raw_status": (
            "" if base_report is None else base_report.get("raw_status", "")
        ),
        "base_structure_hash": "" if base is None else base.get("structure_hash", ""),
        "base_state_hash": "" if base is None else base.get("state_hash", ""),
        "retained_seed_count": len(retained_seeds),
        "retained_seed_hashes": json.dumps(
            [item["seed_hash"] for item in retained_seeds], separators=(",", ":")
        ),
        "phase7_status": phase7_status or "",
        "phase7_result_hash": phase7_result_hash or "",
        "process_state": "completed",
        "case_directory": str(target),
        "case_metadata_file": str(target / "case.json"),
        "construction_report_file": str(target / "construction.json"),
        "initialization_report_file": str(target / "initialization.json"),
        "warning_count": 0,
        "warnings": "[]",
        "error_type": error_type,
        "message": message,
    }


def _generate_case(case: BoundaryCase, config: GenerationConfig) -> dict[str, Any]:
    target = config.output_root / case.case_id
    staging = config.output_root / f".{case.case_id}.tmp-{os.getpid()}-{time.time_ns()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    started = time.perf_counter()
    stage = "exact_construction"
    base: dict[str, Any] | None = None
    construction_document: dict[str, Any] | None = None

    if config.project_root is not None and str(config.project_root) not in sys.path:
        sys.path.insert(0, str(config.project_root))

    try:
        import numpy as np
        from GBOpt.BoundarySpec import PQSpec
        from GBOpt.GBMaker import GBMaker
        from GBOpt.geometry_validation import FeasibilityPolicy, validate_bicrystal_state
        from GBOpt.termination import TerminationPair
        from GBOpt.termination_initialization import (
            ExactBoundaryReconstruction,
            check_decorated_population,
            generate_termination_seeds,
        )

        settings = config.clean_settings
        boundary = PQSpec(
            P=np.asarray(case.P, dtype=object),
            Q=np.asarray(case.Q, dtype=object),
            basis_mode="supplied",
        )
        provenance = {
            "campaign": "zhang_uo2_clean_generation_phase8",
            **_case_source(case, config),
            "generation_signature": config.generation_signature,
        }
        common = {
            "a0": config.lattice_constant,
            "structure": config.structure,
            "atom_types": config.atom_types,
            "boundary": boundary,
            "mode": "exact",
            "repeat_factor": config.repeat_factor,
            "x_dim_min": config.x_dim_min,
            "vacuum": settings.vacuum_angstrom,
            "fixed_region_thickness": settings.fixed_region_thickness_angstrom,
            "surface_buffer_thickness": settings.surface_buffer_thickness_angstrom,
            "interaction_distance": config.interaction_distance,
            "mismatch_tol": config.mismatch_tol,
            "mismatch_max_cells": config.mismatch_max_cells,
            "strain_grain": config.strain_grain,
            "topology": settings.topology,
            "boundary_conditions": settings.boundary_conditions,
            "termination_pair": TerminationPair(),
            "provenance": provenance,
        }
        gb = GBMaker.from_boundary_spec(gb_thickness=0.0, **common)
        if not gb.uses_exact_construction:
            raise RuntimeError("Default construction did not use exact supplied P/Q")
        gb_thickness = config.gb_thickness_periods * max(
            float(gb.spacing["x"]["left"]), float(gb.spacing["x"]["right"])
        )
        if gb_thickness != 0.0:
            gb.gb_thickness = gb_thickness

        reconstruction = ExactBoundaryReconstruction(
            a0=config.lattice_constant,
            structure=config.structure,
            atom_types=config.atom_types,
            boundary=boundary,
            gb_thickness=gb_thickness,
            repeat_factor=config.repeat_factor,
            x_dim_min=config.x_dim_min,
            vacuum=settings.vacuum_angstrom,
            fixed_region_thickness=settings.fixed_region_thickness_angstrom,
            surface_buffer_thickness=settings.surface_buffer_thickness_angstrom,
            interaction_distance=config.interaction_distance,
            mismatch_tol=config.mismatch_tol,
            mismatch_max_cells=config.mismatch_max_cells,
            strain_grain=config.strain_grain,
            topology=settings.topology,
            boundary_conditions=settings.boundary_conditions,
            provenance=provenance,
        )
        left_stoichiometry = _campaign_stoichiometry_check(
            gb.left_grain,
            atom_types=config.atom_types,
            expected_ratio=config.expected_ratio,
            label=f"{case.case_id} left grain",
        )
        right_stoichiometry = _campaign_stoichiometry_check(
            gb.right_grain,
            atom_types=config.atom_types,
            expected_ratio=config.expected_ratio,
            label=f"{case.case_id} right grain",
        )
        whole_stoichiometry = _campaign_stoichiometry_check(
            gb.whole_system,
            atom_types=config.atom_types,
            expected_ratio=config.expected_ratio,
            label=f"{case.case_id} whole system",
        )
        campaign_stoichiometry = {
            "passed": all(
                item["passed"]
                for item in (
                    left_stoichiometry,
                    right_stoichiometry,
                    whole_stoichiometry,
                )
            ),
            "left": left_stoichiometry,
            "right": right_stoichiometry,
            "whole": whole_stoichiometry,
            "reasons": sorted(
                {
                    reason
                    for item in (
                        left_stoichiometry,
                        right_stoichiometry,
                        whole_stoichiometry,
                    )
                    for reason in item["reasons"]
                }
            ),
        }
        population = check_decorated_population(gb)
        effective_policy = FeasibilityPolicy.from_unit_cell(
            gb.unit_cell,
            contact=settings.feasibility_policy.contact,
            void=settings.feasibility_policy.void,
            slab=settings.feasibility_policy.slab,
        )
        base_report = validate_bicrystal_state(
            gb.bicrystal_state,
            policy=effective_policy,
            override=settings.feasibility_override,
        )
        charge_map = dict(zip(config.atom_types, config.charges))
        _save_state_atomic(gb.bicrystal_state, staging / "base_state")
        base_data_sha = _write_lammps_state(
            gb,
            gb.bicrystal_state,
            staging / "base_structure.data",
            charges=charge_map,
            precision=config.precision,
        )
        base = _base_descriptor(
            state=gb.bicrystal_state,
            data_sha=base_data_sha,
            report=base_report,
            population=population,
            campaign_stoichiometry=campaign_stoichiometry,
        )
        construction_deterministic = {
            "source": _case_source(case, config),
            "construction_mode": "exact",
            "basis_mode": "supplied",
            "default_termination_descriptor": TerminationPair().to_dict(),
            "reconstruction": reconstruction.to_dict(),
            "reconstruction_hash": reconstruction.reconstruction_hash,
            "effective_policy": effective_policy.to_dict(),
            "effective_policy_hash": effective_policy.policy_hash,
            "configured_policy_hash": settings.feasibility_policy.policy_hash,
            "feasibility_override": (
                None
                if settings.feasibility_override is None
                else {
                    "status": settings.feasibility_override.status,
                    "reason": settings.feasibility_override.reason,
                }
            ),
            "atom_counts": {
                "left": left_stoichiometry["counts"],
                "right": right_stoichiometry["counts"],
                "whole": whole_stoichiometry["counts"],
            },
            "campaign_stoichiometry_check": campaign_stoichiometry,
            "base": base,
        }
        construction_document = _hashed_document(
            schema=_CONSTRUCTION_SCHEMA,
            generation_signature=config.generation_signature,
            deterministic=construction_deterministic,
        )

        if not population.passed or not campaign_stoichiometry["passed"]:
            construction_reasons = sorted(
                set(population.reasons) | set(campaign_stoichiometry["reasons"])
            )
            initialization_document = _hashed_document(
                schema=_INITIALIZATION_SCHEMA,
                generation_signature=config.generation_signature,
                deterministic={
                    "search_invoked": False,
                    "outcome": "construction_population_check_failed",
                    "phase7_result": None,
                    "retained_seeds": [],
                    "reason_codes": construction_reasons,
                },
            )
            return _finalize_case(
                staging=staging,
                target=target,
                config=config,
                case=case,
                status="construction_failed",
                failure_stage="decorated_population_or_stoichiometry",
                reason_codes=construction_reasons,
                construction_document=construction_document,
                initialization_document=initialization_document,
                retained_seeds=(),
                base=base,
                phase7_status=None,
                phase7_result_hash=None,
                message=(
                    "Exact construction failed decorated-population or campaign "
                    "stoichiometry validation."
                ),
            )

        accepted = {"feasible"}
        if settings.retain_warnings:
            accepted.add("warning")
        if base_report.status in accepted:
            stage = "persist_base_seed"
            seed_ref = _persist_seed(
                seed_root=staging / "seeds",
                order=0,
                state=gb.bicrystal_state,
                report=base_report,
                gb_writer=gb,
                charges=charge_map,
                precision=config.precision,
                descriptor={
                    "kind": "base_zero",
                    "phase7_seed_kind": None,
                    "phase7_seed_hash": None,
                    "termination_pair": TerminationPair().to_dict(),
                    "applied_translation_lab": [0.0, 0.0, 0.0],
                    "population_check": population.to_dict(),
                    "nested_translation_result_hash": None,
                    "reconstruction": reconstruction.to_dict(),
                    "reconstruction_hash": reconstruction.reconstruction_hash,
                },
                generation_signature=config.generation_signature,
            )
            initialization_document = _hashed_document(
                schema=_INITIALIZATION_SCHEMA,
                generation_signature=config.generation_signature,
                deterministic={
                    "search_invoked": False,
                    "outcome": "base_exact_state_accepted",
                    "acceptance_rule": {
                        "retain_warnings": settings.retain_warnings,
                        "accepted_statuses": sorted(accepted),
                    },
                    "configured_translation_domain": settings.translation_domain.to_dict(),
                    "configured_translation_domain_hash": settings.translation_domain.domain_hash,
                    "configured_termination_selection": settings.termination_domain.to_dict(),
                    "configured_termination_selection_hash": (
                        settings.termination_domain.selection_hash
                    ),
                    "phase7_result": None,
                    "retained_seeds": [seed_ref],
                    "reason_codes": [],
                },
            )
            return _finalize_case(
                staging=staging,
                target=target,
                config=config,
                case=case,
                status="feasible_seed_ready",
                failure_stage="none",
                reason_codes=(),
                construction_document=construction_document,
                initialization_document=initialization_document,
                retained_seeds=(seed_ref,),
                base=base,
                phase7_status=None,
                phase7_result_hash=None,
            )

        if not settings.initialization_enabled:
            reasons = [reason.code for reason in base_report.reasons]
            initialization_document = _hashed_document(
                schema=_INITIALIZATION_SCHEMA,
                generation_signature=config.generation_signature,
                deterministic={
                    "search_invoked": False,
                    "outcome": "initialization_disabled",
                    "phase7_result": None,
                    "retained_seeds": [],
                    "reason_codes": reasons,
                },
            )
            return _finalize_case(
                staging=staging,
                target=target,
                config=config,
                case=case,
                status="constructed_infeasible",
                failure_stage="base_feasibility",
                reason_codes=reasons,
                construction_document=construction_document,
                initialization_document=initialization_document,
                retained_seeds=(),
                base=base,
                phase7_status=None,
                phase7_result_hash=None,
                message="Base exact state was not accepted and initialization is disabled.",
            )

        stage = "termination_domain_resolution"
        try:
            termination_domain = _resolve_termination_domain(
                settings.termination_domain, gb
            )
        except CampaignConfigurationError as exc:
            reason = "initialization.unsupported_termination_selection"
            initialization_document = _hashed_document(
                schema=_INITIALIZATION_SCHEMA,
                generation_signature=config.generation_signature,
                deterministic={
                    "search_invoked": False,
                    "outcome": "invalid_configuration",
                    "failure_stage": "termination_domain_resolution",
                    "configured_termination_selection": (
                        settings.termination_domain.to_dict()
                    ),
                    "configured_termination_selection_hash": (
                        settings.termination_domain.selection_hash
                    ),
                    "phase7_result": None,
                    "retained_seeds": [],
                    "reason_codes": [reason],
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return _finalize_case(
                staging=staging,
                target=target,
                config=config,
                case=case,
                status="seed_generation_failed",
                failure_stage="initializer_input",
                reason_codes=(reason,),
                construction_document=construction_document,
                initialization_document=initialization_document,
                retained_seeds=(),
                base=base,
                phase7_status="invalid_input",
                phase7_result_hash=None,
                error_type=type(exc).__name__,
                message=str(exc),
            )
        stage = "phase7_seed_generation"
        phase7 = generate_termination_seeds(
            reconstruction=reconstruction,
            feasibility_policy=effective_policy,
            feasibility_override=settings.feasibility_override,
            termination_domain=termination_domain,
            translation_domain=settings.translation_domain,
            max_seeds=settings.max_seeds,
            retain_warnings=settings.retain_warnings,
        )
        if (
            phase7.source_structure_hash is not None
            and phase7.source_structure_hash != gb.bicrystal_state.structure_hash
        ):
            raise RuntimeError(
                "Phase 7 source reconstruction differs from persisted base structure"
            )
        retained = []
        stage = "persist_retained_seeds"
        for order, seed in enumerate(phase7.seeds):
            retained.append(
                _persist_seed(
                    seed_root=staging / "seeds",
                    order=order,
                    state=seed.state,
                    report=seed.report,
                    gb_writer=gb,
                    charges=charge_map,
                    precision=config.precision,
                    descriptor={
                        "kind": _campaign_seed_kind(seed),
                        "phase7_seed_kind": seed.kind,
                        "phase7_seed_hash": seed.seed_hash,
                        "candidate": seed.candidate.to_dict(),
                        "termination_pair": seed.termination_pair.to_dict(),
                        "applied_translation_lab": list(seed.applied_translation_lab),
                        "population_check": seed.population_check.to_dict(),
                        "nested_translation_result_hash": seed.nested_translation_result_hash,
                        "reconstruction": reconstruction.to_dict(),
                        "reconstruction_hash": reconstruction.reconstruction_hash,
                    },
                    generation_signature=config.generation_signature,
                )
            )
        phase7_payload = phase7.to_dict()
        reasons = _reason_codes_from_phase7(phase7)
        initialization_document = _hashed_document(
            schema=_INITIALIZATION_SCHEMA,
            generation_signature=config.generation_signature,
            deterministic={
                "search_invoked": True,
                "outcome": phase7.status,
                "acceptance_rule": {
                    "retain_warnings": settings.retain_warnings,
                    "accepted_statuses": sorted(accepted),
                },
                "termination_domain": termination_domain.to_dict(),
                "termination_domain_hash": termination_domain.domain_hash,
                "translation_domain": settings.translation_domain.to_dict(),
                "translation_domain_hash": settings.translation_domain.domain_hash,
                "phase7_result": phase7_payload,
                "phase7_result_hash": phase7.result_hash,
                "retained_seeds": retained,
                "reason_codes": reasons,
            },
        )
        if retained:
            status = "feasible_seed_ready"
            failure_stage = "none"
        elif phase7.status == "invalid_input":
            status = "seed_generation_failed"
            failure_stage = "initializer_input"
        elif len(termination_domain.candidates()) == 1:
            status = "translation_search_exhausted"
            failure_stage = "translation_domain"
        else:
            status = "termination_search_exhausted"
            failure_stage = "termination_translation_domain"
        return _finalize_case(
            staging=staging,
            target=target,
            config=config,
            case=case,
            status=status,
            failure_stage=failure_stage,
            reason_codes=reasons,
            construction_document=construction_document,
            initialization_document=initialization_document,
            retained_seeds=retained,
            base=base,
            phase7_status=phase7.status,
            phase7_result_hash=phase7.result_hash,
            message=("" if retained else "Configured clean-generation domain retained no seed."),
        )

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        status = "construction_failed" if base is None else "seed_generation_failed"
        failure_stage = stage
        reason = (
            "construction.exact_build_failed"
            if base is None
            else "initialization.unexpected_failure"
        )
        if construction_document is None:
            construction_document = _hashed_document(
                schema=_CONSTRUCTION_SCHEMA,
                generation_signature=config.generation_signature,
                deterministic={
                    "source": _case_source(case, config),
                    "construction_mode": "exact",
                    "outcome": "error",
                    "failure_stage": stage,
                    "reason_codes": [reason],
                    "error": error,
                    "base": None,
                },
            )
        initialization_document = _hashed_document(
            schema=_INITIALIZATION_SCHEMA,
            generation_signature=config.generation_signature,
            deterministic={
                "search_invoked": base is not None,
                "outcome": "error",
                "failure_stage": stage,
                "phase7_result": None,
                "retained_seeds": [],
                "reason_codes": [reason],
                "error": error,
            },
        )
        try:
            return _finalize_case(
                staging=staging,
                target=target,
                config=config,
                case=case,
                status=status,
                failure_stage=failure_stage,
                reason_codes=(reason,),
                construction_document=construction_document,
                initialization_document=initialization_document,
                retained_seeds=(),
                base=base,
                phase7_status=None,
                phase7_result_hash=None,
                error_type=type(exc).__name__,
                message=detail,
            )
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
    finally:
        _ = time.perf_counter() - started


def _augment_case_diagnostics(
    result: dict[str, Any], *, elapsed: float, peak: float | None, warning_messages: list[str]
) -> None:
    case_dir_text = result.get("case_directory")
    if not case_dir_text:
        return
    case_dir = Path(case_dir_text)
    for filename in ("case.json", "construction.json", "initialization.json"):
        path = case_dir / filename
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not _hashed_document_matches(payload):
                continue
            diagnostics = dict(payload.get("diagnostics", {}))
            diagnostics.update(
                {
                    "elapsed_s": round(elapsed, 6),
                    "peak_rss_mib": None if peak is None else round(peak, 3),
                    "warnings": warning_messages,
                }
            )
            payload["diagnostics"] = diagnostics
            _atomic_json(path, payload)
        except (OSError, ValueError, json.JSONDecodeError):
            continue


def _child_run(case: BoundaryCase, config: GenerationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _generate_case(case, config)
    elapsed = time.perf_counter() - started
    peak = _peak_rss_mib()
    warning_messages = [f"{item.category.__name__}: {item.message}" for item in caught]
    _augment_case_diagnostics(
        result, elapsed=elapsed, peak=peak, warning_messages=warning_messages
    )
    result["elapsed_s"] = round(elapsed, 6)
    result["peak_rss_mib"] = "" if peak is None else round(peak, 3)
    result["warning_count"] = len(warning_messages)
    result["warnings"] = json.dumps(warning_messages, separators=(",", ":"))
    return result


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _verify_seed(case_dir: Path, ref: Mapping[str, Any], signature: str) -> bool:
    try:
        order = int(ref["order"])
        expected_path = Path("seeds") / f"seed_{order:03d}"
        expected_metadata = expected_path / "seed.json"
        if Path(str(ref["path"])) != expected_path:
            return False
        if Path(str(ref["seed_metadata"])) != expected_metadata:
            return False
        seed_dir = case_dir / expected_path
        metadata = _read_json(case_dir / expected_metadata)
        if not _hashed_document_matches(
            metadata, schema="gbopt-clean-generation-seed-v1", signature=signature
        ):
            return False
        deterministic = metadata["deterministic"]
        if metadata["report_hash"] != ref["seed_hash"]:
            return False
        if deterministic["order"] != order or deterministic["kind"] != ref["kind"]:
            return False
        state = deterministic["state"]
        if state["structure_hash"] != ref["structure_hash"]:
            return False
        if state["state_hash"] != ref["state_hash"]:
            return False
        if not _state_directory_matches(
            seed_dir / state["directory"],
            structure_hash=state["structure_hash"],
            state_hash=state["state_hash"],
        ):
            return False
        report_path = seed_dir / deterministic["feasibility_report"]["path"]
        report = _read_json(report_path)
        if not _feasibility_report_matches(report):
            return False
        if report["report_hash"] != deterministic["feasibility_report"]["report_hash"]:
            return False
        if report["report_hash"] != ref["report_hash"]:
            return False
        if report["policy_hash"] != deterministic["feasibility_report"]["policy_hash"]:
            return False
        if report["status"] != deterministic["feasibility_report"]["status"]:
            return False
        if report["raw_status"] != deterministic["feasibility_report"]["raw_status"]:
            return False
        data = deterministic["structure_file"]
        data_path = seed_dir / data["path"]
        return (
            data["sha256"] == ref["data_sha256"]
            and data_path.is_file()
            and _sha256(data_path) == data["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _existing_result(case: BoundaryCase, config: GenerationConfig) -> dict[str, Any] | None:
    case_dir = config.output_root / case.case_id
    case_document = _read_json(case_dir / "case.json")
    construction = _read_json(case_dir / "construction.json")
    initialization = _read_json(case_dir / "initialization.json")
    if not all(
        (
            _hashed_document_matches(
                case_document, schema=_CASE_SCHEMA, signature=config.generation_signature
            ),
            _hashed_document_matches(
                construction,
                schema=_CONSTRUCTION_SCHEMA,
                signature=config.generation_signature,
            ),
            _hashed_document_matches(
                initialization,
                schema=_INITIALIZATION_SCHEMA,
                signature=config.generation_signature,
            ),
        )
    ):
        return None
    try:
        deterministic = case_document["deterministic"]
        if deterministic["source"]["case_id"] != case.case_id:
            return None
        if deterministic["source"]["source_row"] != case.source_row:
            return None
        status = deterministic["status"]
        if status not in _CLEAN_STATUSES:
            return None
        if deterministic["effective_configuration"] != _jsonable(config.clean_settings.to_dict()):
            return None
        if (
            deterministic["effective_configuration_hash"]
            != config.clean_settings.configuration_hash
        ):
            return None
        if deterministic["software"] != _jsonable(config.software_identity):
            return None
        if deterministic["construction_report"]["report_hash"] != construction["report_hash"]:
            return None
        if deterministic["initialization_report"]["report_hash"] != initialization["report_hash"]:
            return None
        base = deterministic["base"]
        if construction["deterministic"].get("base") != base:
            return None
        if base is not None:
            if not _state_directory_matches(
                case_dir / base["state_directory"],
                structure_hash=base["structure_hash"],
                state_hash=base["state_hash"],
            ):
                return None
            base_data = case_dir / base["structure_file"]
            if not base_data.is_file() or _sha256(base_data) != base["structure_file_sha256"]:
                return None
            if not _feasibility_report_matches(base["feasibility_report"]):
                return None
        seed_refs = deterministic["retained_seeds"]
        if initialization["deterministic"].get("retained_seeds") != seed_refs:
            return None
        if any(ref["order"] != index for index, ref in enumerate(seed_refs)):
            return None
        if not all(_verify_seed(case_dir, ref, config.generation_signature) for ref in seed_refs):
            return None
        if status == "feasible_seed_ready" and not seed_refs:
            return None
        phase7_payload = initialization["deterministic"].get("phase7_result")
        phase7_hash = deterministic.get("phase7_result_hash")
        initialization_phase7_hash = initialization["deterministic"].get(
            "phase7_result_hash"
        )
        if phase7_payload is None:
            if phase7_hash is not None or initialization_phase7_hash is not None:
                return None
        else:
            if not _phase7_result_matches(phase7_payload):
                return None
            if phase7_payload["result_hash"] != phase7_hash:
                return None
            if phase7_payload["result_hash"] != initialization_phase7_hash:
                return None
        if status != "feasible_seed_ready" and seed_refs:
            return None
        if status not in {"construction_failed", "seed_generation_failed"} and base is None:
            return None
        base_report = None if base is None else base["feasibility_report"]
        return {
            "case_id": case.case_id,
            "source_row": case.source_row,
            "status": status,
            "failure_stage": deterministic["failure_stage"],
            "reason_codes": json.dumps(
                deterministic["reason_codes"], separators=(",", ":")
            ),
            "resumed": True,
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "topology": deterministic["effective_configuration"]["topology"],
            "boundary_conditions": json.dumps(
                deterministic["effective_configuration"]["boundary_conditions"],
                separators=(",", ":"),
            ),
            "base_natoms": "" if base is None else base["natoms"],
            "base_feasibility_status": (
                "" if base_report is None else base_report["status"]
            ),
            "base_feasibility_raw_status": (
                "" if base_report is None else base_report["raw_status"]
            ),
            "base_structure_hash": "" if base is None else base["structure_hash"],
            "base_state_hash": "" if base is None else base["state_hash"],
            "retained_seed_count": len(seed_refs),
            "retained_seed_hashes": json.dumps(
                [item["seed_hash"] for item in seed_refs], separators=(",", ":")
            ),
            "phase7_status": deterministic.get("phase7_status") or "",
            "phase7_result_hash": deterministic.get("phase7_result_hash") or "",
            "elapsed_s": 0.0,
            "peak_rss_mib": "",
            "process_state": "resumed",
            "returncode": 0,
            "signal": "",
            "case_directory": str(case_dir),
            "case_metadata_file": str(case_dir / "case.json"),
            "construction_report_file": str(case_dir / "construction.json"),
            "initialization_report_file": str(case_dir / "initialization.json"),
            "warning_count": 0,
            "warnings": "[]",
            "error_type": "",
            "message": "Existing complete output passed schema and hash verification.",
            "stdout_tail": "",
            "stderr_tail": "",
        }
    except (KeyError, TypeError, ValueError, OSError):
        return None


def _common_child_args(config: GenerationConfig) -> list[str]:
    result = [
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
    if config.mismatch_tol is not None:
        result.extend(("--mismatch-tol", repr(config.mismatch_tol)))
    if config.project_root is not None:
        result.extend(("--project-root", str(config.project_root)))
    if config.clean_config_file is not None:
        result.extend(("--clean-config", str(config.clean_config_file)))
    # Always transmit the fully effective focused overrides. The config file continues
    # to provide the complete feasibility policy and exact domain selection.
    settings = config.clean_settings
    result.extend(("--topology", settings.topology))
    result.extend(("--boundary-conditions", *settings.boundary_conditions))
    result.extend(("--vacuum", repr(settings.vacuum_angstrom)))
    result.extend(
        ("--fixed-region-thickness", repr(settings.fixed_region_thickness_angstrom))
    )
    result.extend(
        ("--surface-buffer-thickness", repr(settings.surface_buffer_thickness_angstrom))
    )
    result.extend(("--max-seeds", str(settings.max_seeds)))
    result.append("--retain-warnings" if settings.retain_warnings else "--no-retain-warnings")
    result.append("--initialization" if settings.initialization_enabled else "--no-initialization")
    result.extend(
        (
            "--in-plane-components-y",
            *(
                repr(value)
                for value in settings.translation_domain.in_plane_components[0]
            ),
        )
    )
    result.extend(
        (
            "--in-plane-components-z",
            *(
                repr(value)
                for value in settings.translation_domain.in_plane_components[1]
            ),
        )
    )
    result.extend(
        (
            "--normal-offsets",
            *(repr(value) for value in settings.translation_domain.normal_offsets),
        )
    )
    result.extend(("--termination-mode", settings.termination_domain.mode))
    if settings.termination_domain.mode == "explicit":
        for phase in settings.termination_domain.left:
            result.extend(("--left-termination-phase", f"{phase.numerator}/{phase.denominator}"))
        for phase in settings.termination_domain.right:
            result.extend(("--right-termination-phase", f"{phase.numerator}/{phase.denominator}"))
    if settings.feasibility_override is not None:
        result.extend(("--override-status", settings.feasibility_override.status))
        result.extend(("--override-reason", settings.feasibility_override.reason))
    return result


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _tail(text: object, limit: int) -> str:
    normalized = _text(text)
    return normalized if len(normalized) <= limit else normalized[-limit:]


def _signal_name(returncode: int) -> str:
    if returncode >= 0:
        return ""
    try:
        return signal.Signals(-returncode).name
    except ValueError:
        return f"SIG{-returncode}"


def _parse_child_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_RESULT_MARKER):
            try:
                value = json.loads(line[len(_RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _diagnostic_stdout(stdout: str) -> str:
    return "\n".join(
        line for line in stdout.splitlines() if not line.startswith(_RESULT_MARKER)
    )


def _execute_case(
    case: BoundaryCase, *, script_path: Path, config: GenerationConfig
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script_path),
        *_common_child_args(config),
        "--run-one",
        case.case_id,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "case_id": case.case_id,
            "source_row": case.source_row,
            "status": "seed_generation_failed",
            "failure_stage": "child_timeout",
            "reason_codes": '["process.timeout"]',
            "resumed": False,
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "topology": config.clean_settings.topology,
            "boundary_conditions": json.dumps(list(config.clean_settings.boundary_conditions)),
            "retained_seed_count": 0,
            "retained_seed_hashes": "[]",
            "elapsed_s": round(time.perf_counter() - started, 6),
            "peak_rss_mib": "",
            "process_state": "timeout",
            "returncode": "",
            "signal": "",
            "warning_count": 0,
            "warnings": "[]",
            "error_type": "TimeoutExpired",
            "message": f"Case exceeded {config.timeout:g} seconds.",
            "stdout_tail": _tail(exc.stdout or "", config.diagnostic_chars),
            "stderr_tail": _tail(exc.stderr or "", config.diagnostic_chars),
        }
    except OSError as exc:
        return {
            "case_id": case.case_id,
            "source_row": case.source_row,
            "status": "seed_generation_failed",
            "failure_stage": "child_launch",
            "reason_codes": '["process.launch_error"]',
            "resumed": False,
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "topology": config.clean_settings.topology,
            "boundary_conditions": json.dumps(list(config.clean_settings.boundary_conditions)),
            "retained_seed_count": 0,
            "retained_seed_hashes": "[]",
            "elapsed_s": round(time.perf_counter() - started, 6),
            "peak_rss_mib": "",
            "process_state": "launch_error",
            "returncode": "",
            "signal": "",
            "warning_count": 0,
            "warnings": "[]",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "stdout_tail": "",
            "stderr_tail": "",
        }
    stdout = _text(completed.stdout)
    stderr = _text(completed.stderr)
    result = _parse_child_result(stdout)
    signal_name = _signal_name(completed.returncode)
    if result is None:
        process_state = "signaled" if completed.returncode < 0 else "no_result"
        result = {
            "case_id": case.case_id,
            "source_row": case.source_row,
            "status": "seed_generation_failed",
            "failure_stage": "child_process",
            "reason_codes": json.dumps([f"process.{process_state}"]),
            "resumed": False,
            "boundary_type": case.boundary_type,
            "axis_set": case.axis_set,
            "topology": config.clean_settings.topology,
            "boundary_conditions": json.dumps(list(config.clean_settings.boundary_conditions)),
            "retained_seed_count": 0,
            "retained_seed_hashes": "[]",
            "elapsed_s": round(time.perf_counter() - started, 6),
            "peak_rss_mib": "",
            "process_state": process_state,
            "warning_count": 0,
            "warnings": "[]",
            "error_type": signal_name or "MissingChildResult",
            "message": (
                f"Child terminated by {signal_name}."
                if signal_name
                else "Child exited without a clean-generation result record."
            ),
        }
    result["returncode"] = completed.returncode
    result["signal"] = signal_name
    result["stdout_tail"] = _tail(
        _diagnostic_stdout(stdout), config.diagnostic_chars
    )
    result["stderr_tail"] = _tail(stderr, config.diagnostic_chars)
    return result


def _persist_process_failure(
    case: BoundaryCase, config: GenerationConfig, result: dict[str, Any]
) -> None:
    if result.get("case_directory"):
        return
    existing = _existing_result(case, config)
    if existing is not None:
        process_state = result.get("process_state", "")
        returncode = result.get("returncode", "")
        signal_name = result.get("signal", "")
        stdout_tail = result.get("stdout_tail", "")
        stderr_tail = result.get("stderr_tail", "")
        result.clear()
        result.update(existing)
        result.update(
            {
                "process_state": f"recovered_after_{process_state}",
                "returncode": returncode,
                "signal": signal_name,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "message": (
                    "Child result marker was unavailable, but the complete persisted "
                    "case passed schema and hash verification."
                ),
            }
        )
        return
    target = config.output_root / case.case_id
    staging = config.output_root / f".{case.case_id}.tmp-parent-{os.getpid()}-{time.time_ns()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    reason_codes = json.loads(result.get("reason_codes", "[]"))
    construction = _hashed_document(
        schema=_CONSTRUCTION_SCHEMA,
        generation_signature=config.generation_signature,
        deterministic={
            "source": _case_source(case, config),
            "construction_mode": "exact",
            "outcome": "child_process_failure",
            "failure_stage": result.get("failure_stage"),
            "reason_codes": reason_codes,
            "base": None,
        },
        diagnostics={
            "process_state": result.get("process_state"),
            "returncode": result.get("returncode"),
            "signal": result.get("signal"),
            "stdout_tail": result.get("stdout_tail"),
            "stderr_tail": result.get("stderr_tail"),
        },
    )
    initialization = _hashed_document(
        schema=_INITIALIZATION_SCHEMA,
        generation_signature=config.generation_signature,
        deterministic={
            "search_invoked": False,
            "outcome": "child_process_failure",
            "phase7_result": None,
            "retained_seeds": [],
            "reason_codes": reason_codes,
        },
    )
    finalized = _finalize_case(
        staging=staging,
        target=target,
        config=config,
        case=case,
        status="seed_generation_failed",
        failure_stage=str(result.get("failure_stage", "child_process")),
        reason_codes=reason_codes,
        construction_document=construction,
        initialization_document=initialization,
        retained_seeds=(),
        base=None,
        phase7_status=None,
        phase7_result_hash=None,
        error_type=str(result.get("error_type", "")),
        message=str(result.get("message", "")),
    )
    for key in (
        "case_directory",
        "case_metadata_file",
        "construction_report_file",
        "initialization_report_file",
    ):
        result[key] = finalized[key]


def _write_results(path: Path, results: Sequence[dict[str, Any]]) -> None:
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=_RESULT_FIELDS, delimiter="\t", extrasaction="ignore"
    )
    writer.writeheader()
    for result in sorted(results, key=lambda item: int(item["source_row"])):
        writer.writerow({field: result.get(field, "") for field in _RESULT_FIELDS})
    _atomic_text(path, stream.getvalue())


def _campaign_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {"stdout_tail", "stderr_tail", "warnings"}
    return {
        field: result.get(field, "")
        for field in _RESULT_FIELDS
        if field not in excluded
    }


def _write_campaign_outputs(
    output_root: Path, *, config: GenerationConfig, results: Sequence[dict[str, Any]]
) -> None:
    ordered = sorted(results, key=lambda item: int(item["source_row"]))
    entries = [_campaign_entry(item) for item in ordered]
    common = {
        "schema": _CAMPAIGN_REPORT_SCHEMA,
        "generator_schema": _GENERATOR_SCHEMA,
        "generation_signature": config.generation_signature,
        "source_csv": str(config.data_file),
        "source_csv_sha256": config.source_sha256,
        "effective_configuration": config.clean_settings.to_dict(),
        "effective_configuration_hash": config.clean_settings.configuration_hash,
        "software_identity": _jsonable(config.software_identity),
    }
    construction_entries = [
        {
            "case_id": item.get("case_id"),
            "source_row": item.get("source_row"),
            "status": item.get("status"),
            "failure_stage": item.get("failure_stage"),
            "base_structure_hash": item.get("base_structure_hash"),
            "base_state_hash": item.get("base_state_hash"),
            "base_feasibility_status": item.get("base_feasibility_status"),
            "report_file": item.get("construction_report_file"),
        }
        for item in ordered
    ]
    initialization_entries = [
        {
            "case_id": item.get("case_id"),
            "source_row": item.get("source_row"),
            "status": item.get("status"),
            "failure_stage": item.get("failure_stage"),
            "retained_seed_count": item.get("retained_seed_count"),
            "retained_seed_hashes": item.get("retained_seed_hashes"),
            "phase7_status": item.get("phase7_status"),
            "phase7_result_hash": item.get("phase7_result_hash"),
            "report_file": item.get("initialization_report_file"),
        }
        for item in ordered
    ]
    _atomic_json(
        output_root / "construction_report.json",
        {**common, "report_kind": "construction", "cases": construction_entries},
    )
    _atomic_json(
        output_root / "initialization_report.json",
        {**common, "report_kind": "initialization", "cases": initialization_entries},
    )
    _atomic_json(
        output_root / "manifest.json",
        {
            **common,
            "manifest_schema": _MANIFEST_SCHEMA,
            "configuration": _jsonable(asdict(config)),
            "status_counts": dict(Counter(str(item["status"]) for item in ordered)),
            "failure_stage_counts": dict(
                Counter(str(item.get("failure_stage", "")) for item in ordered)
            ),
            "cases": entries,
        },
    )


def _display(index: int, total: int, result: Mapping[str, Any]) -> None:
    status = str(result.get("status", "seed_generation_failed"))
    case_id = str(result.get("case_id", "<unknown>"))
    resumed = " resumed" if result.get("resumed") else ""
    elapsed = result.get("elapsed_s", "")
    seed_count = result.get("retained_seed_count", "")
    suffix = f" {elapsed}s" if elapsed != "" else ""
    if seed_count != "":
        suffix += f" seeds={seed_count}"
    print(f"[{index:03d}/{total:03d}] {status:>29} {case_id}{suffix}{resumed}", flush=True)
    if status != "feasible_seed_ready":
        message = str(result.get("message", "")).strip().splitlines()
        if message:
            print(f"    {message[-1]}", flush=True)


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

    print(f"Data file          : {data_file}")
    print(f"Source SHA-256     : {config.source_sha256}")
    print(f"Boundary rows      : {len(cases)}")
    print(f"Selected           : {len(selected)}")
    print(f"Output root        : {output_root}")
    print(f"Generation signature: {config.generation_signature}")
    print(
        "Clean generation    : exact supplied P/Q -> strict validation -> "
        "Phase 7 termination/translation initialization"
    )
    print(
        f"Topology            : {config.clean_settings.topology}; "
        f"BC={config.clean_settings.boundary_conditions}; "
        f"max_seeds={config.clean_settings.max_seeds}; "
        f"retain_warnings={config.clean_settings.retain_warnings}"
    )

    if args.list:
        for case in selected:
            print(
                f"{case.case_id}\trow={case.source_row}\ttype={case.boundary_type}"
                f"\taxis={case.axis_set}\tUO2={case.uo2_reference_j_m2:g}"
            )
        return 0

    results: list[dict[str, Any]] = []
    pending = []
    for case in selected:
        existing = None if args.force else _existing_result(case, config)
        if existing is None:
            pending.append(case)
        else:
            results.append(existing)
    for index, result in enumerate(results, start=1):
        _display(index, len(selected), result)

    if pending:
        future_to_case: dict[Future[dict[str, Any]], BoundaryCase] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for case in pending:
                future_to_case[
                    executor.submit(
                        _execute_case, case, script_path=script_path, config=config
                    )
                ] = case
            try:
                for future in as_completed(future_to_case):
                    case = future_to_case[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {
                            "case_id": case.case_id,
                            "source_row": case.source_row,
                            "status": "seed_generation_failed",
                            "failure_stage": "parent_execution",
                            "reason_codes": '["process.parent_exception"]',
                            "resumed": False,
                            "boundary_type": case.boundary_type,
                            "axis_set": case.axis_set,
                            "topology": config.clean_settings.topology,
                            "boundary_conditions": json.dumps(
                                list(config.clean_settings.boundary_conditions)
                            ),
                            "retained_seed_count": 0,
                            "retained_seed_hashes": "[]",
                            "elapsed_s": "",
                            "peak_rss_mib": "",
                            "process_state": "parent_exception",
                            "returncode": "",
                            "signal": "",
                            "warning_count": 0,
                            "warnings": "[]",
                            "error_type": type(exc).__name__,
                            "message": "".join(
                                traceback.format_exception(type(exc), exc, exc.__traceback__)
                            ),
                            "stdout_tail": "",
                            "stderr_tail": "",
                        }
                    _persist_process_failure(case, config, result)
                    results.append(result)
                    _display(len(results), len(selected), result)
                    _write_results(output_root / "clean_generation_results.tsv", results)
                    _write_campaign_outputs(output_root, config=config, results=results)
            except KeyboardInterrupt:
                for future in future_to_case:
                    future.cancel()
                print("\nInterrupted; completed results were written.", file=sys.stderr)
                return 130

    _write_results(output_root / "clean_generation_results.tsv", results)
    _write_campaign_outputs(output_root, config=config, results=results)
    counts = Counter(str(result["status"]) for result in results)
    print("\nSummary\n-------")
    for status, count in sorted(counts.items()):
        print(f"{status:>29}: {count}")
    print(f"{'total':>29}: {len(results)}")
    print(f"Results       : {output_root / 'clean_generation_results.tsv'}")
    print(f"Manifest      : {output_root / 'manifest.json'}")
    print(f"Construction  : {output_root / 'construction_report.json'}")
    print(f"Initialization: {output_root / 'initialization_report.json'}")
    return 0 if all(item["status"] == "feasible_seed_ready" for item in results) else 1


def _run_child(args: argparse.Namespace) -> int:
    data_file = _resolve(args.data_file)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _make_config(args, data_file=data_file, output_root=output_root)
    cases = _load_cases(data_file, expected_cases=args.expected_cases)
    case = {item.case_id: item for item in cases}.get(args.run_one)
    if case is None:
        raise KeyError(f"Unknown case ID: {args.run_one}")
    result = _child_run(case, config)
    print(_RESULT_MARKER + json.dumps(_jsonable(result), sort_keys=True), flush=True)
    return 0 if result.get("status") == "feasible_seed_ready" else 1


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
