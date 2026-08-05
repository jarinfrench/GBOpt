"""Generate the deterministic F0 behavior-characterization manifest.

This module is test support, not production API.  It intentionally exercises public
and compatibility entry points before later PRs move their implementations.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from GBOpt.BoundarySpec import CSLApproxSpec, CSLExactSpec, PQSpec
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.FileGrainOwnership import GrainOwnership
from GBOpt.GBMaker import GBMaker
from GBOpt.GBManipulator import GBManipulator, InterfaceCandidate
from GBOpt.GBMinimizer import GeneticAlgorithmMinimizer, MonteCarloMinimizer, PENALTY

SCHEMA_VERSION = 1
SOURCE_ARCHIVE = "gbopt_source.tar(1).gz"
SOURCE_ARCHIVE_SHA256 = "d0c24898b334b26f5445304d209cb3bf3fbc1ac3882eb7a5b029045ac565d6b1"
FLOAT_DIGITS = 12

_SIGMA5_P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
_SIGMA5_Q = [[4, -3, 0], [3, 4, 0], [0, 0, 1]]
_SIGMA5_PQ = PQSpec(P=_SIGMA5_P, Q=_SIGMA5_Q, basis_mode="supplied")
_SIGMA5_CSL = CSLExactSpec(
    axis=[0, 0, 1],
    plane=[1, 0, 0],
    quat=[3, 0, 0, 1],
)
_SIGMA5_APPROX = CSLApproxSpec(
    axis=[0, 0, 1],
    plane=[1, 0, 0],
    angle_deg=36.87,
)
_INCOMMENSURATE_PQ = PQSpec(
    P=[[0, 2, 5], [0, 5, -2], [-1, 0, 0]],
    Q=[[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    basis_mode="supplied",
)


def _normalize_float(value: float) -> float:
    result = round(float(value), FLOAT_DIGITS)
    return 0.0 if result == 0.0 else result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return _normalize_float(value)
    if isinstance(value, Path):
        return value.name
    if isinstance(value, BoundaryNormalTopology):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _structured_rows(atoms: np.ndarray) -> list[list[Any]]:
    names = atoms.dtype.names
    if names is None:
        raise TypeError("Expected a structured atom array")
    rows: list[list[Any]] = []
    for row in atoms:
        rows.append([_jsonable(row[name]) for name in names])
    return rows


def _structured_array_summary(atoms: np.ndarray) -> dict[str, Any]:
    rows = _structured_rows(atoms)
    sorted_rows = sorted(rows, key=lambda row: tuple(str(item) for item in row))
    species, counts = np.unique(atoms["name"], return_counts=True)
    coordinates = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
    return {
        "count": int(len(atoms)),
        "dtype_fields": list(atoms.dtype.names or ()),
        "species_counts": {
            str(name): int(count) for name, count in zip(species, counts)
        },
        "coordinate_min": _jsonable(np.min(coordinates, axis=0)),
        "coordinate_max": _jsonable(np.max(coordinates, axis=0)),
        "order_sensitive_sha256": _sha256(rows),
        "order_insensitive_sha256": _sha256(sorted_rows),
    }


def _warnings_summary(caught: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {
            "category": warning.category.__name__,
            "message": str(warning.message),
        }
        for warning in caught
    ]


def _build_with_warnings(factory: Callable[[], GBMaker]) -> tuple[GBMaker, list[dict[str, str]]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gb = factory()
    return gb, _warnings_summary(caught)


def _gbmaker_summary(gb: GBMaker, caught: list[dict[str, str]]) -> dict[str, Any]:
    manipulator = GBManipulator(gb, seed=19)
    parent = manipulator.parents[0]
    return {
        "warnings": caught,
        "uses_exact_construction": bool(gb.uses_exact_construction),
        "inplane_periodic": list(gb.inplane_periodic),
        "normal_topology": parent.normal_topology.value,
        "dimensions": {
            "x_dim": _normalize_float(gb.x_dim),
            "y_dim": _normalize_float(gb.y_dim),
            "z_dim": _normalize_float(gb.z_dim),
            "box_dims": _jsonable(gb.box_dims),
            "gb_plane_x": _normalize_float(gb.gb_plane_x),
        },
        "spacing": _jsonable(gb.spacing),
        "left_grain": _structured_array_summary(gb.left_grain),
        "right_grain": _structured_array_summary(gb.right_grain),
        "whole_system": _structured_array_summary(gb.whole_system),
    }


def _legacy_gb(*, vacuum: float) -> GBMaker:
    theta = math.radians(36.869898)
    return GBMaker(
        3.52,
        "fcc",
        5.0,
        np.array([theta, 0.0, 0.0, 0.0, -theta / 2.0]),
        "Ni",
        repeat_factor=(1, 1),
        x_dim_min=12.0,
        vacuum=vacuum,
        interaction_distance=4.0,
    )


def _build_gbmaker_cases() -> tuple[dict[str, Any], dict[str, GBMaker]]:
    factories: dict[str, Callable[[], GBMaker]] = {
        "legacy_fcc_periodic": lambda: _legacy_gb(vacuum=0.0),
        "legacy_fcc_slab": lambda: _legacy_gb(vacuum=2.0),
        "exact_pq_fcc_periodic": lambda: GBMaker.from_boundary_spec(
            3.615,
            "fcc",
            "Cu",
            _SIGMA5_PQ,
            mode="exact",
            gb_thickness=5.0,
            repeat_factor=2,
            x_dim_min=12.0,
            vacuum=0.0,
            interaction_distance=3.615,
        ),
        "exact_csl_fcc_periodic": lambda: GBMaker.from_boundary_spec(
            3.615,
            "fcc",
            "Cu",
            _SIGMA5_CSL,
            mode="exact",
            gb_thickness=5.0,
            repeat_factor=2,
            x_dim_min=12.0,
            vacuum=0.0,
            interaction_distance=3.615,
        ),
        "exact_pq_fluorite_slab": lambda: GBMaker.from_boundary_spec(
            5.47,
            "fluorite",
            ("U", "O"),
            _SIGMA5_PQ,
            mode="exact",
            gb_thickness=5.0,
            repeat_factor=1,
            x_dim_min=12.0,
            vacuum=2.0,
            interaction_distance=5.47,
        ),
        "approximate_fcc_non_csl": lambda: GBMaker.from_boundary_spec(
            3.615,
            "fcc",
            "Cu",
            _SIGMA5_APPROX,
            mode="approximate",
            gb_thickness=5.0,
            repeat_factor=1,
            x_dim_min=12.0,
            vacuum=0.0,
            interaction_distance=3.615,
        ),
        "exact_pq_mismatch_accommodation": lambda: GBMaker.from_boundary_spec(
            3.615,
            "sc",
            "Cu",
            _INCOMMENSURATE_PQ,
            mode="exact",
            gb_thickness=0.0,
            repeat_factor=(2, 3),
            x_dim_min=10.0,
            vacuum=0.0,
            interaction_distance=1.0,
            mismatch_tol=0.005,
            strain_grain="both",
        ),
    }
    summaries: dict[str, Any] = {}
    objects: dict[str, GBMaker] = {}
    for name, factory in factories.items():
        gb, caught = _build_with_warnings(factory)
        summaries[name] = _gbmaker_summary(gb, caught)
        objects[name] = gb
    return summaries, objects


def _file_summary(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    text = data.decode("utf-8")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "line_count": len(text.splitlines()),
        "has_tilt_line": "xy xz yz" in text,
        "header": text.splitlines()[:12],
    }


def _writer_cases(gb: GBMaker, directory: Path) -> dict[str, Any]:
    orthogonal = directory / "orthogonal.data"
    triclinic = directory / "triclinic.data"
    gb.write_lammps(
        str(orthogonal),
        type_as_int=True,
        precision=12,
        triclinic=False,
    )
    gb.write_lammps(
        str(triclinic),
        type_as_int=True,
        precision=12,
        triclinic=True,
    )
    return {
        "orthogonal": _file_summary(orthogonal),
        "triclinic": _file_summary(triclinic),
    }


def _candidate_summary(candidate: InterfaceCandidate) -> dict[str, Any]:
    return {
        "atoms": _structured_array_summary(candidate.atoms),
        "grain_labels_sha256": _sha256(candidate.grain_labels.tolist()),
        "left_count": int(np.count_nonzero(candidate.grain_labels == 0)),
        "right_count": int(np.count_nonzero(candidate.grain_labels == 1)),
        "box_dims": _jsonable(candidate.box_dims),
        "gb_plane_x": _normalize_float(candidate.gb_plane_x),
        "left_grain_x_bounds": _jsonable(candidate.left_grain_x_bounds),
        "right_grain_x_bounds": _jsonable(candidate.right_grain_x_bounds),
        "inplane_periodic": list(candidate.inplane_periodic),
        "normal_topology": candidate.normal_topology.value,
        "coordinate_tolerance": _normalize_float(candidate.coordinate_tolerance),
        "interface_separation": _normalize_float(candidate.interface_separation),
    }


def _manipulation_cases(periodic: GBMaker, slab: GBMaker) -> dict[str, Any]:
    periodic_manipulator = GBManipulator(periodic, seed=23)
    slab_manipulator = GBManipulator(slab, seed=23)

    periodic_parent = periodic_manipulator.make_parent_candidate()
    translated = periodic_manipulator.make_translation_candidate(0.125, 0.25)
    terminated = periodic_manipulator.make_termination_candidate(
        left_phase_shift=0.25,
        right_phase_shift=0.5,
        right_dy=0.125,
    )
    periodic_separated = periodic_manipulator.apply_interface_separation(
        terminated,
        interface_separation=0.75,
    )

    slab_parent = slab_manipulator.make_parent_candidate()
    slab_terminated = slab_manipulator.make_slab_termination_candidate(
        left_phase_shift=0.25,
        right_phase_shift=0.5,
        right_dy=0.125,
    )
    slab_separated = slab_manipulator.apply_interface_separation(
        slab_terminated,
        interface_separation=0.75,
    )

    return {
        "periodic_parent": _candidate_summary(periodic_parent),
        "periodic_translation": _candidate_summary(translated),
        "periodic_termination": _candidate_summary(terminated),
        "periodic_termination_then_separation": _candidate_summary(
            periodic_separated
        ),
        "slab_parent": _candidate_summary(slab_parent),
        "slab_termination": _candidate_summary(slab_terminated),
        "slab_termination_then_separation": _candidate_summary(slab_separated),
    }


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _write_candidate(
    gb: GBMaker,
    manipulator: GBManipulator,
    atoms: np.ndarray,
    path: Path,
) -> None:
    gb.write_lammps(
        str(path),
        atoms,
        manipulator.parents[0].box_dims,
        type_as_int=False,
        precision=12,
    )


def _normalized_lineage(value: Any) -> Any:
    if isinstance(value, str):
        if "/" in value or "\\" in value or value.endswith((".data", ".dump")):
            return Path(value).name
        return value
    if isinstance(value, list):
        return [_normalized_lineage(item) for item in value]
    if isinstance(value, tuple):
        return [_normalized_lineage(item) for item in value]
    return _jsonable(value)


def _mc_fixture(gb: GBMaker, directory: Path) -> dict[str, Any]:
    call_index = 0

    def energy_function(GB, manipulator, atom_positions, unique_id, **_kwargs):
        nonlocal call_index
        output = directory / f"mc_{call_index:02d}_{unique_id}.data"
        _write_candidate(GB, manipulator, atom_positions, output)
        energy = 0.0 if call_index == 0 else 10.0
        call_index += 1
        return energy, str(output)

    minimizer = MonteCarloMinimizer(
        gb,
        energy_function,
        ["translate_right_grain"],
        seed=11,
    )
    with _working_directory(directory):
        best = minimizer.run_MC(
            E_accept=1.0e-6,
            max_steps=3,
            E_tol=0.0,
            max_rejections=10,
            cooldown_rate=0.9,
            unique_id=101,
        )
    return {
        "best_energy": _normalize_float(best),
        "GBE_vals": _jsonable(minimizer.GBE_vals),
        "accepted_idx": list(minimizer.accepted_idx),
        "operation_list": _normalized_lineage(minimizer.operation_list),
    }


def _legacy_ga_fixture(gb: GBMaker, directory: Path) -> dict[str, Any]:
    def scalar_energy(GB, manipulator, atom_positions, unique_id):
        if str(unique_id).endswith("g0_c0"):
            raise RuntimeError("characterized scalar evaluator failure")
        output = directory / f"{unique_id}.data"
        _write_candidate(GB, manipulator, atom_positions, output)
        return float(np.mean(atom_positions["x"])), str(output)

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        scalar_energy,
        ["translate_right_grain"],
        seed=13,
        population_size=2,
        generations=1,
        keep_top_pct=50,
        intermediate_pct=100,
    )
    best_energy, best_path = minimizer.run_GA(unique_id=202)
    return {
        "best_energy": _normalize_float(best_energy),
        "best_path": Path(best_path).name,
        "GBE_vals": _jsonable(minimizer.GBE_vals),
        "history": _normalized_lineage(minimizer.history),
        "penalty": PENALTY,
    }


def _write_preserved_id_candidate(
    path: Path,
    atoms: np.ndarray,
    ids: np.ndarray,
    box_dims: np.ndarray,
    *,
    change_species: bool = False,
    crossing_index: int | None = None,
    crossing_x: float | None = None,
) -> None:
    output = np.array(atoms, copy=True)
    if change_species:
        output[0]["name"] = "H"
    if crossing_index is not None and crossing_x is not None:
        output[crossing_index]["x"] = crossing_x
    order = np.arange(len(output))[::-1]
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("F0 owned evaluator output\n\n")
        stream.write(f"{len(output)} atoms\n")
        stream.write(f"{len(set(output['name'].tolist()))} atom types\n")
        for axis, (lower, upper) in zip("xyz", box_dims):
            stream.write(
                f"{float(lower):.12f} {float(upper):.12f} {axis}lo {axis}hi\n"
            )
        stream.write("\nAtoms\n\n")
        for row in order:
            atom = output[row]
            stream.write(
                f"{int(ids[row])} {atom['name']} {float(atom['x']):.12f} "
                f"{float(atom['y']):.12f} {float(atom['z']):.12f}\n"
            )


def _owned_ga_fixture(gb: GBMaker, directory: Path) -> dict[str, Any]:
    seed_path = directory / "owned_initial.data"
    gb.write_lammps(str(seed_path), type_as_int=False, precision=12)
    labels = np.hstack(
        (
            np.zeros(len(gb.left_grain), dtype=np.int8),
            np.ones(len(gb.right_grain), dtype=np.int8),
        )
    )
    ownership = GrainOwnership(
        atom_ids=np.arange(1, len(gb.whole_system) + 1),
        labels=labels,
        gb_plane_x=gb.gb_plane_x,
        inplane_periodic=gb.inplane_periodic,
        left_grain_x_bounds=(gb.box_dims[0, 0], gb.gb_plane_x),
        right_grain_x_bounds=(gb.gb_plane_x, gb.box_dims[0, 1]),
        coordinate_tolerance=gb.epsilon,
        normal_topology=BoundaryNormalTopology.PERIODIC_BICRYSTAL,
    )

    def scalar_energy(GB, manipulator, atom_positions, unique_id):
        output = directory / f"owned_{unique_id}.data"
        candidate_labels = manipulator.candidate_grain_labels
        crossing_index = int(np.flatnonzero(candidate_labels == 0)[0])
        _write_preserved_id_candidate(
            output,
            atom_positions,
            np.arange(1, len(atom_positions) + 1),
            manipulator.parents[0].box_dims,
            crossing_index=crossing_index,
            crossing_x=GB.gb_plane_x + 0.25,
        )
        return 5.0, str(output)

    def batch_energy(GB, manipulators, structures, lineages, unique_ids):
        del lineages
        results = []
        for index, (manipulator, atoms, candidate_id) in enumerate(
            zip(manipulators, structures, unique_ids)
        ):
            output = directory / f"batch_{candidate_id}.data"
            _write_preserved_id_candidate(
                output,
                atoms,
                np.arange(1, len(atoms) + 1),
                manipulator.parents[0].box_dims,
                change_species=index == 0,
            )
            results.append({"energy": float(index), "final_dump": str(output)})
        return results

    minimizer = GeneticAlgorithmMinimizer(
        gb,
        scalar_energy,
        ["translate_right_grain"],
        seed=17,
        initial_structure=str(seed_path),
        initial_ownership=ownership,
        population_size=2,
        generations=1,
        keep_top_pct=50,
        intermediate_pct=100,
        gb_batch_energy_func=batch_energy,
    )
    best_energy, best_path = minimizer.run_GA(unique_id=303)
    evaluations = []
    for record in minimizer.last_generation_evaluations:
        evaluations.append(
            {
                "input_index": record.input_index,
                "energy": record.energy,
                "structure_path": (
                    Path(record.structure_path).name
                    if record.structure_path is not None
                    else None
                ),
                "success": record.success,
                "failure_reason": record.failure_reason,
                "has_mapping": record.mapping is not None,
                "has_manipulator": record.manipulator is not None,
                "labels_sha256": (
                    _sha256(record.manipulator.parents[0].grain_labels.tolist())
                    if record.manipulator is not None
                    else None
                ),
            }
        )
    successful = next(record for record in minimizer.last_generation_evaluations if record.success)
    successful_parent = successful.manipulator.parents[0]
    crossing_left_rows = int(
        np.count_nonzero(
            (successful_parent.grain_labels == 0)
            & (successful_parent.whole_system["x"] > gb.gb_plane_x)
        )
    )
    return {
        "best_energy": best_energy,
        "best_path": Path(best_path).name,
        "evaluations": _jsonable(evaluations),
        "left_count": int(np.count_nonzero(successful_parent.grain_labels == 0)),
        "right_count": int(np.count_nonzero(successful_parent.grain_labels == 1)),
        "left_owned_rows_right_of_plane": crossing_left_rows,
        "gb_plane_x": _normalize_float(successful_parent.gb_plane_x),
    }


def _optimizer_cases(gb: GBMaker, periodic_gb: GBMaker, directory: Path) -> dict[str, Any]:
    return {
        "fixed_seed_mc": _mc_fixture(gb, directory),
        "fixed_seed_legacy_scalar_ga": _legacy_ga_fixture(gb, directory),
        "owned_batch_ga": _owned_ga_fixture(periodic_gb, directory),
    }


def behavior_manifest() -> dict[str, Any]:
    """Return the deterministic behavior portion of the F0 manifest."""
    gbmaker, objects = _build_gbmaker_cases()
    with tempfile.TemporaryDirectory(prefix="gbopt-f0-") as tmp:
        directory = Path(tmp)
        return {
            "gbmaker": gbmaker,
            "writers": _writer_cases(objects["legacy_fcc_periodic"], directory),
            "manipulation": _manipulation_cases(
                objects["legacy_fcc_periodic"],
                objects["legacy_fcc_slab"],
            ),
            "optimization": _optimizer_cases(
                objects["legacy_fcc_periodic"],
                objects["exact_pq_fcc_periodic"],
                directory,
            ),
        }


def environment_record() -> dict[str, Any]:
    """Return informational environment metadata for a generated manifest."""
    versions: dict[str, str] = {}
    for module_name in ("numpy", "scipy", "numba", "pandas", "matplotlib", "spglib"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - informational only
            versions[module_name] = f"unavailable: {type(exc).__name__}: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


def build_manifest() -> dict[str, Any]:
    """Build the complete committed characterization manifest."""
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": {
            "source_archive": SOURCE_ARCHIVE,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "environment": environment_record(),
        },
        "behavior": behavior_manifest(),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("baseline_manifest.json"),
    )
    parser.add_argument(
        "--verify-repeat",
        action="store_true",
        help="Generate behavior twice and fail if the results differ.",
    )
    args = parser.parse_args()

    manifest = build_manifest()
    if args.verify_repeat:
        second = behavior_manifest()
        if manifest["behavior"] != second:
            raise SystemExit("Consecutive F0 behavior manifests differ")
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
