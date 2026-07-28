# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Regression tests for representative Zhang grain-boundary geometries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from GBOpt.geometry_audit import GeometryAuditResult, audit_bicrystal_geometry

_DATA_DIR = Path(__file__).parent / "data"


def _read_lammps_charge_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read positions and orthorhombic bounds from a LAMMPS charge data file."""
    lines = path.read_text(encoding="utf-8").splitlines()

    atom_count: int | None = None
    box = np.full((3, 2), np.nan, dtype=float)
    axis_by_bounds = {
        ("xlo", "xhi"): 0,
        ("ylo", "yhi"): 1,
        ("zlo", "zhi"): 2,
    }

    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[1] == "atoms":
            atom_count = int(fields[0])
        elif len(fields) >= 4:
            axis = axis_by_bounds.get(tuple(fields[-2:]))
            if axis is not None:
                box[axis] = (float(fields[0]), float(fields[1]))

    if atom_count is None:
        raise ValueError(f"{path} does not declare an atom count.")
    if not np.all(np.isfinite(box)):
        raise ValueError(f"{path} does not contain complete orthorhombic bounds.")

    try:
        atoms_header = next(
            index for index, line in enumerate(lines) if line.strip() == "Atoms"
        )
    except StopIteration as exc:
        raise ValueError(f"{path} does not contain an Atoms section.") from exc

    positions: list[tuple[float, float, float]] = []
    for line in lines[atoms_header + 1:]:
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"Malformed atom record in {path}: {line!r}")

        # GBMaker writes either ``id type x y z`` or ``id type q x y z``.
        positions.append(tuple(float(value) for value in fields[-3:]))
        if len(positions) == atom_count:
            break

    if len(positions) != atom_count:
        raise ValueError(
            f"{path} declares {atom_count} atoms but contains "
            f"{len(positions)} records."
        )

    return np.asarray(positions, dtype=float), box


def _load_and_audit_case(
    case_id: str,
) -> tuple[GeometryAuditResult, dict[str, Any]]:
    """Load a generated case, recover grain membership, and rerun the audit."""
    data_path = _DATA_DIR / f"{case_id}.data"
    metadata_path = _DATA_DIR / f"{case_id}.metadata.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    positions, box = _read_lammps_charge_data(data_path)

    assert metadata["case_id"] == case_id
    assert int(metadata["atoms"]["total"]) == len(positions)

    left_count = int(metadata["atoms"]["left_total"])
    right_count = int(metadata["atoms"]["right_total"])
    assert left_count + right_count == len(positions)

    # GBMaker constructs whole_system as left_grain followed by right_grain,
    # and write_lammps preserves that order. The metadata counts therefore
    # recover grain membership even when the two grains overlap in x.
    left = positions[:left_count]
    right = positions[left_count:]

    recorded_box = np.asarray(
        metadata["geometry"]["box_dims_angstrom"],
        dtype=float,
    )
    np.testing.assert_allclose(
        box,
        recorded_box,
        rtol=0.0,
        atol=1.0e-9,
    )

    recorded_audit = metadata["geometry_audit"]
    result = audit_bicrystal_geometry(
        left,
        right,
        box,
        central_plane_x=float(
            metadata["geometry"]["gb_plane_x_angstrom"]
        ),
        bins=(
            int(recorded_audit["bins_y"]),
            int(recorded_audit["bins_z"]),
        ),
    )
    return result, metadata


def test_zhang_001_symmetric_tilt_remains_geometrically_suspicious() -> None:
    """The original voided symmetric-tilt structure must remain detectable."""
    result, metadata = _load_and_audit_case("zhang_001_ST_100")

    expected_reasons = {
        "central_interface_large_gap_range",
        "central_interface_heavy_gap_tail",
        "periodic_interface_large_gap_range",
        "periodic_interface_heavy_gap_tail",
        "central_interface_severe_overlap",
        "periodic_interface_severe_overlap",
    }

    assert metadata["geometry_audit"]["status"] == "suspicious"
    assert result.status == "suspicious"
    assert set(result.reasons) == expected_reasons

    bulk = result.bulk_reference_distance_angstrom
    central_range = result.central_interface.range_angstrom
    periodic_range = result.periodic_interface.range_angstrom

    assert bulk is not None
    assert central_range is not None
    assert periodic_range is not None

    # The observed ranges are about 4.3-4.6 bulk nearest-neighbor distances,
    # well above the classifier threshold of 2.0.
    assert central_range > 4.0 * bulk
    assert periodic_range > 4.0 * bulk
    assert result.nearest_neighbors.periodic_duplicate_count == 0


def test_zhang_041_twist_remains_a_reasonable_geometry_control() -> None:
    """A real Zhang twist case must remain unflagged by the geometry audit."""
    result, metadata = _load_and_audit_case("zhang_041_TW_100")

    assert metadata["geometry_audit"]["status"] == "ok"
    assert metadata["geometry_audit"]["reasons"] == []

    assert result.status == "ok"
    assert result.reasons == ()

    assert result.central_interface.range_angstrom is not None
    assert result.periodic_interface.range_angstrom is not None
    assert result.nearest_neighbors.left_internal_min_angstrom is not None
    assert result.nearest_neighbors.right_internal_min_angstrom is not None
    assert result.nearest_neighbors.central_cross_min_angstrom is not None
    assert result.nearest_neighbors.periodic_cross_min_angstrom is not None
    assert result.nearest_neighbors.periodic_duplicate_count == 0
