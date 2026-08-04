# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Geometry regression for the original voided Zhang first-case structure."""

from pathlib import Path

import numpy as np

from GBOpt.geometry_audit import audit_bicrystal_geometry


_FIXTURE = Path(__file__).parent / "data" / "zhang_001_ST_100.data"


def _read_lammps_atomic_charge_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    box: list[tuple[float, float]] = []
    for line in lines:
        stripped = line.strip()
        if stripped.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            fields = stripped.split()
            box.append((float(fields[0]), float(fields[1])))
        if len(box) == 3:
            break

    atoms_header = lines.index("Atoms")
    positions = []
    for line in lines[atoms_header + 1 :]:
        if not line.strip():
            continue
        fields = line.split()
        positions.append((float(fields[3]), float(fields[4]), float(fields[5])))
    return np.asarray(positions, dtype=float), np.asarray(box, dtype=float)


def test_zhang_001_original_structure_is_geometrically_suspicious():
    positions, box = _read_lammps_atomic_charge_data(_FIXTURE)
    central_plane_x = 0.5 * (box[0, 0] + box[0, 1])
    left = positions[positions[:, 0] < central_plane_x]
    right = positions[positions[:, 0] >= central_plane_x]

    result = audit_bicrystal_geometry(
        left,
        right,
        box,
        central_plane_x=central_plane_x,
    )

    assert result.status == "suspicious"
    assert "central_interface_large_gap_range" in result.reasons
    assert "periodic_interface_large_gap_range" in result.reasons
    assert result.central_interface.range_angstrom is not None
    assert result.central_interface.range_angstrom > 10.0
    assert result.periodic_interface.range_angstrom is not None
    assert result.periodic_interface.range_angstrom > 10.0
    assert result.nearest_neighbors.periodic_duplicate_count == 0
