"""Narrow file-backed grain ownership support for optimizer seed handoff.

LAMMPS atom IDs are used only to align one verified serialized seed with its
persistent left/right grain labels.  They are not optimizer-wide atom identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from GBOpt.Atom import Atom

LEFT_GRAIN_LABEL = 0
RIGHT_GRAIN_LABEL = 1
_SUPPORTED_LABELS = frozenset((LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL))
_INTEGER_TOKEN = re.compile(r"^[+-]?\d+$")


class GrainOwnershipError(ValueError):
    """Raised when explicit file-backed ownership metadata is malformed."""


class LammpsDataError(ValueError):
    """Raised when a LAMMPS data file cannot be read unambiguously."""


def _readonly_copy(values: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _strict_int(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise GrainOwnershipError(f"{name} must be an integer")
    return int(value)


def _strict_id_token(token: str) -> int:
    if not _INTEGER_TOKEN.fullmatch(token):
        raise LammpsDataError(f"atom ID must be an integral token, got {token!r}")
    value = int(token)
    if value <= 0:
        raise LammpsDataError("atom IDs must be positive")
    return value


def _normalize_type_mapping(type_dict: Mapping[object, object] | None) -> dict[int, str]:
    if type_dict is None:
        return {}
    if all(
        isinstance(key, Integral) and not isinstance(key, bool) and isinstance(value, str)
        for key, value in type_dict.items()
    ):
        return {int(key): value for key, value in type_dict.items()}
    if all(
        isinstance(key, str) and isinstance(value, Integral) and not isinstance(value, bool)
        for key, value in type_dict.items()
    ):
        return {int(value): key for key, value in type_dict.items()}
    raise LammpsDataError("type_dict must be a dict[str, int] or dict[int, str]")


@dataclass(frozen=True, slots=True, init=False)
class GrainOwnership:
    """Immutable left/right labels and the topology needed by a file-backed Parent.

    ``atom_ids`` are initial serialization identifiers used only for row alignment.
    The persistent state is ``labels`` aligned with the in-memory atom rows.
    """

    _atom_ids: np.ndarray
    _labels: np.ndarray
    gb_plane_x: float
    inplane_periodic: tuple[bool, bool]
    _right_grain_x_bounds: np.ndarray
    coordinate_tolerance: float
    periodic_outer_x_interface: bool

    def __init__(
        self,
        *,
        atom_ids: np.ndarray,
        labels: np.ndarray,
        gb_plane_x: float,
        inplane_periodic: tuple[bool, bool],
        right_grain_x_bounds: np.ndarray | tuple[float, float],
        coordinate_tolerance: float,
        periodic_outer_x_interface: bool,
    ) -> None:
        raw_ids = np.asarray(atom_ids)
        raw_labels = np.asarray(labels)
        if raw_ids.ndim != 1 or raw_labels.ndim != 1:
            raise GrainOwnershipError("atom_ids and labels must be one-dimensional")
        if raw_ids.size != raw_labels.size:
            raise GrainOwnershipError("ownership-array length must equal atom ID count")

        normalized_ids = np.empty(raw_ids.size, dtype=np.int64)
        normalized_labels = np.empty(raw_labels.size, dtype=np.int8)
        for index, value in enumerate(raw_ids.tolist()):
            normalized_ids[index] = _strict_int("atom ID", value)
        if np.any(normalized_ids <= 0):
            raise GrainOwnershipError("atom IDs must be positive")
        if np.unique(normalized_ids).size != normalized_ids.size:
            raise GrainOwnershipError("atom IDs must be unique")
        for index, value in enumerate(raw_labels.tolist()):
            parsed = _strict_int("grain label", value)
            if parsed not in _SUPPORTED_LABELS:
                raise GrainOwnershipError("grain labels must be exactly 0 (left) or 1 (right)")
            normalized_labels[index] = parsed

        plane = float(gb_plane_x)
        tolerance = float(coordinate_tolerance)
        bounds = np.asarray(right_grain_x_bounds, dtype=float)
        if not np.isfinite(plane):
            raise GrainOwnershipError("gb_plane_x must be finite")
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise GrainOwnershipError("right_grain_x_bounds must contain two finite values")
        if bounds[0] >= bounds[1]:
            raise GrainOwnershipError("right_grain_x_bounds must be strictly ordered")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise GrainOwnershipError("coordinate_tolerance must be finite and positive")
        if not np.isclose(bounds[0], plane, atol=tolerance, rtol=0.0):
            raise GrainOwnershipError("right-grain lower bound must equal gb_plane_x")
        if len(inplane_periodic) != 2:
            raise GrainOwnershipError("inplane_periodic must contain y and z flags")

        object.__setattr__(self, "_atom_ids", _readonly_copy(normalized_ids))
        object.__setattr__(self, "_labels", _readonly_copy(normalized_labels))
        object.__setattr__(self, "gb_plane_x", plane)
        object.__setattr__(
            self, "inplane_periodic", tuple(bool(value) for value in inplane_periodic)
        )
        object.__setattr__(self, "_right_grain_x_bounds", _readonly_copy(bounds))
        object.__setattr__(self, "coordinate_tolerance", tolerance)
        object.__setattr__(
            self, "periodic_outer_x_interface", bool(periodic_outer_x_interface)
        )

    @property
    def atom_ids(self) -> np.ndarray:
        """Return a read-only defensive copy of initial serialization IDs."""
        return _readonly_copy(self._atom_ids)

    @property
    def labels(self) -> np.ndarray:
        """Return a read-only defensive copy of persistent grain labels."""
        return _readonly_copy(self._labels)

    @property
    def right_grain_x_bounds(self) -> np.ndarray:
        return _readonly_copy(self._right_grain_x_bounds)

    def aligned_to(self, atom_ids: np.ndarray) -> "GrainOwnership":
        """Return ownership reordered to the supplied file-row atom IDs."""
        requested = np.asarray(atom_ids)
        if requested.ndim != 1:
            raise GrainOwnershipError("loaded atom IDs must be one-dimensional")
        normalized = np.empty(requested.size, dtype=np.int64)
        for index, value in enumerate(requested.tolist()):
            normalized[index] = _strict_int("loaded atom ID", value)
        if np.any(normalized <= 0) or np.unique(normalized).size != normalized.size:
            raise GrainOwnershipError("loaded atom IDs must be positive and unique")
        if normalized.size != self._atom_ids.size:
            raise GrainOwnershipError("loaded atom ID count does not match ownership")

        expected_order = np.argsort(self._atom_ids, kind="stable")
        expected_ids = self._atom_ids[expected_order]
        loaded_order = np.argsort(normalized, kind="stable")
        if not np.array_equal(normalized[loaded_order], expected_ids):
            raise GrainOwnershipError("loaded atom IDs do not match ownership atom IDs")
        positions = np.searchsorted(expected_ids, normalized)
        ordered_labels = self._labels[expected_order][positions]
        return GrainOwnership(
            atom_ids=normalized,
            labels=ordered_labels,
            gb_plane_x=self.gb_plane_x,
            inplane_periodic=self.inplane_periodic,
            right_grain_x_bounds=self._right_grain_x_bounds,
            coordinate_tolerance=self.coordinate_tolerance,
            periodic_outer_x_interface=self.periodic_outer_x_interface,
        )

    def __copy__(self) -> "GrainOwnership":
        return GrainOwnership(
            atom_ids=self._atom_ids,
            labels=self._labels,
            gb_plane_x=self.gb_plane_x,
            inplane_periodic=self.inplane_periodic,
            right_grain_x_bounds=self._right_grain_x_bounds,
            coordinate_tolerance=self.coordinate_tolerance,
            periodic_outer_x_interface=self.periodic_outer_x_interface,
        )

    def __deepcopy__(self, memo: dict[int, object]) -> "GrainOwnership":
        copied = self.__copy__()
        memo[id(self)] = copied
        return copied


@dataclass(frozen=True, slots=True, init=False)
class LammpsAtomData:
    """Parsed atom IDs, species/coordinates, and orthogonal box bounds."""

    _atom_ids: np.ndarray
    _atoms: np.ndarray
    _box_dims: np.ndarray

    def __init__(self, atom_ids: np.ndarray, atoms: np.ndarray, box_dims: np.ndarray) -> None:
        object.__setattr__(self, "_atom_ids", _readonly_copy(atom_ids, dtype=np.int64))
        object.__setattr__(self, "_atoms", _readonly_copy(atoms, dtype=Atom.atom_dtype))
        object.__setattr__(self, "_box_dims", _readonly_copy(box_dims, dtype=float))

    @property
    def atom_ids(self) -> np.ndarray:
        return _readonly_copy(self._atom_ids)

    @property
    def atoms(self) -> np.ndarray:
        return _readonly_copy(self._atoms)

    @property
    def box_dims(self) -> np.ndarray:
        return _readonly_copy(self._box_dims)


def read_lammps_data_file(
    path: str | Path, *, type_dict: Mapping[object, object] | None = None
) -> LammpsAtomData:
    """Read the orthogonal LAMMPS data format emitted by ``GBMaker.write_lammps``.

    Both the five-column atomic form and the six-column charge form are supported.
    File row order is preserved; callers may align or canonicalize by atom ID.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise LammpsDataError("LAMMPS data file is too short")

    n_atoms: int | None = None
    n_types: int | None = None
    box: dict[str, tuple[float, float]] = {}
    id_to_name = _normalize_type_mapping(type_dict)
    atoms_line: int | None = None
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        parts = stripped.split()
        if len(parts) == 2 and parts[1] == "atoms":
            try:
                n_atoms = int(parts[0])
            except ValueError as exc:
                raise LammpsDataError("invalid atom count") from exc
        elif len(parts) == 3 and parts[1:] == ["atom", "types"]:
            try:
                n_types = int(parts[0])
            except ValueError as exc:
                raise LammpsDataError("invalid atom-type count") from exc
        elif len(parts) >= 4 and parts[-2:] in (["xlo", "xhi"], ["ylo", "yhi"], ["zlo", "zhi"]):
            axis = parts[-2][0]
            try:
                bounds = (float(parts[0]), float(parts[1]))
            except ValueError as exc:
                raise LammpsDataError(f"invalid {axis} box bounds") from exc
            box[axis] = bounds
        elif stripped == "Atom Type Labels":
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            while index < len(lines):
                label_parts = lines[index].strip().split()
                if not label_parts:
                    break
                if len(label_parts) != 2 or not _INTEGER_TOKEN.fullmatch(label_parts[0]):
                    break
                type_id = int(label_parts[0])
                if type_id <= 0:
                    raise LammpsDataError("atom type IDs must be positive")
                id_to_name[type_id] = label_parts[1]
                index += 1
            continue
        elif stripped.startswith("Atoms"):
            atoms_line = index
            break
        index += 1

    if n_atoms is None or n_atoms < 0:
        raise LammpsDataError("missing or invalid atom count")
    if n_types is None or n_types <= 0:
        raise LammpsDataError("missing or invalid atom-type count")
    if set(box) != {"x", "y", "z"}:
        raise LammpsDataError("missing orthogonal box bounds")
    box_dims = np.asarray([box[axis] for axis in "xyz"], dtype=float)
    if not np.all(np.isfinite(box_dims)) or np.any(box_dims[:, 0] >= box_dims[:, 1]):
        raise LammpsDataError("box bounds must be finite and strictly ordered")
    if atoms_line is None:
        raise LammpsDataError("missing Atoms section")

    inverse_default = {number: name for name, number in Atom._numbers.items()}
    rows: list[tuple[str, float, float, float]] = []
    ids: list[int] = []
    index = atoms_line + 1
    while index < len(lines) and len(rows) < n_atoms:
        stripped = lines[index].split("#", 1)[0].strip()
        index += 1
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) not in (5, 6):
            raise LammpsDataError(
                "Atoms rows must use 'id type x y z' or 'id type charge x y z'"
            )
        atom_id = _strict_id_token(parts[0])
        type_token = parts[1]
        coordinate_start = 2 if len(parts) == 5 else 3
        if len(parts) == 6:
            try:
                charge = float(parts[2])
            except ValueError as exc:
                raise LammpsDataError("atom charge must be numeric") from exc
            if not np.isfinite(charge):
                raise LammpsDataError("atom charge must be finite")
        if _INTEGER_TOKEN.fullmatch(type_token):
            type_id = int(type_token)
            if type_id <= 0:
                raise LammpsDataError("atom type IDs must be positive")
            if type_id > n_types:
                raise LammpsDataError(
                    f"atom type id {type_id} exceeds declared atom-type count {n_types}"
                )
            if id_to_name:
                if type_id not in id_to_name:
                    raise LammpsDataError(f"type id {type_id} not found in type mapping")
                name = id_to_name[type_id]
            else:
                if type_id not in inverse_default:
                    raise LammpsDataError(f"unknown atom type id {type_id}")
                name = inverse_default[type_id]
        else:
            name = type_token
        try:
            coordinates = tuple(float(value) for value in parts[coordinate_start:coordinate_start + 3])
        except ValueError as exc:
            raise LammpsDataError("atom coordinates must be numeric") from exc
        if len(coordinates) != 3 or not np.all(np.isfinite(coordinates)):
            raise LammpsDataError("atom coordinates must contain three finite values")
        ids.append(atom_id)
        rows.append((name, *coordinates))

    if len(rows) != n_atoms:
        raise LammpsDataError(f"expected {n_atoms} atom rows, found {len(rows)}")
    # The handoff format emitted by GBMaker ends after Atoms.  Reject additional
    # atom-like records instead of silently trusting a header count that understates
    # the serialized structure.  Standard later LAMMPS sections remain acceptable.
    section_headers = {
        "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
        "Masses", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
        "Dihedral Coeffs", "Improper Coeffs",
    }
    for remaining in lines[index:]:
        stripped = remaining.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped in section_headers or any(
            stripped.startswith(f"{header} ") for header in section_headers
        ):
            break
        raise LammpsDataError("unexpected extra content in the Atoms section")

    atom_ids = np.asarray(ids, dtype=np.int64)
    if np.unique(atom_ids).size != atom_ids.size:
        raise LammpsDataError("atom IDs must be unique")
    atoms = np.asarray(rows, dtype=Atom.atom_dtype)
    if len(np.unique(atoms["name"])) > n_types:
        raise LammpsDataError("atom rows contain more species than declared atom types")
    return LammpsAtomData(atom_ids, atoms, box_dims)
