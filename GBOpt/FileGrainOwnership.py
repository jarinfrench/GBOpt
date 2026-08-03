"""Narrow file-backed grain ownership support for optimizer seed handoff.

LAMMPS atom IDs are used only to align one verified serialized seed with its
persistent left/right grain labels.  They are not optimizer-wide atom identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from GBOpt.Atom import Atom

if TYPE_CHECKING:
    from GBOpt.GBManipulator import GBManipulator

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
    """Parsed atom IDs, species/coordinates, box bounds, and optional BC flags."""

    _atom_ids: np.ndarray
    _atoms: np.ndarray
    _box_dims: np.ndarray
    boundary_periodic: tuple[bool, bool, bool] | None
    selected_frame: int | None

    def __init__(
        self,
        atom_ids: np.ndarray,
        atoms: np.ndarray,
        box_dims: np.ndarray,
        *,
        boundary_periodic: tuple[bool, bool, bool] | None = None,
        selected_frame: int | None = None,
    ) -> None:
        object.__setattr__(self, "_atom_ids", _readonly_copy(atom_ids, dtype=np.int64))
        object.__setattr__(self, "_atoms", _readonly_copy(atoms, dtype=Atom.atom_dtype))
        object.__setattr__(self, "_box_dims", _readonly_copy(box_dims, dtype=float))
        object.__setattr__(self, "boundary_periodic", boundary_periodic)
        object.__setattr__(self, "selected_frame", selected_frame)

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


@dataclass(frozen=True, slots=True, init=False)
class CandidateFileMapping:
    """One candidate's transient serialization map and expected topology.

    IDs are always freshly assigned in deterministic row order (1..N). They are
    valid only for the candidate/evaluator-return round trip represented here.
    Persistent grain identity remains in ``labels``.
    """

    _atom_ids: np.ndarray
    _labels: np.ndarray
    _species: np.ndarray
    _box_dims: np.ndarray
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
        species: np.ndarray,
        box_dims: np.ndarray,
        gb_plane_x: float,
        inplane_periodic: tuple[bool, bool],
        right_grain_x_bounds: np.ndarray | tuple[float, float],
        coordinate_tolerance: float,
        periodic_outer_x_interface: bool,
    ) -> None:
        ownership = GrainOwnership(
            atom_ids=atom_ids,
            labels=labels,
            gb_plane_x=gb_plane_x,
            inplane_periodic=inplane_periodic,
            right_grain_x_bounds=right_grain_x_bounds,
            coordinate_tolerance=coordinate_tolerance,
            periodic_outer_x_interface=periodic_outer_x_interface,
        )
        raw_species = np.asarray(species)
        if raw_species.ndim != 1 or raw_species.size != ownership.atom_ids.size:
            raise GrainOwnershipError(
                "candidate species length must equal candidate atom ID count"
            )
        normalized_species = np.asarray([str(value) for value in raw_species.tolist()], dtype="U8")
        if np.any(normalized_species == ""):
            raise GrainOwnershipError("candidate species names must be nonempty")
        bounds = np.asarray(box_dims, dtype=float)
        if bounds.shape != (3, 2) or not np.all(np.isfinite(bounds)):
            raise GrainOwnershipError("candidate box_dims must have shape (3, 2) and be finite")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise GrainOwnershipError("candidate box bounds must be strictly ordered")
        tolerance = ownership.coordinate_tolerance
        if not bounds[0, 0] < ownership.gb_plane_x < bounds[0, 1]:
            raise GrainOwnershipError("candidate gb_plane_x must lie inside the x box")
        grain_bounds = ownership.right_grain_x_bounds
        if (
            grain_bounds[0] < bounds[0, 0] - tolerance
            or grain_bounds[1] > bounds[0, 1] + tolerance
        ):
            raise GrainOwnershipError(
                "candidate right-grain x bounds must lie inside the candidate box"
            )
        object.__setattr__(self, "_atom_ids", ownership.atom_ids)
        object.__setattr__(self, "_labels", ownership.labels)
        object.__setattr__(self, "_species", _readonly_copy(normalized_species))
        object.__setattr__(self, "_box_dims", _readonly_copy(bounds, dtype=float))
        object.__setattr__(self, "gb_plane_x", ownership.gb_plane_x)
        object.__setattr__(self, "inplane_periodic", ownership.inplane_periodic)
        object.__setattr__(self, "_right_grain_x_bounds", ownership.right_grain_x_bounds)
        object.__setattr__(self, "coordinate_tolerance", tolerance)
        object.__setattr__(
            self, "periodic_outer_x_interface", ownership.periodic_outer_x_interface
        )

    @classmethod
    def from_candidate(
        cls,
        atoms: np.ndarray,
        labels: np.ndarray,
        *,
        box_dims: np.ndarray,
        gb_plane_x: float,
        inplane_periodic: tuple[bool, bool],
        right_grain_x_bounds: np.ndarray | tuple[float, float],
        coordinate_tolerance: float,
        periodic_outer_x_interface: bool,
    ) -> "CandidateFileMapping":
        structured = np.asarray(atoms)
        if structured.ndim != 1 or structured.dtype.names is None or "name" not in structured.dtype.names:
            raise GrainOwnershipError("candidate atoms must be a one-dimensional structured atom array")
        candidate_labels = np.asarray(labels)
        if candidate_labels.ndim != 1 or candidate_labels.size != structured.size:
            raise GrainOwnershipError("ownership length must equal candidate atom count")
        # GBMaker.write_lammps emits this exact fresh deterministic ID sequence.
        atom_ids = np.arange(1, structured.size + 1, dtype=np.int64)
        return cls(
            atom_ids=atom_ids,
            labels=candidate_labels,
            species=np.asarray(structured["name"], dtype="U8"),
            box_dims=box_dims,
            gb_plane_x=gb_plane_x,
            inplane_periodic=inplane_periodic,
            right_grain_x_bounds=right_grain_x_bounds,
            coordinate_tolerance=coordinate_tolerance,
            periodic_outer_x_interface=periodic_outer_x_interface,
        )

    @property
    def atom_ids(self) -> np.ndarray:
        return _readonly_copy(self._atom_ids)

    @property
    def labels(self) -> np.ndarray:
        return _readonly_copy(self._labels)

    @property
    def species(self) -> np.ndarray:
        return _readonly_copy(self._species)

    @property
    def box_dims(self) -> np.ndarray:
        return _readonly_copy(self._box_dims)

    @property
    def right_grain_x_bounds(self) -> np.ndarray:
        return _readonly_copy(self._right_grain_x_bounds)

    @property
    def expected_count(self) -> int:
        return int(self._atom_ids.size)

    def ownership_for_file_ids(self, file_ids: np.ndarray) -> GrainOwnership:
        base = GrainOwnership(
            atom_ids=self._atom_ids,
            labels=self._labels,
            gb_plane_x=self.gb_plane_x,
            inplane_periodic=self.inplane_periodic,
            right_grain_x_bounds=self._right_grain_x_bounds,
            coordinate_tolerance=self.coordinate_tolerance,
            periodic_outer_x_interface=self.periodic_outer_x_interface,
        )
        return base.aligned_to(file_ids)


def _dump_boundary_flags(tokens: list[str]) -> tuple[bool, bool, bool] | None:
    if len(tokens) < 3:
        return None
    flags = tokens[-3:]
    if not all(len(flag) == 2 and set(flag).issubset(set("pfsm")) for flag in flags):
        return None
    return tuple(flag == "pp" for flag in flags)


def read_lammps_dump_file(
    path: str | Path, *, type_dict: Mapping[object, object] | None = None
) -> LammpsAtomData:
    """Read exactly the first LAMMPS dump frame, matching the legacy loader.

    Multi-frame dumps are never concatenated. Validation applies only to frame zero;
    a malformed first frame is an error even if a later frame is valid.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "ITEM: TIMESTEP":
        raise LammpsDataError("LAMMPS dump must begin with ITEM: TIMESTEP")
    try:
        int(lines[1].strip())
    except (IndexError, ValueError) as exc:
        raise LammpsDataError("selected dump frame has an invalid timestep") from exc
    index = 2
    if index >= len(lines) or lines[index].strip() != "ITEM: NUMBER OF ATOMS":
        raise LammpsDataError("selected dump frame is missing NUMBER OF ATOMS")
    try:
        n_atoms = int(lines[index + 1].strip())
    except (IndexError, ValueError) as exc:
        raise LammpsDataError("selected dump frame has an invalid atom count") from exc
    if n_atoms < 0:
        raise LammpsDataError("selected dump frame atom count must be nonnegative")
    index += 2
    if index >= len(lines) or not lines[index].startswith("ITEM: BOX BOUNDS"):
        raise LammpsDataError("selected dump frame is missing BOX BOUNDS")
    boundary_periodic = _dump_boundary_flags(lines[index].split()[3:])
    bounds_rows: list[tuple[float, float]] = []
    for offset in range(1, 4):
        try:
            parts = lines[index + offset].split()
            if len(parts) != 2:
                raise ValueError
            lower, upper = float(parts[0]), float(parts[1])
        except (IndexError, ValueError) as exc:
            raise LammpsDataError(
                "explicit ownership supports orthogonal two-column dump bounds only"
            ) from exc
        bounds_rows.append((lower, upper))
    box_dims = np.asarray(bounds_rows, dtype=float)
    if not np.all(np.isfinite(box_dims)) or np.any(box_dims[:, 0] >= box_dims[:, 1]):
        raise LammpsDataError("selected dump frame box bounds are invalid")
    index += 4
    if index >= len(lines) or not lines[index].startswith("ITEM: ATOMS"):
        raise LammpsDataError("selected dump frame is missing ATOMS")
    attributes = lines[index].split()[2:]
    for required in ("id", "x", "y", "z"):
        if required not in attributes:
            raise LammpsDataError(f"selected dump frame is missing atom attribute {required!r}")
    if "typelabel" in attributes:
        species_attr = "typelabel"
    elif "type" in attributes:
        species_attr = "type"
    else:
        raise LammpsDataError("selected dump frame requires type or typelabel")
    attr_index = {name: attributes.index(name) for name in ("id", species_attr, "x", "y", "z")}
    id_to_name = _normalize_type_mapping(type_dict)
    inverse_default = {number: name for name, number in Atom._numbers.items()}
    ids: list[int] = []
    rows: list[tuple[str, float, float, float]] = []
    index += 1
    for row_index in range(n_atoms):
        if index + row_index >= len(lines) or lines[index + row_index].startswith("ITEM:"):
            raise LammpsDataError(
                f"selected dump frame expected {n_atoms} atom rows, found {row_index}"
            )
        parts = lines[index + row_index].split()
        if len(parts) < len(attributes):
            raise LammpsDataError("selected dump frame contains a short atom row")
        atom_id = _strict_id_token(parts[attr_index["id"]])
        species_token = parts[attr_index[species_attr]]
        if species_attr == "typelabel":
            species = species_token
        else:
            if not _INTEGER_TOKEN.fullmatch(species_token):
                raise LammpsDataError("dump type values must be integral")
            type_id = int(species_token)
            if type_id <= 0:
                raise LammpsDataError("dump type IDs must be positive")
            if id_to_name:
                if type_id not in id_to_name:
                    raise LammpsDataError(f"type id {type_id} not found in type mapping")
                species = id_to_name[type_id]
            elif type_id in inverse_default:
                species = inverse_default[type_id]
            else:
                raise LammpsDataError(f"unknown atom type id {type_id}")
        try:
            xyz = tuple(float(parts[attr_index[axis]]) for axis in ("x", "y", "z"))
        except ValueError as exc:
            raise LammpsDataError("dump coordinates must be numeric") from exc
        if not np.all(np.isfinite(xyz)):
            raise LammpsDataError("dump coordinates must be finite")
        ids.append(atom_id)
        rows.append((species, *xyz))
    next_index = index + n_atoms
    while next_index < len(lines) and not lines[next_index].strip():
        next_index += 1
    if (
        next_index < len(lines)
        and lines[next_index].strip() != "ITEM: TIMESTEP"
    ):
        raise LammpsDataError(
            "selected dump frame contains unexpected content after its atom rows"
        )
    atom_ids = np.asarray(ids, dtype=np.int64)
    if np.unique(atom_ids).size != atom_ids.size:
        raise LammpsDataError("atom IDs must be unique")
    return LammpsAtomData(
        atom_ids,
        np.asarray(rows, dtype=Atom.atom_dtype),
        box_dims,
        boundary_periodic=boundary_periodic,
        selected_frame=0,
    )


def read_lammps_structure_file(
    path: str | Path, *, type_dict: Mapping[object, object] | None = None
) -> LammpsAtomData:
    """Read a supported data file or the first frame of a LAMMPS dump."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    with file_path.open(encoding="utf-8") as stream:
        first = stream.readline().strip()
    if first == "ITEM: TIMESTEP":
        return read_lammps_dump_file(file_path, type_dict=type_dict)
    return read_lammps_data_file(file_path, type_dict=type_dict)


def reload_explicit_manipulator(
    returned_structure: str | Path,
    *,
    candidate_mapping: CandidateFileMapping,
    unit_cell: Any,
    gb_thickness: float,
    type_dict: Mapping[object, object] | None = None,
) -> "GBManipulator":
    """Validate and reconstruct one evaluator-returned explicit-ownership candidate.

    This is the authoritative reload path for explicit-ownership GA execution.
    """
    snapshot = read_lammps_structure_file(returned_structure, type_dict=type_dict)
    file_ids = snapshot.atom_ids
    if snapshot.atoms.size != candidate_mapping.expected_count:
        raise GrainOwnershipError(
            "evaluator output atom count does not match the candidate"
        )
    expected_ids = candidate_mapping.atom_ids
    if not np.array_equal(np.sort(file_ids), expected_ids):
        raise GrainOwnershipError("evaluator output atom IDs do not match the candidate")
    order = np.argsort(file_ids, kind="stable")
    expected_species = candidate_mapping.species
    actual_species = np.asarray(snapshot.atoms["name"], dtype="U8")[order]
    if not np.array_equal(actual_species, expected_species):
        raise GrainOwnershipError(
            "evaluator output changed species/type for one or more atom IDs"
        )
    tolerance = candidate_mapping.coordinate_tolerance
    if not np.allclose(
        snapshot.box_dims, candidate_mapping.box_dims, atol=tolerance, rtol=0.0
    ):
        raise GrainOwnershipError(
            "evaluator output changed box bounds; variable-cell relaxation is unsupported"
        )
    if snapshot.selected_frame is not None and snapshot.boundary_periodic is None:
        raise GrainOwnershipError(
            "evaluator dump does not encode unambiguous boundary topology"
        )
    if snapshot.boundary_periodic is not None:
        expected_periodic = (
            candidate_mapping.periodic_outer_x_interface,
            *candidate_mapping.inplane_periodic,
        )
        if snapshot.boundary_periodic != expected_periodic:
            raise GrainOwnershipError("evaluator output changed boundary topology")
    ownership = candidate_mapping.ownership_for_file_ids(file_ids)
    # Local import avoids a module cycle with GBManipulator -> FileGrainOwnership.
    from GBOpt.GBManipulator import GBManipulator

    manipulator = GBManipulator(
        str(returned_structure),
        unit_cell=unit_cell,
        gb_thickness=gb_thickness,
        type_dict=type_dict,
        grain_ownership=ownership,
    )
    parent = manipulator.parents[0]
    if parent.grain_labels is None or len(parent.grain_labels) != candidate_mapping.expected_count:
        raise GrainOwnershipError("reloaded ownership length does not match atom count")
    return manipulator
