"""LAMMPS structure I/O for explicit grain-ownership handoff.

This module parses and validates external structure syntax and transient atom-ID
mappings. Persistent grain identity is defined in :mod:`GBOpt.InterfaceDomain`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from GBOpt.Atom import Atom
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.InterfaceDomain import (
    GrainOwnership,
    GrainOwnershipError,
    _readonly_copy,
    _strict_finite_real,
)

_INTEGER_TOKEN = re.compile(r"^[+-]?\d+$")
_INT64_MAX = np.iinfo(np.int64).max


class LammpsDataError(ValueError):
    """Raised when a LAMMPS data file cannot be read unambiguously."""


def _strict_box_dims(value: object) -> np.ndarray:
    """Validate orthogonal candidate box bounds.

    :param value: Box bounds to validate.
    :return: A floating-point array with shape ``(3, 2)``.
    :raises GrainOwnershipError: If the bounds are malformed or unordered.
    """
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3, 2):
        raise GrainOwnershipError("candidate box_dims must have shape (3, 2)")

    bounds = np.empty((3, 2), dtype=float)
    for axis in range(3):
        bounds[axis, 0] = _strict_finite_real(
            f"candidate box_dims[{axis}, 0]",
            raw[axis, 0],
        )
        bounds[axis, 1] = _strict_finite_real(
            f"candidate box_dims[{axis}, 1]",
            raw[axis, 1],
        )

    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise GrainOwnershipError("candidate box bounds must be strictly ordered")
    return bounds


def _strict_id_token(token: str) -> int:
    """Validate one positive file atom-ID token.

    :param token: Raw atom-ID token read from a LAMMPS file.
    :return: A positive Python integer representable as signed ``int64``.
    :raises LammpsDataError: If the token is nonintegral, nonpositive, or outside the
        supported ``int64`` range.
    """
    if not _INTEGER_TOKEN.fullmatch(token):
        raise LammpsDataError(f"atom ID must be an integral token, got {token!r}")

    value = int(token)
    if value <= 0:
        raise LammpsDataError("atom IDs must be positive")
    if value > _INT64_MAX:
        raise LammpsDataError(
            "atom ID must be representable as a signed 64-bit integer"
        )
    return value


def _strict_species_name(value: object) -> str:
    """Validate one GBOpt element symbol.

    :param value: Species value to validate.
    :return: The validated element symbol.
    :raises LammpsDataError: If the value is not a supported element symbol.
    """
    if not isinstance(value, str) or value not in Atom._numbers:
        raise LammpsDataError(f"unsupported atom species label: {value!r}")
    return value


def _strict_candidate_species(value: object) -> str:
    """Validate one candidate species symbol.

    :param value: Species value to validate.
    :return: The validated element symbol.
    :raises GrainOwnershipError: If the value is unsupported.
    """
    if not isinstance(value, str) or value not in Atom._numbers:
        raise GrainOwnershipError(f"unsupported candidate species label: {value!r}")
    return value


def _strict_type_id(value: object) -> int:
    """Validate one positive LAMMPS atom-type ID.

    :param value: Value to validate.
    :return: A positive Python integer.
    :raises LammpsDataError: If the value is Boolean, nonintegral, or
        nonpositive.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise LammpsDataError("atom type IDs must be integers")

    normalized = int(value)
    if normalized <= 0:
        raise LammpsDataError("atom type IDs must be positive")
    return normalized


def _normalize_type_mapping(
    type_dict: Mapping[object, object] | None,
) -> dict[int, str]:
    """Normalize a type map to ``type ID -> element symbol``.

    :param type_dict: Mapping in either ``str -> int`` or ``int -> str`` form.
    :return: A validated type-ID-to-species mapping.
    :raises LammpsDataError: If the mapping is malformed or ambiguous.
    """
    if type_dict is None:
        return {}
    if not isinstance(type_dict, Mapping):
        raise LammpsDataError(
            "type_dict must be a mapping[str, int] or mapping[int, str]"
        )
    if not type_dict:
        return {}

    items = list(type_dict.items())

    id_to_name_form = all(
        isinstance(key, Integral)
        and not isinstance(key, (bool, np.bool_))
        and isinstance(value, str)
        for key, value in items
    )
    name_to_id_form = all(
        isinstance(key, str)
        and isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        for key, value in items
    )

    if id_to_name_form:
        return {
            _strict_type_id(type_id): _strict_species_name(species)
            for type_id, species in items
        }

    if name_to_id_form:
        normalized: dict[int, str] = {}
        for species_value, type_id_value in items:
            species = _strict_species_name(species_value)
            type_id = _strict_type_id(type_id_value)

            previous = normalized.get(type_id)
            if previous is not None and previous != species:
                raise LammpsDataError(
                    f"atom type ID {type_id} is mapped to both "
                    f"{previous!r} and {species!r}"
                )
            normalized[type_id] = species

        return normalized

    raise LammpsDataError(
        "type_dict must be a mapping[str, int] or mapping[int, str]"
    )


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
        """Construct one immutable parsed LAMMPS atom snapshot.

        :param atom_ids: Serialization-local atom IDs in file-row order.
        :param atoms: Structured atom rows containing species and Cartesian
            coordinates.
        :param box_dims: Orthogonal lower and upper bounds for x, y, and z.
        :param boundary_periodic: Keyword argument, optional. Per-axis periodicity
            decoded from a dump header; defaults to ``None``.
        :param selected_frame: Keyword argument, optional. Selected zero-based dump
            frame index; defaults to ``None`` for data files.
        """
        object.__setattr__(self, "_atom_ids", _readonly_copy(atom_ids, dtype=np.int64))
        object.__setattr__(self, "_atoms", _readonly_copy(atoms, dtype=Atom.atom_dtype))
        object.__setattr__(self, "_box_dims", _readonly_copy(box_dims, dtype=float))
        object.__setattr__(self, "boundary_periodic", boundary_periodic)
        object.__setattr__(self, "selected_frame", selected_frame)

    @property
    def atom_ids(self) -> np.ndarray:
        """Return the parsed atom IDs in file-row order.

        :return: A read-only defensive copy of the atom IDs.
        """
        return _readonly_copy(self._atom_ids)

    @property
    def atoms(self) -> np.ndarray:
        """Return the parsed structured atom rows.

        :return: A read-only defensive copy of the atom array.
        """
        return _readonly_copy(self._atoms)

    @property
    def box_dims(self) -> np.ndarray:
        """Return the parsed orthogonal box bounds.

        :return: A read-only defensive copy with shape ``(3, 2)``.
        """
        return _readonly_copy(self._box_dims)


def read_lammps_data_file(
    path: str | Path, *, type_dict: Mapping[object, object] | None = None
) -> LammpsAtomData:
    """Read the orthogonal LAMMPS data format emitted by ``GBMaker``.

    Both the five-column atomic form and the six-column charge form are supported.
    File row order is preserved so callers can align ownership by atom ID.

    :param path: Path to the LAMMPS data file.
    :param type_dict: Keyword argument, optional. Mapping in ``species -> type ID``
        or ``type ID -> species`` form; defaults to ``None``.
    :return: Immutable parsed atom IDs, atom rows, and box bounds.
    :raises FileNotFoundError: If ``path`` does not identify an existing file.
    :raises LammpsDataError: If the file or type mapping is malformed, ambiguous,
        unsupported, or inconsistent with its declared counts and bounds.
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
                type_id = _strict_type_id(int(label_parts[0]))
                if n_types is not None and type_id > n_types:
                    raise LammpsDataError(
                        f"atom type id {type_id} exceeds declared atom-type count "
                        f"{n_types}"
                    )

                species = _strict_species_name(label_parts[1])
                existing = id_to_name.get(type_id)
                if existing is not None and existing != species:
                    raise LammpsDataError(
                        f"conflicting labels for atom type ID {type_id}: "
                        f"{existing!r} and {species!r}"
                    )

                id_to_name[type_id] = species
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
            type_id = _strict_type_id(int(type_token))
            if type_id > n_types:
                raise LammpsDataError(
                    f"atom type id {type_id} exceeds declared atom-type count "
                    f"{n_types}"
                )

            if id_to_name:
                try:
                    name = id_to_name[type_id]
                except KeyError as exc:
                    raise LammpsDataError(
                        f"type id {type_id} not found in type mapping"
                    ) from exc
            else:
                try:
                    name = inverse_default[type_id]
                except KeyError as exc:
                    raise LammpsDataError(
                        f"unknown atom type id {type_id}"
                    ) from exc

            name = _strict_species_name(name)
        else:
            name = _strict_species_name(type_token)
        try:
            coordinates = tuple(float(value)
                                for value in parts[coordinate_start:coordinate_start + 3])
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
    """One candidate's transient serialization map and expected geometry."""

    _atom_ids: np.ndarray
    _labels: np.ndarray
    _species: np.ndarray
    _box_dims: np.ndarray
    gb_plane_x: float
    inplane_periodic: tuple[bool, bool]
    _left_grain_x_bounds: np.ndarray
    _right_grain_x_bounds: np.ndarray
    coordinate_tolerance: float
    _normal_topology: BoundaryNormalTopology

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
        periodic_outer_x_interface: bool | None = None,
        left_grain_x_bounds: np.ndarray | tuple[float, float] | None = None,
        normal_topology: BoundaryNormalTopology | str | None = None,
    ) -> None:
        """Construct one deterministic candidate/file round-trip mapping.

        Candidate IDs are serialization-local identifiers assigned in candidate row
        order and must therefore be exactly ``1..N``.

        :param atom_ids: Keyword argument. Candidate-local IDs in row order.
        :param labels: Keyword argument. Persistent grain labels aligned with
            candidate rows.
        :param species: Keyword argument. Expected element symbols aligned with
            candidate rows.
        :param box_dims: Keyword argument. Expected orthogonal box bounds.
        :param gb_plane_x: Keyword argument. Nominal central interface plane in
            angstroms.
        :param inplane_periodic: Keyword argument. Explicit y/z periodicity flags.
        :param right_grain_x_bounds: Keyword argument. Physical right-grain x bounds.
        :param coordinate_tolerance: Keyword argument. Positive geometry tolerance in
            angstroms.
        :param periodic_outer_x_interface: Keyword argument, optional. Legacy topology
            compatibility flag; defaults to ``None``.
        :param left_grain_x_bounds: Keyword argument, optional. Physical left-grain x
            bounds; defaults to ``None``.
        :param normal_topology: Keyword argument, optional. Explicit boundary-normal
            topology; defaults to ``None``.
        :raises GrainOwnershipError: If the mapping, species, geometry, IDs, or
            topology are invalid.
        """
        bounds = _strict_box_dims(box_dims)
        plane = _strict_finite_real("candidate gb_plane_x", gb_plane_x)
        if not bounds[0, 0] < plane < bounds[0, 1]:
            raise GrainOwnershipError(
                "candidate gb_plane_x must lie inside the x box"
            )

        if left_grain_x_bounds is None:
            left_grain_x_bounds = (float(bounds[0, 0]), plane)

        ownership = GrainOwnership(
            atom_ids=atom_ids,
            labels=labels,
            gb_plane_x=plane,
            inplane_periodic=inplane_periodic,
            left_grain_x_bounds=left_grain_x_bounds,
            right_grain_x_bounds=right_grain_x_bounds,
            coordinate_tolerance=coordinate_tolerance,
            periodic_outer_x_interface=periodic_outer_x_interface,
            normal_topology=normal_topology,
        )

        ownership_ids = ownership.atom_ids
        canonical_ids = np.arange(
            1,
            ownership_ids.size + 1,
            dtype=np.int64,
        )
        if not np.array_equal(ownership_ids, canonical_ids):
            raise GrainOwnershipError(
                "candidate atom_ids must be exactly 1..N in candidate row order"
            )

        raw_species = np.asarray(species, dtype=object)
        if raw_species.ndim != 1 or raw_species.size != ownership_ids.size:
            raise GrainOwnershipError(
                "candidate species length must equal candidate atom ID count"
            )

        normalized_species = np.asarray(
            [
                _strict_candidate_species(value)
                for value in raw_species.tolist()
            ],
            dtype=Atom.atom_dtype["name"],
        )

        tolerance = ownership.coordinate_tolerance
        left_bounds = ownership.left_grain_x_bounds
        if left_bounds is None:
            raise GrainOwnershipError(
                "candidate mapping requires explicit left-grain x bounds"
            )
        right_bounds = ownership.right_grain_x_bounds

        if (
            left_bounds[0] < bounds[0, 0] - tolerance
            or right_bounds[1] > bounds[0, 1] + tolerance
        ):
            raise GrainOwnershipError(
                "candidate physical grain x bounds must lie inside the "
                "candidate box"
            )

        if (
            left_bounds[1] > plane + tolerance
            or right_bounds[0] < plane - tolerance
        ):
            raise GrainOwnershipError(
                "candidate physical grain bounds must not cross gb_plane_x"
            )

        object.__setattr__(
            self,
            "_atom_ids",
            _readonly_copy(ownership_ids),
        )
        object.__setattr__(
            self,
            "_labels",
            ownership.labels,
        )
        object.__setattr__(
            self,
            "_species",
            _readonly_copy(normalized_species),
        )
        object.__setattr__(
            self,
            "_box_dims",
            _readonly_copy(bounds, dtype=float),
        )
        object.__setattr__(self, "gb_plane_x", ownership.gb_plane_x)
        object.__setattr__(
            self,
            "inplane_periodic",
            ownership.inplane_periodic,
        )
        object.__setattr__(
            self,
            "_left_grain_x_bounds",
            left_bounds,
        )
        object.__setattr__(
            self,
            "_right_grain_x_bounds",
            right_bounds,
        )
        object.__setattr__(self, "coordinate_tolerance", tolerance)
        object.__setattr__(
            self,
            "_normal_topology",
            ownership.normal_topology,
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
        periodic_outer_x_interface: bool | None = None,
        left_grain_x_bounds: np.ndarray | tuple[float, float] | None = None,
        normal_topology: BoundaryNormalTopology | str | None = None,
    ) -> CandidateFileMapping:
        """Construct a mapping from candidate atom and ownership arrays.

        :param atoms: One-dimensional structured candidate atom array containing a
            ``name`` field.
        :param labels: Persistent grain labels aligned with candidate rows.
        :param box_dims: Keyword argument. Expected orthogonal box bounds.
        :param gb_plane_x: Keyword argument. Nominal central interface plane in
            angstroms.
        :param inplane_periodic: Keyword argument. Explicit y/z periodicity flags.
        :param right_grain_x_bounds: Keyword argument. Physical right-grain x bounds.
        :param coordinate_tolerance: Keyword argument. Positive geometry tolerance in
            angstroms.
        :param periodic_outer_x_interface: Keyword argument, optional. Legacy topology
            compatibility flag; defaults to ``None``.
        :param left_grain_x_bounds: Keyword argument, optional. Physical left-grain x
            bounds; defaults to ``None``.
        :param normal_topology: Keyword argument, optional. Explicit boundary-normal
            topology; defaults to ``None``.
        :return: Deterministic candidate-to-file mapping with consecutive atom IDs.
        :raises GrainOwnershipError: If candidate arrays or mapping metadata are
            malformed or inconsistent.
        """
        structured = np.asarray(atoms)
        if (
            structured.ndim != 1
            or structured.dtype.names is None
            or "name" not in structured.dtype.names
        ):
            raise GrainOwnershipError(
                "candidate atoms must be a one-dimensional structured atom array"
            )
        candidate_labels = np.asarray(labels)
        if candidate_labels.ndim != 1 or candidate_labels.size != structured.size:
            raise GrainOwnershipError(
                "ownership length must equal candidate atom count")
        atom_ids = np.arange(1, structured.size + 1, dtype=np.int64)
        return cls(
            atom_ids=atom_ids,
            labels=candidate_labels,
            species=np.asarray(structured["name"], dtype="U8"),
            box_dims=box_dims,
            gb_plane_x=gb_plane_x,
            inplane_periodic=inplane_periodic,
            left_grain_x_bounds=left_grain_x_bounds,
            right_grain_x_bounds=right_grain_x_bounds,
            coordinate_tolerance=coordinate_tolerance,
            periodic_outer_x_interface=periodic_outer_x_interface,
            normal_topology=normal_topology,
        )

    @classmethod
    def from_interface_candidate(cls, candidate: Any) -> CandidateFileMapping:
        """Construct a mapping from an interface-candidate value object.

        :param candidate: Interface-candidate-like object providing atoms, ownership,
            geometry, periodicity, and topology.
        :return: Deterministic candidate-to-file mapping with consecutive atom IDs.
        :raises GrainOwnershipError: If candidate data violate a mapping invariant.
        """
        return cls.from_candidate(
            candidate.atoms,
            candidate.grain_labels,
            box_dims=candidate.box_dims,
            gb_plane_x=candidate.gb_plane_x,
            inplane_periodic=candidate.inplane_periodic,
            left_grain_x_bounds=candidate.left_grain_x_bounds,
            right_grain_x_bounds=candidate.right_grain_x_bounds,
            coordinate_tolerance=candidate.coordinate_tolerance,
            normal_topology=candidate.normal_topology,
        )

    @property
    def atom_ids(self) -> np.ndarray:
        """Return the candidate-local serialization IDs.

        :return: A read-only defensive copy of the IDs.
        """
        return _readonly_copy(self._atom_ids)

    @property
    def labels(self) -> np.ndarray:
        """Return the persistent candidate grain labels.

        :return: A read-only defensive copy of the labels.
        """
        return _readonly_copy(self._labels)

    @property
    def species(self) -> np.ndarray:
        """Return the expected species for each candidate atom ID.

        :return: A read-only defensive copy of the species array.
        """
        return _readonly_copy(self._species)

    @property
    def box_dims(self) -> np.ndarray:
        """Return the expected orthogonal candidate box bounds.

        :return: A read-only defensive copy with shape ``(3, 2)``.
        """
        return _readonly_copy(self._box_dims)

    @property
    def left_grain_x_bounds(self) -> np.ndarray:
        """Return the expected physical left-grain x bounds.

        :return: A read-only defensive copy of the left-grain bounds.
        """
        return _readonly_copy(self._left_grain_x_bounds)

    @property
    def right_grain_x_bounds(self) -> np.ndarray:
        """Return the expected physical right-grain x bounds.

        :return: A read-only defensive copy of the right-grain bounds.
        """
        return _readonly_copy(self._right_grain_x_bounds)

    @property
    def normal_topology(self) -> BoundaryNormalTopology:
        """Return the expected boundary-normal topology.

        :return: The normalized boundary-normal topology value object.
        """
        return self._normal_topology

    @property
    def periodic_outer_x_interface(self) -> bool:
        """Report the expected outer-x periodic-interface state.

        :return: ``True`` when the expected topology has a periodic outer x interface;
            otherwise ``False``.
        """
        return self._normal_topology.periodic_outer_x_interface

    @property
    def expected_count(self) -> int:
        """Return the expected number of evaluator-output atoms.

        :return: Candidate atom count as a Python integer.
        """
        return int(self._atom_ids.size)

    def ownership_for_file_ids(self, file_ids: np.ndarray) -> GrainOwnership:
        """Align mapped ownership to evaluator-output file-row IDs.

        :param file_ids: Evaluator-output atom IDs in file-row order.
        :return: Immutable ownership metadata aligned to ``file_ids``.
        :raises GrainOwnershipError: If the IDs are malformed or do not exactly match
            the candidate mapping.
        """
        base = GrainOwnership(
            atom_ids=self._atom_ids,
            labels=self._labels,
            gb_plane_x=self.gb_plane_x,
            inplane_periodic=self.inplane_periodic,
            left_grain_x_bounds=self._left_grain_x_bounds,
            right_grain_x_bounds=self._right_grain_x_bounds,
            coordinate_tolerance=self.coordinate_tolerance,
            normal_topology=self._normal_topology,
        )
        return base.aligned_to(file_ids)


def _dump_boundary_flags(tokens: list[str]) -> tuple[bool, bool, bool] | None:
    """Decode orthogonal LAMMPS dump boundary flags.

    :param tokens: Tokens following ``ITEM: BOX BOUNDS``.
    :return: Per-axis periodicity when three valid flags are present; otherwise
        ``None``.
    """
    if len(tokens) < 3:
        return None
    flags = tokens[-3:]
    if not all(len(flag) == 2 and set(flag).issubset(set("pfsm")) for flag in flags):
        return None
    return tuple(flag == "pp" for flag in flags)


def read_lammps_dump_file(
    path: str | Path, *, type_dict: Mapping[object, object] | None = None
) -> LammpsAtomData:
    """Read exactly the first frame of an orthogonal LAMMPS dump.

    Multi-frame dumps are never concatenated. Validation applies only to frame
    zero; a malformed first frame is an error even if a later frame is valid.

    :param path: Path to the LAMMPS dump file.
    :param type_dict: Keyword argument, optional. Mapping in ``species -> type ID``
        or ``type ID -> species`` form; defaults to ``None``.
    :return: Immutable parsed data for dump frame zero.
    :raises FileNotFoundError: If ``path`` does not identify an existing file.
    :raises LammpsDataError: If the type mapping, frame structure, bounds, topology,
        atom attributes, IDs, species, or coordinates are malformed or ambiguous.
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
            raise LammpsDataError(
                f"selected dump frame is missing atom attribute {required!r}")
    if "typelabel" in attributes:
        species_attr = "typelabel"
    elif "type" in attributes:
        species_attr = "type"
    else:
        raise LammpsDataError("selected dump frame requires type or typelabel")
    attr_index = {name: attributes.index(name)
                  for name in ("id", species_attr, "x", "y", "z")}
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
            species = _strict_species_name(species_token)
        else:
            if not _INTEGER_TOKEN.fullmatch(species_token):
                raise LammpsDataError("dump type values must be integral")

            type_id = _strict_type_id(int(species_token))

            if id_to_name:
                try:
                    species = id_to_name[type_id]
                except KeyError as exc:
                    raise LammpsDataError(
                        f"type id {type_id} not found in type mapping"
                    ) from exc
            else:
                try:
                    species = inverse_default[type_id]
                except KeyError as exc:
                    raise LammpsDataError(
                        f"unknown atom type id {type_id}"
                    ) from exc

            species = _strict_species_name(species)
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
    """Read a supported LAMMPS data file or first dump frame.

    :param path: Path to a LAMMPS data or dump file.
    :param type_dict: Keyword argument, optional. Mapping in ``species -> type ID``
        or ``type ID -> species`` form; defaults to ``None``.
    :return: Immutable parsed atom and box data.
    :raises FileNotFoundError: If ``path`` does not identify an existing file.
    :raises LammpsDataError: If the selected reader rejects malformed, unsupported,
        or ambiguous file content or type metadata.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    with file_path.open(encoding="utf-8") as stream:
        first = stream.readline().strip()
    if first == "ITEM: TIMESTEP":
        return read_lammps_dump_file(file_path, type_dict=type_dict)
    return read_lammps_data_file(file_path, type_dict=type_dict)


def validate_explicit_structure(
    returned_structure: str | Path,
    *,
    candidate_mapping: CandidateFileMapping,
    type_dict: Mapping[object, object] | None = None,
) -> GrainOwnership:
    """Validate an evaluator-returned structure and recover aligned ownership.

    :param returned_structure: Path to the evaluator-returned LAMMPS structure.
    :param candidate_mapping: Keyword argument, required. Expected candidate IDs,
        species, ownership, geometry, and topology.
    :param type_dict: Keyword argument, optional, defaults to ``None``. Mapping in
        ``species -> type ID`` or ``type ID -> species`` form.
    :return: Explicit ownership aligned to the returned file-row atom IDs.
    :raises FileNotFoundError: If the returned structure file does not exist.
    :raises LammpsDataError: If the returned LAMMPS file or type mapping is
        malformed, unsupported, or ambiguous.
    :raises GrainOwnershipError: If evaluator output changes candidate atom count,
        IDs, species, box bounds, topology, or ownership alignment.
    """
    snapshot = read_lammps_structure_file(returned_structure, type_dict=type_dict)
    file_ids = snapshot.atom_ids
    if snapshot.atoms.size != candidate_mapping.expected_count:
        raise GrainOwnershipError(
            "evaluator output atom count does not match the candidate"
        )
    expected_ids = candidate_mapping.atom_ids
    if not np.array_equal(np.sort(file_ids), expected_ids):
        raise GrainOwnershipError(
            "evaluator output atom IDs do not match the candidate"
        )
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
    return candidate_mapping.ownership_for_file_ids(file_ids)
