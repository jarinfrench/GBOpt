# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer supercell enumeration for ``GBMaker`` coherent-boundary construction.

This module converts canonical crystallographic orientation rows into integer supercell
matrices and enumerates conventional-cell origins inside repeated supercells. It also
enumerates exact rational decorated sites without floating-point membership tests. It is
``GBMaker``-facing glue, not core CSL/PQ/plane arithmetic.

TODO: Move this module into ``GBOpt.GBMaker`` when ``GBMaker`` is split into a package.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from math import prod
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np

from GBOpt.crystallography.integer import (
    as_int_array,
    cross_int3,
    integer_adj3,
    integer_det3,
    row_gcd_reduce,
)
from GBOpt.crystallography.types import CrystallographyValueError

if TYPE_CHECKING:
    from GBOpt.UnitCell import RationalBasis


def _positive_integer(value: object, *, name: str) -> int:
    """Return ``value`` as a positive Python integer, excluding booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return integer


def _validated_repeats(
    repeat_x: object,
    repeat_y: object,
    repeat_z: object,
) -> tuple[int, int, int]:
    """Return validated positive repeat counts as Python integers."""
    return (
        _positive_integer(repeat_x, name="repeat_x"),
        _positive_integer(repeat_y, name="repeat_y"),
        _positive_integer(repeat_z, name="repeat_z"),
    )


def _exact_integer_rows(values: object, *, name: str) -> tuple[tuple[int, ...], ...]:
    """Return a rectangular two-dimensional tuple of exact Python integer rows."""
    try:
        array = np.asarray(values, dtype=object)
    except ValueError as exc:
        raise ValueError(f"{name} must be a rectangular two-dimensional array.") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array; got {array.shape}.")

    rows = []
    for row in array:
        exact_row = []
        for value in row:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"{name} must contain only integers; got {value!r}."
                )
            exact_row.append(int(value))
        rows.append(tuple(exact_row))
    return tuple(rows)


def _readonly_object_array(rows: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Return a defensive read-only object array from exact integer rows."""
    array = np.array(rows, dtype=object, copy=True)
    array.setflags(write=False)
    return array


def _readonly_integer_array(values: tuple[int, ...]) -> np.ndarray:
    """Return a defensive read-only platform-integer array."""
    array = np.array(values, dtype=int, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, init=False)
class SupercellSites:
    """Immutable exact decorated sites in a repeated integer supercell.

    ``coordinate_numerators / coordinate_denominator`` are canonical coordinates in
    the row basis of ``supercell_matrix``. Axis ``i`` lies in the half-open interval
    ``[0, repeats[i])``. Conventional-cell coordinates can therefore be reconstructed
    exactly as ``coordinate_numerators @ supercell_matrix / coordinate_denominator``.
    ``basis_indices`` maps each row back to the corresponding rational-basis row.

    Site order is the existing quotient-lattice origin order, followed by rational
    decorated-basis row order for each origin.
    """

    basis_denominator: int
    supercell_index: int
    repeats: tuple[int, int, int]
    basis_size: int
    _coordinate_rows: tuple[tuple[int, int, int], ...] = field(repr=False)
    _basis_index_values: tuple[int, ...] = field(repr=False)
    _supercell_rows: tuple[tuple[int, int, int], ...] = field(repr=False)

    def __init__(
        self,
        *,
        coordinate_numerators: np.ndarray,
        basis_denominator: int,
        basis_indices: np.ndarray,
        supercell_matrix: np.ndarray,
        repeats: tuple[int, int, int],
        basis_size: int,
    ) -> None:
        denominator = _positive_integer(
            basis_denominator,
            name="basis_denominator",
        )
        validated_repeats = _validated_repeats(*repeats)
        validated_basis_size = _positive_integer(basis_size, name="basis_size")

        try:
            int_supercell = as_int_array(supercell_matrix, (3, 3), "S")
        except CrystallographyValueError as exc:
            raise ValueError(str(exc)) from exc
        determinant = integer_det3(int_supercell)
        if determinant == 0:
            raise ValueError("SupercellSites requires non-singular S.")
        supercell_index = abs(determinant)
        supercell_rows = tuple(
            tuple(int(value) for value in row) for row in int_supercell
        )

        coordinate_rows = _exact_integer_rows(
            coordinate_numerators,
            name="coordinate_numerators",
        )
        if coordinate_rows and len(coordinate_rows[0]) != 3:
            raise ValueError(
                "coordinate_numerators must have shape (site_count, 3); "
                f"got ({len(coordinate_rows)}, {len(coordinate_rows[0])})."
            )
        if not coordinate_rows:
            raise ValueError("coordinate_numerators must contain at least one site.")
        if any(len(row) != 3 for row in coordinate_rows):
            raise ValueError("coordinate_numerators must have shape (site_count, 3).")

        raw_basis_indices = np.asarray(basis_indices, dtype=object)
        if raw_basis_indices.ndim != 1 or len(raw_basis_indices) != len(coordinate_rows):
            raise ValueError(
                "basis_indices must have shape (site_count,) parallel to "
                f"coordinate_numerators; got {raw_basis_indices.shape}."
            )
        basis_index_values = []
        for value in raw_basis_indices:
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"basis_indices must contain only integers; got {value!r}."
                )
            index = int(value)
            if index < 0 or index >= validated_basis_size:
                raise ValueError(
                    "basis_indices must lie in the half-open interval "
                    f"[0, basis_size); got {index}."
                )
            basis_index_values.append(index)
        basis_index_tuple = tuple(basis_index_values)

        coordinate_denominator = denominator * supercell_index
        upper_bounds = tuple(
            repeat * coordinate_denominator for repeat in validated_repeats
        )
        for row in coordinate_rows:
            if any(
                value < 0 or value >= upper_bounds[axis]
                for axis, value in enumerate(row)
            ):
                raise ValueError(
                    "coordinate_numerators must lie in the repeated half-open "
                    "supercell coordinate bounds."
                )

        expected_per_basis = supercell_index * prod(validated_repeats)
        expected_sites = validated_basis_size * expected_per_basis
        if len(coordinate_rows) != expected_sites:
            raise ValueError(
                f"SupercellSites expected {expected_sites} sites but received "
                f"{len(coordinate_rows)}."
            )
        counts = Counter(basis_index_tuple)
        if any(counts[index] != expected_per_basis for index in range(validated_basis_size)):
            raise ValueError(
                "SupercellSites basis-index populations do not match the exact "
                "quotient-lattice origin count."
            )

        decorated_representatives = tuple(zip(coordinate_rows, basis_index_tuple))
        if len(decorated_representatives) != len(set(decorated_representatives)):
            raise ValueError(
                "SupercellSites contains duplicate wrapped exact representatives "
                "for the same decorated basis identity."
            )

        object.__setattr__(self, "basis_denominator", denominator)
        object.__setattr__(self, "supercell_index", supercell_index)
        object.__setattr__(self, "repeats", validated_repeats)
        object.__setattr__(self, "basis_size", validated_basis_size)
        object.__setattr__(self, "_coordinate_rows", coordinate_rows)
        object.__setattr__(self, "_basis_index_values", basis_index_tuple)
        object.__setattr__(self, "_supercell_rows", supercell_rows)

    @property
    def coordinate_denominator(self) -> int:
        """Return the positive common denominator of exact supercell coordinates."""
        return self.basis_denominator * self.supercell_index

    @property
    def coordinate_numerators(self) -> np.ndarray:
        """Return a defensive read-only copy of exact supercell-coordinate numerators."""
        return _readonly_object_array(self._coordinate_rows)

    @property
    def basis_indices(self) -> np.ndarray:
        """Return a defensive read-only copy of decorated basis-row indices."""
        return _readonly_integer_array(self._basis_index_values)

    @property
    def supercell_matrix(self) -> np.ndarray:
        """Return a defensive read-only copy of the integer supercell matrix."""
        return _readonly_object_array(self._supercell_rows)

    @property
    def site_count(self) -> int:
        """Return the number of exact decorated representatives."""
        return len(self._coordinate_rows)


def _integer_membership(
    origin,
    adj_S: list,
    det_S: int,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> bool:
    """Return whether an integer conventional-cell origin lies inside a repeated
    supercell.

    Computes fractional supercell coordinates as integer numerators via ``origin @
    adj(S)``. If ``det(S)`` is negative, the numerators are sign-flipped before checking
    ``0 <= u_num[i] < repeat[i] * abs(det(S))`` for each axis.

    This helper supports both signs of ``det_S``; production callers currently guarantee
    ``det_S > 0``.

    :param origin: Integer 3-vector giving the candidate conventional-cell origin.
    :param adj_S: Adjugate of ``S`` as a 3 by 3 list-of-lists from ``integer_adj3``.
    :param det_S: Integer determinant of ``S`` from ``integer_det3``.
    :param repeat_x: Number of repeats along the boundary-normal direction.
    :param repeat_y: Number of repeats along the first in-plane direction.
    :param repeat_z: Number of repeats along the second in-plane direction.
    :return: ``True`` if ``origin`` lies inside the repeated supercell.
    """
    abs_det = abs(det_S)
    # u_num[j] = sum_k origin[k] * adj_S[k][j]   (row-vector @ matrix)
    u_num = [sum(int(origin[k]) * adj_S[k][j] for k in range(3)) for j in range(3)]
    if det_S < 0:
        u_num = [-u for u in u_num]
    return (
        0 <= u_num[0] < repeat_x * abs_det
        and 0 <= u_num[1] < repeat_y * abs_det
        and 0 <= u_num[2] < repeat_z * abs_det
    )


def build_supercell_matrix(P: np.ndarray) -> np.ndarray:
    """Build the integer supercell matrix ``S = [s0; s1; s2]`` from canonical ``P``.

    For a canonical orientation matrix ``P`` whose rows have already been GCD-reduced
    and made right-handed by ``canonicalize_pq``, ``s1 = P[1]``, ``s2 = P[2]``, and ``s0
    = P[0]``. This relies on ``P[0]`` equaling ``gcd_reduce(cross(P[1], P[2]))``.

    :param P: 3 by 3 canonical orientation matrix with integer-valued rows.
    :return: 3 by 3 integer ndarray ``S`` with rows ``[s0, s1, s2]``.
    :raises ValueError: If ``P`` cannot be converted to an exact 3 by 3 integer matrix,
        ``S`` is singular, or ``P[0]`` does not equal ``gcd_reduce(cross(P[1], P[2]))``.
    """
    try:
        supercell_obj = as_int_array(P, (3, 3), "P (supercell matrix)")
    except CrystallographyValueError as exc:
        raise ValueError(str(exc)) from exc

    supercell = np.asarray(supercell_obj, dtype=int)
    det_S = integer_det3(supercell)
    if det_S == 0:
        raise ValueError(
            f"Supercell matrix S derived from P is singular (det_S=0). "
            f"P = {supercell_obj.tolist()}. The in-plane rows P[1], P[2] must be "
            "linearly independent."
        )
    expected_s0 = row_gcd_reduce(
        np.array(cross_int3(supercell[1], supercell[2]), dtype=object)
    )
    if not np.array_equal(expected_s0.astype(int), supercell[0]):
        raise ValueError(
            f"P[0]={supercell[0].tolist()} does not equal gcd_reduce(cross(P[1], P[2]))"
            f"={expected_s0.tolist()}; P must be canonical and right-handed."
        )
    return supercell


def _enumerate_validated_origins(
    int_supercell: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
    det_S: int,
) -> np.ndarray:
    """Enumerate origin representatives for a validated nonsingular supercell."""
    adj_S = integer_adj3(int_supercell)

    # Bounding box from the 8 parallelepiped corners. Iteration over ``product`` is
    # lexicographic and intentionally defines the existing deterministic origin order.
    corners = np.array(
        [
            i * repeat_x * int_supercell[0]
            + j * repeat_y * int_supercell[1]
            + k * repeat_z * int_supercell[2]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ],
        dtype=object,
    )
    lower_bound = [int(corners[:, d].min()) - 1 for d in range(3)]
    upper_bound = [int(corners[:, d].max()) + 1 for d in range(3)]
    ranges = [range(lower_bound[d], upper_bound[d] + 1) for d in range(3)]

    accepted = [
        tuple(row)
        for row in product(*ranges)
        if _integer_membership(
            row,
            adj_S,
            det_S,
            repeat_x,
            repeat_y,
            repeat_z,
        )
    ]

    expected = repeat_x * repeat_y * repeat_z * abs(det_S)
    if len(accepted) != expected:
        raise ValueError(
            f"expected {expected} origins "
            f"(repeat={repeat_x},{repeat_y},{repeat_z}, |det|={abs(det_S)}) "
            f"but got {len(accepted)}. supercell = {int_supercell.tolist()}"
        )
    return np.array(accepted, dtype=object)


def enumerate_supercell_origins(
    supercell: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> np.ndarray:
    """Enumerate all integer conventional-cell origins inside the repeated supercell.

    The repeated supercell is spanned by ``repeat_x * s0``, ``repeat_y * s1``, and
    ``repeat_z * s2``. Candidates are drawn from the integer bounding box of the eight
    parallelepiped corners, padded by one lattice step. Membership is tested with
    ``_integer_membership``, so no floating-point selection is used.

    :param supercell: 3 by 3 integer supercell matrix with rows ``s0``, ``s1``, and
        ``s2``.
    :param repeat_x: Number of repeats along ``s0``.
    :param repeat_y: Number of repeats along ``s1``.
    :param repeat_z: Number of repeats along ``s2``.
    :return: Array of shape ``(N, 3)`` of accepted integer origins, where ``N ==
        repeat_x * repeat_y * repeat_z * abs(det(S))``.
    :raises ValueError: If ``supercell`` cannot be converted to an exact integer matrix,
        ``supercell`` is singular or has negative determinant, a repeat count is not a
        positive integer, or the accepted count does not match the expected value.
    """
    try:
        int_supercell = as_int_array(supercell, (3, 3), "S")
    except CrystallographyValueError as exc:
        raise ValueError(str(exc)) from exc

    repeat_x, repeat_y, repeat_z = _validated_repeats(
        repeat_x,
        repeat_y,
        repeat_z,
    )

    det_S = integer_det3(int_supercell)
    if det_S == 0:
        raise ValueError("enumerate_supercell_origins requires non-singular S.")
    if det_S < 0:
        raise ValueError(
            "S must have positive determinant; ensure P was produced by "
            "canonicalize_pq with right-handed orientation rows. "
            f"Got det(S)={det_S}, S={int_supercell.tolist()}."
        )
    return np.asarray(
        _enumerate_validated_origins(
            int_supercell,
            repeat_x,
            repeat_y,
            repeat_z,
            det_S,
        ),
        dtype=int,
    )


def _validated_rational_basis(
    rational_basis: RationalBasis | None,
) -> tuple[tuple[str, ...], tuple[tuple[int, int, int], ...], int]:
    """Return independently validated exact decorated-basis metadata."""
    if rational_basis is None:
        raise ValueError(
            "Exact decorated-site enumeration requires UnitCell.rational_basis; "
            "arbitrary floating-point basis coordinates are not rationalized."
        )

    try:
        raw_names = rational_basis.names
        raw_numerators = rational_basis.numerators
        raw_denominator = rational_basis.denominator
    except AttributeError as exc:
        raise ValueError(
            "rational_basis must provide names, numerators, and denominator metadata."
        ) from exc

    if isinstance(raw_names, str):
        raise ValueError("rational_basis names must be a sequence of strings.")
    try:
        names = tuple(raw_names)
    except TypeError as exc:
        raise ValueError("rational_basis names must be a sequence of strings.") from exc
    if not names:
        raise ValueError("rational_basis must contain at least one decorated site.")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("rational_basis names must contain only strings.")

    denominator = _positive_integer(
        raw_denominator,
        name="rational_basis denominator",
    )

    try:
        numerator_array = np.asarray(raw_numerators, dtype=object)
    except ValueError as exc:
        raise ValueError(
            "rational_basis numerators must have shape (basis_size, 3)."
        ) from exc
    if numerator_array.ndim != 2 or numerator_array.shape[1] != 3:
        raise ValueError(
            "rational_basis numerators must have shape (basis_size, 3); "
            f"got {numerator_array.shape}."
        )
    if numerator_array.shape[0] != len(names):
        raise ValueError(
            "rational_basis species names and numerator rows must have equal lengths."
        )

    numerator_rows = _exact_integer_rows(
        numerator_array,
        name="rational_basis numerators",
    )
    for row in numerator_rows:
        if any(value < 0 or value >= denominator for value in row):
            raise ValueError(
                "rational_basis coordinates must lie in the canonical half-open "
                "interval [0, denominator)."
            )

    decorated_rows = tuple(zip(names, numerator_rows))
    if len(decorated_rows) != len(set(decorated_rows)):
        raise ValueError("rational_basis contains duplicate decorated basis rows.")

    return names, numerator_rows, denominator


def enumerate_supercell_sites(
    supercell: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
    *,
    rational_basis: RationalBasis | None,
) -> SupercellSites:
    """Enumerate exact decorated sites inside a repeated integer supercell.

    Existing quotient-lattice origin representatives are traversed in their established
    deterministic order. Every rational decorated-basis row is then attached in basis
    order. Each site is transformed with integer adjugate arithmetic and wrapped into
    the repeated half-open supercell. No floating-point membership decision or
    Cartesian clipping is performed.

    :param supercell: Nonsingular 3 by 3 integer supercell matrix ``S``. Either
        determinant sign is accepted.
    :param repeat_x: Number of repeats along ``S[0]``.
    :param repeat_y: Number of repeats along ``S[1]``.
    :param repeat_z: Number of repeats along ``S[2]``.
    :param rational_basis: Exact decorated basis from ``UnitCell.rational_basis``.
        ``None`` is rejected rather than approximating a floating-point basis.
    :return: Immutable exact supercell-coordinate representatives and basis-row indices.
    :raises ValueError: If an input is malformed, the supercell is singular, rational
        metadata is absent or invalid, or an internal exact invariant fails.
    """
    try:
        int_supercell = as_int_array(supercell, (3, 3), "S")
    except CrystallographyValueError as exc:
        raise ValueError(str(exc)) from exc

    repeats = _validated_repeats(repeat_x, repeat_y, repeat_z)
    _, basis_rows, basis_denominator = _validated_rational_basis(rational_basis)

    det_S = integer_det3(int_supercell)
    if det_S == 0:
        raise ValueError("enumerate_supercell_sites requires non-singular S.")
    abs_det = abs(det_S)
    adj_S = np.asarray(integer_adj3(int_supercell), dtype=object)
    origins = _enumerate_validated_origins(
        int_supercell,
        *repeats,
        det_S,
    )

    coordinate_denominator = basis_denominator * abs_det
    wrap_limits = tuple(
        repeat * coordinate_denominator for repeat in repeats
    )
    coordinate_rows: list[tuple[int, int, int]] = []
    basis_indices: list[int] = []

    for origin in origins:
        for basis_index, basis_row in enumerate(basis_rows):
            site_numerator = np.asarray(
                tuple(
                    basis_denominator * int(origin[axis]) + basis_row[axis]
                    for axis in range(3)
                ),
                dtype=object,
            )
            supercell_numerator = site_numerator @ adj_S
            if det_S < 0:
                supercell_numerator = -supercell_numerator
            wrapped = tuple(
                int(supercell_numerator[axis]) % wrap_limits[axis]
                for axis in range(3)
            )
            coordinate_rows.append(wrapped)
            basis_indices.append(basis_index)

    expected = len(basis_rows) * abs_det * prod(repeats)
    if len(coordinate_rows) != expected:
        raise ValueError(
            f"enumerate_supercell_sites expected {expected} sites but produced "
            f"{len(coordinate_rows)}."
        )

    return SupercellSites(
        coordinate_numerators=np.array(coordinate_rows, dtype=object),
        basis_denominator=basis_denominator,
        basis_indices=np.array(basis_indices, dtype=int),
        supercell_matrix=int_supercell,
        repeats=repeats,
        basis_size=len(basis_rows),
    )


def supercell_axis_numerators(
    supercell: np.ndarray,
    origins: np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Return exact supercell-coordinate numerators for integer origins.

    For a supercell matrix ``S``, an integer origin has reduced supercell coordinates
    ``u = origin @ inv(S)``. This function returns the selected numerator column of
    ``origin @ adj(S)``, leaving the common denominator ``det(S)`` implicit. Distinct
    numerator values identify fine integer layers along the selected supercell axis.

    :param supercell: 3 by 3 integer supercell matrix ``S``.
    :param origins: Integer conventional-cell origins with shape ``(N, 3)``.
    :param axis: Supercell coordinate axis to return. Must be 0, 1, or 2.
    :return: Integer numerator coordinates parallel to ``origins``.
    :raises ValueError: If inputs cannot be converted to exact integer arrays, if
        ``axis`` is invalid, or if ``supercell`` is singular.
    """
    int_supercell = as_int_array(supercell, (3, 3), "S")

    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis!r}.")
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2; got {axis!r}.")

    origins_arr = np.asarray(origins, dtype=object)
    if origins_arr.ndim != 2 or origins_arr.shape[1] != 3:
        raise ValueError(f"origins must have shape (N, 3); got {origins_arr.shape}.")

    int_origins = as_int_array(origins_arr, origins_arr.shape, "origins")

    det_S = integer_det3(int_supercell)
    if det_S == 0:
        raise ValueError("supercell_axis_numerators requires non-singular S.")

    adj_S = np.array(integer_adj3(int_supercell), dtype=object)
    numerators = int_origins @ adj_S[:, axis]

    if det_S < 0:
        numerators = -numerators

    return np.asarray([int(value) for value in numerators], dtype=object)


__all__ = [
    "SupercellSites",
    "build_supercell_matrix",
    "enumerate_supercell_origins",
    "enumerate_supercell_sites",
    "supercell_axis_numerators",
]
