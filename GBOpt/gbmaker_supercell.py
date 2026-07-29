# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact integer supercell enumeration for ``GBMaker`` coherent-boundary construction.

This module converts canonical crystallographic orientation rows into integer supercell
matrices and enumerates conventional-cell origins inside repeated supercells. It is
``GBMaker``-facing glue, not core CSL/PQ/plane arithmetic.

TODO: Move this module into ``GBOpt.GBMaker`` when ``GBMaker`` is split into a package.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral

import numpy as np

from GBOpt.crystallography.integer import (
    as_int_array,
    cross_int3,
    integer_adj3,
    integer_det3,
    row_gcd_reduce,
)
from GBOpt.crystallography.types import CrystallographyValueError


def _readonly_exact_array(values: np.ndarray, *, name: str) -> np.ndarray:
    """Return a defensive, read-only object array of exact Python integers."""
    array = np.asarray(values, dtype=object)
    exact = np.empty(array.shape, dtype=object)
    for index, value in np.ndenumerate(array):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise ValueError(f"{name} must contain only integers; got {value!r}.")
        exact[index] = int(value)
    exact.setflags(write=False)
    return exact


def _positive_integer(value: object, *, name: str) -> int:
    """Return ``value`` as a positive Python integer, excluding booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}.")
    return integer


@dataclass(frozen=True, slots=True, eq=False)
class SupercellSites:
    """Exact decorated sites inside a repeated integer supercell.

    ``crystal_numerators / denominator`` are wrapped conventional-cell
    coordinates. ``supercell_coordinate_numerators /
    supercell_coordinate_denominator`` are the corresponding wrapped coordinates
    in the row basis of the unrepeated supercell matrix. ``supercell_index`` is
    ``abs(det(S))`` and therefore determines the latter denominator together with
    ``denominator``. ``basis_indices`` maps each site back to its input basis row.
    """

    crystal_numerators: np.ndarray
    denominator: int
    basis_indices: np.ndarray
    supercell_coordinate_numerators: np.ndarray
    supercell_index: int

    def __post_init__(self) -> None:
        denominator = _positive_integer(self.denominator, name="denominator")
        supercell_index = _positive_integer(
            self.supercell_index,
            name="supercell_index",
        )

        crystal_numerators = np.asarray(self.crystal_numerators, dtype=object)
        if crystal_numerators.ndim != 2 or crystal_numerators.shape[1] != 3:
            raise ValueError(
                "crystal_numerators must have shape (site_count, 3); "
                f"got {crystal_numerators.shape}."
            )
        crystal_numerators = _readonly_exact_array(
            crystal_numerators,
            name="crystal_numerators",
        )

        supercell_numerators = np.asarray(
            self.supercell_coordinate_numerators,
            dtype=object,
        )
        if (
            supercell_numerators.ndim != 2
            or supercell_numerators.shape != crystal_numerators.shape
        ):
            raise ValueError(
                "supercell_coordinate_numerators must have the same "
                "(site_count, 3) shape as crystal_numerators; "
                f"got {supercell_numerators.shape}."
            )
        supercell_numerators = _readonly_exact_array(
            supercell_numerators,
            name="supercell_coordinate_numerators",
        )

        raw_basis_indices = np.asarray(self.basis_indices, dtype=object)
        if raw_basis_indices.ndim != 1 or len(raw_basis_indices) != len(
            crystal_numerators
        ):
            raise ValueError(
                "basis_indices must have shape (site_count,) parallel to the "
                f"coordinate arrays; got {raw_basis_indices.shape}."
            )
        basis_indices = np.empty(raw_basis_indices.shape, dtype=int)
        for index, value in np.ndenumerate(raw_basis_indices):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"basis_indices must contain only integers; got {value!r}."
                )
            integer = int(value)
            if integer < 0:
                raise ValueError(
                    f"basis_indices must be nonnegative; got {integer}."
                )
            basis_indices[index] = integer
        basis_indices.setflags(write=False)

        object.__setattr__(self, "crystal_numerators", crystal_numerators)
        object.__setattr__(self, "denominator", denominator)
        object.__setattr__(self, "basis_indices", basis_indices)
        object.__setattr__(
            self,
            "supercell_coordinate_numerators",
            supercell_numerators,
        )
        object.__setattr__(self, "supercell_index", supercell_index)

    @property
    def supercell_coordinate_denominator(self) -> int:
        """Return the exact common denominator of supercell coordinates."""
        return self.denominator * self.supercell_index


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


def _enumerate_validated_origins(
    int_supercell: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
    det_S: int,
) -> np.ndarray:
    """Enumerate origin representatives for a validated nonsingular supercell."""
    adj_S = integer_adj3(int_supercell)

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


def enumerate_supercell_sites(
    supercell: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
    *,
    basis_numerators: np.ndarray,
    basis_denominator: int,
) -> SupercellSites:
    """Enumerate exact decorated sites inside a repeated integer supercell.

    Integer conventional-cell origins are representatives of the quotient lattice
    defined by the repeated supercell. For each rational basis offset ``b / d``,
    this function translates every representative, converts it to exact supercell
    coordinates, wraps those coordinates into the repeated half-open cell, and
    reconstructs exact conventional coordinates. No floating-point clipping or
    search halo is used.

    :param supercell: Nonsingular 3 by 3 integer supercell matrix ``S``. Either
        determinant sign is accepted.
    :param repeat_x: Number of repeats along ``S[0]``.
    :param repeat_y: Number of repeats along ``S[1]``.
    :param repeat_z: Number of repeats along ``S[2]``.
    :param basis_numerators: Exact fractional conventional-cell basis numerators
        with shape ``(basis_size, 3)``.
    :param basis_denominator: Positive common denominator for
        ``basis_numerators``.
    :return: Immutable exact coordinates and basis-row indices for every decorated
        site. The site count is ``basis_size * repeat_x * repeat_y * repeat_z *
        abs(det(S))``.
    :raises ValueError: If any input is malformed, the supercell is singular, the
        basis is empty or noncanonical, or an internal exact divisibility/count
        invariant fails.
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
    denominator = _positive_integer(
        basis_denominator,
        name="basis_denominator",
    )

    raw_basis = np.asarray(basis_numerators, dtype=object)
    if raw_basis.ndim != 2 or raw_basis.shape[1] != 3:
        raise ValueError(
            "basis_numerators must have shape (basis_size, 3); "
            f"got {raw_basis.shape}."
        )
    if raw_basis.shape[0] == 0:
        raise ValueError("basis_numerators must contain at least one basis site.")
    try:
        int_basis = as_int_array(
            raw_basis,
            raw_basis.shape,
            "basis_numerators",
        )
    except CrystallographyValueError as exc:
        raise ValueError(str(exc)) from exc

    for value in int_basis.flat:
        integer = int(value)
        if integer < 0 or integer >= denominator:
            raise ValueError(
                "basis_numerators must lie in the half-open interval "
                "[0, basis_denominator)."
            )
    basis_rows = [tuple(int(value) for value in row) for row in int_basis]
    if len(basis_rows) != len(set(basis_rows)):
        raise ValueError("basis_numerators must not contain duplicate basis sites.")

    det_S = integer_det3(int_supercell)
    if det_S == 0:
        raise ValueError("enumerate_supercell_sites requires non-singular S.")
    abs_det = abs(det_S)
    adj_S = np.asarray(integer_adj3(int_supercell), dtype=object)
    origins = _enumerate_validated_origins(
        int_supercell,
        repeat_x,
        repeat_y,
        repeat_z,
        det_S,
    )

    site_count = len(origins) * len(int_basis)
    crystal_numerators = np.empty((site_count, 3), dtype=object)
    supercell_numerators = np.empty((site_count, 3), dtype=object)
    basis_indices = np.empty(site_count, dtype=int)
    supercell_denominator = denominator * abs_det
    wrap_limits = np.asarray(
        (repeat_x, repeat_y, repeat_z),
        dtype=object,
    ) * supercell_denominator

    site_index = 0
    for origin in origins:
        for basis_index, basis_row in enumerate(int_basis):
            site_numerator = denominator * origin + basis_row
            u_numerator = site_numerator @ adj_S
            if det_S < 0:
                u_numerator = -u_numerator
            wrapped_u = np.mod(u_numerator, wrap_limits)
            reconstructed_scaled = wrapped_u @ int_supercell
            if any(int(value) % abs_det for value in reconstructed_scaled):
                raise ValueError(
                    "enumerate_supercell_sites produced a non-integral exact "
                    "coordinate reconstruction."
                )

            supercell_numerators[site_index] = [
                int(value) for value in wrapped_u
            ]
            crystal_numerators[site_index] = [
                int(value) // abs_det for value in reconstructed_scaled
            ]
            basis_indices[site_index] = basis_index
            site_index += 1

    expected = (
        len(int_basis) * repeat_x * repeat_y * repeat_z * abs_det
    )
    if site_index != expected:
        raise ValueError(
            f"enumerate_supercell_sites expected {expected} sites but produced "
            f"{site_index}."
        )
    crystal_rows = {
        tuple(int(value) for value in row) for row in crystal_numerators
    }
    if len(crystal_rows) != expected:
        raise ValueError(
            "enumerate_supercell_sites produced duplicate wrapped decorated sites."
        )

    return SupercellSites(
        crystal_numerators=crystal_numerators,
        denominator=denominator,
        basis_indices=basis_indices,
        supercell_coordinate_numerators=supercell_numerators,
        supercell_index=abs_det,
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
