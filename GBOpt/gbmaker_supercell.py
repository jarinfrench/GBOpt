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
from typing import TYPE_CHECKING

import numpy as np

from GBOpt.crystallography.integer import (
    as_int_array,
    as_positive_int,
    cross_int3,
    integer_adj3,
    integer_det3,
    row_gcd_reduce,
)
from GBOpt.crystallography.types import CrystallographyValueError
from GBOpt.Utils.integer_normal_forms import ExactNormalFormError, smith_normal_form_3x3

if TYPE_CHECKING:
    from GBOpt.UnitCell import RationalBasis


def _positive_integer(value: object, *, name: str) -> int:
    """Return ``value`` as a positive Python integer.

    Validation is delegated to the shared exact-integer utility so boolean rejection,
    exact integer coercion, and error wording remain centralized.

    :param value: Candidate Python or NumPy integer scalar.
    :param name: Keyword argument naming the value in validation messages.
    :return: Validated value as a Python ``int``.
    :raises ValueError: If ``value`` is not an exact positive integer.
    """
    try:
        return as_positive_int(value, name)
    except CrystallographyValueError as exc:
        raise ValueError(str(exc)) from exc


def _validated_repeats(
    repeat_x: object,
    repeat_y: object,
    repeat_z: object,
) -> tuple[int, int, int]:
    """Return validated positive supercell repeat counts.

    :param repeat_x: Repeat count along the first supercell row.
    :param repeat_y: Repeat count along the second supercell row.
    :param repeat_z: Repeat count along the third supercell row.
    :return: ``(repeat_x, repeat_y, repeat_z)`` as Python integers.
    :raises ValueError: If any repeat count is not an exact positive integer.
    """
    return (
        _positive_integer(repeat_x, name="repeat_x"),
        _positive_integer(repeat_y, name="repeat_y"),
        _positive_integer(repeat_z, name="repeat_z"),
    )


def _exact_integer_rows(values: object, *, name: str) -> tuple[tuple[int, ...], ...]:
    """Return rectangular two-dimensional rows of exact Python integers.

    :param values: Candidate rectangular two-dimensional array-like object.
    :param name: Keyword argument naming the input in validation messages.
    :return: Immutable rectangular rows containing Python ``int`` values.
    :raises ValueError: If ``values`` cannot be represented as a rectangular
        two-dimensional exact-integer array.
    """
    try:
        raw_array = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a rectangular two-dimensional array."
        ) from exc

    if raw_array.ndim != 2:
        raise ValueError(
            f"{name} must be a two-dimensional array; got {raw_array.shape}."
        )

    try:
        exact_array = as_int_array(raw_array, raw_array.shape, name)
    except CrystallographyValueError as exc:
        raise ValueError(str(exc)) from exc

    return tuple(
        tuple(int(value) for value in row)
        for row in exact_array
    )


def _readonly_object_array(rows: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Return a defensive read-only object array containing Python integers.

    :param rows: Exact integer rows to copy.
    :return: Read-only object-dtype array with no writable alias to ``rows``.
    """
    array = np.array(rows, dtype=object, copy=True)
    array.setflags(write=False)
    return array


def _readonly_integer_array(values: tuple[int, ...]) -> np.ndarray:
    """Return a defensive read-only platform-integer array.

    This helper is appropriate for bounded basis indices, not unbounded crystallographic
    coordinates.

    :param values: Bounded integer index values to copy.
    :return: Read-only platform-integer array with no writable alias to ``values``.
    """
    array = np.array(values, dtype=int, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, init=False)
class SupercellSites:
    """Immutable exact decorated sites in a repeated integer supercell.

    ``coordinate_numerators / coordinate_denominator`` are canonical coordinates in the
    row basis of ``supercell_matrix``. Axis ``i`` lies in the half-open interval ``[0,
    repeats[i])``. Conventional-cell coordinates can therefore be reconstructed exactly
    as ``coordinate_numerators @ supercell_matrix / coordinate_denominator``.
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
        """Initialize validated immutable exact decorated-site state.

        :param coordinate_numerators: Keyword argument containing exact canonical
            supercell-coordinate numerators with shape ``(site_count, 3)``.
        :param basis_denominator: Keyword argument containing the positive denominator
            of the rational conventional basis.
        :param basis_indices: Keyword argument containing the rational-basis row index
            for every coordinate row.
        :param supercell_matrix: Keyword argument containing the nonsingular exact 3 by
            3 supercell matrix.
        :param repeats: Keyword argument containing three positive supercell repeat
            counts.
        :param basis_size: Keyword argument containing the number of rows in the
            rational decorated basis.
        :raises ValueError: If an input is malformed or any exact population,
            coordinate-bound, or uniqueness invariant fails.
        """
        denominator = _positive_integer(
            basis_denominator,
            name="basis_denominator",
        )

        try:
            repeat_values = tuple(repeats)
        except TypeError as exc:
            raise ValueError(
                "repeats must be a length-3 sequence of positive integers."
            ) from exc
        if len(repeat_values) != 3:
            raise ValueError(
                "repeats must be a length-3 sequence of positive integers; got length "
                f"{len(repeat_values)}."
            )
        validated_repeats = _validated_repeats(*repeat_values)

        validated_basis_size = _positive_integer(
            basis_size,
            name="basis_size",
        )

        try:
            int_supercell = as_int_array(supercell_matrix, (3, 3), "S")
        except CrystallographyValueError as exc:
            raise ValueError(str(exc)) from exc

        determinant = integer_det3(int_supercell)
        if determinant == 0:
            raise ValueError("SupercellSites requires non-singular S.")

        supercell_index = abs(determinant)
        supercell_rows = tuple(
            tuple(int(value) for value in row)
            for row in int_supercell
        )

        coordinate_rows = _exact_integer_rows(
            coordinate_numerators,
            name="coordinate_numerators",
        )
        if not coordinate_rows:
            raise ValueError(
                "coordinate_numerators must contain at least one site."
            )
        if any(len(row) != 3 for row in coordinate_rows):
            raise ValueError(
                "coordinate_numerators must have shape (site_count, 3)."
            )

        try:
            raw_basis_indices = np.asarray(basis_indices, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "basis_indices must be a one-dimensional exact-integer array."
            ) from exc

        if (
            raw_basis_indices.ndim != 1
            or len(raw_basis_indices) != len(coordinate_rows)
        ):
            raise ValueError(
                "basis_indices must have shape (site_count,) parallel to "
                f"coordinate_numerators; got {raw_basis_indices.shape}."
            )

        try:
            exact_basis_indices = as_int_array(
                raw_basis_indices,
                raw_basis_indices.shape,
                "basis_indices",
            )
        except CrystallographyValueError as exc:
            raise ValueError(str(exc)) from exc

        basis_index_tuple = tuple(
            int(value)
            for value in exact_basis_indices
        )
        for index in basis_index_tuple:
            if index < 0 or index >= validated_basis_size:
                raise ValueError(
                    "basis_indices must lie in the half-open interval [0, "
                    f"basis_size); got {index}."
                )

        coordinate_denominator = denominator * supercell_index
        upper_bounds = tuple(
            repeat * coordinate_denominator
            for repeat in validated_repeats
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
        if any(
            counts[index] != expected_per_basis
            for index in range(validated_basis_size)
        ):
            raise ValueError(
                "SupercellSites basis-index populations do not match the exact "
                "quotient-lattice origin count."
            )

        # Basis identity does not make a duplicated physical coordinate valid.
        if len(coordinate_rows) != len(set(coordinate_rows)):
            raise ValueError(
                "SupercellSites contains duplicate wrapped exact coordinate "
                "representatives."
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
        """Return the positive common denominator of exact supercell coordinates.

        :return: Positive common denominator of the exact supercell coordinates.
        """
        return self.basis_denominator * self.supercell_index

    @property
    def coordinate_numerators(self) -> np.ndarray:
        """Return a defensive read-only copy of exact supercell-coordinate numerators.

        :return: Read-only object array of exact supercell-coordinate numerators.
        """
        return _readonly_object_array(self._coordinate_rows)

    @property
    def basis_indices(self) -> np.ndarray:
        """Return a defensive read-only copy of decorated basis-row indices.

        :return: Read-only integer array of decorated rational-basis row indices.
        """
        return _readonly_integer_array(self._basis_index_values)

    @property
    def supercell_matrix(self) -> np.ndarray:
        """Return a defensive read-only copy of the integer supercell matrix.

        :return: Read-only object array containing the exact supercell matrix.
        """
        return _readonly_object_array(self._supercell_rows)

    @property
    def site_count(self) -> int:
        """Return the number of exact decorated representatives.

        :return: Number of exact decorated representatives.
        """
        return len(self._coordinate_rows)


def _integer_membership(
    origin,
    adj_S: list,
    det_S: int,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> bool:
    """Return whether an integer origin lies inside a repeated supercell.

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
    # u_num[j] = sum_k origin[k] * adj_S[k][j] (row-vector @ matrix)
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
            "Supercell matrix S derived from P is singular (det_S=0). P = "
            f"{supercell_obj.tolist()}. The in-plane rows P[1], P[2] must be linearly "
            "independent."
        )
    expected_s0 = row_gcd_reduce(
        np.array(cross_int3(supercell[1], supercell[2]), dtype=object)
    )
    if not np.array_equal(expected_s0.astype(int), supercell[0]):
        raise ValueError(
            f"P[0]={supercell[0].tolist()} does not equal gcd_reduce(cross(P[1], "
            f"P[2]))={expected_s0.tolist()}; P must be canonical and right-handed."
        )
    return supercell


def _enumerate_validated_origins(
    int_supercell: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
    det_S: int,
) -> np.ndarray:
    """Enumerate repeated-supercell origin representatives output-proportionally.

    Smith normal form enumerates exactly the quotient-lattice classes. Each class is
    then mapped into the original repeated half-open supercell using exact adjugate
    arithmetic. The final lexicographic sort preserves the established order produced by
    the former Cartesian bounding-box traversal.

    :param int_supercell: Validated nonsingular 3 by 3 exact integer matrix.
    :param repeat_x: Positive repeat count along row zero.
    :param repeat_y: Positive repeat count along row one.
    :param repeat_z: Positive repeat count along row two.
    :param det_S: Exact determinant of ``int_supercell``.
    :return: Object-dtype array containing one Python-integer representative per
        repeated-supercell quotient-lattice class.
    :raises ValueError: If ``det_S`` is inconsistent, normal-form construction fails, an
        exact divisibility invariant fails, or the resulting representatives do not
        satisfy count, uniqueness, or membership invariants.
    """
    actual_det = integer_det3(int_supercell)
    if actual_det != det_S:
        raise ValueError(
            "det_S does not match the validated supercell determinant: expected "
            f"{actual_det}, got {det_S}."
        )

    repeated_supercell = np.array(
        int_supercell,
        dtype=object,
        copy=True,
    )
    repeated_supercell[0, :] *= repeat_x
    repeated_supercell[1, :] *= repeat_y
    repeated_supercell[2, :] *= repeat_z

    repeated_det = integer_det3(repeated_supercell)
    expected = repeat_x * repeat_y * repeat_z * abs(det_S)
    if abs(repeated_det) != expected:
        raise ValueError(
            "Repeated-supercell determinant does not match the expected quotient "
            f"index: expected {expected}, got {abs(repeated_det)}."
        )

    try:
        decomposition = smith_normal_form_3x3(repeated_supercell)
    except ExactNormalFormError as exc:
        raise ValueError(
            f"Could not enumerate supercell origins by Smith normal form: {exc}"
        ) from exc

    smith_diagonal = tuple(
        abs(int(decomposition.D[index, index]))
        for index in range(3)
    )
    if prod(smith_diagonal) != expected:
        raise ValueError(
            "Smith normal-form diagonal does not reproduce the repeated supercell "
            f"index {expected}: got {smith_diagonal}."
        )

    transform_det = integer_det3(decomposition.V)
    if abs(transform_det) != 1:
        raise ValueError(
            "Smith normal-form right transformation is not unimodular."
        )

    inverse_transform = np.asarray(
        integer_adj3(decomposition.V),
        dtype=object,
    )
    if transform_det < 0:
        inverse_transform = -inverse_transform

    repeated_adjugate = np.asarray(
        integer_adj3(repeated_supercell),
        dtype=object,
    )
    repeated_index = abs(repeated_det)

    representatives: list[tuple[int, int, int]] = []
    for residue in product(
        *(range(diagonal) for diagonal in smith_diagonal)
    ):
        quotient_representative = (
            np.asarray(residue, dtype=object) @ inverse_transform
        )

        coordinate_numerators = (
            quotient_representative @ repeated_adjugate
        )
        if repeated_det < 0:
            coordinate_numerators = -coordinate_numerators

        wrapped_numerators = np.asarray(
            [
                int(value) % repeated_index
                for value in coordinate_numerators
            ],
            dtype=object,
        )
        conventional_numerators = (
            wrapped_numerators @ repeated_supercell
        )

        conventional_row = []
        for value in conventional_numerators:
            quotient, remainder = divmod(
                int(value),
                repeated_index,
            )
            if remainder != 0:
                raise ValueError(
                    "Exact origin reconstruction was not divisible by the "
                    "repeated-supercell index."
                )
            conventional_row.append(quotient)

        representatives.append(tuple(conventional_row))

    representatives.sort()

    if len(representatives) != expected:
        raise ValueError(
            f"Expected {expected} origins but produced {len(representatives)}."
        )
    if len(set(representatives)) != expected:
        raise ValueError(
            "Origin enumeration produced duplicate quotient-lattice representatives."
        )

    original_adjugate = integer_adj3(int_supercell)
    if any(
        not _integer_membership(
            row,
            original_adjugate,
            det_S,
            repeat_x,
            repeat_y,
            repeat_z,
        )
        for row in representatives
    ):
        raise ValueError(
            "Origin enumeration produced a representative outside the repeated "
            "half-open supercell."
        )

    return np.asarray(representatives, dtype=object)


def enumerate_supercell_origins(
    supercell: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
) -> np.ndarray:
    """Enumerate integer conventional-cell origins in a repeated supercell.

    The repeated supercell is spanned by ``repeat_x * s0``, ``repeat_y * s1``, and
    ``repeat_z * s2``. Membership and enumeration use exact integer arithmetic only.

    :param supercell: Exact 3 by 3 integer supercell matrix with rows ``s0``, ``s1``,
        and ``s2``.
    :param repeat_x: Positive repeat count along ``s0``.
    :param repeat_y: Positive repeat count along ``s1``.
    :param repeat_z: Positive repeat count along ``s2``.
    :return: Object-dtype array of shape ``(N, 3)`` containing Python integers, where
        ``N == repeat_x * repeat_y * repeat_z * abs(det(S))``.
    :raises ValueError: If ``supercell`` is malformed, singular, or left-handed; if a
        repeat count is invalid; or if an exact enumeration invariant fails.
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
        raise ValueError(
            "enumerate_supercell_origins requires non-singular S."
        )
    if det_S < 0:
        raise ValueError(
            "S must have positive determinant; ensure P was produced by "
            f"canonicalize_pq with right-handed orientation rows. Got det(S)={det_S}, "
            f"S={int_supercell.tolist()}."
        )

    return _enumerate_validated_origins(
        int_supercell,
        repeat_x,
        repeat_y,
        repeat_z,
        det_S,
    )


def _validated_rational_basis(
    rational_basis: RationalBasis | None,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[int, int, int], ...],
    int,
]:
    """Return exact metadata from a validated ``RationalBasis`` value object.

    ``RationalBasis`` owns validation of species names, numerator shape and exactness,
    denominator positivity, canonical coordinate bounds, defensive copying, and
    immutability. This adapter only enforces that the declared value-object contract was
    actually supplied.

    :param rational_basis: Exact immutable basis metadata from ``UnitCell``.
    :return: Species names, immutable exact numerator rows, and the positive common
        denominator.
    :raises ValueError: If rational metadata is absent or is not a ``RationalBasis``
        instance.
    """
    if rational_basis is None:
        raise ValueError(
            "Exact decorated-site enumeration requires UnitCell.rational_basis; "
            "arbitrary floating-point basis coordinates are not rationalized."
        )

    # Local import avoids introducing a module-import cycle while still enforcing the
    # declared runtime contract.
    from GBOpt.UnitCell import RationalBasis

    if not isinstance(rational_basis, RationalBasis):
        raise ValueError(
            "rational_basis must be a validated UnitCell.RationalBasis instance."
        )

    numerator_rows = tuple(
        tuple(int(value) for value in row)
        for row in rational_basis.numerators
    )
    return (
        rational_basis.names,
        numerator_rows,
        rational_basis.denominator,
    )


def enumerate_supercell_sites(
    supercell: np.ndarray,
    repeat_x: int,
    repeat_y: int,
    repeat_z: int,
    *,
    rational_basis: RationalBasis | None,
) -> SupercellSites:
    """Enumerate exact decorated sites inside a repeated integer supercell.

    Quotient-lattice origin representatives are traversed in their established
    deterministic order, followed by rational decorated-basis row order. Each decorated
    site is transformed with exact adjugate arithmetic, verified against the defining
    reconstruction identity, and wrapped into the repeated half-open supercell. No
    floating-point membership decision is performed.

    :param supercell: Nonsingular exact 3 by 3 supercell matrix ``S``. Either
        determinant sign is accepted.
    :param repeat_x: Positive repeat count along ``S[0]``.
    :param repeat_y: Positive repeat count along ``S[1]``.
    :param repeat_z: Positive repeat count along ``S[2]``.
    :param rational_basis: Keyword argument containing validated exact basis metadata
        from ``UnitCell.rational_basis``.
    :return: Immutable exact supercell representatives and corresponding rational-basis
        row indices.
    :raises ValueError: If an input is malformed, ``S`` is singular, rational metadata
        is unavailable, or an exact count, reconstruction, wrapping, population, or
        uniqueness invariant fails.
    """
    try:
        int_supercell = as_int_array(supercell, (3, 3), "S")
    except CrystallographyValueError as exc:
        raise ValueError(str(exc)) from exc

    repeats = _validated_repeats(
        repeat_x,
        repeat_y,
        repeat_z,
    )
    _, basis_rows, basis_denominator = _validated_rational_basis(
        rational_basis
    )

    det_S = integer_det3(int_supercell)
    if det_S == 0:
        raise ValueError(
            "enumerate_supercell_sites requires non-singular S."
        )

    abs_det = abs(det_S)
    adj_S = np.asarray(
        integer_adj3(int_supercell),
        dtype=object,
    )
    origins = _enumerate_validated_origins(
        int_supercell,
        *repeats,
        det_S,
    )

    expected_origins = abs_det * prod(repeats)
    if len(origins) != expected_origins:
        raise ValueError(
            "Origin enumeration returned an unexpected exact population: expected "
            f"{expected_origins}, got {len(origins)}."
        )

    coordinate_denominator = basis_denominator * abs_det
    wrap_limits = tuple(
        repeat * coordinate_denominator
        for repeat in repeats
    )
    expected_sites = len(basis_rows) * expected_origins

    coordinate_numerators = np.empty(
        (expected_sites, 3),
        dtype=object,
    )
    basis_indices = np.empty(
        expected_sites,
        dtype=np.intp,
    )

    site_index = 0
    for origin in origins:
        exact_origin = tuple(int(value) for value in origin)

        for basis_index, basis_row in enumerate(basis_rows):
            site_numerator = np.asarray(
                tuple(
                    basis_denominator * exact_origin[axis]
                    + basis_row[axis]
                    for axis in range(3)
                ),
                dtype=object,
            )

            unwrapped_supercell_numerator = site_numerator @ adj_S
            if det_S < 0:
                unwrapped_supercell_numerator = (
                    -unwrapped_supercell_numerator
                )

            # Verify the row-vector adjugate identity before periodic wrapping: (site @
            # signed-adj(S)) @ S == site * abs(det(S)).
            reconstructed_numerator = (
                unwrapped_supercell_numerator @ int_supercell
            )
            expected_numerator = site_numerator * abs_det
            if not np.array_equal(
                reconstructed_numerator,
                expected_numerator,
            ):
                raise ValueError(
                    "Exact decorated-site transformation failed the adjugate "
                    "reconstruction identity."
                )

            coordinate_numerators[site_index, :] = tuple(
                int(unwrapped_supercell_numerator[axis])
                % wrap_limits[axis]
                for axis in range(3)
            )
            basis_indices[site_index] = basis_index
            site_index += 1

    if site_index != expected_sites:
        raise ValueError(
            f"enumerate_supercell_sites expected {expected_sites} sites but produced "
            f"{site_index}."
        )

    return SupercellSites(
        coordinate_numerators=coordinate_numerators,
        basis_denominator=basis_denominator,
        basis_indices=basis_indices,
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
    :param axis: Optional keyword argument selecting supercell coordinate axis 0, 1,
        or 2; defaults to ``0``.
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
