# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Pure orientation and in-plane periodicity resolution for GBMaker construction.

Contains the misorientation Euler-angle decomposition, integer-row reduction and
angular-error arithmetic, floating-point rotation-row/matrix integer approximation,
per-grain rotation-matrix and periodic-Miller-row assignment, boundary-normal x-period
calculation, and in-plane periodicity determination extracted from ``GBOpt.GBMaker``.
``resolve_orientation`` is the only genuinely public entry point, promoted from
``GBOpt.gbmaker``'s curated surface: given a misorientation array and an optional
resolved ``BoundaryEmbedding``, it produces an ``OrientationState`` without mutating
any ``GBMaker`` instance. The underscore-prefixed helpers below it are internals that
``GBOpt.GBMaker`` still imports directly for its own thin wrappers and remaining
unextracted call sites, the same way it already imports ``gbmaker.config``'s
``_validate_scalar``. Exact P/Q/CSL-derived rows carried by an exact embedding are
never routed through the float re-approximation path; only the legacy Euler-angle
path and non-exact embeddings use integer-row approximation. No supercell
construction, strain accommodation, or final box-dimension planning belongs here.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from GBOpt.BoundarySpec import BoundaryEmbedding
from GBOpt.gbmaker.types import GBMakerConstructionValueError, OrientationState


def _reduce_integer_row(row: np.ndarray) -> np.ndarray:
    """Reduce an integer row by its GCD.

    :param row: Integer row vector.
    :return: GCD-reduced integer row vector.
    """
    reduced = np.asarray(row, dtype=int).copy()
    non_zero = np.abs(reduced[reduced != 0])
    if not non_zero.size:
        return reduced
    gcd = np.gcd.reduce(non_zero)
    if gcd > 1:
        reduced //= gcd
    return reduced


def _row_angle_error_deg(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute the angular error in degrees between two vectors.

    :param reference: Reference float vector.
    :param candidate: Candidate integer vector.
    :return: Angle between the two vectors in degrees.
    """
    ref_norm = np.linalg.norm(reference)
    cand_norm = np.linalg.norm(candidate)
    if np.isclose(ref_norm, 0) or np.isclose(cand_norm, 0):
        return 180.0
    cosine = np.dot(reference, candidate) / (ref_norm * cand_norm)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _approximate_rotation_row_as_int(
    row: np.ndarray,
    angle_tol_deg: float = 0.5,
    max_scale: int = 10000,
) -> np.ndarray:
    """Approximate one floating-point rotation row by an integer Miller row.

    Searches integer scale factors ``k`` from one through ``max_scale`` and rounds
    ``k * row`` to the nearest integer row. Each candidate is primitive reduced, and
    the first candidate within ``angle_tol_deg`` is returned after retaining the
    smallest angular-error candidate encountered so far.

    :param row: Floating-point row vector to approximate.
    :param angle_tol_deg: Maximum allowed angular error in degrees. Keyword
        parameter, optional, defaults to ``0.5``.
    :param max_scale: Maximum integer scale factor to try. Keyword parameter,
        optional, defaults to ``10000``.
    :return: Primitive integer row approximating ``row``.
    """
    row = np.asarray(row, dtype=np.float64)
    best: np.ndarray | None = None
    best_err = 180.0
    batch_size = 1000

    for k_start in range(1, max_scale + 1, batch_size):
        k_end = min(k_start + batch_size, max_scale + 1)

        for k in range(k_start, k_end):
            candidate = _reduce_integer_row(np.round(row * k).astype(int))
            err = _row_angle_error_deg(row, candidate)

            if best is None or err < best_err or (
                err == best_err
                and np.linalg.norm(candidate) < np.linalg.norm(best)
            ):
                best_err = err
                best = candidate

            if best_err <= angle_tol_deg:
                break

        if best_err <= angle_tol_deg:
            break

    return best if best is not None else np.round(row).astype(int)


def _approximate_rotation_matrix_as_int(
    m: np.ndarray, precision: float = 5
) -> np.ndarray:
    """Approximate a rotation matrix in integer format given the original matrix and
    the desired precision.

    :param m: The matrix to approximate.
    :param precision: Decimal precision to use during calculations, defaults to 5.
    :return: Integer approximation of the rotation matrix ``m``.
    """
    max_scale = max(1000, 10 ** max(int(precision) - 1, 0))
    return np.vstack(
        [
            _approximate_rotation_row_as_int(row, angle_tol_deg=0.5, max_scale=max_scale)
            for row in np.asarray(m, dtype=np.float64)
        ]
    ).astype(int)


def _miller_row_norm(row: Sequence[object] | np.ndarray) -> float:
    """Return the Euclidean norm of a nonzero integer Miller-index row.

    Computes ``sqrt(h*h + k*k + l*l)`` using Python ``int`` arithmetic for the squared
    norm. This avoids fixed-width NumPy integer overflow and avoids object-dtype NumPy
    ufuncs.

    Duplicates ``GBOpt.GBMaker._miller_row_norm``: that copy still has call sites
    outside the orientation stage (exact-repeat and box-basis geometry, R07/R08
    territory), and this module cannot import it back from ``GBOpt.GBMaker`` without a
    circular import. See ``REFACTOR_CLEANUP.md`` for the planned consolidation.

    :param row: Nonzero integer Miller-index row ``(h, k, l)``.
    :return: Euclidean norm of the Miller-index row.
    :raises GBMakerConstructionValueError: If ``row`` is not a three-component integer
        row or if the row is zero.
    """
    values = tuple(row)
    if len(values) != 3:
        raise GBMakerConstructionValueError(
            f"Miller-index row must have exactly three components; got {values!r}."
        )

    integers: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise GBMakerConstructionValueError(
                f"Miller-index row components must be integers; got {values!r}."
            )
        integers.append(int(value))

    squared_norm = sum(value * value for value in integers)
    if squared_norm == 0:
        raise GBMakerConstructionValueError("Miller-index row must be nonzero.")

    return math.sqrt(squared_norm)


def _x_period(periodic_miller_rows: np.ndarray, a0: float) -> float:
    """Return one full x-period length for a grain.

    The x-period is the distance between equivalent crystallographic repeats along
    the boundary-normal direction. It is computed from the first integer periodic
    Miller row as ``a0 * ||periodic_miller_rows[0]||``.

    :param periodic_miller_rows: Three-row integer periodic Miller matrix for one
        grain. Row 0 defines the boundary-normal x-period.
    :param a0: Crystal lattice parameter (Angstroms).
    :return: Boundary-normal x-period in Angstroms.
    :raises GBMakerConstructionValueError: If row 0 is not a nonzero three-component
        integer Miller row.
    """
    return a0 * _miller_row_norm(periodic_miller_rows[0])


def _decompose_misorientation(
    misorientation: np.ndarray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Decompose a five-element misorientation/inclination array into rotation matrices.

    :param misorientation: Five-element array. Elements 0-2 are ZXZ misorientation
        Euler angles (radians); elements 3-4 are the inclination rotations about y and
        z (radians).
    :return: Four-tuple ``(misorientation_angles, inclination_angles, R_mis, R_incl)``.
    """
    misorientation = np.asarray(misorientation, dtype=np.float64)
    mis_angles = misorientation[:3]
    incl_angles = misorientation[3:]
    R_mis = Rotation.from_euler("ZXZ", mis_angles).as_matrix()
    R_incl = (
        Rotation.from_euler("z", misorientation[4])
        * Rotation.from_euler("y", misorientation[3])
    ).as_matrix()
    return mis_angles, incl_angles, R_mis, R_incl


def resolve_orientation(
    misorientation: np.ndarray,
    *,
    embedding: BoundaryEmbedding | None,
    a0: float,
    threshold: float | None = None,
) -> OrientationState:
    """Resolve per-grain rotations, periodic Miller rows, and in-plane periodicity.

    When ``embedding`` carries exact integer P/Q rows (``embedding.exact`` is True and
    ``embedding.P`` is not None), those rows are used directly as the periodic Miller
    rows; they are never routed through the integer-approximation path. Otherwise, the
    corresponding rotation matrix (``embedding.R_left``/``R_right``, or the legacy-path
    rotation composed from the misorientation/inclination rotation matrices when
    ``embedding`` is ``None``) is integer-approximated.

    In-plane periodicity is taken directly from ``embedding.coherent`` for any
    embedding other than one sourced from a legacy five-DOF spec. For a ``None``
    embedding or a five-DOF-sourced embedding, periodicity is instead determined per
    axis by comparing ``a0 * max(left, right periodic-row norm)`` against
    ``threshold``, warning for each axis that exceeds it -- matching the legacy
    heuristic this stage replaces.

    :param misorientation: Five-element misorientation/inclination array (see
        ``_decompose_misorientation``).
    :param embedding: Boundary embedding providing left/right rotations and, for exact
        construction, integer P/Q periodic Miller rows. Keyword parameter, required.
        ``None`` selects the legacy Euler-angle rotation path.
    :param a0: Crystal lattice parameter (Angstroms). Keyword parameter, required.
    :param threshold: Maximum periodic spacing (Angstroms) before an in-plane axis is
        treated as non-periodic under the legacy/five-DOF periodicity heuristic.
        Keyword parameter, optional, defaults to ``a0 * 15``.
    :return: Resolved orientation state. Does not mutate ``embedding`` or any
        ``GBMaker`` instance.
    :raises GBMakerConstructionValueError: If a periodic Miller row cannot be resolved
        to a nonzero three-component integer row.
    """
    if threshold is None:
        threshold = a0 * 15

    mis_angles, incl_angles, R_mis, R_incl = _decompose_misorientation(misorientation)

    if embedding is not None:
        R_left = embedding.R_left
        R_right = embedding.R_right
        if embedding.exact and embedding.P is not None:
            left_rows = np.asarray(embedding.P, dtype=object)
            right_rows = np.asarray(embedding.Q, dtype=object)
        else:
            left_rows = _approximate_rotation_matrix_as_int(R_left).astype(object)
            right_rows = _approximate_rotation_matrix_as_int(R_right).astype(object)
    else:
        R_left = R_incl
        R_right = np.dot(R_incl, R_mis)
        left_rows = _approximate_rotation_matrix_as_int(R_left).astype(object)
        right_rows = _approximate_rotation_matrix_as_int(R_right).astype(object)

    if embedding is not None and embedding.source != "five_dof":
        coherent = embedding.coherent
        inplane_periodic = (coherent, coherent)
    else:
        flags: list[bool] = []
        for axis, left_row, right_row in (
            ("y", left_rows[1], right_rows[1]),
            ("z", left_rows[2], right_rows[2]),
        ):
            spacing = max(a0 * _miller_row_norm(left_row), a0 * _miller_row_norm(right_row))
            is_periodic = spacing <= threshold
            flags.append(is_periodic)
            if not is_periodic:
                warnings.warn(
                    f"Required {axis}-spacing {spacing:.4f} A exceeds threshold "
                    f"{threshold:.4f} A; boundary is non-periodic along {axis}."
                )
        inplane_periodic = (flags[0], flags[1])

    return OrientationState(
        embedding=embedding,
        misorientation=mis_angles,
        inclination=incl_angles,
        R_mis=R_mis,
        R_incl=R_incl,
        R_left=R_left,
        R_right=R_right,
        left_periodic_miller_rows=left_rows,
        right_periodic_miller_rows=right_rows,
        inplane_periodic=inplane_periodic,
    )
