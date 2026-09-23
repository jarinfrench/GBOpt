# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Pure dimension planning for GBMaker construction.

Contains the commensurate-repeat search, per-axis strain accommodation (``both``,
``left``, and ``right`` policies), minimum in-plane dimension enforcement,
interaction-distance-driven resizing, boundary-normal x-extent planning, and final
simulation-box dimension calculation extracted from ``GBOpt.GBMaker``.
``plan_periodic_spacing`` and ``plan_dimensions`` are the genuinely public entry
points, promoted from ``GBOpt.gbmaker``'s curated surface: given orientation state
(periodic Miller rows), build configuration, and -- for ``plan_dimensions`` -- the
already-planned x extent, they produce periodic-spacing metadata and a validated
``DimensionPlan`` without mutating any ``GBMaker`` instance. The underscore-prefixed
helpers below them are internals that ``GBOpt.GBMaker`` still imports directly for its
own thin wrappers and remaining unextracted call sites, the same way it already
imports ``gbmaker.orientation``'s helpers. ``_miller_row_norm`` is imported from
``gbmaker.geometry``, the canonical implementation as of R07 (previously a private
copy here; see ``REFACTOR_CLEANUP.md`` for the R05/R06/R07 history of that
duplication). No supercell construction or grain enumeration belongs here.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.gbmaker.geometry import _miller_row_norm
from GBOpt.gbmaker.types import (
    AxisAccommodation,
    DimensionPlan,
    GBMakerConstructionValueError,
    StrainGrainPolicy,
)


def _find_commensurate_pair(
    d1: float,
    d2: float,
    *,
    tol: float = 0.005,
    max_n: int = 50,
) -> tuple[int, int, float, float] | None:
    """Find a small commensurate repeat pair for two one-dimensional periods.

    Searches for integer repeat counts ``n1`` and ``n2`` such that the repeated lengths
    ``n1*d1`` and ``n2*d2`` match within the requested relative mismatch tolerance.
    Candidate pairs are ordered by shared length first, then mismatch, total repeat
    count, ``n1``, and ``n2``.

    The mismatch is computed as::

        abs(n1*d1 - n2*d2) / max(n1*d1, n2*d2)

    :param d1: Period of the first grain along the selected in-plane axis (Angstroms).
    :param d2: Period of the second grain along the selected in-plane axis (Angstroms).
    :param tol: Maximum allowed relative mismatch. Keyword parameter, optional, defaults
        to ``0.005``.
    :param max_n: Maximum repeat count allowed for either grain. Keyword parameter,
        optional, defaults to ``50``.
    :return: ``(n1, n2, n1*d1, n2*d2)`` for the best admissible pair, or ``None`` if no
        admissible pair exists within ``max_n``.
    :raises GBMakerConstructionValueError: If ``d1`` or ``d2`` is not finite and
        positive, if ``tol`` is not finite and non-negative, or if ``max_n`` is not a
        positive integer.
    """
    if isinstance(d1, (bool, np.bool_)):
        raise GBMakerConstructionValueError(
            f"d1 must be a finite positive period; got {d1!r}."
        )
    if isinstance(d2, (bool, np.bool_)):
        raise GBMakerConstructionValueError(
            f"d2 must be a finite positive period; got {d2!r}."
        )
    if isinstance(tol, (bool, np.bool_)):
        raise GBMakerConstructionValueError(
            f"tol must be finite and non-negative; got {tol!r}."
        )
    if isinstance(max_n, (bool, np.bool_)) or not isinstance(max_n, (int, np.integer)):
        raise GBMakerConstructionValueError(
            f"max_n must be a positive integer; got {max_n!r}."
        )

    try:
        d1 = float(d1)
        d2 = float(d2)
        tol = float(tol)
    except (TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(
            "d1 and d2 must be finite positive periods, and tol must be finite and "
            "non-negative."
        ) from exc

    max_n = int(max_n)

    if not math.isfinite(d1) or d1 <= 0.0:
        raise GBMakerConstructionValueError(
            f"d1 must be a finite positive period; got {d1!r}."
        )
    if not math.isfinite(d2) or d2 <= 0.0:
        raise GBMakerConstructionValueError(
            f"d2 must be a finite positive period; got {d2!r}."
        )
    if not math.isfinite(tol) or tol < 0.0:
        raise GBMakerConstructionValueError(
            f"tol must be finite and non-negative; got {tol!r}."
        )
    if max_n < 1:
        raise GBMakerConstructionValueError(
            f"max_n must be a positive integer; got {max_n!r}."
        )

    best: tuple[int, int, float, float] | None = None
    best_key: tuple[float, float, int, int, int] | None = None
    seen: set[tuple[int, int]] = set()

    def consider(n1: int, n2: int) -> None:
        """Evaluate one integer repeat-count pair against the current best pair.

        Operates on the enclosing helper's non-local search state. Out-of-bounds and
        previously checked pairs are ignored. Admissible pairs update ``best`` and
        ``best_key`` when they improve the current candidate under the enclosing
        helper's ordering: shared length, mismatch, total repeat count, ``n1``, then
        ``n2``.

        :param n1: Integer repeat count for the first grain.
        :param n2: Integer repeat count for the second grain.
        :return: ``None``. The enclosing ``best``, ``best_key``, and ``seen`` state
            may be updated.
        """
        nonlocal best, best_key

        if n1 < 1 or n2 < 1 or n1 > max_n or n2 > max_n:
            return

        pair = (n1, n2)
        if pair in seen:
            return
        seen.add(pair)

        l1 = n1 * d1
        l2 = n2 * d2
        size = max(l1, l2)
        mismatch = abs(l1 - l2) / size

        if mismatch <= tol:
            key = (size, mismatch, n1 + n2, n1, n2)
            if best_key is None or key < best_key:
                best = (n1, n2, l1, l2)
                best_key = key

    # n1*d1 ~= n2*d2 is equivalent to n1/n2 ~= d2/d1. Continued-fraction convergents and
    # intermediate convergents give the relevant small rational candidates without
    # scanning all O(max_n**2) repeat pairs.
    ratio = d2 / d1
    x = ratio

    p_prev2, q_prev2 = 0, 1
    p_prev1, q_prev1 = 1, 0

    for _ in range(256):
        a = int(math.floor(x))

        if p_prev1 == 0:
            k_limit_p = max_n if p_prev2 <= max_n else 0
        else:
            k_limit_p = (max_n - p_prev2) // p_prev1

        if q_prev1 == 0:
            k_limit_q = max_n if q_prev2 <= max_n else 0
        else:
            k_limit_q = (max_n - q_prev2) // q_prev1

        k_limit = min(a, k_limit_p, k_limit_q)
        for k in range(1, k_limit + 1):
            consider(k * p_prev1 + p_prev2, k * q_prev1 + q_prev2)

        p_next = a * p_prev1 + p_prev2
        q_next = a * q_prev1 + q_prev2

        frac = x - a
        if frac <= 1e-15 * max(1.0, abs(x)):
            break

        p_prev2, q_prev2 = p_prev1, q_prev1
        p_prev1, q_prev1 = p_next, q_next

        if p_prev1 > max_n or q_prev1 > max_n:
            break

        x = 1.0 / frac
    else:
        raise GBMakerConstructionValueError(
            "Commensurate-period search exceeded the continued-fraction iteration limit"
            f" before completing; got max_n={max_n!r}."
        )

    return best


def _plan_axis_accommodation(
    axis_name: str,
    *,
    a0: float,
    left_periodic_miller_rows: np.ndarray,
    right_periodic_miller_rows: np.ndarray,
    tol: float,
    max_n: int,
    strain_grain: StrainGrainPolicy,
    require_pair: bool,
) -> AxisAccommodation | None:
    """Build commensurate repeat and strain metadata for one in-plane axis.

    Computes the left- and right-grain unstrained periods for the selected in-plane
    axis, searches for a small commensurate integer repeat pair, and returns the
    repeat counts, unstrained lengths, shared box length, scale factors, and residual
    mismatch for that axis.

    The selected axis is mapped to the corresponding periodic Miller row: ``"y"`` uses
    row 1 and ``"z"`` uses row 2. The period for each grain is computed as ``a0 *
    ||row||``.

    If no admissible repeat pair is found, the behavior depends on ``require_pair``.
    Exact construction passes ``True`` and raises ``GBMakerConstructionValueError``.
    Approximate construction passes ``False``, emits a ``UserWarning``, and returns
    ``None`` so the legacy repeat-factor box can be used.

    :param axis_name: In-plane axis name, either ``"y"`` or ``"z"``.
    :param a0: Crystal lattice parameter (Angstroms). Keyword parameter, required.
    :param left_periodic_miller_rows: Left-grain 3 by 3 integer periodic Miller-row
        matrix. Keyword parameter, required.
    :param right_periodic_miller_rows: Right-grain counterpart of
        ``left_periodic_miller_rows``. Keyword parameter, required.
    :param tol: Maximum allowed relative mismatch. Keyword parameter, required.
    :param max_n: Maximum repeat count allowed for either grain. Keyword parameter,
        required.
    :param strain_grain: In-plane strain policy, ``"both"``, ``"left"``, or ``"right"``.
        Keyword parameter, required.
    :param require_pair: Whether failure to find a commensurate pair is fatal. Keyword
        parameter, required.
    :return: Strain-accommodation metadata for the selected axis, or ``None`` when no
        pair is found and ``require_pair`` is ``False``.
    :raises GBMakerConstructionValueError: If ``axis_name`` is not ``"y"`` or ``"z"``,
        if ``require_pair`` is not boolean, if the Miller rows are invalid, if the
        commensurate-pair search receives invalid parameters, if no pair is found when
        ``require_pair`` is ``True``, or if ``strain_grain`` is invalid.
    """
    if axis_name not in {"y", "z"}:
        raise GBMakerConstructionValueError(
            f"axis_name must be 'y' or 'z'; got {axis_name!r}."
        )

    if not isinstance(require_pair, bool):
        raise GBMakerConstructionValueError(
            f"require_pair must be boolean; got {require_pair!r}."
        )

    axis_row = 1 if axis_name == "y" else 2
    d1 = a0 * _miller_row_norm(left_periodic_miller_rows[axis_row])
    d2 = a0 * _miller_row_norm(right_periodic_miller_rows[axis_row])

    result = _find_commensurate_pair(d1, d2, tol=tol, max_n=max_n)

    if result is None:
        residual = abs(d1 - d2) / max(d1, d2)
        msg = (
            f"No commensurate {axis_name} pair found within "
            f"mismatch_max_cells={max_n} for mismatch_tol={tol}. Residual "
            f"one-period mismatch is {residual:.4%}."
        )

        if require_pair:
            raise GBMakerConstructionValueError(
                f"{msg} Exact strain accommodation cannot build this boundary "
                "within the requested tolerance."
            )

        warnings.warn(
            f"{msg} Falling back to max(d_left, d_right) * repeat_factor.",
            UserWarning,
            stacklevel=5,
        )
        return None

    n1, n2, l1, l2 = result

    if strain_grain == "both":
        box_length = (l1 + l2) / 2.0
    elif strain_grain == "left":
        box_length = l2
    elif strain_grain == "right":
        box_length = l1
    else:
        raise GBMakerConstructionValueError(f"Invalid strain_grain={strain_grain!r}.")

    mismatch = abs(l1 - l2) / max(l1, l2)

    return AxisAccommodation(
        left_repeats=n1,
        right_repeats=n2,
        left_unstrained_length=l1,
        right_unstrained_length=l2,
        box_length=box_length,
        left_scale=box_length / l1,
        right_scale=box_length / l2,
        mismatch=mismatch,
    )


def _plan_inplane_axis_dim(
    axis_name: str,
    dim: float,
    spacing: float,
    epsilon: float,
) -> tuple[float, int]:
    """Return a validated in-plane box length and its synchronized repeat factor.

    The repeat factor is synchronized as the smallest positive integer whose
    unstrained spacing-based box length is at least ``dim``.

    :param axis_name: In-plane axis name, either ``"y"`` or ``"z"``.
    :param dim: New box length for this axis (Angstroms).
    :param spacing: Periodic spacing for this axis (Angstroms).
    :param epsilon: Numerical tolerance used for the repeat-factor ceiling.
    :return: ``(dim, repeat_factor)``.
    :raises GBMakerConstructionValueError: If ``axis_name`` is not ``"y"`` or ``"z"``,
        if ``dim`` is not finite and positive, or if ``spacing`` is not finite and
        positive.
    """
    if axis_name not in {"y", "z"}:
        raise GBMakerConstructionValueError(
            f"axis_name must be 'y' or 'z'; got {axis_name!r}."
        )

    try:
        dim = float(dim)
    except (TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(
            f"{axis_name}_dim must be finite and positive; got {dim!r}."
        ) from exc

    if not math.isfinite(dim) or dim <= 0.0:
        raise GBMakerConstructionValueError(
            f"{axis_name}_dim must be finite and positive; got {dim!r}."
        )

    if not math.isfinite(spacing) or spacing <= 0.0:
        raise GBMakerConstructionValueError(
            f"{axis_name}-spacing must be finite and positive; got {spacing!r}."
        )

    repeat_factor = max(1, int(math.ceil(dim / spacing - epsilon)))
    return dim, repeat_factor


def _plan_inplane_minimum_dim(
    axis_name: str,
    *,
    current_dim: float,
    repeat_factor: int,
    spacing: float,
    epsilon: float,
    cutoff: float,
    accommodation: AxisAccommodation | None,
) -> tuple[float, int, AxisAccommodation | None]:
    """Resize one in-plane axis to satisfy a minimum box-length cutoff.

    If ``current_dim`` already satisfies ``cutoff``, nothing changes. When
    ``accommodation`` is supplied, the commensurate repeat pair is multiplied by a
    positive integer resize factor. Otherwise, the spacing-based repeat factor is
    increased.

    :param axis_name: In-plane axis name, either ``"y"`` or ``"z"``.
    :param current_dim: Current box length for this axis (Angstroms). Keyword
        parameter, required.
    :param repeat_factor: Current repeat factor for this axis. Keyword parameter,
        required.
    :param spacing: Periodic spacing for this axis (Angstroms). Keyword parameter,
        required.
    :param epsilon: Numerical tolerance used for ceiling computations. Keyword
        parameter, required.
    :param cutoff: Minimum required box length for the axis (Angstroms). Keyword
        parameter, required.
    :param accommodation: Active strain accommodation for this axis, or ``None``.
        Keyword parameter, required.
    :return: ``(dim, repeat_factor, accommodation)``, unchanged when ``current_dim``
        already satisfies ``cutoff``.
    :raises GBMakerConstructionValueError: If ``axis_name`` is not ``"y"`` or ``"z"``,
        if ``cutoff`` is not finite and non-negative, or if ``spacing`` is not finite
        and positive when no accommodation is active.
    """
    if axis_name not in {"y", "z"}:
        raise GBMakerConstructionValueError(
            f"axis_name must be 'y' or 'z'; got {axis_name!r}."
        )

    try:
        cutoff = float(cutoff)
    except (TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(
            f"cutoff must be finite and non-negative; got {cutoff!r}."
        ) from exc

    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise GBMakerConstructionValueError(
            f"cutoff must be finite and non-negative; got {cutoff!r}."
        )

    if current_dim >= cutoff:
        return current_dim, repeat_factor, accommodation

    if accommodation is not None:
        resize_factor = max(1, int(math.ceil(cutoff / accommodation.box_length - epsilon)))
        accommodation = accommodation.resized(resize_factor)
        dim, repeat_factor = _plan_inplane_axis_dim(
            axis_name, accommodation.box_length, spacing, epsilon
        )

        warnings.warn(
            f"Commensurate repeat pair in {axis_name} multiplied by "
            f"{resize_factor} to satisfy the minimum in-plane dimension "
            f"cutoff of {cutoff:.6g} A.",
            UserWarning,
            stacklevel=4,
        )
        return dim, repeat_factor, accommodation

    if not math.isfinite(spacing) or spacing <= 0.0:
        raise GBMakerConstructionValueError(
            f"{axis_name}-spacing must be finite and positive; got {spacing!r}."
        )

    repeat = max(1, int(math.ceil(cutoff / spacing - epsilon)))
    dim, repeat_factor = _plan_inplane_axis_dim(axis_name, repeat * spacing, spacing, epsilon)

    warnings.warn(
        f"Repeat factor in {axis_name} modified to {repeat} to satisfy the "
        f"minimum in-plane dimension cutoff of {cutoff:.6g} A.",
        UserWarning,
        stacklevel=4,
    )
    return dim, repeat_factor, accommodation


def _plan_box_dims(
    x_dim: float,
    vacuum_thickness: float,
    y_dim: float,
    z_dim: float,
) -> np.ndarray:
    """Return the 3 by 2 array of box bounds.

    :param x_dim: Combined left- and right-grain x extent (Angstroms).
    :param vacuum_thickness: Vacuum thickness applied along x (Angstroms).
    :param y_dim: Box length along y (Angstroms).
    :param z_dim: Box length along z (Angstroms).
    :return: 3 by 2 array containing xlo, xhi, ylo, yhi, zlo, and zhi.
    """
    return np.array(
        [
            [0, x_dim + 2 * vacuum_thickness],
            [0, y_dim],
            [0, z_dim],
        ]
    )


def plan_periodic_spacing(
    *,
    a0: float,
    left_periodic_miller_rows: np.ndarray,
    right_periodic_miller_rows: np.ndarray,
    x_dim_min: float,
    epsilon: float,
    inplane_periodic: tuple[bool, bool],
    threshold: float,
    legacy_periodicity_heuristic: bool,
) -> tuple[dict[str, float | dict[str, float]], float, float, float]:
    """Plan per-axis periodic spacing and the equalized boundary-normal x extent.

    The periodic distance in each direction is the lattice parameter multiplied by the
    norm of the Miller indices in that direction (the usual interplanar-spacing
    formula, ``d = a / sqrt(h**2+k**2+l**2)``, simplified as described in
    ``GBOpt.GBMaker``'s historical implementation). The x extents for the left and
    right grains are equalized to the smaller of the two grains' minimum-satisfying
    multiples so that ``x_dim_min`` is met on both sides with the same shared boundary
    plane.

    In-plane periodicity was already resolved upstream by
    ``GBOpt.gbmaker.orientation.resolve_orientation``; this only reapplies the
    resulting flags to the returned y/z spacing values used for box-dimension
    planning. When ``legacy_periodicity_heuristic`` is ``True`` (a ``None`` embedding
    or one sourced from a legacy five-DOF spec), a non-periodic axis's spacing is
    replaced outright by ``threshold``. Otherwise, a non-periodic axis's spacing is
    clamped to at most ``threshold``.

    :param a0: Crystal lattice parameter (Angstroms). Keyword parameter, required.
    :param left_periodic_miller_rows: Left-grain 3 by 3 integer periodic Miller-row
        matrix. Keyword parameter, required.
    :param right_periodic_miller_rows: Right-grain counterpart of
        ``left_periodic_miller_rows``. Keyword parameter, required.
    :param x_dim_min: Minimum size of one grain along x (Angstroms). Keyword
        parameter, required.
    :param epsilon: Numerical tolerance used for ceiling computations. Keyword
        parameter, required.
    :param inplane_periodic: Per-axis in-plane periodicity flags ``(y, z)``. Keyword
        parameter, required.
    :param threshold: Maximum periodic spacing (Angstroms) before an in-plane axis is
        treated as non-periodic. Keyword parameter, required.
    :param legacy_periodicity_heuristic: Whether the legacy/five-DOF non-periodic
        spacing replacement applies, rather than the coherent-embedding clamp. Keyword
        parameter, required.
    :return: ``(spacing, left_x, right_x, x_dim)`` where ``spacing`` is a dict keyed
        by ``"x"`` (itself a dict with ``"left"``/``"right"`` entries) and ``"y"``/
        ``"z"`` (each a float), matching ``GBMaker.spacing``'s established shape.
    :raises GBMakerConstructionValueError: If a periodic Miller row cannot be resolved
        to a nonzero three-component integer row.
    """
    spacing_left = {
        axis: a0 * _miller_row_norm(vec)
        for axis, vec in zip(("x", "y", "z"), left_periodic_miller_rows)
    }
    spacing_right = {
        axis: a0 * _miller_row_norm(vec)
        for axis, vec in zip(("x", "y", "z"), right_periodic_miller_rows)
    }

    left_x = math.ceil(x_dim_min / spacing_left["x"]) * spacing_left["x"]
    right_x = math.ceil(x_dim_min / spacing_right["x"]) * spacing_right["x"]
    target = max(left_x, right_x)
    left_x = math.ceil(target / spacing_left["x"] - epsilon) * spacing_left["x"]
    right_x = math.ceil(target / spacing_right["x"] - epsilon) * spacing_right["x"]
    x_dim = left_x + right_x

    spacing: dict[str, float | dict[str, float]] = {
        "x": {"left": spacing_left["x"], "right": spacing_right["x"]},
    }
    spacing.update(
        {axis: max(spacing_left[axis], spacing_right[axis]) for axis in ("y", "z")}
    )

    if legacy_periodicity_heuristic:
        for axis, is_periodic in zip(("y", "z"), inplane_periodic):
            if not is_periodic:
                spacing[axis] = threshold
    else:
        if not all(inplane_periodic):
            for axis in ("y", "z"):
                spacing[axis] = min(spacing[axis], threshold)

    return spacing, left_x, right_x, x_dim


def plan_dimensions(
    *,
    a0: float,
    left_periodic_miller_rows: np.ndarray,
    right_periodic_miller_rows: np.ndarray,
    spacing_y: float,
    spacing_z: float,
    repeat_factor: tuple[int, int],
    mismatch_tol: float | None,
    mismatch_max_cells: int,
    strain_grain: StrainGrainPolicy,
    require_exact_pair: bool,
    interaction_distance: float,
    x_dim: float,
    vacuum_thickness: float,
    normal_topology: BoundaryNormalTopology,
    epsilon: float,
) -> tuple[DimensionPlan, tuple[int, int]]:
    """Plan in-plane box dimensions, strain accommodation, and final box bounds.

    Computes the nominal y and z box lengths from ``repeat_factor`` and the supplied
    periodic spacing, searches for commensurate-repeat strain accommodation on each
    in-plane axis when ``mismatch_tol`` is not ``None``, enforces the
    interaction-distance-driven minimum in-plane dimension (``2 *
    interaction_distance``) on each axis, and assembles the resulting box dimensions.

    :param a0: Crystal lattice parameter (Angstroms). Keyword parameter, required.
    :param left_periodic_miller_rows: Left-grain 3 by 3 integer periodic Miller-row
        matrix. Keyword parameter, required.
    :param right_periodic_miller_rows: Right-grain counterpart of
        ``left_periodic_miller_rows``. Keyword parameter, required.
    :param spacing_y: Periodic spacing along y (Angstroms). Keyword parameter,
        required.
    :param spacing_z: Periodic spacing along z (Angstroms). Keyword parameter,
        required.
    :param repeat_factor: Current in-plane repeat counts ``(y, z)``. Keyword
        parameter, required.
    :param mismatch_tol: Maximum relative in-plane mismatch permitted by the
        commensurate-repeat search, or ``None`` to disable mismatch accommodation.
        Keyword parameter, required.
    :param mismatch_max_cells: Maximum repeat count searched per axis when mismatch
        accommodation is active. Keyword parameter, required.
    :param strain_grain: In-plane strain policy used when mismatch accommodation is
        active. Keyword parameter, required.
    :param require_exact_pair: Whether failure to find a commensurate pair is fatal
        (the exact construction path). Keyword parameter, required.
    :param interaction_distance: Maximum atom interaction distance (Angstroms).
        Keyword parameter, required.
    :param x_dim: Combined left- and right-grain x extent, already equalized by
        ``plan_periodic_spacing`` (Angstroms). Keyword parameter, required.
    :param vacuum_thickness: Vacuum thickness applied along x (Angstroms). Keyword
        parameter, required.
    :param normal_topology: Physical topology along the grain-boundary normal.
        Keyword parameter, required.
    :param epsilon: Numerical tolerance used for ceiling computations. Keyword
        parameter, required.
    :return: ``(plan, repeat_factor)`` -- the validated ``DimensionPlan`` and the
        in-plane repeat-factor pair, synchronized with any commensurate-repeat strain
        accommodation or minimum-dimension resizing that occurred while planning.
    :raises GBMakerConstructionValueError: If a periodic Miller row cannot be
        resolved, if no commensurate pair is found and ``require_exact_pair`` is
        ``True``, or if any planned dimension fails ``DimensionPlan`` validation.
    """
    repeat_y, repeat_z = repeat_factor
    accommodation: dict[str, AxisAccommodation] = {}

    y_dim = repeat_y * spacing_y
    z_dim = repeat_z * spacing_z

    if mismatch_tol is not None:
        for axis_name in ("y", "z"):
            axis_accommodation = _plan_axis_accommodation(
                axis_name,
                a0=a0,
                left_periodic_miller_rows=left_periodic_miller_rows,
                right_periodic_miller_rows=right_periodic_miller_rows,
                tol=mismatch_tol,
                max_n=mismatch_max_cells,
                strain_grain=strain_grain,
                require_pair=require_exact_pair,
            )
            if axis_accommodation is not None:
                accommodation[axis_name] = axis_accommodation
                spacing = spacing_y if axis_name == "y" else spacing_z
                dim, repeat = _plan_inplane_axis_dim(
                    axis_name, axis_accommodation.box_length, spacing, epsilon
                )
                if axis_name == "y":
                    y_dim, repeat_y = dim, repeat
                else:
                    z_dim, repeat_z = dim, repeat

    cutoff = 2 * interaction_distance
    for axis_name in ("y", "z"):
        current_dim = y_dim if axis_name == "y" else z_dim
        current_repeat = repeat_y if axis_name == "y" else repeat_z
        spacing = spacing_y if axis_name == "y" else spacing_z
        dim, repeat, axis_accommodation = _plan_inplane_minimum_dim(
            axis_name,
            current_dim=current_dim,
            repeat_factor=current_repeat,
            spacing=spacing,
            epsilon=epsilon,
            cutoff=cutoff,
            accommodation=accommodation.get(axis_name),
        )
        if axis_accommodation is not None:
            accommodation[axis_name] = axis_accommodation
        if axis_name == "y":
            y_dim, repeat_y = dim, repeat
        else:
            z_dim, repeat_z = dim, repeat

    box_dims = _plan_box_dims(x_dim, vacuum_thickness, y_dim, z_dim)

    plan = DimensionPlan(
        box_dims=box_dims,
        vacuum_thickness=vacuum_thickness,
        normal_topology=normal_topology,
        periodic_spacing={"y": spacing_y, "z": spacing_z},
        accommodation=accommodation,
    )
    return plan, (repeat_y, repeat_z)
