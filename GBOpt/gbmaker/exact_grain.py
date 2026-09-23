# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Exact decorated-site grain construction extracted from ``GBOpt.GBMaker``.

Contains ``build_exact_grain`` and its supercell-repeat-count helper
``_exact_grain_repeats``, extracted from ``GBOpt.GBMaker.__build_exact_grain`` and
``GBOpt.GBMaker.__exact_grain_repeats``. Both consume a ``GrainBuildRequest`` and
either return a ``GrainBuildResult`` or raise ``GBMakerConstructionValueError`` --
neither accepts a ``GBMaker`` instance. ``GBOpt.GBMaker`` keeps a thin wrapper around
``build_exact_grain`` that supplies the request and translates
``GBMakerConstructionValueError`` back to the established ``GBMakerValueError``, the
same pattern used for every other ``gbmaker`` extraction boundary.

Enumerates every rational decorated basis site in the repeated integer supercell
before any floating-point conversion; the exact-to-floating conversion happens exactly
once, when exact conventional-cell coordinate numerators are scaled into Cartesian
crystal positions. Existing rotation, in-plane strain, placement, and periodic
in-plane wrapping are applied without Cartesian membership clipping or complete-origin
deletion, matching issue #69's non-goal that the exact path must never reintroduce
float membership clipping or layer deletion.

This module depends on ``gbmaker.geometry`` and ``gbmaker.types`` only, and on the
existing ``GBOpt.gbmaker_supercell`` exact-enumeration module; it does not import
``gbmaker.approximate_grain`` or any other ``gbmaker`` sibling.
"""

from __future__ import annotations

import numpy as np

from GBOpt.gbmaker.geometry import (
    _cartesian_from_box_coordinates,
    _miller_row_norm,
    _reduced_box_coordinates,
    _reduced_coordinate_tolerance,
    _selection_basis_vectors,
    wrap_reduced_coordinate,
)
from GBOpt.gbmaker.types import (
    GBMakerConstructionValueError,
    GrainBuildRequest,
    GrainBuildResult,
)
from GBOpt.gbmaker_supercell import build_supercell_matrix, enumerate_supercell_sites


def _exact_grain_repeats(
    request: GrainBuildRequest,
) -> tuple[np.ndarray, int, int, int]:
    """Compute exact-path supercell repeat counts for one grain.

    Builds the integer supercell matrix for ``request.periodic_matrix`` and derives
    the number of repeated supercell periods needed along the boundary-normal x
    direction and the two in-plane directions. The x repeat count is derived from
    ``request.x_length``. The y and z repeat counts are derived from
    ``request.inplane_box_lengths``, unless ``request.y_repeats``/``request.z_repeats``
    supplied explicit mismatch-accommodation repeat counts for that axis.

    :param request: Grain build request. ``request.grain_side`` selects nothing here
        directly; it is only used in diagnostic messages upstream.
    :return: ``(supercell, repeat_x, repeat_y, repeat_z)`` where ``supercell`` is the
        validated integer supercell matrix and the remaining values are positive
        Python integers.
    :raises GBMakerConstructionValueError: If the supercell matrix cannot be built, or
        if the x, y, or z box length is not commensurate with this grain's
        corresponding period.
    """
    try:
        supercell = build_supercell_matrix(request.periodic_matrix)
    except ValueError as exc:
        raise GBMakerConstructionValueError(str(exc)) from exc

    a0 = request.material.a0
    x_period = a0 * _miller_row_norm(supercell[0])
    y_period = a0 * _miller_row_norm(supercell[1])
    z_period = a0 * _miller_row_norm(supercell[2])

    tol = 1e-6

    def commensurate_repeat(box_length: float, period: float, axis_name: str) -> int:
        """Return a positive repeat count for one exact box/period pair.

        :param box_length: Box length along this axis (Angstroms).
        :param period: Grain period along this axis (Angstroms).
        :param axis_name: Axis label used in error messages.
        :return: Positive integer repeat count.
        :raises GBMakerConstructionValueError: If ``box_length`` is not an integer
            multiple of ``period`` within the repeat-count tolerance.
        """
        repeat_raw = box_length / period
        repeat = int(round(repeat_raw))

        if abs(repeat_raw - repeat) > tol:
            raise GBMakerConstructionValueError(
                f"Exact construction requires the {axis_name} box "
                f"({box_length:.6f}A) to be an integer multiple of this grain's "
                f"{axis_name}-period ({period:.6f} A), but got repeat_{axis_name} "
                f"= {repeat_raw:.8f}. Use mode='approximate' or adjust "
                "repeat_factor until both grains' periods divide the shared box "
                "exactly. See the commensurability note in from_boundary_spec for "
                "details."
            )

        if repeat <= 0:
            raise GBMakerConstructionValueError(
                f"Exact construction requires positive {axis_name} repeats; got "
                f"{repeat}."
            )

        return repeat

    repeat_x = commensurate_repeat(request.x_length, x_period, "x")

    y_dim, z_dim = request.inplane_box_lengths
    repeat_y = (
        request.y_repeats
        if request.y_repeats is not None
        else commensurate_repeat(y_dim, y_period, "y")
    )
    repeat_z = (
        request.z_repeats
        if request.z_repeats is not None
        else commensurate_repeat(z_dim, z_period, "z")
    )

    return supercell, repeat_x, repeat_y, repeat_z


def build_exact_grain(request: GrainBuildRequest) -> GrainBuildResult:
    """Build one grain from exact decorated repeated-supercell sites.

    Enumerates every rational decorated basis site in the repeated integer supercell
    before any floating-point conversion. Exact conventional coordinates are
    reconstructed from exact site metadata and converted once to Cartesian crystal
    positions. Existing rotation, in-plane strain, placement, and periodic in-plane
    wrapping are then applied without Cartesian membership clipping or complete-origin
    deletion.

    ``origin_ids`` in the returned result assigns one origin id per contiguous
    ``basis_size``-sized block of enumerated sites, following the established
    quotient-lattice origin order documented on
    ``GBOpt.gbmaker_supercell.enumerate_supercell_sites``. The exact path never
    filters, clips, or deduplicates by origin the way the approximate path does, so
    this metadata is carried through for contract uniformity rather than consumed
    downstream.

    :param request: Grain build request with ``request.exact`` expected ``True``.
    :return: Grain build result for the complete decorated grain.
    :raises GBMakerConstructionValueError: If rational basis metadata is unavailable,
        exact enumeration violates a population invariant, or final coordinates are
        non-finite or outside the intended grain box.
    """
    unit_cell = request.material.unit_cell
    rational_basis = unit_cell.rational_basis if unit_cell is not None else None
    if rational_basis is None:
        raise GBMakerConstructionValueError(
            "Exact grain generation requires UnitCell.rational_basis; arbitrary "
            "floating-point basis coordinates are not accepted."
        )

    supercell, repeat_x, repeat_y, repeat_z = _exact_grain_repeats(request)

    try:
        sites = enumerate_supercell_sites(
            supercell,
            repeat_x,
            repeat_y,
            repeat_z,
            rational_basis=rational_basis,
        )
    except ValueError as exc:
        raise GBMakerConstructionValueError(
            f"Exact decorated-site enumeration failed for the {request.grain_side} "
            f"grain: {exc}"
        ) from exc

    basis_size = sites.basis_size
    basis_indices = sites.basis_indices
    expected_site_count = sites.site_count

    structured_basis = unit_cell.asarray()
    structured_names = tuple(str(name) for name in structured_basis["name"])

    if len(structured_basis) != basis_size:
        raise GBMakerConstructionValueError(
            "UnitCell rational-basis and structured-basis sizes disagree: "
            f"{basis_size} exact sites versus {len(structured_basis)} structured "
            "atoms."
        )

    if structured_names != rational_basis.names:
        raise GBMakerConstructionValueError(
            "UnitCell rational-basis and structured-basis species order disagree: "
            f"exact={rational_basis.names!r}, structured={structured_names!r}."
        )

    atoms = np.empty(expected_site_count, dtype=structured_basis.dtype)
    atoms["name"] = structured_basis["name"][basis_indices]

    # This is the sole exact-to-floating conversion in grain construction. Exact site
    # metadata stores canonical repeated-supercell coordinates; multiplying by S
    # reconstructs exact conventional-cell coordinate numerators.
    conventional_numerators = sites.coordinate_numerators @ sites.supercell_matrix

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            crystal_positions = np.asarray(conventional_numerators, dtype=np.float64)
            crystal_positions *= request.material.a0 / sites.coordinate_denominator

        rotation = np.asarray(request.rotation, dtype=np.float64)
        inplane_orientation_rows = np.asarray(
            request.periodic_matrix[1:], dtype=np.float64
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(
            "Exact decorated-site coordinates or orientation rows cannot be "
            f"represented as finite Cartesian values for the {request.grain_side} "
            "grain."
        ) from exc

    if (
        not np.all(np.isfinite(crystal_positions))
        or not np.all(np.isfinite(rotation))
        or not np.all(np.isfinite(inplane_orientation_rows))
    ):
        raise GBMakerConstructionValueError(
            "Exact decorated-site coordinates or orientation rows became "
            f"non-finite during Cartesian conversion for the {request.grain_side} "
            "grain."
        )

    rotated = crystal_positions @ rotation.T

    strain_scales = np.array([1.0, request.y_scale, request.z_scale], dtype=np.float64)

    rotated *= strain_scales
    rotated[:, 0] += request.x_offset

    rotated_unit_cell_basis = unit_cell.conventional @ rotation.T
    primitive_periods = inplane_orientation_rows @ rotated_unit_cell_basis
    strained_periods = primitive_periods * strain_scales
    selection_basis = _selection_basis_vectors(
        strained_periods,
        request.inplane_periodic,
        request.inplane_box_lengths,
        request.epsilon,
    )

    box_coordinates = _reduced_box_coordinates(
        rotated, selection_basis, request.epsilon
    )

    for row_index, is_periodic in enumerate(request.inplane_periodic):
        if not is_periodic:
            continue

        coordinate_index = row_index + 1
        tolerance = _reduced_coordinate_tolerance(
            selection_basis[row_index], request.epsilon
        )
        box_coordinates[:, coordinate_index] = wrap_reduced_coordinate(
            box_coordinates[:, coordinate_index],
            tolerance,
        )

    rotated = _cartesian_from_box_coordinates(box_coordinates, selection_basis)
    if not np.all(np.isfinite(rotated)):
        raise GBMakerConstructionValueError(
            "Exact decorated-site conversion produced non-finite Cartesian "
            f"coordinates for the {request.grain_side} grain."
        )
    atoms["x"], atoms["y"], atoms["z"] = rotated.T

    lower_x = float(request.x_offset)
    upper_x = lower_x + float(request.x_length)
    x_coordinates = atoms["x"]

    near_lower = (x_coordinates < lower_x) & (x_coordinates >= lower_x - request.epsilon)
    near_upper = (x_coordinates >= upper_x) & (
        x_coordinates < upper_x + request.epsilon
    )

    # Preserve the physical side of the upper termination while ensuring that its
    # floating representation remains strictly half-open.
    x_coordinates[near_lower] = lower_x
    x_coordinates[near_upper] = np.nextafter(upper_x, lower_x)

    outside_x = (x_coordinates < lower_x) | (x_coordinates >= upper_x)
    if np.any(outside_x):
        offending = x_coordinates[outside_x]
        raise GBMakerConstructionValueError(
            "Exact decorated-site conversion produced atoms outside the "
            f"{request.grain_side} half-open x slab [{lower_x:.8f}, {upper_x:.8f}): "
            f"min={float(np.min(offending)):.8f}, "
            f"max={float(np.max(offending)):.8f}."
        )

    y_dim, z_dim = request.inplane_box_lengths
    for axis_name, dimension, is_periodic in zip(
        ("y", "z"),
        (y_dim, z_dim),
        request.inplane_periodic,
    ):
        if not is_periodic:
            continue
        coordinates = atoms[axis_name]
        near_lower = (coordinates < 0.0) & (coordinates >= -request.epsilon)
        near_upper = (coordinates >= dimension) & (
            coordinates < dimension + request.epsilon
        )
        coordinates[near_lower | near_upper] = 0.0

        outside = (coordinates < 0.0) | (coordinates >= dimension)
        if np.any(outside):
            offending = coordinates[outside]
            raise GBMakerConstructionValueError(
                "Exact decorated-site conversion produced atoms outside the "
                f"periodic {axis_name} box [0, {dimension:.8f}): "
                f"min={float(np.min(offending)):.8f}, "
                f"max={float(np.max(offending)):.8f}."
            )

    origin_ids = np.repeat(
        np.arange(expected_site_count // basis_size, dtype=np.int64), basis_size
    )

    return GrainBuildResult(
        grain_side=request.grain_side,
        atoms=atoms,
        origin_ids=origin_ids,
        basis_size=basis_size,
    )
