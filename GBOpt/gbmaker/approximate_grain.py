# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Approximate floating-point grain construction extracted from ``GBOpt.GBMaker``.

Contains ``build_approximate_grain``, extracted from
``GBOpt.GBMaker.__generate_grain_result``, plus
``filter_grain_result_complete_origins`` and ``trim_grain_result_to_upper_x``,
extracted from ``GBOpt.GBMaker.__filter_float_result_complete_origins`` and
``GBOpt.GBMaker.__trim_float_result_to_upper_x``. All three consume and return
``GrainBuildResult`` (replacing the former ``GBMaker.py``-local
``_FloatGrainBuildResult``, which was the same shape) and either succeed or raise
``GBMakerConstructionValueError`` -- none accepts a ``GBMaker`` instance.
``GBOpt.GBMaker`` keeps thin wrapper methods that supply instance state explicitly and
translate ``GBMakerConstructionValueError`` back to the established
``GBMakerValueError``, the same pattern used for every other ``gbmaker`` extraction
boundary.

Enumerates conventional-cell lattice coefficients over a conservative slab, expands
each retained origin to the full conventional-cell basis, rotates atoms into the lab
frame, applies any requested lab-frame in-plane strain, selects/wraps periodic
in-plane coordinates, clips to the Cartesian x slab, and removes duplicate
complete-origin groups -- preserving the current complete-origin selection, mismatch
behavior, atom ordering, and duplicate elimination per issue #69's acceptance
criteria.

This module depends on ``gbmaker.geometry`` and ``gbmaker.types`` only; it does not
import ``gbmaker.exact_grain`` or any other ``gbmaker`` sibling.
"""

from __future__ import annotations

import math

import numpy as np

from GBOpt.gbmaker.geometry import (
    _clip_complete_origins_to_cartesian_box,
    _deduplicate_complete_origins,
    _filter_complete_origins,
    _reduce_integer_row,
    _select_complete_origins_in_box_basis,
    _selection_basis_vectors,
    _x_index_range,
)
from GBOpt.gbmaker.types import (
    GBMakerConstructionValueError,
    GrainBuildRequest,
    GrainBuildResult,
)


def build_approximate_grain(request: GrainBuildRequest) -> GrainBuildResult:
    """Build one grain using the floating-point lattice-enumeration path.

    Enumerates conventional-cell lattice coefficients over a conservative slab,
    expands each retained origin to the full conventional-cell basis, rotates atoms
    into the lab frame, applies any requested lab-frame in-plane strain,
    selects/wraps periodic in-plane coordinates, clips to the Cartesian x slab, and
    removes duplicate complete-origin groups.

    ``origin_ids`` is returned in parallel with the atom array so later trimming
    operations (``filter_grain_result_complete_origins``,
    ``trim_grain_result_to_upper_x``) can keep or remove complete conventional-cell
    origins.

    :param request: Grain build request with ``request.exact`` expected ``False``.
        ``request.periodic_matrix`` rows 1-2 define the primitive in-plane y/z period
        vectors used by the floating-point selection basis.
    :return: Float-path grain build result containing the atom array, parallel
        origin-ID array, and conventional-cell basis size.
    :raises GBMakerConstructionValueError: If selection or clipping cannot preserve
        complete origin groups, or if no complete origins remain after filtering.
    """
    unit_cell = request.material.unit_cell
    if unit_cell is None:
        raise GBMakerConstructionValueError(
            "Approximate grain generation requires a resolved MaterialState.unit_cell."
        )

    x_bounds = np.array(
        [request.x_offset, request.x_offset + request.x_length], dtype=np.float64
    )
    inplane_periodic = request.inplane_periodic
    epsilon = request.epsilon
    strain_scales = np.array([1.0, request.y_scale, request.z_scale], dtype=np.float64)

    R_grain = np.asarray(request.rotation, dtype=np.float64)
    periodic_miller_rows = np.asarray(request.periodic_matrix, dtype=np.float64)

    rotated_unit_cell_basis = unit_cell.conventional @ R_grain.T

    primitive_periods = periodic_miller_rows[1:]
    primitive_periods = primitive_periods @ rotated_unit_cell_basis

    # Selection and wrapping operate on strained lab-frame coordinates, so the period
    # vectors passed to those helpers must carry the same lab-frame y/z strain as the
    # atoms.
    strained_periods = primitive_periods * strain_scales

    reduced_periods = np.linalg.solve(
        rotated_unit_cell_basis.T, primitive_periods.T
    ).T
    x_direction_lattice = np.cross(reduced_periods[0], reduced_periods[1])
    rounded_direction = np.rint(x_direction_lattice)
    if np.allclose(
        x_direction_lattice, rounded_direction, atol=epsilon, rtol=0.0
    ) and np.any(rounded_direction):
        x_direction_lattice = _reduce_integer_row(
            rounded_direction.astype(int)
        ).astype(np.float64)

    # Build the final strained selection basis, then map it back through the lab-frame
    # strain before converting to lattice coordinates. This keeps the coefficient
    # search conservative for the unstrained lattice that is enumerated before atom
    # positions are strained.
    selection_box_basis = _selection_basis_vectors(
        strained_periods, inplane_periodic, request.inplane_box_lengths, epsilon
    ).copy()
    axis_dims = request.inplane_box_lengths
    for row_index, (is_periodic, axis_dim) in enumerate(
        zip(inplane_periodic, axis_dims)
    ):
        if not is_periodic:
            selection_box_basis[row_index] *= axis_dim

    prestrain_selection_box_basis = selection_box_basis / strain_scales
    selection_box_basis_lattice = np.linalg.solve(
        rotated_unit_cell_basis.T, prestrain_selection_box_basis.T
    ).T

    local_x_bounds = np.array([0.0, x_bounds[1] - x_bounds[0]], dtype=np.float64)
    nx_range = _x_index_range(
        primitive_periods,
        rotated_unit_cell_basis,
        local_x_bounds,
        inplane_periodic,
        request.inplane_box_lengths,
        epsilon,
    )

    lattice_bound_corners = []
    for nx in (nx_range[0], nx_range[-1]):
        x_base = nx * x_direction_lattice
        for uy in (0.0, 1.0):
            for uz in (0.0, 1.0):
                cell_origin = (
                    x_base
                    + uy * selection_box_basis_lattice[0]
                    + uz * selection_box_basis_lattice[1]
                )
                for cell_corner in np.ndindex((2, 2, 2)):
                    lattice_bound_corners.append(
                        cell_origin + np.array(cell_corner, dtype=np.float64)
                    )

    lattice_bound_corners = np.asarray(lattice_bound_corners, dtype=np.float64)
    lattice_min = np.floor(np.min(lattice_bound_corners, axis=0)).astype(int) - 1
    lattice_max = np.ceil(np.max(lattice_bound_corners, axis=0)).astype(int) + 1

    coefficient_ranges = [
        np.arange(lower, upper + 1, dtype=int)
        for lower, upper in zip(lattice_min, lattice_max)
    ]
    lattice_coefficients = (
        np.array(np.meshgrid(*coefficient_ranges, indexing="ij")).reshape(3, -1).T
    )

    structured_basis = unit_cell.asarray()
    basis_size = len(structured_basis)
    corners = lattice_coefficients @ unit_cell.conventional
    atoms = np.tile(structured_basis, len(corners))
    translations = np.repeat(corners, basis_size, axis=0)
    atoms["x"] += translations[:, 0]
    atoms["y"] += translations[:, 1]
    atoms["z"] += translations[:, 2]
    origin_ids = np.repeat(
        np.arange(len(lattice_coefficients), dtype=np.int64), basis_size
    )

    positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
    rotated_positions = positions @ R_grain.T
    rotated_positions *= strain_scales
    rotated_positions[:, 0] += x_bounds[0]
    atoms["x"], atoms["y"], atoms["z"] = rotated_positions.T

    if any(inplane_periodic):
        atoms, origin_ids = _select_complete_origins_in_box_basis(
            atoms,
            origin_ids,
            strained_periods,
            x_bounds,
            basis_size,
            inplane_periodic,
            request.inplane_box_lengths,
            epsilon,
        )

    atoms, origin_ids = _clip_complete_origins_to_cartesian_box(
        atoms,
        origin_ids,
        x_bounds,
        basis_size,
        inplane_periodic,
        request.inplane_box_lengths,
        epsilon,
    )
    atoms, origin_ids = _deduplicate_complete_origins(
        atoms, origin_ids, basis_size, epsilon
    )

    if len(atoms) == 0:
        raise GBMakerConstructionValueError(
            f"Float grain generation removed all complete origins for the "
            f"{request.grain_side} grain."
        )

    return GrainBuildResult(
        grain_side=request.grain_side,
        atoms=atoms,
        origin_ids=origin_ids,
        basis_size=basis_size,
    )


def filter_grain_result_complete_origins(
    result: GrainBuildResult,
    atom_mask: np.ndarray,
) -> GrainBuildResult:
    """Filter a float-path build result by complete origin groups.

    Applies ``atom_mask`` to ``result.atoms`` through complete-origin filtering,
    preserving only conventional-cell origins for which every atom in the origin
    group passes the mask. The returned result carries the filtered atom array,
    filtered parallel origin IDs, and the original basis size.

    :param result: Float-path grain build result to filter.
    :param atom_mask: Boolean atom-level mask parallel to ``result.atoms``.
    :return: Filtered float-path grain build result.
    :raises GBMakerConstructionValueError: If complete-origin filtering rejects the
        mask, origin IDs, or basis size.
    """
    atoms, origin_ids = _filter_complete_origins(
        result.atoms,
        result.origin_ids,
        atom_mask,
        result.basis_size,
    )
    return GrainBuildResult(
        grain_side=result.grain_side,
        atoms=atoms,
        origin_ids=origin_ids,
        basis_size=result.basis_size,
    )


def trim_grain_result_to_upper_x(
    result: GrainBuildResult,
    upper_x: float,
    epsilon: float,
) -> GrainBuildResult:
    """Trim a float-path grain to an upper x bound by complete origins.

    Retains only complete conventional-cell origins whose atoms all lie below
    ``upper_x`` using the same half-open upper-bound convention as the rest of the
    grain-generation pipeline.

    :param result: Float-path grain build result to trim.
    :param upper_x: Upper x bound in Angstroms.
    :param epsilon: Numerical tolerance used for the upper-bound comparison.
    :return: Trimmed float-path grain build result.
    :raises GBMakerConstructionValueError: If ``upper_x`` is not finite or if
        complete-origin filtering rejects the result metadata.
    """
    try:
        upper_x = float(upper_x)
    except (TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(
            f"upper_x must be a finite number; got {upper_x!r}."
        ) from exc

    if not math.isfinite(upper_x):
        raise GBMakerConstructionValueError(
            f"upper_x must be a finite number; got {upper_x!r}."
        )

    atom_mask = result.atoms["x"] < upper_x - epsilon
    return filter_grain_result_complete_origins(result, atom_mask)
