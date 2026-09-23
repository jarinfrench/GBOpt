# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Pure geometry kernels for GBMaker grain construction.

Contains the reduced-coordinate wrapping and tolerance calculation, scaled periodic
and selection/box basis construction, reduced/Cartesian box-coordinate transforms,
boundary-normal lattice-index range estimation, complete conventional-cell origin
masking/filtering, Cartesian box clipping, deterministic origin deduplication, and the
composed periodic-box selection stage extracted from ``GBOpt.GBMaker``. Every kernel
here receives bases, bounds, periodicity, and tolerance explicitly; none accepts a
``GBMaker`` instance. ``GBOpt.GBMaker`` keeps its own private wrapper methods that
supply this instance state explicitly and translate
``GBMakerConstructionValueError`` back to the established ``GBMakerValueError``, the
same pattern used for ``gbmaker.config``/``gbmaker.orientation``/``gbmaker.dimension``.

``_miller_row_norm`` is the single canonical implementation; ``gbmaker.orientation``
and ``gbmaker.dimension`` import it from here instead of each carrying a private copy
(see ``REFACTOR_CLEANUP.md`` for the R05/R06 history of that duplication). This module
is a leaf with respect to its ``gbmaker`` siblings -- it imports only from
``gbmaker.types`` -- specifically so ``orientation`` and ``dimension`` can depend on it
without a cycle. ``_reduce_integer_row`` duplicates
``GBOpt.gbmaker.orientation._reduce_integer_row`` for the same reason: importing it
from ``orientation`` here would create exactly that cycle, since ``orientation`` now
imports ``_miller_row_norm`` from this module.

Exact-path decorated-site enumeration and float-path lattice-coefficient enumeration
(``__build_exact_grain``/``__generate_grain_result`` and everything that composes
their results into full grains) are grain-generation orchestration, not geometry
kernels, and remain in ``GBOpt.GBMaker`` pending R08.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from GBOpt.gbmaker.types import GBMakerConstructionValueError


def _miller_row_norm(row: Sequence[object] | np.ndarray) -> float:
    """Return the Euclidean norm of a nonzero integer Miller-index row.

    Computes ``sqrt(h*h + k*k + l*l)`` using Python ``int`` arithmetic for the squared
    norm. This avoids fixed-width NumPy integer overflow and avoids object-dtype NumPy
    ufuncs.

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


def wrap_reduced_coordinate(
    reduced_coord: np.ndarray, tol: float = 1e-8
) -> np.ndarray:
    """
    Wrap reduced coordinates into [0, 1) and snap both periodic faces to 0.

    :param reduced_coord: Reduced coordinates to wrap.
    :param tol: Tolerance in reduced-coordinate units. Optional, defaults to 1e-8
    :return: Wrapped reduced coordinates in [0, 1).
    :raises GBMakerConstructionValueError: If ``tol`` is not finite or is negative.
    """
    if not math.isfinite(tol):
        raise GBMakerConstructionValueError(
            "Reduced-coordinate tolerance must be finite."
        )
    if tol < 0:
        raise GBMakerConstructionValueError(
            "Reduced-coordinate tolerance must be non-negative."
        )

    wrapped = np.mod(np.asarray(reduced_coord, dtype=np.float64), 1.0)
    return np.where(
        (wrapped < tol) | ((1.0 - wrapped) < tol),
        0.0,
        wrapped,
    )


def _reduce_integer_row(row: np.ndarray) -> np.ndarray:
    """Reduce an integer row by its GCD.

    Duplicates ``GBOpt.gbmaker.orientation._reduce_integer_row``: ``orientation``
    imports ``_miller_row_norm`` from this module, so this module cannot import back
    from ``orientation`` without a circular import.

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


def _reduced_coordinate_tolerance(basis_vector: np.ndarray, epsilon: float) -> float:
    """
    Convert the Cartesian epsilon to reduced-coordinate units for a basis vector.

    :param basis_vector: Cartesian basis vector used to define the coordinate scale.
    :param epsilon: Cartesian numerical tolerance (Angstroms).
    :return: Reduced-coordinate tolerance corresponding to ``epsilon``.
    """
    basis_vector = np.asarray(basis_vector, dtype=np.float64)
    basis_length = np.linalg.norm(basis_vector)

    return epsilon / basis_length


def _scaled_periodic_basis_vector(
    period_vector: np.ndarray, box_length: float, axis_index: int
) -> np.ndarray:
    """
    Scale a periodic basis vector so one axis projection matches the box length.

    :param period_vector: Cartesian periodic basis vector.
    :param box_length: Desired box length along the selected axis.
    :param axis_index: Axis whose projection should match ``box_length``.
    :return: Scaled periodic basis vector.
    :raises GBMakerConstructionValueError: If ``box_length`` is not strictly positive
        or if the scaled vector is not finite.
    """
    period_vector = np.asarray(period_vector, dtype=np.float64)
    box_length = float(box_length)
    if box_length <= 0.0:
        raise GBMakerConstructionValueError("box_length must be strictly positive.")
    axis_index = int(axis_index)

    # We ignore overflow/invalid values because the check immediately after catches
    # those states and raises a GBMakerConstructionValueError
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = box_length / period_vector[axis_index]
        scaled_vector = period_vector * scale
    if not np.all(np.isfinite(scaled_vector)):
        raise GBMakerConstructionValueError(
            "Scaled periodic basis vector must be finite."
        )
    return scaled_vector


def _box_periodic_basis(
    primitive_periods: np.ndarray,
    inplane_periodic: tuple[bool, bool],
    box_lengths: tuple[float, float],
    epsilon: float,
) -> np.ndarray:
    """
    Build the in-plane box basis from primitive periodic vectors.

    :param primitive_periods: 2x3 array containing primitive y/z period vectors.
    :param inplane_periodic: ``(y_periodic, z_periodic)`` periodicity flags.
    :param box_lengths: ``(y_dim, z_dim)`` box lengths (Angstroms).
    :param epsilon: Cartesian numerical tolerance (Angstroms).
    :return: 2x3 array containing the box basis vectors for y and z.
    :raises GBMakerConstructionValueError: If a periodic axis has a near-zero
        projection on itself.
    """
    primitive_periods = np.asarray(primitive_periods, dtype=np.float64)

    box_basis = np.zeros((2, 3), dtype=np.float64)

    for row_index, (is_periodic, box_length) in enumerate(
        zip(inplane_periodic, box_lengths)
    ):
        if not is_periodic:
            continue

        axis_index = row_index + 1
        axis_projection = primitive_periods[row_index, axis_index]
        if np.isclose(axis_projection, 0.0, atol=epsilon, rtol=0.0):
            raise GBMakerConstructionValueError(
                "primitive_periods must have a non-zero projection on the "
                "selected box axis."
            )
        box_basis[row_index] = _scaled_periodic_basis_vector(
            primitive_periods[row_index], box_length, axis_index
        )

    return box_basis


def _selection_basis_vectors(
    primitive_periods: np.ndarray,
    inplane_periodic: tuple[bool, bool],
    box_lengths: tuple[float, float],
    epsilon: float,
) -> np.ndarray:
    """
    Build the canonical in-plane selection basis for y/z box coordinates.

    Periodic axes use the box-periodic basis vectors; non-periodic axes fall back
    to the corresponding Cartesian unit vectors.

    :param primitive_periods: 2x3 array containing primitive y/z period vectors.
    :param inplane_periodic: ``(y_periodic, z_periodic)`` periodicity flags.
    :param box_lengths: ``(y_dim, z_dim)`` box lengths (Angstroms).
    :param epsilon: Cartesian numerical tolerance (Angstroms).
    :return: 2x3 array containing the y/z selection basis vectors.
    """
    selection_basis = _box_periodic_basis(
        primitive_periods, inplane_periodic, box_lengths, epsilon
    )

    for row_index, is_periodic in enumerate(inplane_periodic):
        if is_periodic:
            continue
        selection_basis[row_index, row_index + 1] = 1.0

    return selection_basis


def _x_index_range(
    primitive_periods: np.ndarray,
    rotated_unit_cell_basis: np.ndarray,
    x_bounds: np.ndarray,
    inplane_periodic: tuple[bool, bool],
    box_lengths: tuple[float, float],
    epsilon: float,
) -> np.ndarray:
    """
    Build a conservative contiguous lattice-index range along the x-period vector.

    The x-period direction is derived in lattice space as the cross product of the
    two in-plane primitive periods expressed in the rotated unit-cell basis. The
    returned integer range is padded conservatively so translated unit cells cover
    the requested x slab after in-plane box tilts and unit-cell extent are applied.

    :param primitive_periods: 2x3 array containing primitive y/z period vectors.
    :param rotated_unit_cell_basis: 3x3 array containing the rotated unit-cell
        basis vectors as rows.
    :param x_bounds: Length-2 array-like containing ``[x_min, x_max]``.
    :param inplane_periodic: ``(y_periodic, z_periodic)`` periodicity flags.
    :param box_lengths: ``(y_dim, z_dim)`` box lengths (Angstroms).
    :param epsilon: Cartesian numerical tolerance (Angstroms).
    :return: Contiguous integer array of lattice indices along the x-period
        direction.
    :raises GBMakerConstructionValueError: If ``rotated_unit_cell_basis`` is
        singular, if ``primitive_periods`` does not define distinct in-plane
        directions, or if the x-period direction has a zero x projection.
    """
    primitive_periods = np.asarray(primitive_periods, dtype=np.float64)
    rotated_unit_cell_basis = np.asarray(rotated_unit_cell_basis, dtype=np.float64)
    x_bounds = np.asarray(x_bounds, dtype=np.float64)

    determinant = np.linalg.det(rotated_unit_cell_basis)
    if np.isclose(determinant, 0.0, atol=epsilon, rtol=0.0):
        raise GBMakerConstructionValueError(
            "rotated_unit_cell_basis must form an invertible 3x3 basis."
        )

    reduced_periods = np.linalg.solve(
        rotated_unit_cell_basis.T, primitive_periods.T
    ).T
    x_direction_lattice = np.cross(reduced_periods[0], reduced_periods[1])
    if np.linalg.norm(x_direction_lattice) <= epsilon:
        raise GBMakerConstructionValueError(
            "primitive_periods must define distinct in-plane directions."
        )

    rounded_direction = np.rint(x_direction_lattice)
    if np.allclose(
        x_direction_lattice, rounded_direction, atol=epsilon, rtol=0.0
    ) and np.any(rounded_direction):
        x_direction_lattice = _reduce_integer_row(
            rounded_direction.astype(int)
        ).astype(np.float64)

    x_period_vector = x_direction_lattice @ rotated_unit_cell_basis
    x_projection = float(x_period_vector[0])
    if np.isclose(x_projection, 0.0, atol=epsilon, rtol=0.0):
        raise GBMakerConstructionValueError(
            "x-period direction must have a non-zero projection on x."
        )
    if x_projection < 0.0:
        x_projection = -x_projection

    box_basis = _box_periodic_basis(
        primitive_periods, inplane_periodic, box_lengths, epsilon
    )
    box_corners_x = np.array(
        [
            0.0,
            box_basis[0, 0],
            box_basis[1, 0],
            box_basis[0, 0] + box_basis[1, 0],
        ],
        dtype=np.float64,
    )
    cell_corners_x = np.array(
        [
            np.sum(
                rotated_unit_cell_basis[np.array(mask, dtype=bool), 0],
                dtype=np.float64,
            )
            for mask in np.ndindex((2, 2, 2))
        ],
        dtype=np.float64,
    )

    x_offset_min = float(np.min(box_corners_x) + np.min(cell_corners_x))
    x_offset_max = float(np.max(box_corners_x) + np.max(cell_corners_x))

    n_min = math.floor((x_bounds[0] - x_offset_max) / x_projection) - 1
    n_max = math.ceil((x_bounds[1] - x_offset_min) / x_projection) + 1
    return np.arange(n_min, n_max + 1, dtype=int)


def _reduced_box_coordinates(
    cartesian_coordinates: np.ndarray, box_basis: np.ndarray, epsilon: float
) -> np.ndarray:
    """
    Convert Cartesian coordinates to mixed box coordinates ``[x_cart, u_y, u_z]``.

    The mixed basis is ``[e_x, A_y, A_z]`` where ``e_x`` is the Cartesian x-axis
    and ``A_y``/``A_z`` are the in-plane box basis vectors.

    :param cartesian_coordinates: Cartesian coordinates with shape ``(..., 3)``.
    :param box_basis: 2x3 array containing ``A_y`` and ``A_z``.
    :param epsilon: Cartesian numerical tolerance (Angstroms).
    :return: Mixed box coordinates with shape ``(..., 3)``.
    :raises GBMakerConstructionValueError: If ``box_basis``'s y/z projections do
        not form an invertible 2x2 basis.
    """
    cartesian_coordinates = np.asarray(cartesian_coordinates, dtype=np.float64)
    box_basis = np.asarray(box_basis, dtype=np.float64)

    yz_basis = box_basis[:, 1:].T
    determinant = np.linalg.det(yz_basis)
    if np.isclose(determinant, 0.0, atol=epsilon, rtol=0.0):
        raise GBMakerConstructionValueError(
            "box_basis y/z projections must form an invertible 2x2 basis."
        )

    yz_coordinates = cartesian_coordinates[..., 1:]
    reduced_yz = np.linalg.solve(
        yz_basis, yz_coordinates.reshape(-1, 2).T
    ).T.reshape(yz_coordinates.shape)
    x_cart = (
        cartesian_coordinates[..., 0]
        - reduced_yz[..., 0] * box_basis[0, 0]
        - reduced_yz[..., 1] * box_basis[1, 0]
    )
    return np.concatenate((x_cart[..., np.newaxis], reduced_yz), axis=-1)


def _cartesian_from_box_coordinates(
    box_coordinates: np.ndarray, box_basis: np.ndarray
) -> np.ndarray:
    """
    Convert mixed box coordinates ``[x_cart, u_y, u_z]`` to Cartesian coordinates.

    :param box_coordinates: Mixed box coordinates with shape ``(..., 3)``.
    :param box_basis: 2x3 array containing ``A_y`` and ``A_z``.
    :return: Cartesian coordinates with shape ``(..., 3)``.
    """
    box_coordinates = np.asarray(box_coordinates, dtype=np.float64)
    box_basis = np.asarray(box_basis, dtype=np.float64)

    cartesian_coordinates = np.array(box_coordinates, copy=True)
    cartesian_coordinates[..., 0] += np.tensordot(
        box_coordinates[..., 1:], box_basis[:, 0], axes=([-1], [0])
    )
    cartesian_coordinates[..., 1:] = np.tensordot(
        box_coordinates[..., 1:], box_basis[:, 1:], axes=([-1], [0])
    )
    return cartesian_coordinates


def _complete_origin_atom_mask(
    atom_mask: np.ndarray,
    origin_ids: np.ndarray,
    basis_size: int,
) -> np.ndarray:
    """Promote an atom-level mask to a complete-origin atom mask.

    An origin is retained only when exactly ``basis_size`` atoms are present for
    that origin and every atom from that origin passes ``atom_mask``. The returned
    mask is parallel to ``atom_mask`` and ``origin_ids``; retained atoms are marked
    ``True``.

    A fast grouped-origin path is used when the input already consists of
    contiguous, unique complete-origin groups. Otherwise, the method falls back to
    an origin-ID count.

    :param atom_mask: One-dimensional boolean atom-level mask.
    :param origin_ids: One-dimensional integer array parallel to ``atom_mask``. Each
        value identifies the conventional-cell origin that produced the
        corresponding atom.
    :param basis_size: Number of atoms expected in one complete origin group.
    :return: Boolean atom-level mask that keeps only complete retained origins.
    :raises GBMakerConstructionValueError: If the arrays are not one-dimensional and
        parallel, if ``origin_ids`` is not integer-valued, or if ``basis_size`` is not
        a positive integer.
    """
    atom_mask = np.asarray(atom_mask)
    origin_ids = np.asarray(origin_ids)

    if atom_mask.ndim != 1:
        raise GBMakerConstructionValueError("atom_mask must be a one-dimensional array.")
    if not np.issubdtype(atom_mask.dtype, np.bool_):
        raise GBMakerConstructionValueError("atom_mask must be a boolean array.")

    if origin_ids.ndim != 1:
        raise GBMakerConstructionValueError(
            "origin_ids must be a one-dimensional array."
        )
    if not np.issubdtype(origin_ids.dtype, np.integer):
        raise GBMakerConstructionValueError("origin_ids must contain integer values.")

    if len(atom_mask) != len(origin_ids):
        raise GBMakerConstructionValueError(
            "atom_mask and origin_ids must have equal length."
        )

    if isinstance(basis_size, (bool, np.bool_)) or not isinstance(
        basis_size, (int, np.integer)
    ):
        raise GBMakerConstructionValueError(
            f"basis_size must be a positive integer; got {basis_size!r}."
        )

    basis_size = int(basis_size)
    if basis_size < 1:
        raise GBMakerConstructionValueError(
            f"basis_size must be a positive integer; got {basis_size!r}."
        )

    if len(atom_mask) == 0:
        return atom_mask.copy()

    if len(atom_mask) % basis_size == 0:
        grouped_ids = origin_ids.reshape(-1, basis_size)
        group_ids = grouped_ids[:, 0]
        grouped_complete = np.all(grouped_ids == group_ids[:, None])
        grouped_unique = len(np.unique(group_ids)) == len(group_ids)

        if grouped_complete and grouped_unique:
            grouped_mask = atom_mask.reshape(-1, basis_size)
            return np.repeat(np.all(grouped_mask, axis=1), basis_size)

    unique_ids, inverse = np.unique(origin_ids, return_inverse=True)
    total_counts = np.bincount(inverse, minlength=len(unique_ids))
    pass_counts = np.bincount(inverse[atom_mask], minlength=len(unique_ids))

    keep_origin = (total_counts == basis_size) & (pass_counts == basis_size)
    return keep_origin[inverse]


def _filter_complete_origins(
    atoms: np.ndarray,
    origin_ids: np.ndarray,
    atom_mask: np.ndarray,
    basis_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter atoms and origin IDs while preserving complete origin groups.

    Converts ``atom_mask`` into a complete-origin mask using
    ``_complete_origin_atom_mask``. An origin is kept only when every one of its
    ``basis_size`` atoms passes the input mask. The returned arrays are copies.

    :param atoms: Structured atom array to filter.
    :param origin_ids: Integer origin-ID array parallel to ``atoms``.
    :param atom_mask: Boolean atom-level mask parallel to ``atoms``.
    :param basis_size: Number of atoms expected in one complete origin group.
    :return: ``(filtered_atoms, filtered_origin_ids)``.
    :raises GBMakerConstructionValueError: If ``atoms`` and ``origin_ids`` are not
        parallel, or if ``_complete_origin_atom_mask`` rejects the mask, origin IDs,
        or basis size.
    """
    if len(atoms) != len(origin_ids):
        raise GBMakerConstructionValueError(
            "atoms and origin_ids must have equal length."
        )

    keep_atoms = _complete_origin_atom_mask(
        atom_mask,
        origin_ids,
        basis_size,
    )
    return atoms[keep_atoms].copy(), origin_ids[keep_atoms].copy()


def _clip_complete_origins_to_cartesian_box(
    atoms: np.ndarray,
    origin_ids: np.ndarray,
    x_bounds: np.ndarray,
    basis_size: int,
    inplane_periodic: tuple[bool, bool],
    box_lengths: tuple[float, float],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip atoms to the Cartesian grain box by complete origin groups.

    Atoms are tested against the half-open x interval ``[x_bounds[0], x_bounds[1])``
    using ``epsilon``. Non-periodic in-plane axes are also clipped to their Cartesian
    box dimensions. Periodic in-plane axes are not clipped here because they have
    already been selected and wrapped by the mixed-basis selection path.

    Complete-origin filtering is applied after the atom-level box mask is
    constructed, so an origin is retained only when all atoms in that origin remain
    inside the requested box.

    :param atoms: Structured atom array to clip.
    :param origin_ids: Integer origin-ID array parallel to ``atoms``.
    :param x_bounds: Length-2 array containing lower and upper x bounds (Angstroms).
    :param basis_size: Number of atoms expected in one complete origin group.
    :param inplane_periodic: ``(y_periodic, z_periodic)`` periodicity flags.
    :param box_lengths: ``(y_dim, z_dim)`` box lengths (Angstroms).
    :param epsilon: Cartesian numerical tolerance (Angstroms).
    :return: ``(clipped_atoms, clipped_origin_ids)``.
    :raises GBMakerConstructionValueError: If ``x_bounds`` is not a finite increasing
        two-value interval, or if complete-origin filtering rejects the inputs.
    """
    try:
        x_bounds = np.asarray(x_bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(
            f"x_bounds must be a finite two-value interval; got {x_bounds!r}."
        ) from exc

    if x_bounds.shape != (2,):
        raise GBMakerConstructionValueError(
            f"x_bounds must be a finite two-value interval; got {x_bounds!r}."
        )
    if not np.all(np.isfinite(x_bounds)) or x_bounds[1] <= x_bounds[0]:
        raise GBMakerConstructionValueError(
            f"x_bounds must be a finite increasing interval; got {x_bounds!r}."
        )

    inside_box = (
        (atoms["x"] >= x_bounds[0] - epsilon)
        & (atoms["x"] < x_bounds[1] - epsilon)
    )

    axis_names = ("y", "z")

    for axis_name, axis_dim, is_periodic in zip(
        axis_names,
        box_lengths,
        inplane_periodic,
    ):
        if is_periodic:
            continue

        inside_box &= (
            (atoms[axis_name] >= -epsilon)
            & (atoms[axis_name] < axis_dim)
        )

    clipped_atoms, clipped_origin_ids = _filter_complete_origins(
        atoms,
        origin_ids,
        inside_box,
        basis_size,
    )

    for axis_name, is_periodic in zip(axis_names, inplane_periodic):
        if is_periodic:
            continue

        clipped_atoms[axis_name] = np.where(
            (clipped_atoms[axis_name] < 0.0)
            & (clipped_atoms[axis_name] >= -epsilon),
            0.0,
            clipped_atoms[axis_name],
        )

    return clipped_atoms, clipped_origin_ids


def _deduplicate_complete_origins(
    atoms: np.ndarray,
    origin_ids: np.ndarray,
    basis_size: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove duplicate complete-origin groups by full atom signatures.

    Each origin group is expected to contain exactly ``basis_size`` contiguous
    atoms with a single origin ID. Duplicate groups are identified by the full
    ordered basis signature: atom names plus quantized Cartesian positions. The
    first occurrence of each unique complete-origin group is retained.

    :param atoms: Structured atom array containing complete contiguous origin
        groups.
    :param origin_ids: Integer origin-ID array parallel to ``atoms``.
    :param basis_size: Number of atoms expected in one complete origin group.
    :param epsilon: Cartesian numerical tolerance (Angstroms) used to quantize
        positions before deduplication.
    :return: ``(deduplicated_atoms, deduplicated_origin_ids)``.
    :raises GBMakerConstructionValueError: If the inputs are not parallel, if
        ``origin_ids`` is not integer-valued, if ``basis_size`` is not a
        positive integer, or if the atom array cannot be reshaped into complete
        contiguous origin groups.
    """
    origin_ids = np.asarray(origin_ids)

    if origin_ids.ndim != 1:
        raise GBMakerConstructionValueError(
            "origin_ids must be a one-dimensional array."
        )
    if not np.issubdtype(origin_ids.dtype, np.integer):
        raise GBMakerConstructionValueError("origin_ids must contain integer values.")

    if isinstance(basis_size, (bool, np.bool_)) or not isinstance(
        basis_size, (int, np.integer)
    ):
        raise GBMakerConstructionValueError(
            f"basis_size must be a positive integer; got {basis_size!r}."
        )

    basis_size = int(basis_size)
    if basis_size < 1:
        raise GBMakerConstructionValueError(
            f"basis_size must be a positive integer; got {basis_size!r}."
        )

    if len(atoms) != len(origin_ids):
        raise GBMakerConstructionValueError(
            "atoms and origin_ids must have equal length."
        )

    if len(atoms) == 0:
        return atoms.copy(), origin_ids.copy()

    if len(atoms) % basis_size != 0:
        raise GBMakerConstructionValueError(
            "Complete-origin deduplication requires full origin groups."
        )

    n_origins = len(atoms) // basis_size
    grouped_origin_ids = origin_ids.reshape(n_origins, basis_size)
    if not np.all(grouped_origin_ids == grouped_origin_ids[:, :1]):
        raise GBMakerConstructionValueError(
            "Complete-origin deduplication requires contiguous origin groups."
        )

    positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
    quantized = np.round(positions / epsilon).astype(np.int64)

    signature_dtype = np.dtype(
        [
            ("name", atoms.dtype["name"], (basis_size,)),
            ("position", np.int64, (basis_size, 3)),
        ]
    )
    signatures = np.empty(n_origins, dtype=signature_dtype)
    signatures["name"] = atoms["name"].reshape(n_origins, basis_size)
    signatures["position"] = quantized.reshape(n_origins, basis_size, 3)

    _, unique_group_indices = np.unique(signatures, return_index=True)

    keep_groups = np.zeros(n_origins, dtype=bool)
    keep_groups[np.sort(unique_group_indices)] = True

    grouped_atoms = atoms.reshape(n_origins, basis_size)
    grouped_ids = origin_ids.reshape(n_origins, basis_size)

    return (
        grouped_atoms[keep_groups].reshape(-1).copy(),
        grouped_ids[keep_groups].reshape(-1).copy(),
    )


def _select_complete_origins_in_box_basis(
    atoms: np.ndarray,
    origin_ids: np.ndarray,
    primitive_periods: np.ndarray,
    x_bounds: np.ndarray,
    basis_size: int,
    inplane_periodic: tuple[bool, bool],
    box_lengths: tuple[float, float],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select and wrap in-plane coordinates while preserving complete origins.

    Builds the y/z selection basis from ``primitive_periods`` and filters atoms into
    the in-plane simulation box. Periodic in-plane axes are selected in reduced
    coordinates and wrapped onto the periodic box. Non-periodic in-plane axes are
    selected against their Cartesian box extents.

    When the selection basis has no x component, this uses an axis-aligned fast
    path. Otherwise, atoms are converted into mixed box coordinates ``[x_cart, u_y,
    u_z]``, selected/wrapped there, converted back to Cartesian coordinates, and then
    re-filtered by complete origins against the x slab.

    :param atoms: Structured atom array to select and wrap.
    :param origin_ids: Integer origin-ID array parallel to ``atoms``.
    :param primitive_periods: Two-row array containing the in-plane y and z
        primitive period vectors in strained lab-frame Cartesian coordinates.
    :param x_bounds: Length-2 array containing lower and upper x bounds (Angstroms).
    :param basis_size: Number of atoms expected in one complete origin group.
    :param inplane_periodic: ``(y_periodic, z_periodic)`` periodicity flags.
    :param box_lengths: ``(y_dim, z_dim)`` box lengths (Angstroms).
    :param epsilon: Cartesian numerical tolerance (Angstroms).
    :return: ``(selected_atoms, selected_origin_ids)`` after complete-origin
        selection and periodic wrapping.
    :raises GBMakerConstructionValueError: If ``x_bounds`` is not a finite increasing
        interval, if the selection basis is singular, or if complete-origin
        filtering rejects the inputs.
    """
    try:
        x_bounds = np.asarray(x_bounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(
            f"x_bounds must be a finite two-value interval; got {x_bounds!r}."
        ) from exc

    if x_bounds.shape != (2,):
        raise GBMakerConstructionValueError(
            f"x_bounds must be a finite two-value interval; got {x_bounds!r}."
        )
    if not np.all(np.isfinite(x_bounds)) or x_bounds[1] <= x_bounds[0]:
        raise GBMakerConstructionValueError(
            f"x_bounds must be a finite increasing interval; got {x_bounds!r}."
        )

    selection_basis = _selection_basis_vectors(
        primitive_periods, inplane_periodic, box_lengths, epsilon
    )
    positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))

    if np.allclose(selection_basis[:, 0], 0.0, atol=epsilon, rtol=0.0):
        inside_box = np.ones(len(atoms), dtype=bool)
        for row_index, is_periodic in enumerate(inplane_periodic):
            axis = row_index + 1
            period = selection_basis[row_index, axis]
            coord = positions[:, axis]

            if is_periodic:
                tol = _reduced_coordinate_tolerance(
                    selection_basis[row_index], epsilon
                )
                reduced_coord = coord / period
                inside_box &= (
                    (reduced_coord >= -tol)
                    & (reduced_coord < 1.0 + tol)
                )

        selected_atoms, selected_origin_ids = _filter_complete_origins(
            atoms,
            origin_ids,
            inside_box,
            basis_size,
        )
        if len(selected_atoms) == 0:
            return selected_atoms, selected_origin_ids

        for row_index, is_periodic in enumerate(inplane_periodic):
            if not is_periodic:
                continue

            axis_name = ("y", "z")[row_index]
            axis = row_index + 1
            period = selection_basis[row_index, axis]
            tol = _reduced_coordinate_tolerance(selection_basis[row_index], epsilon)

            wrapped = np.mod(selected_atoms[axis_name], period)
            selected_atoms[axis_name] = np.where(
                (wrapped < tol * period) | ((period - wrapped) < tol * period),
                0.0,
                wrapped,
            )

        inside_x = (
            (selected_atoms["x"] >= x_bounds[0] - epsilon)
            & (selected_atoms["x"] < x_bounds[1] - epsilon)
        )
        return _filter_complete_origins(
            selected_atoms,
            selected_origin_ids,
            inside_x,
            basis_size,
        )

    box_coordinates = _reduced_box_coordinates(positions, selection_basis, epsilon)

    inside_box = np.ones(len(atoms), dtype=bool)

    for row_index, (axis_dim, is_periodic) in enumerate(
        zip(box_lengths, inplane_periodic)
    ):
        reduced_axis = box_coordinates[:, row_index + 1]

        if is_periodic:
            tol = _reduced_coordinate_tolerance(selection_basis[row_index], epsilon)
            inside_box &= (
                (reduced_axis >= -tol)
                & (reduced_axis < 1.0 + tol)
            )
        else:
            inside_box &= (
                (reduced_axis >= -epsilon)
                & (reduced_axis < axis_dim)
            )

    selected_atoms, selected_origin_ids = _filter_complete_origins(
        atoms,
        origin_ids,
        inside_box,
        basis_size,
    )
    if len(selected_atoms) == 0:
        return selected_atoms, selected_origin_ids

    selected_mask = _complete_origin_atom_mask(
        inside_box,
        origin_ids,
        basis_size,
    )
    selected_box_coordinates = box_coordinates[selected_mask].copy()

    for row_index, is_periodic in enumerate(inplane_periodic):
        coordinate_index = row_index + 1

        if is_periodic:
            tol = _reduced_coordinate_tolerance(selection_basis[row_index], epsilon)
            selected_box_coordinates[:, coordinate_index] = wrap_reduced_coordinate(
                selected_box_coordinates[:, coordinate_index],
                tol,
            )
            continue

        selected_box_coordinates[:, coordinate_index] = np.where(
            (
                (selected_box_coordinates[:, coordinate_index] < 0.0)
                & (
                    selected_box_coordinates[:, coordinate_index]
                    >= -epsilon
                )
            ),
            0.0,
            selected_box_coordinates[:, coordinate_index],
        )

    wrapped_positions = _cartesian_from_box_coordinates(
        selected_box_coordinates,
        selection_basis,
    )
    selected_atoms["x"], selected_atoms["y"], selected_atoms["z"] = (
        wrapped_positions.T
    )

    inside_x = (
        (selected_atoms["x"] >= x_bounds[0] - epsilon)
        & (selected_atoms["x"] < x_bounds[1] - epsilon)
    )
    return _filter_complete_origins(
        selected_atoms,
        selected_origin_ids,
        inside_x,
        basis_size,
    )


def _triclinic_tilt_params(
    *,
    inplane_periodic: tuple[bool, bool],
    left_periodic_miller_rows: np.ndarray,
    right_periodic_miller_rows: np.ndarray,
    R_left: np.ndarray,
    R_right: np.ndarray,
    conventional_basis: np.ndarray,
    y_dim: float,
    z_dim: float,
    epsilon: float,
) -> tuple[float, float, float, float]:
    """Compute LAMMPS restricted-triclinic tilt factors.

    The y-period in the lab frame is ``R_grain @ (g_y * a0)``. For an exact CSL
    boundary this is exactly ``||g_y|| * a0 * e_y``; for non-CSL it has small x and z
    components. To satisfy LAMMPS's restriction that the b-vector lies in the xy-plane,
    everything is rotated about the x-axis by ``theta = -atan2(A2[2], A2[1])``.

    :param inplane_periodic: ``(y_periodic, z_periodic)`` periodicity flags.
    :param left_periodic_miller_rows: Left-grain periodic Miller-row matrix.
    :param right_periodic_miller_rows: Right-grain periodic Miller-row matrix.
    :param R_left: Left-grain rotation matrix.
    :param R_right: Right-grain rotation matrix.
    :param conventional_basis: Conventional unit-cell basis vectors as rows.
    :param y_dim: Simulation box length along y (Angstroms).
    :param z_dim: Simulation box length along z (Angstroms).
    :param epsilon: Cartesian numerical tolerance (Angstroms).
    :return: ``(xy, xz, yz, theta)`` -- the three tilt scalars and the rotation angle
        to apply to atom coordinates.
    :raises GBMakerConstructionValueError: If ``inplane_periodic`` is not periodic
        along both y and z, or if the selected grain's primitive periods have a
        near-zero projection on their own box axis.
    """
    if not all(inplane_periodic):
        raise GBMakerConstructionValueError(
            "Triclinic output requires periodic y and z directions."
        )

    # Use grain with larger y-period, consistent with how spacing["y"] is chosen.
    if (
        np.linalg.norm(left_periodic_miller_rows[1])
        >= np.linalg.norm(right_periodic_miller_rows[1])
    ):
        R_grain = R_left
        R_grain_approx = left_periodic_miller_rows
    else:
        R_grain = R_right
        R_grain_approx = right_periodic_miller_rows

    # conventional_basis stores basis vectors as rows: C = [a1; a2; a3].
    # Rotating each row vector to the lab frame gives [R@a1; R@a2; R@a3],
    # which in batch form is (R @ C.T).T = C @ R.T.
    rotated_unit_cell_basis = conventional_basis @ R_grain.T
    primitive_periods = (
        np.asarray(R_grain_approx[1:], dtype=np.float64) @ rotated_unit_cell_basis
    )
    A2_lab, A3_lab = _box_periodic_basis(
        primitive_periods, inplane_periodic, (y_dim, z_dim), epsilon
    )

    # Rotate about x to bring A2 into the xy-plane (LAMMPS restricted-triclinic
    # requires b-vector in the xy-plane). x-components are unaffected by this
    # rotation.
    theta = -math.atan2(float(A2_lab[2]), float(A2_lab[1]))
    ct, st = math.cos(theta), math.sin(theta)

    # The x-rotation matrix is [[1,0,0],[0,ct,-st],[0,st,ct]]. The x-components of
    # A2_lab and A3_lab are unchanged by it, so xy and xz can be read directly from
    # the pre-rotation vectors. yz requires the full rotation.
    xy = float(A2_lab[0])
    xz = float(A3_lab[0])
    yz = float(ct * A3_lab[1] - st * A3_lab[2])

    return xy, xz, yz, theta
