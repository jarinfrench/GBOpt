# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Pure bicrystal assembly and end-to-end construction pipeline for GBMaker.

Contains the left/right grain placement, exact-vs-float construction-path selection,
periodic-gap equalization, grain concatenation, grain-boundary-region selection, and
final ``BicrystalResult`` packaging extracted from ``GBOpt.GBMaker``'s
``__generate_gb``/``__set_gb_region``. ``assemble_bicrystal`` is the assembly stage:
given already-resolved material identity, per-grain rotations/periodic Miller rows,
boundary embedding, boundary-normal x placement, and planned box dimensions/strain
accommodation, it builds both grains (through ``gbmaker.exact_grain``/
``gbmaker.approximate_grain``), equalizes the periodic x gap, concatenates the grains,
selects the grain-boundary-region window, and returns a ``BicrystalResult``.
``build_bicrystal`` is the full pure pipeline: given a resolved ``MaterialState`` and
the same scalar construction knobs ``GBOpt.GBMaker`` normalizes in its constructor, it
composes orientation resolution (``gbmaker.orientation.resolve_orientation``),
periodic-spacing and box-dimension planning (``gbmaker.dimension.plan_periodic_spacing``/
``plan_dimensions``), and ``assemble_bicrystal`` into one call that builds a complete
bicrystal without instantiating ``GBMaker``.

This module sits above ``gbmaker``'s other construction-stage modules in the package's
dependency graph (leaf module first): ``types`` (no intra-package imports) <-
``geometry`` (imports only ``types``) <- ``orientation``/``dimension``/``exact_grain``/
``approximate_grain`` (each imports only ``geometry``/``types``; these four are mutual
leaves -- none imports another) <- ``assembly`` (this module; imports ``types``,
``orientation``, ``dimension``, ``exact_grain``, and ``approximate_grain``, since
composing construction stages together is exactly what ``build_bicrystal`` does).
Nothing in ``orientation``/``dimension``/``exact_grain``/``approximate_grain``/
``geometry``/``types`` imports this module (verified by grep), so no cycle is
introduced.

``GBOpt.GBMaker`` continues to call ``gbmaker.orientation.resolve_orientation`` and
``gbmaker.dimension.plan_periodic_spacing``/``plan_dimensions`` directly for its own
cached, setter-driven recomputation (``__calculate_periodic_spacing``/``__update_dims``
resolve orientation and dimensions at different times, since e.g. box-dimension-only
setters do not re-resolve orientation) -- only the previously-facade-coupled assembly
portion (``__generate_gb``/``__set_gb_region``) is rewired to delegate to
``assemble_bicrystal`` here. ``build_bicrystal`` exists as a genuinely standalone
pipeline usable without any ``GBMaker`` instance, per issue #70's acceptance criteria.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from GBOpt.BoundarySpec import BoundaryEmbedding
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.gbmaker.approximate_grain import (
    build_approximate_grain,
    trim_grain_result_to_upper_x,
)
from GBOpt.gbmaker.dimension import plan_dimensions, plan_periodic_spacing
from GBOpt.gbmaker.exact_grain import build_exact_grain
from GBOpt.gbmaker.orientation import _x_period, resolve_orientation
from GBOpt.gbmaker.types import (
    AxisAccommodation,
    BicrystalResult,
    GBMakerConstructionValueError,
    GrainBuildRequest,
    GrainBuildResult,
    GrainSide,
    MaterialState,
    StrainGrainPolicy,
)


def _grain_x_bounds(
    left_x: float,
    right_x: float,
    x_dim: float,
    vacuum_thickness: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return initial lab-frame x bounds for the left and right grains.

    :param left_x: Left-grain equalized x-slab thickness (Angstroms).
    :param right_x: Right-grain equalized x-slab thickness (Angstroms).
    :param x_dim: Combined left- and right-grain x extent (Angstroms).
    :param vacuum_thickness: Vacuum thickness applied along x (Angstroms).
    :return: ``(left_bounds, right_bounds)``, where each array contains ``[x_min,
        x_max]`` in Angstroms.
    """
    left_bounds = np.array(
        [vacuum_thickness, left_x + vacuum_thickness], dtype=np.float64
    )
    right_bounds = np.array(
        [left_x + vacuum_thickness, x_dim + vacuum_thickness], dtype=np.float64
    )
    return left_bounds, right_bounds


def _use_exact_grain_generation(embedding: BoundaryEmbedding | None) -> bool:
    """Return whether the exact integer grain-generation path should be used.

    :param embedding: Boundary embedding, or ``None`` for the legacy path.
    :return: ``True`` when ``embedding`` is exact, coherent, and carries both integer
        P and Q orientation matrices.
    :raises GBMakerConstructionValueError: If an exact coherent embedding is present
        but does not carry both P and Q.
    """
    if embedding is None:
        return False

    if not (embedding.exact and embedding.coherent):
        return False

    if embedding.P is None or embedding.Q is None:
        raise GBMakerConstructionValueError(
            "Exact coherent grain generation requires both embedding.P and "
            "embedding.Q."
        )

    return True


def _grain_strain_scales(
    grain_side: GrainSide,
    strain_accommodation: Mapping[str, AxisAccommodation],
) -> tuple[float, float]:
    """Return lab-frame in-plane strain scale factors for one grain.

    :param grain_side: Grain side, either ``"left"`` or ``"right"``.
    :param strain_accommodation: Mapping from in-plane axis name (``"y"``/``"z"``) to
        its commensurate-repeat accommodation. Empty when mismatch accommodation is
        inactive.
    :return: ``(y_scale, z_scale)`` for the selected grain.
    :raises GBMakerConstructionValueError: If ``grain_side`` is not ``"left"`` or
        ``"right"``.
    """
    if grain_side not in {"left", "right"}:
        raise GBMakerConstructionValueError(
            f"grain_side must be 'left' or 'right'; got {grain_side!r}."
        )

    y_accommodation = strain_accommodation.get("y")
    z_accommodation = strain_accommodation.get("z")

    y_scale = 1.0
    z_scale = 1.0

    if y_accommodation is not None:
        y_scale = (
            y_accommodation.left_scale
            if grain_side == "left"
            else y_accommodation.right_scale
        )

    if z_accommodation is not None:
        z_scale = (
            z_accommodation.left_scale
            if grain_side == "left"
            else z_accommodation.right_scale
        )

    return y_scale, z_scale


def _grain_build_request(
    material: MaterialState,
    R_grain: np.ndarray,
    periodic_matrix: np.ndarray,
    x_length: float,
    x_offset: float,
    grain_side: GrainSide,
    *,
    inplane_periodic: tuple[bool, bool],
    inplane_box_lengths: tuple[float, float],
    epsilon: float,
    strain_accommodation: Mapping[str, AxisAccommodation],
    exact: bool,
) -> GrainBuildRequest:
    """Build the ``GrainBuildRequest`` shared by both grain-build paths.

    :param material: Crystal identity shared by both grains.
    :param R_grain: Proper rotation matrix for this grain.
    :param periodic_matrix: 3x3 integer orientation matrix: the canonical P/Q matrix
        on the exact path, or the periodic Miller-row matrix on the approximate path.
    :param x_length: Equalized x-slab thickness for this grain (Angstroms).
    :param x_offset: Lab x-coordinate of the grain's lower face (Angstroms).
    :param grain_side: Grain side, either ``"left"`` or ``"right"``.
    :param inplane_periodic: Per-axis in-plane periodicity flags ``(y, z)``. Keyword
        argument, required.
    :param inplane_box_lengths: Shared in-plane box lengths ``(y_dim, z_dim)``
        (Angstroms). Keyword argument, required.
    :param epsilon: Numerical tolerance used for geometric comparisons. Keyword
        argument, required.
    :param strain_accommodation: Mapping from in-plane axis name to its commensurate-
        repeat accommodation. Keyword argument, required.
    :param exact: Whether this request targets the exact decorated-site path. Keyword
        argument, required.
    :return: Grain build request for this grain.
    :raises GBMakerConstructionValueError: If ``grain_side`` is not ``"left"`` or
        ``"right"``, or if any field fails ``GrainBuildRequest`` validation.
    """
    if grain_side not in {"left", "right"}:
        raise GBMakerConstructionValueError(
            f"grain_side must be 'left' or 'right'; got {grain_side!r}."
        )

    y_scale, z_scale = _grain_strain_scales(grain_side, strain_accommodation)

    y_accommodation = strain_accommodation.get("y")
    z_accommodation = strain_accommodation.get("z")
    y_repeats = None
    z_repeats = None
    if exact and y_accommodation is not None:
        y_repeats = (
            y_accommodation.left_repeats
            if grain_side == "left"
            else y_accommodation.right_repeats
        )
    if exact and z_accommodation is not None:
        z_repeats = (
            z_accommodation.left_repeats
            if grain_side == "left"
            else z_accommodation.right_repeats
        )

    return GrainBuildRequest(
        material=material,
        rotation=R_grain,
        periodic_matrix=periodic_matrix,
        grain_side=grain_side,
        x_offset=x_offset,
        x_length=x_length,
        inplane_periodic=inplane_periodic,
        inplane_box_lengths=inplane_box_lengths,
        epsilon=epsilon,
        y_scale=y_scale,
        z_scale=z_scale,
        y_repeats=y_repeats,
        z_repeats=z_repeats,
        exact=exact,
    )


def _generate_exact_grains(
    *,
    material: MaterialState,
    embedding: BoundaryEmbedding | None,
    R_left: np.ndarray,
    R_right: np.ndarray,
    left_x: float,
    right_x: float,
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    inplane_periodic: tuple[bool, bool],
    inplane_box_lengths: tuple[float, float],
    epsilon: float,
    strain_accommodation: Mapping[str, AxisAccommodation],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate both grains using exact decorated-site enumeration.

    :return: ``(left_atoms, right_atoms)``.
    :raises GBMakerConstructionValueError: If the exact embedding is missing P or Q,
        rational basis metadata is unavailable, exact site enumeration fails, exact
        populations disagree with the unit-cell basis, or either grain produces
        invalid Cartesian coordinates.
    """
    if embedding is None or embedding.P is None or embedding.Q is None:
        raise GBMakerConstructionValueError(
            "Exact grain generation requires an embedding with both P and Q."
        )

    left_request = _grain_build_request(
        material,
        R_left,
        embedding.P,
        left_x,
        float(left_bounds[0]),
        "left",
        inplane_periodic=inplane_periodic,
        inplane_box_lengths=inplane_box_lengths,
        epsilon=epsilon,
        strain_accommodation=strain_accommodation,
        exact=True,
    )
    left_atoms = build_exact_grain(left_request).atoms

    right_request = _grain_build_request(
        material,
        R_right,
        embedding.Q,
        right_x,
        float(right_bounds[0]),
        "right",
        inplane_periodic=inplane_periodic,
        inplane_box_lengths=inplane_box_lengths,
        epsilon=epsilon,
        strain_accommodation=strain_accommodation,
        exact=True,
    )
    right_atoms = build_exact_grain(right_request).atoms

    return left_atoms, right_atoms


def _generate_float_grains(
    *,
    material: MaterialState,
    R_left: np.ndarray,
    R_right: np.ndarray,
    left_periodic_miller_rows: np.ndarray,
    right_periodic_miller_rows: np.ndarray,
    left_bounds: np.ndarray,
    right_effective_bounds: np.ndarray,
    vacuum_thickness: float,
    inplane_periodic: tuple[bool, bool],
    inplane_box_lengths: tuple[float, float],
    epsilon: float,
    strain_accommodation: Mapping[str, AxisAccommodation],
) -> tuple[np.ndarray, np.ndarray, GrainBuildResult, np.ndarray, bool, float]:
    """Generate both grains using the floating-point path.

    For ``vacuum=0``, trims one complete right-grain x period from the high-x side
    when enough thickness remains. The trim is origin-complete so multi-species
    conventional-cell groups are preserved.

    :return: ``(left_atoms, right_atoms, right_float_result, right_effective_bounds,
        vacuum0_trim_applied, x_period_right)``.
    """
    left_request = _grain_build_request(
        material,
        R_left,
        left_periodic_miller_rows,
        float(left_bounds[1] - left_bounds[0]),
        float(left_bounds[0]),
        "left",
        inplane_periodic=inplane_periodic,
        inplane_box_lengths=inplane_box_lengths,
        epsilon=epsilon,
        strain_accommodation=strain_accommodation,
        exact=False,
    )
    left_result = build_approximate_grain(left_request)
    left_atoms = left_result.atoms

    x_period_right = _x_period(right_periodic_miller_rows, material.a0)
    vacuum0_trim_applied = False

    right_effective_bounds = np.array(right_effective_bounds, dtype=np.float64)
    right_request = _grain_build_request(
        material,
        R_right,
        right_periodic_miller_rows,
        float(right_effective_bounds[1] - right_effective_bounds[0]),
        float(right_effective_bounds[0]),
        "right",
        inplane_periodic=inplane_periodic,
        inplane_box_lengths=inplane_box_lengths,
        epsilon=epsilon,
        strain_accommodation=strain_accommodation,
        exact=False,
    )
    right_float_result = build_approximate_grain(right_request)

    right_width = right_effective_bounds[1] - right_effective_bounds[0]
    if (
        vacuum_thickness == 0
        and right_width > x_period_right * (1.0 + epsilon)
    ):
        new_upper = right_effective_bounds[1] - x_period_right
        trial_result = trim_grain_result_to_upper_x(
            right_float_result, new_upper, epsilon
        )

        if len(trial_result.atoms) == 0:
            warnings.warn(
                "Vacuum=0 trim would remove all atoms from the right grain. "
                "Skipping trim to preserve a non-empty grain.",
                UserWarning,
                stacklevel=2,
            )
        else:
            right_float_result = trial_result
            right_effective_bounds[1] = new_upper
            vacuum0_trim_applied = True

    right_atoms = right_float_result.atoms

    return (
        left_atoms,
        right_atoms,
        right_float_result,
        right_effective_bounds,
        vacuum0_trim_applied,
        x_period_right,
    )


def _current_gap_metrics(
    left_atoms: np.ndarray,
    right_atoms: np.ndarray,
    left_bounds: np.ndarray,
    right_effective_bounds: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return current central and periodic x-gap metrics.

    :return: ``(central_gap, periodic_gap, left_min_x, right_max_x)``.
    """
    left_min_x = float(np.min(left_atoms["x"]))
    left_max_x = float(np.max(left_atoms["x"]))
    right_min_x = float(np.min(right_atoms["x"]))
    right_max_x = float(np.max(right_atoms["x"]))

    central_gap = right_min_x - left_max_x
    periodic_gap = (
        right_effective_bounds[1] - right_max_x
    ) + (left_min_x - left_bounds[0])

    return central_gap, periodic_gap, left_min_x, right_max_x


def _equalize_float_periodic_gap(
    *,
    right_atoms: np.ndarray,
    central_gap: float,
    left_min_x: float,
    right_max_x: float,
    left_bounds: np.ndarray,
    right_effective_bounds: np.ndarray,
    right_float_result: GrainBuildResult,
    x_period_right: float,
    epsilon: float,
) -> np.ndarray:
    """Equalize the periodic gap by removing whole right-grain x periods.

    Removal is performed through complete-origin filtering so atom groups from the
    same conventional-cell origin are not split.

    :return: Right-grain atom array after equalization (or ``right_atoms`` unchanged
        if equalization was skipped to preserve a non-empty grain).
    """
    excess = right_max_x - (right_effective_bounds[1] - central_gap)
    n_remove = max(1, math.ceil(excess / x_period_right))
    new_upper = right_effective_bounds[1] - n_remove * x_period_right

    if new_upper <= right_effective_bounds[0]:
        warnings.warn(
            f"Gap equalization would remove all atoms from the right grain "
            f"({n_remove} x-periods; right_x = "
            f"{right_effective_bounds[1] - right_effective_bounds[0]:.4f} A, "
            f"x_period = {x_period_right:.4f} A). Skipping equalization to "
            "preserve a non-empty grain.",
            UserWarning,
            stacklevel=2,
        )
        return right_atoms

    grain_width = right_effective_bounds[1] - right_effective_bounds[0]
    if n_remove * x_period_right > grain_width / 2.0:
        warnings.warn(
            f"Gap equalization removed {n_remove} x-period(s) "
            f"({n_remove * x_period_right:.4f} A), more than half the right "
            "grain. The resulting bicrystal may be unusable.",
            UserWarning,
            stacklevel=2,
        )

    trial_result = trim_grain_result_to_upper_x(right_float_result, new_upper, epsilon)

    if len(trial_result.atoms) == 0:
        warnings.warn(
            f"Gap equalization would remove all atoms from the right grain "
            f"({n_remove} x-periods; right_x = "
            f"{right_effective_bounds[1] - right_effective_bounds[0]:.4f} A, "
            f"x_period = {x_period_right:.4f} A). Skipping equalization to "
            "preserve a non-empty grain.",
            UserWarning,
            stacklevel=2,
        )
        return right_atoms

    new_right_atoms = trial_result.atoms

    final_periodic_gap = (
        right_effective_bounds[1] - float(np.max(new_right_atoms["x"]))
    ) + (left_min_x - left_bounds[0])

    if final_periodic_gap < central_gap - epsilon:
        warnings.warn(
            f"Float gap equalization: periodic_gap "
            f"({final_periodic_gap:.4f} A) < central_gap "
            f"({central_gap:.4f} A). Stoichiometry preserved; matching would "
            "require splitting an origin or deleting the right grain.",
            UserWarning,
            stacklevel=2,
        )

    return new_right_atoms


def _equalize_periodic_gap(
    *,
    left_atoms: np.ndarray,
    right_atoms: np.ndarray,
    left_bounds: np.ndarray,
    right_effective_bounds: np.ndarray,
    use_exact: bool,
    right_float_result: GrainBuildResult | None,
    vacuum0_trim_applied: bool,
    x_period_right: float | None,
    vacuum_thickness: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Handle a periodic x-gap mismatch for the selected construction path.

    The floating path retains its existing complete-origin trimming behavior. The
    exact decorated-site path never deletes atomic layers merely to reverse projected
    central/periodic gap ordering. Both projected gaps must nevertheless remain
    nonnegative within the Cartesian tolerance.

    :return: ``(left_atoms, right_atoms)``, with ``right_atoms`` possibly updated by
        float-path equalization.
    :raises GBMakerConstructionValueError: If an exact grain crosses the central or
        periodic x boundary, or if required float metadata is missing.
    """
    central_gap, periodic_gap, left_min_x, right_max_x = _current_gap_metrics(
        left_atoms, right_atoms, left_bounds, right_effective_bounds
    )

    if use_exact:
        if central_gap < -epsilon or periodic_gap < -epsilon:
            raise GBMakerConstructionValueError(
                "Exact decorated-site construction produced an invalid x-boundary "
                f"overlap: central_gap={central_gap:.8f} A, "
                f"periodic_gap={periodic_gap:.8f} A."
            )
        return left_atoms, right_atoms

    if periodic_gap >= central_gap - epsilon:
        return left_atoms, right_atoms

    if vacuum_thickness == 0 and vacuum0_trim_applied:
        return left_atoms, right_atoms

    if right_float_result is None or x_period_right is None:
        raise GBMakerConstructionValueError(
            "Float gap equalization requires a right-grain float build result "
            "and right-grain x period."
        )

    new_right_atoms = _equalize_float_periodic_gap(
        right_atoms=right_atoms,
        central_gap=central_gap,
        left_min_x=left_min_x,
        right_max_x=right_max_x,
        left_bounds=left_bounds,
        right_effective_bounds=right_effective_bounds,
        right_float_result=right_float_result,
        x_period_right=x_period_right,
        epsilon=epsilon,
    )
    return left_atoms, new_right_atoms


def _select_gb_region(
    left_atoms: np.ndarray,
    right_atoms: np.ndarray,
    vacuum_thickness: float,
    left_x: float,
    gb_thickness: float,
) -> np.ndarray:
    """Select the atoms within the grain-boundary-region window.

    :return: Structured atom array for the atoms within ``gb_thickness`` of the
        grain-boundary plane.
    """
    x_gb = vacuum_thickness + left_x
    left_cut = x_gb - gb_thickness / 2.0
    right_cut = x_gb + gb_thickness / 2.0
    left_gb = left_atoms[left_atoms["x"] > left_cut]
    right_gb = right_atoms[right_atoms["x"] < right_cut]
    return np.hstack((left_gb, right_gb))


def _build_grains_for_path(
    *,
    material: MaterialState,
    embedding: BoundaryEmbedding | None,
    R_left: np.ndarray,
    R_right: np.ndarray,
    left_periodic_miller_rows: np.ndarray,
    right_periodic_miller_rows: np.ndarray,
    left_x: float,
    right_x: float,
    x_dim: float,
    vacuum_thickness: float,
    inplane_periodic: tuple[bool, bool],
    inplane_box_lengths: tuple[float, float],
    epsilon: float,
    strain_accommodation: Mapping[str, AxisAccommodation],
) -> tuple[
    np.ndarray,
    np.ndarray,
    bool,
    GrainBuildResult | None,
    np.ndarray,
    np.ndarray,
    bool,
    float | None,
]:
    """Select and run the exact-vs-float grain-generation path.

    :return: ``(left_atoms, right_atoms, use_exact, right_float_result, left_bounds,
        right_effective_bounds, vacuum0_trim_applied, x_period_right)``.
    """
    left_bounds, right_bounds = _grain_x_bounds(left_x, right_x, x_dim, vacuum_thickness)
    right_effective_bounds = right_bounds.copy()

    use_exact = _use_exact_grain_generation(embedding)
    right_float_result: GrainBuildResult | None = None
    vacuum0_trim_applied = False
    x_period_right: float | None = None

    if use_exact:
        left_atoms, right_atoms = _generate_exact_grains(
            material=material,
            embedding=embedding,
            R_left=R_left,
            R_right=R_right,
            left_x=left_x,
            right_x=right_x,
            left_bounds=left_bounds,
            right_bounds=right_bounds,
            inplane_periodic=inplane_periodic,
            inplane_box_lengths=inplane_box_lengths,
            epsilon=epsilon,
            strain_accommodation=strain_accommodation,
        )
    else:
        (
            left_atoms,
            right_atoms,
            right_float_result,
            right_effective_bounds,
            vacuum0_trim_applied,
            x_period_right,
        ) = _generate_float_grains(
            material=material,
            R_left=R_left,
            R_right=R_right,
            left_periodic_miller_rows=left_periodic_miller_rows,
            right_periodic_miller_rows=right_periodic_miller_rows,
            left_bounds=left_bounds,
            right_effective_bounds=right_effective_bounds,
            vacuum_thickness=vacuum_thickness,
            inplane_periodic=inplane_periodic,
            inplane_box_lengths=inplane_box_lengths,
            epsilon=epsilon,
            strain_accommodation=strain_accommodation,
        )

    return (
        left_atoms,
        right_atoms,
        use_exact,
        right_float_result,
        left_bounds,
        right_effective_bounds,
        vacuum0_trim_applied,
        x_period_right,
    )


def assemble_bicrystal(
    *,
    material: MaterialState,
    embedding: BoundaryEmbedding | None,
    R_left: np.ndarray,
    R_right: np.ndarray,
    left_periodic_miller_rows: np.ndarray,
    right_periodic_miller_rows: np.ndarray,
    left_x: float,
    right_x: float,
    x_dim: float,
    vacuum_thickness: float,
    inplane_periodic: tuple[bool, bool],
    inplane_box_lengths: tuple[float, float],
    epsilon: float,
    strain_accommodation: Mapping[str, AxisAccommodation],
    gb_thickness: float,
    box_dims: NDArray[np.floating],
    normal_topology: BoundaryNormalTopology,
    gb_id: int,
) -> BicrystalResult:
    """Assemble a complete bicrystal from resolved orientation and dimension state.

    Builds both grains (exact decorated-site path when ``embedding`` is exact and
    coherent with integer P/Q, otherwise the floating-point lattice-enumeration path),
    equalizes the periodic x gap, concatenates the grains, selects the grain-boundary
    region window, and packages the result. All parameters are keyword-only and, apart
    from ``box_dims``/``normal_topology`` (passed through unchanged from the caller's
    ``gbmaker.dimension.plan_dimensions`` stage), mirror the identically-named
    ``GBOpt.GBMaker`` instance state they are extracted from.

    :param left_x: Left-grain equalized x-slab thickness (Angstroms).
    :param right_x: Right-grain equalized x-slab thickness (Angstroms).
    :param x_dim: Combined left- and right-grain x extent (Angstroms).
    :param strain_accommodation: Mapping from in-plane axis name to its commensurate-
        repeat accommodation.
    :param gb_thickness: Grain-boundary-region window thickness (Angstroms).
    :param box_dims: Planned 3x2 simulation-box bounds (Angstroms), from
        ``gbmaker.dimension.plan_dimensions``.
    :param gb_id: Grain-boundary identifier carried onto the assembled result.
    :return: Fully assembled bicrystal.
    :raises GBMakerConstructionValueError: If exact grain generation requires missing
        P/Q data, if float-path gap equalization lacks right-grain build metadata, if
        an exact grain crosses the central or periodic x boundary, or if a downstream
        grain-generation stage fails.
    """
    (
        left_atoms,
        right_atoms,
        use_exact,
        right_float_result,
        left_bounds,
        right_effective_bounds,
        vacuum0_trim_applied,
        x_period_right,
    ) = _build_grains_for_path(
        material=material,
        embedding=embedding,
        R_left=R_left,
        R_right=R_right,
        left_periodic_miller_rows=left_periodic_miller_rows,
        right_periodic_miller_rows=right_periodic_miller_rows,
        left_x=left_x,
        right_x=right_x,
        x_dim=x_dim,
        vacuum_thickness=vacuum_thickness,
        inplane_periodic=inplane_periodic,
        inplane_box_lengths=inplane_box_lengths,
        epsilon=epsilon,
        strain_accommodation=strain_accommodation,
    )

    left_atoms, right_atoms = _equalize_periodic_gap(
        left_atoms=left_atoms,
        right_atoms=right_atoms,
        left_bounds=left_bounds,
        right_effective_bounds=right_effective_bounds,
        use_exact=use_exact,
        right_float_result=right_float_result,
        vacuum0_trim_applied=vacuum0_trim_applied,
        x_period_right=x_period_right,
        vacuum_thickness=vacuum_thickness,
        epsilon=epsilon,
    )

    whole_system = np.hstack((left_atoms, right_atoms))
    gb_region_atoms = _select_gb_region(
        left_atoms, right_atoms, vacuum_thickness, left_x, gb_thickness
    )

    return BicrystalResult(
        atoms=whole_system,
        left_atoms=left_atoms,
        right_atoms=right_atoms,
        gb_region_atoms=gb_region_atoms,
        box_dims=box_dims,
        normal_topology=normal_topology,
        gb_id=gb_id,
    )


def _plan_orientation_and_dimensions(
    *,
    material: MaterialState,
    misorientation: np.ndarray,
    embedding: BoundaryEmbedding | None,
    x_dim_min: float,
    vacuum_thickness: float,
    normal_topology: BoundaryNormalTopology,
    interaction_distance: float,
    epsilon: float,
    repeat_factor: tuple[int, int],
    mismatch_tol: float | None,
    mismatch_max_cells: int,
    strain_grain: StrainGrainPolicy,
    threshold: float,
):
    """Resolve orientation, then plan periodic spacing and box dimensions.

    Composes ``gbmaker.orientation.resolve_orientation`` and
    ``gbmaker.dimension.plan_periodic_spacing``/``plan_dimensions``, the orientation-
    and dimension-planning stages ``build_bicrystal`` sits above.

    :return: ``(orientation, left_x, right_x, x_dim, plan, box_dims, y_dim, z_dim)``.
    """
    orientation = resolve_orientation(
        np.asarray(misorientation, dtype=np.float64),
        embedding=embedding,
        a0=material.a0,
        threshold=threshold,
    )

    legacy_periodicity_heuristic = (
        embedding is None or embedding.source == "five_dof"
    )

    spacing, left_x, right_x, x_dim = plan_periodic_spacing(
        a0=material.a0,
        left_periodic_miller_rows=orientation.left_periodic_miller_rows,
        right_periodic_miller_rows=orientation.right_periodic_miller_rows,
        x_dim_min=x_dim_min,
        epsilon=epsilon,
        inplane_periodic=orientation.inplane_periodic,
        threshold=threshold,
        legacy_periodicity_heuristic=legacy_periodicity_heuristic,
    )

    use_exact_dims = (
        embedding is not None and embedding.exact and embedding.P is not None
    )

    # ``plan_periodic_spacing``'s return type allows any key to be either a float or
    # the nested "x" dict, since the dict is only ever present under "x". Narrow "y"/
    # "z" to a genuine scalar with a clear domain error rather than letting a
    # malformed spacing mapping reach ``plan_dimensions`` as an unexplained TypeError.
    spacing_y = spacing["y"]
    spacing_z = spacing["z"]
    if isinstance(spacing_y, dict) or isinstance(spacing_z, dict):
        raise GBMakerConstructionValueError(
            "periodic spacing for 'y'/'z' must be scalar"
        )

    plan, _resolved_repeat_factor = plan_dimensions(
        a0=material.a0,
        left_periodic_miller_rows=orientation.left_periodic_miller_rows,
        right_periodic_miller_rows=orientation.right_periodic_miller_rows,
        spacing_y=spacing_y,
        spacing_z=spacing_z,
        repeat_factor=repeat_factor,
        mismatch_tol=mismatch_tol,
        mismatch_max_cells=mismatch_max_cells,
        strain_grain=strain_grain,
        require_exact_pair=use_exact_dims,
        interaction_distance=interaction_distance,
        x_dim=x_dim,
        vacuum_thickness=vacuum_thickness,
        normal_topology=normal_topology,
        epsilon=epsilon,
    )

    box_dims = np.array(plan.box_dims, dtype=float)
    y_dim = float(plan.box_dims[1][1])
    z_dim = float(plan.box_dims[2][1])

    return orientation, left_x, right_x, x_dim, plan, box_dims, y_dim, z_dim


def build_bicrystal(
    *,
    material: MaterialState,
    misorientation: np.ndarray,
    embedding: BoundaryEmbedding | None,
    x_dim_min: float,
    vacuum_thickness: float,
    normal_topology: BoundaryNormalTopology,
    interaction_distance: float,
    gb_thickness: float,
    gb_id: int,
    epsilon: float,
    repeat_factor: tuple[int, int],
    mismatch_tol: float | None,
    mismatch_max_cells: int,
    strain_grain: StrainGrainPolicy,
    threshold: float | None = None,
) -> BicrystalResult:
    """Build a complete bicrystal from resolved material identity and construction knobs.

    Pure end-to-end construction pipeline composing orientation resolution
    (``gbmaker.orientation.resolve_orientation``), periodic-spacing and box-dimension
    planning (``gbmaker.dimension.plan_periodic_spacing``/``plan_dimensions``), grain
    building (``gbmaker.exact_grain``/``gbmaker.approximate_grain``, via
    ``assemble_bicrystal``), and bicrystal assembly. Configuration normalization and
    material resolution are represented by this function's ``material`` parameter and
    scalar keyword arguments, matching the shape ``GBOpt.gbmaker.config`` and
    ``GBOpt.gbmaker.material`` already produce for both of ``GBOpt.GBMaker``'s
    construction entry points (the legacy constructor and ``from_boundary_spec``); this
    function does not re-implement that normalization, since it already exists as its
    own reusable pure stage.

    Runs without any ``GBMaker`` instance, satisfying issue #70's "full exact and
    approximate construction can be executed without instantiating GBMaker"
    acceptance criterion. All parameters are keyword-only.

    :param misorientation: Five-element misorientation/inclination array
        (misorientation Euler angles, then inclination Euler angles).
    :param x_dim_min: Minimum size of one grain along x (Angstroms).
    :param interaction_distance: Interatomic interaction cutoff distance (Angstroms).
    :param gb_thickness: Grain-boundary-region window thickness (Angstroms).
    :param repeat_factor: Nominal ``(y, z)`` unit-cell repeat factors.
    :param mismatch_tol: Mismatch-accommodation tolerance, or ``None`` to disable
        accommodation.
    :param mismatch_max_cells: Maximum unit-cell repeats searched for a commensurate
        pair.
    :param strain_grain: Which grain(s) absorb in-plane strain.
    :param threshold: Maximum periodic spacing (Angstroms) before an in-plane axis is
        treated as non-periodic. Optional, defaults to ``material.a0 * 15``.
    :return: Fully assembled bicrystal.
    :raises GBMakerConstructionValueError: If any composed construction stage fails.
    """
    if threshold is None:
        threshold = material.a0 * 15

    (
        orientation,
        left_x,
        right_x,
        x_dim,
        plan,
        box_dims,
        y_dim,
        z_dim,
    ) = _plan_orientation_and_dimensions(
        material=material,
        misorientation=misorientation,
        embedding=embedding,
        x_dim_min=x_dim_min,
        vacuum_thickness=vacuum_thickness,
        normal_topology=normal_topology,
        interaction_distance=interaction_distance,
        epsilon=epsilon,
        repeat_factor=repeat_factor,
        mismatch_tol=mismatch_tol,
        mismatch_max_cells=mismatch_max_cells,
        strain_grain=strain_grain,
        threshold=threshold,
    )

    return assemble_bicrystal(
        material=material,
        embedding=embedding,
        R_left=orientation.R_left,
        R_right=orientation.R_right,
        left_periodic_miller_rows=orientation.left_periodic_miller_rows,
        right_periodic_miller_rows=orientation.right_periodic_miller_rows,
        left_x=left_x,
        right_x=right_x,
        x_dim=x_dim,
        vacuum_thickness=vacuum_thickness,
        inplane_periodic=orientation.inplane_periodic,
        inplane_box_lengths=(y_dim, z_dim),
        epsilon=epsilon,
        strain_accommodation=dict(plan.accommodation),
        gb_thickness=gb_thickness,
        box_dims=box_dims,
        normal_topology=normal_topology,
        gb_id=gb_id,
    )
