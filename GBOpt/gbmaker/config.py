# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Normalize GBMaker constructor and factory inputs into construction contracts.

Pure, testable normalization for both the legacy ``GBMaker(...)`` constructor path and
the ``GBMaker.from_boundary_spec(...)`` path: scalar/config validation, unit-cell
material resolution, and boundary-spec-to-embedding dispatch. Existing ``BoundarySpec``
and crystallography conversion utilities remain authoritative; this module only
sequences and packages their results. No orientation-matrix assignment, dimension
planning, or grain-generation logic belongs here.

``GBMaker`` itself keeps its own ``__validate`` instance method and per-field static
validators for its property setters, but those now delegate to the pure functions here
so there is a single implementation of each validation rule.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from numbers import Number
from typing import Any, cast

import numpy as np

from GBOpt.BoundarySpec import (
    BoundarySpecError,
    CSLApproxSpec,
    CSLExactSpec,
    FiveDOFSpec,
    PQSpec,
)
from GBOpt.crystallography import (
    csl_approx_spec_to_embedding,
    csl_exact_spec_to_embedding,
    exactify_five_dof,
    five_dof_spec_to_embedding,
    pq_spec_to_embedding,
)
from GBOpt.crystallography.types import CrystallographyError

from .material import resolve_material_state
from .types import (
    _VALID_BOUNDARY_MODES,
    _VALID_STRAIN_GRAIN,
    BoundaryMode,
    GBBuildConfig,
    GBMakerConstructionTypeError,
    GBMakerConstructionValueError,
    ResolvedBoundaryInput,
    StrainGrainPolicy,
)


def _validate_scalar(
    value: Any,
    expected_types: type | tuple[type, ...],
    parameter_name: str,
    *,
    positive: bool = False,
    expected_length: int | None = None,
    strictly_positive: bool = False,
):
    """Validate one legacy constructor/setter scalar field.

    This is the single implementation behind ``GBMaker.__validate`` (used by the legacy
    constructor and every property setter) and ``normalize_legacy_config``.

    :param value: The value to validate.
    :param expected_types: Single type or tuple containing the valid types for value.
    :param parameter_name: The name of the parameter.
    :param positive: Whether or not the value should be positive (>= 0), optional,
        defaults to False.
    :param expected_length: Specific to sequences or arrays. The expected length of the
        sequence or array, optional, defaults to None.
    :param strictly_positive: Supercedes ``positive`` by enforcing value > 0. Optional,
        defaults to False.
    :raises GBMakerConstructionTypeError: If the type of the value does not match the
        expected type(s).
    :raises GBMakerConstructionValueError: If invalid values are given for the
        specified parameter.
    :return: The validated value.
    """
    if not isinstance(expected_types, tuple):
        expected_types = (expected_types,)
    if not any(isinstance(value, t) for t in expected_types) and not isinstance(
        value, np.generic
    ):
        expected_type_names = ", ".join(t.__name__ for t in expected_types)
        raise GBMakerConstructionTypeError(
            f"{parameter_name} must be of type {expected_type_names}."
        )

    if strictly_positive and isinstance(value, Number):
        if value <= 0:
            raise GBMakerConstructionValueError(
                f"{parameter_name} must be strictly positive"
            )
        if value < np.finfo(np.float64).eps:
            warnings.warn(
                f"{parameter_name} ({value}) is below machine epsilon "
                f"({np.finfo(np.float64).eps:.2e}) and may not have any "
                "practical effect."
            )
    elif positive and isinstance(value, Number) and value < 0:
        raise GBMakerConstructionValueError(
            f"{parameter_name} must be a positive value.")

    if (
        isinstance(value, (Sequence, np.ndarray))
        and all([isinstance(val, Number) for val in value])
        and positive
    ):
        for val in value:
            if val < 0:
                raise GBMakerConstructionValueError(
                    f"{parameter_name} must have all positive values."
                )

    if (
        expected_length is not None
        and isinstance(value, (Sequence, np.ndarray))
        and len(value) != expected_length
    ):
        raise GBMakerConstructionValueError(
            f"{parameter_name} must have {expected_length} elements."
        )

    if parameter_name == "structure" and value not in [
        "fcc",
        "bcc",
        "sc",
        "diamond",
        "fluorite",
        "rocksalt",
        "zincblende",
    ]:
        raise GBMakerConstructionValueError(
            f"{parameter_name} ({value}) must be one of ['fcc', 'bcc', 'sc', "
            + "'diamond', 'fluorite', 'rocksalt', 'zincblende']."
        )

    if parameter_name == "repeat_factor":
        if isinstance(value, int):
            values = [value, value]
        else:
            values = list(value)
            if not all(isinstance(val, int) for val in values):
                raise GBMakerConstructionValueError(
                    "repeat_factor must be a sequence of type int."
                )

        if any(val < 2 for val in values):
            warnings.warn(
                "Recommended repeat factor is at least 2.",
                UserWarning,
                stacklevel=2,
            )

        value = values
    return value


def validate_mismatch_tol(value: object) -> float | None:
    """Return a validated mismatch-accommodation tolerance.

    ``None`` disables mismatch accommodation. Otherwise, the value is converted to
    ``float`` and interpreted as the maximum allowed relative mismatch in the
    one-dimensional commensurability search.

    :param value: Candidate mismatch tolerance.
    :return: ``None`` if mismatch accommodation is disabled; otherwise a finite,
        non-negative floating-point tolerance.
    :raises GBMakerConstructionValueError: If ``value`` is boolean, non-numeric,
        infinite, NaN, or negative.
    """
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        raise GBMakerConstructionValueError(
            f"mismatch_tol must be finite and non-negative; got {value!r}."
        )

    try:
        tol = float(value)
    except (TypeError, ValueError) as exc:
        raise GBMakerConstructionValueError(
            f"mismatch_tol must be finite and non-negative; got {value!r}."
        ) from exc

    if not np.isfinite(tol) or tol < 0.0:
        raise GBMakerConstructionValueError(
            f"mismatch_tol must be finite and non-negative; got {value!r}."
        )

    return tol


def validate_mismatch_max_cells(value: object) -> int:
    """Return a validated commensurability-search repeat-count bound.

    :param value: Candidate maximum repeat count.
    :return: Positive integer repeat-count bound.
    :raises GBMakerConstructionValueError: If ``value`` is boolean, non-integral, or
        less than one.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise GBMakerConstructionValueError(
            f"mismatch_max_cells must be a positive integer; got {value!r}."
        )

    max_cells = int(value)
    if max_cells < 1:
        raise GBMakerConstructionValueError(
            f"mismatch_max_cells must be a positive integer; got {value!r}."
        )

    return max_cells


def validate_strain_grain(value: str) -> str:
    """Return a validated mismatch-strain policy.

    :param value: Grain strain policy. Supported values are ``"both"``, ``"left"``, and
        ``"right"``.
    :return: Validated strain policy.
    :raises GBMakerConstructionValueError: If ``value`` is not one of ``"both"``,
        ``"left"``, or ``"right"``.
    """
    if value not in _VALID_STRAIN_GRAIN:
        raise GBMakerConstructionValueError(
            f"Invalid strain_grain={value!r}. "
            f"Must be one of {sorted(_VALID_STRAIN_GRAIN)}."
        )
    return value


def validate_boundary_mode(value: str) -> str:
    """Return a validated boundary-spec construction mode.

    :param value: Boundary-spec construction mode. Supported values are ``"exact"``,
        ``"approximate"``, and ``"prefer_exact"``.
    :return: Validated construction mode.
    :raises GBMakerConstructionValueError: If ``value`` is not one of the supported
        modes.
    """
    if not isinstance(value, str):
        raise GBMakerConstructionValueError(
            f"mode must be one of {sorted(_VALID_BOUNDARY_MODES)}; got {value!r}."
        )

    if value not in _VALID_BOUNDARY_MODES:
        raise GBMakerConstructionValueError(
            f"mode must be one of {sorted(_VALID_BOUNDARY_MODES)}; got {value!r}."
        )

    return value


def validate_exact_limit(value: object, name: str) -> int:
    """Return a validated positive exact-construction limit."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise GBMakerConstructionValueError(
            f"{name} must be a positive integer; got {value!r}."
        )

    limit = int(value)
    if limit <= 0:
        raise GBMakerConstructionValueError(
            f"{name} must be a positive integer; got {value!r}."
        )

    return limit


def normalize_legacy_config(
    a0: float,
    structure: str,
    gb_thickness: float,
    atom_types: str | tuple[str, ...],
    *,
    repeat_factor: int | Sequence[int] = 2,
    x_dim_min: float = 50,
    vacuum: float = 10,
    interaction_distance: float = 15.0,
    gb_id: int = 1,
    epsilon: float = 1e-10,
    mismatch_tol: float | None = None,
    mismatch_max_cells: int = 50,
    strain_grain: str = "both",
) -> GBBuildConfig:
    """Normalize legacy ``GBMaker(...)`` constructor inputs into a ``GBBuildConfig``.

    Applies the same field-by-field validation as the legacy constructor, in the same
    order, then resolves the crystal unit cell via :func:`resolve_material_state`.

    :param a0: Crystal lattice parameter (Angstroms).
    :param structure: Crystal structure name.
    :param gb_thickness: Width of the grain-boundary region (Angstroms).
    :param atom_types: Atom type string or tuple of atom type strings.
    :param repeat_factor: In-plane repeat factor(s). Keyword argument, optional,
        defaults to ``2``.
    :param x_dim_min: Minimum grain thickness in x (Angstroms). Keyword argument,
        optional, defaults to ``50``.
    :param vacuum: Vacuum thickness (Angstroms). Keyword argument, optional, defaults to
        ``10``.
    :param interaction_distance: Maximum atom interaction distance (Angstroms). Keyword
        argument, optional, defaults to ``15.0``.
    :param gb_id: Grain-boundary identifier. Keyword argument, optional, defaults to
        ``1``.
    :param epsilon: Numerical tolerance for geometric comparisons. Keyword argument,
        optional, defaults to ``1e-10``.
    :param mismatch_tol: Maximum relative in-plane mismatch, or ``None`` to disable
        mismatch accommodation. Keyword argument, optional, defaults to ``None``.
    :param mismatch_max_cells: Maximum repeat count searched per axis. Keyword
        argument, optional, defaults to ``50``.
    :param strain_grain: In-plane strain policy. Keyword argument, optional, defaults to
        ``"both"``.
    :return: Normalized build configuration with its material identity resolved.
    :raises GBMakerConstructionTypeError: If a field has an unsupported type.
    :raises GBMakerConstructionValueError: If a field value is invalid.
    :raises UnitCellError: If unit-cell construction fails for the resolved material
        identity.
    """
    a0 = _validate_scalar(a0, Number, "a0", positive=True)
    structure = _validate_scalar(structure, str, "structure")
    gb_thickness = _validate_scalar(
        gb_thickness, Number, "gb_thickness", positive=True
    )
    epsilon = _validate_scalar(epsilon, Number, "epsilon", strictly_positive=True)
    validated_repeat_factor = _validate_scalar(
        repeat_factor,
        (int, Sequence),
        "repeat_factor",
        expected_length=2,
        positive=True,
    )
    x_dim_min = _validate_scalar(x_dim_min, Number, "x_dim_min", positive=True)
    vacuum = _validate_scalar(vacuum, Number, "vacuum_thickness", positive=True)
    interaction_distance = _validate_scalar(
        interaction_distance, Number, "interaction_distance", positive=True
    )
    gb_id = _validate_scalar(gb_id, int, "id", positive=True)
    mismatch_tol = validate_mismatch_tol(mismatch_tol)
    mismatch_max_cells = validate_mismatch_max_cells(mismatch_max_cells)
    strain_grain = validate_strain_grain(strain_grain)

    material = resolve_material_state(a0, structure, atom_types)

    return GBBuildConfig(
        material=material,
        gb_thickness=gb_thickness,
        repeat_factor=cast("tuple[int, int]", tuple(validated_repeat_factor)),
        x_dim_min=x_dim_min,
        vacuum=vacuum,
        interaction_distance=interaction_distance,
        gb_id=gb_id,
        epsilon=epsilon,
        mismatch_tol=mismatch_tol,
        mismatch_max_cells=mismatch_max_cells,
        strain_grain=cast(StrainGrainPolicy, strain_grain),
    )


def resolve_boundary_input(
    boundary: PQSpec | CSLExactSpec | CSLApproxSpec | FiveDOFSpec,
    mode: str,
    *,
    max_primitive_area_index: int,
    max_pq_determinant: int,
) -> ResolvedBoundaryInput:
    """Resolve a boundary-spec dataclass into a canonical embedding under a mode.

    Mirrors ``GBMaker.from_boundary_spec``'s exact/prefer_exact/approximate dispatch
    table. Existing ``BoundarySpec`` and crystallography conversion utilities remain
    authoritative; this function only sequences their results.

    :param boundary: Boundary specification to resolve.
    :param mode: Construction mode: ``"exact"``, ``"approximate"``, or
        ``"prefer_exact"``. Must already be validated by :func:`validate_boundary_mode`.
    :param max_primitive_area_index: Exact-cell primitive-reconstruction area-index
        limit. Keyword argument, required.
    :param max_pq_determinant: Exact-cell P/Q determinant limit. Keyword argument,
        required.
    :return: Resolved boundary input carrying the canonical embedding.
    :raises BoundarySpecError: If the requested mode is incompatible with the boundary
        type, exact boundary conversion or exactification fails, or an exact-cell limit
        is exceeded.
    :raises NotImplementedError: If the boundary type is unsupported or the requested
        type/mode combination is recognized but not implemented.
    """
    if isinstance(boundary, PQSpec):
        if mode == "approximate":
            raise NotImplementedError(
                f"Construction mode '{mode}' is not yet supported for PQSpec; "
                f"use mode='exact' or mode='prefer_exact'."
            )
        embedding = pq_spec_to_embedding(
            boundary,
            max_primitive_area_index=max_primitive_area_index,
            max_pq_determinant=max_pq_determinant,
        )

    elif isinstance(boundary, CSLExactSpec):
        if mode == "approximate":
            raise NotImplementedError(
                f"Construction mode '{mode}' is not yet supported for CSLExactSpec; "
                f"use mode='exact' or mode='prefer_exact'."
            )
        embedding = csl_exact_spec_to_embedding(
            boundary,
            max_primitive_area_index=max_primitive_area_index,
            max_pq_determinant=max_pq_determinant,
        )

    elif isinstance(boundary, CSLApproxSpec):
        if mode == "exact":
            raise BoundarySpecError(
                "CSLApproxSpec cannot be used with mode='exact': no integer "
                "quaternion is available for exactification. Use CSLExactSpec "
                "for an exact construction, or mode='approximate'."
            )
        if mode == "prefer_exact":
            warnings.warn(
                "CSLApproxSpec cannot be exactified from a floating-point "
                "angle; falling back to mode='approximate'.",
                UserWarning,
                stacklevel=2,
            )
        embedding = csl_approx_spec_to_embedding(boundary)

    elif isinstance(boundary, FiveDOFSpec):
        params = np.asarray(boundary.params, dtype=float)

        if mode == "exact":
            try:
                P, Q = exactify_five_dof(
                    params,
                    max_primitive_area_index=max_primitive_area_index,
                    max_pq_determinant=max_pq_determinant,
                )
            except CrystallographyError as exc:
                raise BoundarySpecError(str(exc)) from exc

            embedding = pq_spec_to_embedding(
                PQSpec(P=P, Q=Q, basis_mode="primitive"),
                max_primitive_area_index=max_primitive_area_index,
                max_pq_determinant=max_pq_determinant,
            )

        elif mode == "prefer_exact":
            try:
                P, Q = exactify_five_dof(
                    params,
                    max_primitive_area_index=max_primitive_area_index,
                    max_pq_determinant=max_pq_determinant,
                )
            except (BoundarySpecError, CrystallographyError) as exc:
                warnings.warn(
                    "FiveDOFSpec exactification failed; falling back to "
                    f"mode='approximate'. Reason: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                embedding = five_dof_spec_to_embedding(boundary)
            else:
                embedding = pq_spec_to_embedding(
                    PQSpec(P=P, Q=Q, basis_mode="primitive"),
                    max_primitive_area_index=max_primitive_area_index,
                    max_pq_determinant=max_pq_determinant,
                )

        else:
            embedding = five_dof_spec_to_embedding(boundary)

    else:
        raise NotImplementedError(
            "from_boundary_spec does not yet support boundary objects of type "
            f"{type(boundary).__name__}."
        )

    return ResolvedBoundaryInput(
        embedding=embedding,
        mode=cast(BoundaryMode, mode),
        max_primitive_area_index=max_primitive_area_index,
        max_pq_determinant=max_pq_determinant,
    )
