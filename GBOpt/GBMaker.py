# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Grain boundary builder utilities."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np

from GBOpt.BoundarySpec import (
    CSLApproxSpec,
    CSLExactSpec,
    FiveDOFSpec,
    PQSpec,
)
from GBOpt.BoundaryTopology import BoundaryNormalTopology
from GBOpt.crystallography._limits import (
    DEFAULT_MAX_PQ_DETERMINANT,
    DEFAULT_MAX_PRIMITIVE_AREA_INDEX,
)
from GBOpt.gbmaker.config import (
    _validate_scalar,
    normalize_legacy_config,
    resolve_boundary_input,
    validate_boundary_mode,
    validate_exact_limit,
    validate_mismatch_max_cells,
    validate_mismatch_tol,
    validate_strain_grain,
)
from GBOpt.gbmaker.dimension import (
    _find_commensurate_pair as _plan_find_commensurate_pair,
)
from GBOpt.gbmaker.dimension import (
    _plan_box_dims,
    plan_dimensions,
    plan_periodic_spacing,
)
from GBOpt.gbmaker.orientation import (
    _decompose_misorientation,
    _reduce_integer_row,
    _x_period,
    resolve_orientation,
)
from GBOpt.gbmaker.types import (
    AxisAccommodation,
    GBMakerConstructionTypeError,
    GBMakerConstructionValueError,
)
from GBOpt.gbmaker_supercell import (
    build_supercell_matrix,
    enumerate_supercell_sites,
)
from GBOpt.UnitCell import UnitCell

_LEGACY_CONSTRUCTOR_DEPRECATION = (
    "GBMaker(...) is deprecated; use GBMaker.from_boundary_spec(...)."
)


def _normalize_vacuum_topology(
    vacuum: float,
    *,
    tolerance: float,
) -> tuple[float, BoundaryNormalTopology]:
    """Normalize vacuum thickness and its boundary-normal topology.

    :param vacuum: Validated nonnegative vacuum thickness in angstroms.
    :param tolerance: Keyword argument, required. Coordinate tolerance in angstroms.
    :return: Normalized vacuum thickness and explicit topology.
    """
    if np.isclose(vacuum, 0.0, atol=tolerance, rtol=0.0):
        return 0.0, BoundaryNormalTopology.PERIODIC_BICRYSTAL
    return float(vacuum), BoundaryNormalTopology.SINGLE_INTERFACE_SLAB


class GBMakerError(Exception):
    """Base class for Exceptions in the GBMaker class."""


class GBMakerTypeError(GBMakerError, TypeError):
    """Exception raised when an invalid type is assigned to a GBMaker attribute."""


class GBMakerValueError(GBMakerError, ValueError):
    """Exception raised when an invalid value is assigned to a GBMaker attribute."""


def _find_commensurate_pair(
    d1: float,
    d2: float,
    *,
    tol: float = 0.005,
    max_n: int = 50,
) -> tuple[int, int, float, float] | None:
    """Find a small commensurate repeat pair for two one-dimensional periods.

    Thin wrapper delegating to
    ``GBOpt.gbmaker.dimension._find_commensurate_pair``, kept as a module-level
    compatibility alias since existing code imports
    ``GBOpt.GBMaker._find_commensurate_pair`` directly.

    :param d1: Period of the first grain along the selected in-plane axis (Angstroms).
    :param d2: Period of the second grain along the selected in-plane axis (Angstroms).
    :param tol: Maximum allowed relative mismatch. Keyword parameter, optional, defaults
        to ``0.005``.
    :param max_n: Maximum repeat count allowed for either grain. Keyword parameter,
        optional, defaults to ``50``.
    :return: ``(n1, n2, n1*d1, n2*d2)`` for the best admissible pair, or ``None`` if no
        admissible pair exists within ``max_n``.
    :raises GBMakerValueError: If ``d1`` or ``d2`` is not finite and positive, if
        ``tol`` is not finite and non-negative, or if ``max_n`` is not a positive
        integer.
    """
    try:
        return _plan_find_commensurate_pair(d1, d2, tol=tol, max_n=max_n)
    except GBMakerConstructionValueError as exc:
        raise GBMakerValueError(str(exc)) from exc


@dataclass(frozen=True)
class _FloatGrainBuildResult:
    """Float-path atoms with conventional-cell origin metadata.

    Carries the result of the floating-point grain-build path through trimming,
    clipping, wrapping, and deduplication operations that must preserve complete
    conventional-cell origins.

    ``atoms`` and ``origin_ids`` are parallel one-dimensional arrays. Each atom has one
    origin identifier, and atoms sharing an origin identifier belong to the same
    generated conventional-cell origin group. ``basis_size`` gives the expected number
    of atoms in each complete origin group.

    The dataclass is frozen to prevent rebinding the result fields, but the underlying
    NumPy arrays remain mutable because the generated atom arrays are later assigned
    into GBMaker state and may be modified by downstream geometry operations.

    :param atoms: Structured atom array for the generated grain after float-path
        selection and filtering.
    :param origin_ids: Integer array parallel to ``atoms``. Each value identifies the
        generated conventional-cell origin that produced the corresponding atom.
    :param basis_size: Number of atoms generated per conventional-cell origin.
        Complete-origin filtering assumes retained atom groups have this size.
    """

    atoms: np.ndarray
    origin_ids: np.ndarray
    basis_size: int


def wrap_reduced_coordinate(reduced_coord: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Wrap reduced coordinates into [0, 1) and snap both periodic faces to 0.

    :param reduced_coord: Reduced coordinates to wrap.
    :param tol: Tolerance in reduced-coordinate units. Optional, defaults to 1e-8
    :return: Wrapped reduced coordinates in [0, 1).
    """
    if not math.isfinite(tol):
        raise GBMakerValueError("Reduced-coordinate tolerance must be finite.")
    if tol < 0:
        raise GBMakerValueError("Reduced-coordinate tolerance must be non-negative.")

    wrapped = np.mod(np.asarray(reduced_coord, dtype=np.float64), 1.0)
    return np.where(
        (wrapped < tol) | ((1.0 - wrapped) < tol),
        0.0,
        wrapped,
    )


def _miller_row_norm(row: Sequence[object] | np.ndarray) -> float:
    """Return the Euclidean norm of a nonzero integer Miller-index row.

    Computes ``sqrt(h*h + k*k + l*l)`` using Python ``int`` arithmetic for the squared
    norm. This avoids fixed-width NumPy integer overflow and avoids object-dtype NumPy
    ufuncs.

    :param row: Nonzero integer Miller-index row ``(h, k, l)``.
    :return: Euclidean norm of the Miller-index row.
    :raises GBMakerValueError: If ``row`` is not a three-component integer row or if the
        row is zero.
    """
    values = tuple(row)
    if len(values) != 3:
        raise GBMakerValueError(
            f"Miller-index row must have exactly three components; got {values!r}."
        )

    integers: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise GBMakerValueError(
                f"Miller-index row components must be integers; got {values!r}."
            )
        integers.append(int(value))

    squared_norm = sum(value * value for value in integers)
    if squared_norm == 0:
        raise GBMakerValueError("Miller-index row must be nonzero.")

    return math.sqrt(squared_norm)


class GBMaker:
    """Create a grain-boundary structure from user-defined parameters.

    The grain-boundary normal is aligned with the lab-frame x-axis. Direct
    construction uses the legacy Euler-angle/misorientation path. Exact
    boundary-spec construction should use ``from_boundary_spec``.

    :param a0: Crystal lattice parameter (Angstroms).
    :param structure: Crystal structure string. Supported values are ``"fcc"``,
        ``"bcc"``, ``"sc"``, ``"diamond"``, ``"fluorite"``, ``"rocksalt"``, and
        ``"zincblende"``.
    :param gb_thickness: Width of the grain-boundary region (Angstroms).
    :param misorientation: Misorientation angles ``(alpha, beta, gamma, theta, phi)`` in
        radians. ``alpha``, ``beta``, and ``gamma`` are ZXZ Euler angles; ``theta`` and
        ``phi`` are additional rotations about y and z.
    :param atom_types: Atom type string or tuple of atom type strings.
    :param repeat_factor: Number of repeats in the y and z directions. A single integer
        applies to both directions; a two-value sequence applies to y and z
        respectively. Keyword parameter, optional, defaults to ``2``.
    :param x_dim_min: Minimum size of one grain in the x dimension (Angstroms). Keyword
        parameter, optional, defaults to ``50``.
    :param vacuum: Vacuum thickness around the grains in the x dimension (Angstroms).
        Keyword parameter, optional, defaults to ``10``.
    :param interaction_distance: Maximum atom interaction distance (Angstroms). Keyword
        parameter, optional, defaults to ``15.0``.
    :param gb_id: Grain-boundary identifier. Keyword parameter, optional, defaults to
        ``1``.
    :param epsilon: Numerical tolerance used for geometric comparisons. Keyword
        parameter, optional, defaults to ``1e-10``.

    Internal keyword parameters:

    :param _embedding: Boundary embedding supplied by ``_from_boundary_embedding``. When
        present, the embedding provides the left/right rotations and, for exact coherent
        boundaries, the integer P/Q periodic Miller rows. Internal keyword parameter,
        optional, defaults to ``None``.
    :param _mismatch_tol: Maximum allowed relative mismatch for commensurate in-plane
        repeat search. ``None`` disables mismatch accommodation. Internal keyword
        parameter, optional, defaults to ``None``.
    :param _mismatch_max_cells: Maximum repeat count allowed for either grain in each
        one-dimensional commensurability search. Internal keyword parameter, optional,
        defaults to ``50``.
    :param _strain_grain: Grain strain policy used when mismatch accommodation is
        active. Supported values are ``"both"``, ``"left"``, and ``"right"``. Internal
        keyword parameter, optional, defaults to ``"both"``.
    """

    def __init__(self, a0: float, structure: str, gb_thickness: float,
                 misorientation: np.ndarray, atom_types: str | tuple[str, ...], *,
                 _embedding=None,
                 _mismatch_tol=None,
                 _mismatch_max_cells: int = 50,
                 _strain_grain: str = "both",
                 repeat_factor: int | Sequence[int] = 2, x_dim_min: float = 50,
                 vacuum: float = 10, interaction_distance: float = 15.0,
                 gb_id: int = 1, epsilon: float = 1e-10):
        if _embedding is None:
            warnings.warn(
                _LEGACY_CONSTRUCTOR_DEPRECATION,
                DeprecationWarning,
                stacklevel=2,
            )

        try:
            config = normalize_legacy_config(
                a0, structure, gb_thickness, atom_types,
                repeat_factor=repeat_factor,
                x_dim_min=x_dim_min,
                vacuum=vacuum,
                interaction_distance=interaction_distance,
                gb_id=gb_id,
                epsilon=epsilon,
                mismatch_tol=_mismatch_tol,
                mismatch_max_cells=_mismatch_max_cells,
                strain_grain=_strain_grain,
            )
        except GBMakerConstructionTypeError as exc:
            raise GBMakerTypeError(str(exc)) from exc
        except GBMakerConstructionValueError as exc:
            raise GBMakerValueError(str(exc)) from exc

        self.__a0 = config.material.a0
        self.__structure = config.material.structure
        self.__gb_thickness = config.gb_thickness
        self.__epsilon = config.epsilon
        self.__assign_orientations(
            self.__validate(
                np.asarray(misorientation),
                np.ndarray,
                "misorientation",
                expected_length=5,
            )
        )
        self.__repeat_factor = list(config.repeat_factor)
        self.__x_dim_min = config.x_dim_min
        self.__vacuum_thickness, self.__normal_topology = (
            _normalize_vacuum_topology(
                config.vacuum,
                tolerance=self.__epsilon,
            )
        )
        self.__interaction_distance = config.interaction_distance
        self.__id = config.gb_id
        self.__inplane_periodic = (True, True)
        self.__embedding = _embedding
        self.__mismatch_tol = config.mismatch_tol
        self.__mismatch_max_cells = config.mismatch_max_cells
        self.__strain_grain = config.strain_grain
        # Maps axis name ("y" or "z") to commensurate repeat metadata when
        # mismatch accommodation is active; empty when mismatch_tol is None.
        self.__strain_accommodation: dict[str, AxisAccommodation] = {}

        self.__unit_cell = config.material.unit_cell
        self.__spacing = self.__calculate_periodic_spacing()  # periodic distances dict
        self.__update_dims()

        self.__radius = config.material.a0 * self.__unit_cell.radius  # atom radius
        self.__box_dims = self.__calculate_box_dimensions()

    @classmethod
    def _from_boundary_embedding(
        cls,
        embedding,
        *,
        a0: float,
        structure: str,
        atom_types,
        misorientation=None,
        gb_thickness: float = 0.0,
        repeat_factor=2,
        x_dim_min: float = 50,
        vacuum: float = 10,
        interaction_distance: float = 15.0,
        gb_id: int = 1,
        mismatch_tol=None,
        mismatch_max_cells: int = 50,
        strain_grain: str = "both",
    ) -> GBMaker:
        """Build a GBMaker from a BoundaryEmbedding.

        :param embedding: A BoundaryEmbedding produced by an input adapter. When
            ``embedding.exact`` is True and P/Q are present, the integer matrices are
            used directly as the approx rotation matrices, bypassing integer-row
            approximation (see ``GBOpt.gbmaker.orientation.resolve_orientation``).
            When ``embedding.exact`` is False, R_left/R_right are used on the existing
            floating-point approximation path. ``embedding.coherent`` sets
            ``inplane_periodic``.
        :param a0: Crystal lattice parameter (Angstroms).
        :param structure: Crystal structure string.
        :param atom_types: Atom type string or tuple of strings.
        :param misorientation: Optional legacy 5-DOF parameters to retain on
            the constructed object. Exact embeddings default to zeros.
        :param gb_thickness: Width of the GB region (Angstroms), default 0.
        :param repeat_factor: In-plane repeat factor(s), default 2.
        :param x_dim_min: Minimum grain thickness in x (Angstroms), default 50.
        :param vacuum: Vacuum thickness (Angstroms), default 10.
        :param interaction_distance: Maximum atom interaction distance, default 15.
        :param gb_id: Grain boundary identifier, default 1.
        :param mismatch_tol: Maximum allowed relative mismatch for commensurate in-plane
            repeat search. ``None`` disables mismatch accommodation. Keyword parameter,
            optional, defaults to ``None``.
        :param mismatch_max_cells: Maximum repeat count allowed for either grain in each
            one-dimensional commensurability search. Keyword parameter, optional,
            defaults to ``50``.
        :param strain_grain: Grain strain policy used when mismatch accommodation is
            active. Supported values are ``"both"``, ``"left"``, and ``"right"``.
            Keyword parameter, optional, defaults to ``"both"``.
        :return: Fully initialized GBMaker instance.
        """
        if misorientation is None:
            misorientation = np.zeros(5)
        return cls(
            a0, structure, gb_thickness, np.asarray(misorientation), atom_types,
            _embedding=embedding,
            _mismatch_tol=mismatch_tol,
            _mismatch_max_cells=mismatch_max_cells,
            _strain_grain=strain_grain,
            repeat_factor=repeat_factor,
            x_dim_min=x_dim_min,
            vacuum=vacuum,
            interaction_distance=interaction_distance,
            gb_id=gb_id,
        )

    @classmethod
    def from_boundary_spec(
        cls,
        a0: float,
        structure: str,
        atom_types: str | tuple[str, ...],
        boundary: PQSpec | CSLExactSpec | CSLApproxSpec | FiveDOFSpec,
        mode: str = "exact",
        *,
        max_primitive_area_index: int = DEFAULT_MAX_PRIMITIVE_AREA_INDEX,
        max_pq_determinant: int = DEFAULT_MAX_PQ_DETERMINANT,
        gb_thickness: float = 0.0,
        repeat_factor: int | Sequence[int] = 2,
        x_dim_min: float = 50,
        vacuum: float = 10,
        interaction_distance: float = 15.0,
        gb_id: int = 1,
        mismatch_tol: float | None = None,
        mismatch_max_cells: int = 50,
        strain_grain: str = "both",
    ) -> GBMaker:
        """Build a grain boundary from a boundary-spec dataclass.

        The supported boundary types and construction behavior are:

        ================ ================ ================== =========================
        Boundary type    ``exact``        ``approximate``    ``prefer_exact``
        ================ ================ ================== =========================
        ``PQSpec``       exact P/Q        not implemented    exact P/Q
        ``CSLExactSpec`` exact CSL        not implemented    exact CSL
        ``CSLApproxSpec`` rejected        approximate        warning, then approximate
        ``FiveDOFSpec``  exactify or fail approximate        exactify; warning fallback
        ================ ================ ================== =========================

        For ``FiveDOFSpec``, exact construction is currently available only when the
        floating-point boundary can be rationalized into a supported cubic CSL within
        the configured exactification bounds and tolerances. Under ``mode="exact"``,
        failure is reported as an exception. Under ``mode="prefer_exact"``, failure
        emits a warning and uses the approximate orientation path.

        ``max_primitive_area_index`` and ``max_pq_determinant`` are separate exact-cell
        limits. The former bounds the minimal in-plane CSL topology where primitive
        reconstruction is performed. It does not apply to
        ``PQSpec(basis_mode="supplied")``. The latter bounds the absolute determinants
        of the exact P/Q matrices used for grain construction. Both arguments are
        validated as positive integers on every call, although they affect only
        exact-construction paths.

        When ``mismatch_tol`` is ``None``, the shared in-plane simulation box is derived
        from ``repeat_factor`` and the larger left/right period along each in-plane
        axis. Exact construction requires that box to be commensurate with both grains.

        When ``mismatch_tol`` is provided, integer repeat pairs satisfying ``n_left *
        d_left ~= n_right * d_right`` are searched up to ``mismatch_max_cells``. The
        resulting lengths are reconciled according to ``strain_grain``. Exact
        construction fails when no admissible pair exists; approximate construction
        warns and falls back to the repeat-factor box.

        :param a0: Crystal lattice parameter in Angstroms.
        :param structure: Crystal structure name. Supported values are ``"fcc"``,
            ``"bcc"``, ``"sc"``, ``"diamond"``, ``"fluorite"``, ``"rocksalt"``, and
            ``"zincblende"``.
        :param atom_types: Atom type string or tuple of atom type strings accepted by
            ``UnitCell``.
        :param boundary: Boundary specification to construct. Supported values are
            ``PQSpec``, ``CSLExactSpec``, ``CSLApproxSpec``, and ``FiveDOFSpec``.
        :param mode: Construction policy: ``"exact"``, ``"approximate"``, or
            ``"prefer_exact"``. Optional, defaults to ``"exact"``.
        :param max_primitive_area_index: Maximum permitted minimal in-plane CSL area
            index for exact primitive reconstruction. This limit does not apply to
            supplied-mode P/Q embeddings. Keyword argument, optional, defaults to
            ``10000``.
        :param max_pq_determinant: Maximum permitted absolute determinant of each exact
            P/Q matrix used for construction. Keyword argument, optional, defaults to
            ``10000``.
        :param gb_thickness: Width of the grain-boundary region in Angstroms. Keyword
            argument, optional, defaults to ``0.0``.
        :param repeat_factor: In-plane repeat factor. A single integer applies to both
            in-plane axes; a two-value sequence applies to y and z respectively. Keyword
            argument, optional, defaults to ``2``.
        :param x_dim_min: Minimum size of one grain along x in Angstroms. Keyword
            argument, optional, defaults to ``50``.
        :param vacuum: Vacuum thickness around the bicrystal along x in Angstroms.
            Keyword argument, optional, defaults to ``10``.
        :param interaction_distance: Maximum atom interaction distance in Angstroms.
            In-plane dimensions are enlarged when necessary to satisfy twice this
            distance. Keyword argument, optional, defaults to ``15.0``.
        :param gb_id: Grain-boundary identifier. Keyword argument, optional, defaults to
            ``1``.
        :param mismatch_tol: Maximum permitted relative mismatch for the in-plane
            commensurate-repeat search. ``None`` disables mismatch accommodation. For
            example, ``0.005`` permits 0.5 percent mismatch. Keyword argument, optional,
            defaults to ``None``.
        :param mismatch_max_cells: Maximum repeat count allowed for either grain in each
            one-dimensional commensurability search. Keyword argument, optional,
            defaults to ``50``.
        :param strain_grain: In-plane strain policy when mismatch accommodation is
            active. ``"both"`` uses the average unstrained length, ``"left"`` preserves
            the right-grain length, and ``"right"`` preserves the left-grain length.
            Ignored when ``mismatch_tol`` is ``None``. Keyword argument, optional,
            defaults to ``"both"``.
        :return: Fully initialized ``GBMaker`` instance.
        :raises BoundarySpecError: If the requested mode is incompatible with the
            boundary type, exact boundary conversion or exactification fails, an
            exact-cell limit is exceeded, or another boundary-spec construction error
            occurs.
        :raises NotImplementedError: If the boundary type is unsupported or the
            requested type/mode combination is recognized but not implemented.
        """
        mode = cls.__validate_boundary_mode(mode)
        mismatch_tol = cls.__validate_mismatch_tol(mismatch_tol)
        mismatch_max_cells = cls.__validate_mismatch_max_cells(mismatch_max_cells)
        strain_grain = cls.__validate_strain_grain(strain_grain)
        max_primitive_area_index = cls.__validate_exact_limit(
            max_primitive_area_index,
            "max_primitive_area_index",
        )
        max_pq_determinant = cls.__validate_exact_limit(
            max_pq_determinant,
            "max_pq_determinant",
        )

        resolved = resolve_boundary_input(
            boundary,
            mode,
            max_primitive_area_index=max_primitive_area_index,
            max_pq_determinant=max_pq_determinant,
        )
        misorientation = boundary.params if isinstance(boundary, FiveDOFSpec) else None
        return cls._from_boundary_embedding(
            resolved.embedding,
            a0=a0,
            structure=structure,
            atom_types=atom_types,
            misorientation=misorientation,
            gb_thickness=gb_thickness,
            repeat_factor=repeat_factor,
            x_dim_min=x_dim_min,
            vacuum=vacuum,
            interaction_distance=interaction_distance,
            gb_id=gb_id,
            mismatch_tol=mismatch_tol,
            mismatch_max_cells=mismatch_max_cells,
            strain_grain=strain_grain,
        )

    @staticmethod
    def __translate_construction_error(func, *args, **kwargs):
        """Call a ``gbmaker`` pure validation function, translating its exception.

        Single shared implementation behind every ``GBMaker`` validator wrapper below
        (and ``__validate`` itself): translates ``GBOpt.gbmaker.types``'s
        ``GBMakerConstructionTypeError``/``GBMakerConstructionValueError`` back to the
        established public ``GBMakerTypeError``/``GBMakerValueError``, so property
        setters and the legacy constructor see the same exception identities they
        always have while the validation logic itself lives in ``gbmaker.config``.

        :param func: Pure validation function to call.
        :param args: Positional arguments forwarded to ``func``.
        :param kwargs: Keyword arguments forwarded to ``func``.
        :return: Whatever ``func`` returns.
        :raises GBMakerTypeError: If ``func`` raises ``GBMakerConstructionTypeError``.
        :raises GBMakerValueError: If ``func`` raises ``GBMakerConstructionValueError``.
        """
        try:
            return func(*args, **kwargs)
        except GBMakerConstructionTypeError as exc:
            raise GBMakerTypeError(str(exc)) from exc
        except GBMakerConstructionValueError as exc:
            raise GBMakerValueError(str(exc)) from exc

    @staticmethod
    def __validate_mismatch_tol(value: object) -> float | None:
        """Return a validated mismatch-accommodation tolerance.

        Thin wrapper delegating to the single pure implementation in
        ``GBOpt.gbmaker.config.validate_mismatch_tol``.

        :param value: Candidate mismatch tolerance.
        :return: ``None`` if mismatch accommodation is disabled; otherwise a finite,
            non-negative floating-point tolerance.
        :raises GBMakerValueError: If ``value`` is boolean, non-numeric, infinite, NaN,
            or negative.
        """
        return GBMaker.__translate_construction_error(validate_mismatch_tol, value)

    @staticmethod
    def __validate_mismatch_max_cells(value: object) -> int:
        """Return a validated commensurability-search repeat-count bound.

        Thin wrapper delegating to
        ``GBOpt.gbmaker.config.validate_mismatch_max_cells``.

        :param value: Candidate maximum repeat count.
        :return: Positive integer repeat-count bound.
        :raises GBMakerValueError: If ``value`` is boolean, non-integral, or less than
            one.
        """
        return GBMaker.__translate_construction_error(
            validate_mismatch_max_cells, value
        )

    @staticmethod
    def __validate_strain_grain(value: str) -> str:
        """Return a validated mismatch-strain policy.

        Thin wrapper delegating to ``GBOpt.gbmaker.config.validate_strain_grain``.

        :param value: Grain strain policy. Supported values are ``"both"``, ``"left"``,
            and ``"right"``.
        :return: Validated strain policy.
        :raises GBMakerValueError: If ``value`` is not one of ``"both"``, ``"left"``, or
            ``"right"``.
        """
        return GBMaker.__translate_construction_error(validate_strain_grain, value)

    @staticmethod
    def __validate_boundary_mode(value: str) -> str:
        """Return a validated boundary-spec construction mode.

        Thin wrapper delegating to ``GBOpt.gbmaker.config.validate_boundary_mode``.

        :param value: Boundary-spec construction mode. Supported values are
            ``"exact"``, ``"approximate"``, and ``"prefer_exact"``.
        :return: Validated construction mode.
        :raises GBMakerValueError: If ``value`` is not one of the supported modes.
        """
        return GBMaker.__translate_construction_error(validate_boundary_mode, value)

    @staticmethod
    def __validate_exact_limit(value: object, name: str) -> int:
        """Return a validated positive exact-construction limit.

        Thin wrapper delegating to ``GBOpt.gbmaker.config.validate_exact_limit``.
        """
        return GBMaker.__translate_construction_error(
            validate_exact_limit, value, name
        )

    @staticmethod
    def __reduce_integer_row(row: np.ndarray) -> np.ndarray:
        """Reduce an integer row by its GCD.

        Thin wrapper delegating to ``GBOpt.gbmaker.orientation._reduce_integer_row``,
        kept for the geometry call sites (exact-repeat and box-basis direction
        reduction) that have not yet moved out of ``GBMaker`` themselves.

        :param row: Integer row vector
        :return: GCD-reduced integer row vector
        """
        return _reduce_integer_row(row)

    # Private class methods
    def __assign_orientations(self, misorientation: np.ndarray) -> None:
        """ Private method to separate the misorientation and inclination from the
        passed in misorientation array.

        Thin wrapper delegating to
        ``GBOpt.gbmaker.orientation._decompose_misorientation``.

        :param misorientation: Array containing the misorientation and inclination Euler
            angles. Misorientation is the first three, and inclination is the last two.
            Note that misorientation is in the ZXZ Euler angle format.
        """
        (
            self.__misorientation,
            self.__inclination,
            self.__Rmis,
            self.__Rincl,
        ) = self.__translate_construction_error(
            _decompose_misorientation, misorientation
        )

    def __calculate_box_dimensions(self) -> np.ndarray:
        """Private method to calculate the box dimensions

        Thin wrapper delegating to ``GBOpt.gbmaker.dimension._plan_box_dims``.

        :return: The 3x2 array containing xlo, xhi, ylo, yhi, zlo, and zi.
        """
        return np.array(
            _plan_box_dims(
                self.__x_dim, self.__vacuum_thickness, self.__y_dim, self.__z_dim
            )
        )

    def __exact_grain_repeats(
        self,
        P_or_Q: np.ndarray,
        x_length: float,
        grain_side: str,
    ) -> tuple[np.ndarray, int, int, int]:
        """Compute exact-path supercell repeat counts for one grain.

        Builds the integer supercell matrix for the supplied canonical orientation
        matrix and derives the number of repeated supercell periods needed along the
        boundary-normal x direction and the two in-plane directions. The x repeat count
        is derived from the grain's already equalized x-slab thickness. The y and z
        repeat counts are derived from the shared in-plane box dimensions, unless
        mismatch accommodation supplied explicit left/right repeat counts for that axis.

        :param P_or_Q: Canonical 3 by 3 integer orientation matrix for this grain.
        :param x_length: Equalized x-slab thickness for this grain (Angstroms).
        :param grain_side: Grain side, either ``"left"`` or ``"right"``. Used to select
            the appropriate repeat count when mismatch accommodation is active.
        :return: ``(supercell, repeat_x, repeat_y, repeat_z)`` where ``supercell`` is
            the validated integer supercell matrix and the remaining values are positive
            Python integers.
        :raises GBMakerValueError: If ``grain_side`` is not ``"left"`` or ``"right"``,
            if the supercell matrix cannot be built, or if the x, y, or z box length is
            not commensurate with this grain's corresponding period.
        """
        if grain_side not in {"left", "right"}:
            raise GBMakerValueError(
                f"grain_side must be 'left' or 'right'; got {grain_side!r}."
            )

        try:
            supercell = build_supercell_matrix(P_or_Q)
        except ValueError as exc:
            raise GBMakerValueError(str(exc)) from exc

        a0 = self.__a0
        x_period = a0 * _miller_row_norm(supercell[0])
        y_period = a0 * _miller_row_norm(supercell[1])
        z_period = a0 * _miller_row_norm(supercell[2])

        tol = 1e-6

        def commensurate_repeat(
            box_length: float,
            period: float,
            axis_name: str,
        ) -> int:
            """Return a positive repeat count for one exact box/period pair.

            :param box_length: Box length along this axis (Angstroms).
            :param period: Grain period along this axis (Angstroms).
            :param axis_name: Axis label used in error messages.
            :return: Positive integer repeat count.
            :raises GBMakerValueError: If ``box_length`` is not an integer multiple of
                ``period`` within the repeat-count tolerance.
            """
            repeat_raw = box_length / period
            repeat = int(round(repeat_raw))

            if abs(repeat_raw - repeat) > tol:
                raise GBMakerValueError(
                    f"Exact construction requires the {axis_name} box "
                    f"({box_length:.6f}A) to be an integer multiple of this grain's "
                    f"{axis_name}-period ({period:.6f} A), but got repeat_{axis_name} "
                    f"= {repeat_raw:.8f}. Use mode='approximate' or adjust "
                    "repeat_factor until both grains' periods divide the shared box "
                    "exactly. See the commensurability note in from_boundary_spec for "
                    "details."
                )

            if repeat <= 0:
                raise GBMakerValueError(
                    f"Exact construction requires positive {axis_name} repeats; got "
                    f"{repeat}."
                )

            return repeat

        repeat_x = commensurate_repeat(x_length, x_period, "x")

        y_accommodation = self.__strain_accommodation.get("y")
        if y_accommodation is None:
            repeat_y = commensurate_repeat(self.__y_dim, y_period, "y")
        elif grain_side == "left":
            repeat_y = y_accommodation.left_repeats
        else:
            repeat_y = y_accommodation.right_repeats

        z_accommodation = self.__strain_accommodation.get("z")
        if z_accommodation is None:
            repeat_z = commensurate_repeat(self.__z_dim, z_period, "z")
        elif grain_side == "left":
            repeat_z = z_accommodation.left_repeats
        else:
            repeat_z = z_accommodation.right_repeats

        return supercell, repeat_x, repeat_y, repeat_z

    def __build_exact_grain(
        self,
        R_grain: np.ndarray,
        P_or_Q: np.ndarray,
        x_length: float,
        x_offset: float,
        grain_side: str,
    ) -> np.ndarray:
        """Build one grain from exact decorated repeated-supercell sites.

        Enumerates every rational decorated basis site in the repeated integer supercell
        before any floating-point conversion. Exact conventional coordinates are
        reconstructed from exact site metadata and converted once to Cartesian crystal
        positions. Existing rotation, in-plane strain, placement, and periodic in-plane
        wrapping are then applied without Cartesian membership clipping or
        complete-origin deletion.

        :param R_grain: Proper rotation matrix for this grain.
        :param P_or_Q: 3x3 canonical integer orientation matrix.
        :param x_length: Equalized x-slab thickness (Angstroms).
        :param x_offset: Lab x-coordinate of the grain's lower face (Angstroms).
        :param grain_side: Grain side, either ``"left"`` or ``"right"``.
        :return: Structured atom array for the complete decorated grain.
        :raises GBMakerValueError: If rational basis metadata is unavailable, exact
            enumeration violates a population invariant, or final coordinates are
            non-finite or outside the intended grain box.
        """
        rational_basis = self.__unit_cell.rational_basis
        if rational_basis is None:
            raise GBMakerValueError(
                "Exact grain generation requires UnitCell.rational_basis; arbitrary "
                "floating-point basis coordinates are not accepted."
            )

        supercell, repeat_x, repeat_y, repeat_z = self.__exact_grain_repeats(
            P_or_Q, x_length, grain_side
        )

        try:
            sites = enumerate_supercell_sites(
                supercell,
                repeat_x,
                repeat_y,
                repeat_z,
                rational_basis=rational_basis,
            )
        except ValueError as exc:
            raise GBMakerValueError(
                f"Exact decorated-site enumeration failed for the {grain_side} grain: "
                f"{exc}"
            ) from exc

        basis_size = sites.basis_size
        basis_indices = sites.basis_indices
        expected_site_count = sites.site_count

        structured_basis = self.__unit_cell.asarray()
        structured_names = tuple(
            str(name) for name in structured_basis["name"]
        )

        if len(structured_basis) != basis_size:
            raise GBMakerValueError(
                "UnitCell rational-basis and structured-basis sizes disagree: "
                f"{basis_size} exact sites versus {len(structured_basis)} structured "
                "atoms."
            )

        if structured_names != rational_basis.names:
            raise GBMakerValueError(
                "UnitCell rational-basis and structured-basis species order disagree: "
                f"exact={rational_basis.names!r}, structured={structured_names!r}."
            )

        atoms = np.empty(expected_site_count, dtype=structured_basis.dtype)
        atoms["name"] = structured_basis["name"][basis_indices]

        # This is the sole exact-to-floating conversion in grain construction. Exact
        # site metadata stores canonical repeated-supercell coordinates; multiplying by
        # S reconstructs exact conventional-cell coordinate numerators.
        conventional_numerators = sites.coordinate_numerators @ sites.supercell_matrix

        try:
            with np.errstate(over="ignore", invalid="ignore"):
                crystal_positions = np.asarray(
                    conventional_numerators,
                    dtype=np.float64,
                )
                crystal_positions *= self.__a0 / sites.coordinate_denominator

            rotation = np.asarray(R_grain, dtype=np.float64)
            inplane_orientation_rows = np.asarray(P_or_Q[1:], dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as exc:
            raise GBMakerValueError(
                "Exact decorated-site coordinates or orientation rows cannot be "
                f"represented as finite Cartesian values for the {grain_side} grain."
            ) from exc

        if (
            not np.all(np.isfinite(crystal_positions))
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(inplane_orientation_rows))
        ):
            raise GBMakerValueError(
                "Exact decorated-site coordinates or orientation rows became "
                f"non-finite during Cartesian conversion for the {grain_side} grain."
            )

        rotated = crystal_positions @ rotation.T

        y_scale, z_scale = self.__grain_strain_scales(grain_side)
        strain_scales = np.array([1.0, y_scale, z_scale], dtype=np.float64)

        rotated *= strain_scales
        rotated[:, 0] += x_offset

        rotated_unit_cell_basis = self.__unit_cell.conventional @ rotation.T
        primitive_periods = inplane_orientation_rows @ rotated_unit_cell_basis
        strained_periods = primitive_periods * strain_scales
        selection_basis = self.__selection_basis_vectors(strained_periods)

        box_coordinates = self.__reduced_box_coordinates(rotated, selection_basis)

        for row_index, is_periodic in enumerate(self.__inplane_periodic):
            if not is_periodic:
                continue

            coordinate_index = row_index + 1
            tolerance = self.__reduced_coordinate_tolerance(selection_basis[row_index])
            box_coordinates[:, coordinate_index] = wrap_reduced_coordinate(
                box_coordinates[:, coordinate_index],
                tolerance,
            )

        rotated = self.__cartesian_from_box_coordinates(
            box_coordinates,
            selection_basis,
        )
        if not np.all(np.isfinite(rotated)):
            raise GBMakerValueError(
                "Exact decorated-site conversion produced non-finite Cartesian "
                f"coordinates for the {grain_side} grain."
            )
        atoms["x"], atoms["y"], atoms["z"] = rotated.T

        lower_x = float(x_offset)
        upper_x = lower_x + float(x_length)
        x_coordinates = atoms["x"]

        near_lower = (
            (x_coordinates < lower_x)
            & (x_coordinates >= lower_x - self.__epsilon)
        )
        near_upper = (
            (x_coordinates >= upper_x)
            & (x_coordinates < upper_x + self.__epsilon)
        )

        # Preserve the physical side of the upper termination while ensuring that its
        # floating representation remains strictly half-open.
        x_coordinates[near_lower] = lower_x
        x_coordinates[near_upper] = np.nextafter(upper_x, lower_x)

        outside_x = ((x_coordinates < lower_x) | (x_coordinates >= upper_x))
        if np.any(outside_x):
            offending = x_coordinates[outside_x]
            raise GBMakerValueError(
                "Exact decorated-site conversion produced atoms outside the "
                f"{grain_side} half-open x slab [{lower_x:.8f}, {upper_x:.8f}): "
                f"min={float(np.min(offending)):.8f}, "
                f"max={float(np.max(offending)):.8f}."
            )

        for axis_name, dimension, is_periodic in zip(
            ("y", "z"),
            (self.__y_dim, self.__z_dim),
            self.__inplane_periodic,
        ):
            if not is_periodic:
                continue
            coordinates = atoms[axis_name]
            near_lower = (coordinates < 0.0) & (coordinates >= -self.__epsilon)
            near_upper = (coordinates >= dimension) & (
                coordinates < dimension + self.__epsilon
            )
            coordinates[near_lower | near_upper] = 0.0

            outside = (coordinates < 0.0) | (coordinates >= dimension)
            if np.any(outside):
                offending = coordinates[outside]
                raise GBMakerValueError(
                    "Exact decorated-site conversion produced atoms outside the "
                    f"periodic {axis_name} box [0, {dimension:.8f}): "
                    f"min={float(np.min(offending)):.8f}, "
                    f"max={float(np.max(offending)):.8f}."
                )

        return atoms

    def __grain_x_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return initial lab-frame x bounds for the left and right grains.

        :return: ``(left_bounds, right_bounds)``, where each array contains ``[x_min,
            x_max]`` in Angstroms.
        """
        left_bounds = np.array(
            [
                self.__vacuum_thickness,
                self.__left_x + self.__vacuum_thickness,
            ],
            dtype=np.float64,
        )
        right_bounds = np.array(
            [
                self.__left_x + self.__vacuum_thickness,
                self.__x_dim + self.__vacuum_thickness,
            ],
            dtype=np.float64,
        )
        return left_bounds, right_bounds

    def __use_exact_grain_generation(self) -> bool:
        """Return whether the exact integer grain-generation path should be used.

        :return: ``True`` when the current embedding is exact, coherent, and carries
            both integer P and Q orientation matrices.
        :raises GBMakerValueError: If an exact coherent embedding is present but does
            not carry both P and Q.
        """
        if self.__embedding is None:
            return False

        if not (self.__embedding.exact and self.__embedding.coherent):
            return False

        if self.__embedding.P is None or self.__embedding.Q is None:
            raise GBMakerValueError(
                "Exact coherent grain generation requires both embedding.P and "
                "embedding.Q."
            )

        return True

    def __generate_exact_grains(
        self,
        left_bounds: np.ndarray,
        right_bounds: np.ndarray,
    ) -> None:
        """Generate both grains using exact decorated-site enumeration.

        :param left_bounds: Length-2 x-bound array for the left grain.
        :param right_bounds: Length-2 x-bound array for the right grain.
        :return: ``None``. Updates ``self.__left_grain`` and ``self.__right_grain``.
        :raises GBMakerValueError: If the exact embedding is missing P or Q, rational
            basis metadata is unavailable, exact site enumeration fails, exact
            populations disagree with the unit-cell basis, or either grain produces
            invalid Cartesian coordinates.
        """
        if (
            self.__embedding is None
            or self.__embedding.P is None
            or self.__embedding.Q is None
        ):
            raise GBMakerValueError(
                "Exact grain generation requires an embedding with both P and Q."
            )

        self.__left_grain = self.__build_exact_grain(
            self.__R_left,
            self.__embedding.P,
            self.__left_x,
            left_bounds[0],
            "left",
        )
        self.__right_grain = self.__build_exact_grain(
            self.__R_right,
            self.__embedding.Q,
            self.__right_x,
            right_bounds[0],
            "right",
        )

    def __generate_float_grains(
        self,
        left_bounds: np.ndarray,
        right_effective_bounds: np.ndarray,
    ) -> tuple[_FloatGrainBuildResult, np.ndarray, bool, float]:
        """Generate both grains using the floating-point path.

        For ``vacuum=0``, trims one complete right-grain x period from the high-x side
        when enough thickness remains. The trim is origin-complete so multi-species
        conventional-cell groups are preserved.

        :param left_bounds: Length-2 x-bound array for the left grain.
        :param right_effective_bounds: Length-2 right-grain x-bound array. The upper
            bound may be reduced if the vacuum-zero trim is applied.
        :return: ``(right_float_result, right_effective_bounds, vacuum0_trim_applied,
            x_period_right)``.
        """
        left_float_result = self.__generate_grain_result(
            self.__R_left,
            self.__left_periodic_miller_rows,
            left_bounds,
            grain_side="left",
        )
        self.__left_grain = left_float_result.atoms

        x_period_right = self.__x_period(self.__right_periodic_miller_rows)
        vacuum0_trim_applied = False

        right_float_result = self.__generate_grain_result(
            self.__R_right,
            self.__right_periodic_miller_rows,
            right_effective_bounds,
            grain_side="right",
        )

        right_width = right_effective_bounds[1] - right_effective_bounds[0]
        if (
            self.__vacuum_thickness == 0
            and right_width > x_period_right * (1.0 + self.__epsilon)
        ):
            new_upper = right_effective_bounds[1] - x_period_right
            trial_result = self.__trim_float_result_to_upper_x(
                right_float_result,
                new_upper,
            )

            if len(trial_result.atoms) == 0:
                warnings.warn(
                    "Vacuum=0 trim would remove all atoms from the right grain. "
                    "Skipping trim to preserve a non-empty grain.",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                right_float_result = trial_result
                right_effective_bounds[1] = new_upper
                vacuum0_trim_applied = True

        self.__right_grain = right_float_result.atoms

        return (
            right_float_result,
            right_effective_bounds,
            vacuum0_trim_applied,
            x_period_right,
        )

    def __current_gap_metrics(
        self,
        left_bounds: np.ndarray,
        right_effective_bounds: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Return current central and periodic x-gap metrics.

        :param left_bounds: Effective left-grain x bounds.
        :param right_effective_bounds: Effective right-grain x bounds.
        :return: ``(central_gap, periodic_gap, left_min_x, right_max_x)``.
        """
        left_min_x = float(np.min(self.__left_grain["x"]))
        left_max_x = float(np.max(self.__left_grain["x"]))
        right_min_x = float(np.min(self.__right_grain["x"]))
        right_max_x = float(np.max(self.__right_grain["x"]))

        central_gap = right_min_x - left_max_x
        periodic_gap = (
            right_effective_bounds[1] - right_max_x
        ) + (left_min_x - left_bounds[0])

        return central_gap, periodic_gap, left_min_x, right_max_x

    def __equalize_float_periodic_gap(
        self,
        *,
        central_gap: float,
        left_min_x: float,
        right_max_x: float,
        left_bounds: np.ndarray,
        right_effective_bounds: np.ndarray,
        right_float_result: _FloatGrainBuildResult,
        x_period_right: float,
    ) -> None:
        """Equalize the periodic gap by removing whole right-grain x periods.

        Removal is performed through complete-origin filtering so atom groups from the
        same conventional-cell origin are not split.

        :param central_gap: Current central GB gap (Angstroms).
        :param left_min_x: Minimum left-grain x coordinate (Angstroms).
        :param right_max_x: Maximum right-grain x coordinate before equalization
            (Angstroms).
        :param left_bounds: Effective left-grain x bounds.
        :param right_effective_bounds: Effective right-grain x bounds.
        :param right_float_result: Right-grain float build result to trim.
        :param x_period_right: Right-grain x period (Angstroms).
        :return: ``None``. May update ``self.__right_grain``.
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
                stacklevel=3,
            )
            return

        grain_width = right_effective_bounds[1] - right_effective_bounds[0]
        if n_remove * x_period_right > grain_width / 2.0:
            warnings.warn(
                f"Gap equalization removed {n_remove} x-period(s) "
                f"({n_remove * x_period_right:.4f} A), more than half the right "
                "grain. The resulting bicrystal may be unusable.",
                UserWarning,
                stacklevel=3,
            )

        trial_result = self.__trim_float_result_to_upper_x(
            right_float_result,
            new_upper,
        )

        if len(trial_result.atoms) == 0:
            warnings.warn(
                f"Gap equalization would remove all atoms from the right grain "
                f"({n_remove} x-periods; right_x = "
                f"{right_effective_bounds[1] - right_effective_bounds[0]:.4f} A, "
                f"x_period = {x_period_right:.4f} A). Skipping equalization to "
                "preserve a non-empty grain.",
                UserWarning,
                stacklevel=3,
            )
            return

        self.__right_grain = trial_result.atoms

        final_periodic_gap = (
            right_effective_bounds[1] - float(np.max(self.__right_grain["x"]))
        ) + (left_min_x - left_bounds[0])

        if final_periodic_gap < central_gap - self.__epsilon:
            warnings.warn(
                f"Float gap equalization: periodic_gap "
                f"({final_periodic_gap:.4f} A) < central_gap "
                f"({central_gap:.4f} A). Stoichiometry preserved; matching would "
                "require splitting an origin or deleting the right grain.",
                UserWarning,
                stacklevel=3,
            )

    def __equalize_periodic_gap(
        self,
        *,
        left_bounds: np.ndarray,
        right_effective_bounds: np.ndarray,
        use_exact: bool,
        right_float_result: _FloatGrainBuildResult | None,
        vacuum0_trim_applied: bool,
        x_period_right: float | None,
    ) -> None:
        """Handle a periodic x-gap mismatch for the selected construction path.

        The floating path retains its existing complete-origin trimming behavior. The
        exact decorated-site path never deletes atomic layers merely to reverse
        projected central/periodic gap ordering. Both projected gaps must nevertheless
        remain nonnegative within the Cartesian tolerance.

        :param left_bounds: Effective left-grain x bounds.
        :param right_effective_bounds: Effective right-grain x bounds.
        :param use_exact: Whether the exact decorated-site path was used.
        :param right_float_result: Right-grain build metadata for the float path.
        :param vacuum0_trim_applied: Whether the vacuum-zero pre-trim was applied.
        :param x_period_right: Right-grain x period for the float path.
        :return: ``None``. May update ``self.__right_grain``.
        :raises GBMakerValueError: If an exact grain crosses the central or periodic x
            boundary, or if required float metadata is missing.
        """

        (
            central_gap,
            periodic_gap,
            left_min_x,
            right_max_x,
        ) = self.__current_gap_metrics(left_bounds, right_effective_bounds)

        if use_exact:
            if (
                central_gap < -self.__epsilon
                or periodic_gap < -self.__epsilon
            ):
                raise GBMakerValueError(
                    "Exact decorated-site construction produced an invalid x-boundary "
                    f"overlap: central_gap={central_gap:.8f} A, "
                    f"periodic_gap={periodic_gap:.8f} A."
                )
            return

        if periodic_gap >= central_gap - self.__epsilon:
            return

        if self.__vacuum_thickness == 0 and vacuum0_trim_applied:
            return

        if right_float_result is None or x_period_right is None:
            raise GBMakerValueError(
                "Float gap equalization requires a right-grain float build result "
                "and right-grain x period."
            )

        self.__equalize_float_periodic_gap(
            central_gap=central_gap,
            left_min_x=left_min_x,
            right_max_x=right_max_x,
            left_bounds=left_bounds,
            right_effective_bounds=right_effective_bounds,
            right_float_result=right_float_result,
            x_period_right=x_period_right,
        )

    def __generate_gb(self) -> None:
        """Generate the left grain, right grain, and combined GB atom array.

        Builds each grain using the exact integer path when a coherent exact boundary
        embedding with integer P/Q matrices is available; otherwise uses the
        floating-point grain-generation path. Periodic x-gap equalization remains
        available only to the floating path. Exact decorated grains retain every
        enumerated site and are assembled without x-layer deletion.

        :return: ``None``. Updates ``self.__left_grain``, ``self.__right_grain``, and
            ``self.__whole_system``.
        :raises GBMakerValueError: If exact grain generation requires missing P/Q data,
            if float-path gap equalization lacks right-grain build metadata, or if a
            downstream grain-generation helper fails.
        """
        left_bounds, right_bounds = self.__grain_x_bounds()
        right_effective_bounds = right_bounds.copy()

        use_exact = self.__use_exact_grain_generation()
        right_float_result: _FloatGrainBuildResult | None = None
        vacuum0_trim_applied = False
        x_period_right: float | None = None

        if use_exact:
            self.__generate_exact_grains(left_bounds, right_bounds)
        else:
            (
                right_float_result,
                right_effective_bounds,
                vacuum0_trim_applied,
                x_period_right,
            ) = self.__generate_float_grains(left_bounds, right_effective_bounds)

        self.__equalize_periodic_gap(
            left_bounds=left_bounds,
            right_effective_bounds=right_effective_bounds,
            use_exact=use_exact,
            right_float_result=right_float_result,
            vacuum0_trim_applied=vacuum0_trim_applied,
            x_period_right=x_period_right,
        )

        self.__whole_system = np.hstack((self.__left_grain, self.__right_grain))

    def __calculate_periodic_spacing(self, threshold: float = None) -> dict:
        """
        Calculate the periodic spacing based on the rotation matrix.

        :param threshold: The maximum allowed value that any spacing can take. Default
            is 15 * a0.
        :return: Dict containing the periodic spacing along the 'x', 'y', and 'z'
            directions for the given misorientation.
        """
        if threshold is None:
            threshold = self.__a0 * 15

        # Rotation-matrix and periodic-Miller-row assignment, and in-plane periodicity
        # determination, are a pure construction stage; see
        # ``GBOpt.gbmaker.orientation.resolve_orientation``. Exact P/Q rows are never
        # routed through integer-row approximation there.
        orientation = self.__translate_construction_error(
            resolve_orientation,
            np.hstack((self.__misorientation, self.__inclination)),
            embedding=self.__embedding,
            a0=self.__a0,
            threshold=threshold,
        )
        self.__Rmis = orientation.R_mis
        self.__Rincl = orientation.R_incl
        self.__R_left = orientation.R_left
        self.__R_right = orientation.R_right
        self.__left_periodic_miller_rows = orientation.left_periodic_miller_rows
        self.__right_periodic_miller_rows = orientation.right_periodic_miller_rows
        self.__inplane_periodic = orientation.inplane_periodic

        # Periodic-spacing and boundary-normal x-extent arithmetic is a pure
        # construction stage; see ``GBOpt.gbmaker.dimension.plan_periodic_spacing``.
        # In-plane periodicity was already resolved by ``resolve_orientation`` above
        # (including, on the legacy/five-DOF path, its own threshold warning); this
        # only reapplies the resulting flags to the returned spacing values used for
        # box-dimension planning.
        spacing, self.__left_x, self.__right_x, self.__x_dim = (
            self.__translate_construction_error(
                plan_periodic_spacing,
                a0=self.__a0,
                left_periodic_miller_rows=self.__left_periodic_miller_rows,
                right_periodic_miller_rows=self.__right_periodic_miller_rows,
                x_dim_min=self.__x_dim_min,
                epsilon=self.__epsilon,
                inplane_periodic=self.__inplane_periodic,
                threshold=threshold,
                legacy_periodicity_heuristic=(
                    self.__embedding is None
                    or self.__embedding.source == "five_dof"
                ),
            )
        )

        return spacing

    def __get_triclinic_params(self):
        """
        Computes the LAMMPS restricted-triclinic tilt factors. The y-period in the lab
        frame is R_grain @ (g_y * a0). For an exact CSL boundary this is exactly
        ||g_y|| * a0 * e_y; for non-CSL it has small x and z components. To satisfy
        LAMMPS's restriction that the b-vector lies in the xy-plane, rotate everything
        about the x-axis by theta = -atan2(A2[2], A2[1]).

        :return: (xy, xz, yz, theta) - the three tilt scalars and the rotation angle to
                                       apply to atom coordinates
        """
        if not all(self.__inplane_periodic):
            raise GBMakerValueError(
                "Triclinic output requires periodic y and z directions."
            )

        # Use grain with larger y-period, consistent with how spacing["y"] is chosen
        if (np.linalg.norm(self.__left_periodic_miller_rows[1])
                >= np.linalg.norm(self.__right_periodic_miller_rows[1])):
            R_grain = self.__R_left
            R_grain_approx = self.__left_periodic_miller_rows
        else:
            R_grain = self.__R_right
            R_grain_approx = self.__right_periodic_miller_rows

        # conventional stores basis vectors as rows: C = [a1; a2; a3].
        # Rotating each row vector to the lab frame gives [R@a1; R@a2; R@a3],
        # which in batch form is (R @ C.T).T = C @ R.T.
        rotated_unit_cell_basis = self.__unit_cell.conventional @ R_grain.T
        primitive_periods = (
            np.asarray(R_grain_approx[1:], dtype=np.float64) @ rotated_unit_cell_basis
        )
        A2_lab, A3_lab = self.__box_periodic_basis(primitive_periods)

        # Rotate about x to bring A2 into the xy-plane (LAMMPS restricted-triclinic
        # requires b-vector in the xy-plane). x-components are unaffected by this
        # rotation
        theta = -math.atan2(float(A2_lab[2]), float(A2_lab[1]))
        ct, st = math.cos(theta), math.sin(theta)

        # The x-rotation matrix is [[1,0,0],[0,ct,-st],[0,st,ct]]. The x-components of
        # A2_lab and A3_lab are unchanged by it, so xy and xz can be read direcly from
        # the pre-rotation vectors. yz requires the full rotation.
        xy = float(A2_lab[0])
        xz = float(A3_lab[0])
        yz = float(ct * A3_lab[1] - st * A3_lab[2])

        return xy, xz, yz, theta

    def __init_unit_cell(self, atom_types: str | tuple[str, ...]) -> UnitCell:
        """
        Initializes the unit cell.

        :return: The unit cell initialized by structure.
        """
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self.__structure, self.__a0, atom_types)
        return unit_cell

    def __x_period(self, periodic_miller_rows: np.ndarray) -> float:
        """Return one full x-period length for a grain.

        Thin wrapper delegating to ``GBOpt.gbmaker.orientation._x_period``.

        The x-period is the distance between equivalent crystallographic repeats along
        the boundary-normal direction. It is computed from the first integer periodic
        Miller row as ``a0 * ||periodic_miller_rows[0]||``.

        :param periodic_miller_rows: Three-row integer periodic Miller matrix for one
            grain. Row 0 defines the boundary-normal x-period.
        :return: Boundary-normal x-period in Angstroms.
        :raises GBMakerValueError: If row 0 is not a nonzero three-component integer
            Miller row.
        """
        return self.__translate_construction_error(
            _x_period, periodic_miller_rows, self.__a0
        )

    def __generate_grain_result(
        self,
        R_grain: np.ndarray,
        periodic_miller_rows: np.ndarray,
        x_bounds: np.ndarray,
        *,
        grain_side: str | None = None,
    ) -> _FloatGrainBuildResult:
        """Generate one grain using the floating-point lattice-enumeration path.

        Enumerates conventional-cell lattice coefficients over a conservative slab,
        expands each retained origin to the full conventional-cell basis, rotates atoms
        into the lab frame, applies any requested lab-frame in-plane strain,
        selects/wraps periodic in-plane coordinates, clips to the Cartesian x slab, and
        removes duplicate complete-origin groups.

        ``origin_ids`` is preserved in parallel with the atom array so later trimming
        operations can keep or remove complete conventional-cell origins.

        :param R_grain: Proper rotation matrix for this grain.
        :param periodic_miller_rows: Three-row integer periodic Miller matrix for this
            grain. Rows 1 and 2 define the primitive in-plane y/z period vectors used by
            the floating-point selection basis.
        :param x_bounds: Length-2 array-like containing the lower and upper x bounds for
            this grain in the lab frame (Angstroms).
        :param grain_side: Grain side, either ``"left"`` or ``"right"``, when
            mismatch-accommodation strain scales should be applied. ``None`` applies no
            strain. Keyword parameter, optional, defaults to ``None``.
        :return: Float-path grain build result containing the atom array, parallel
            origin-ID array, and conventional-cell basis size.
        :raises GBMakerValueError: If ``grain_side`` is invalid, if selection or
            clipping cannot preserve complete origin groups, or if no complete origins
            remain after filtering.
        """
        x_bounds = np.asarray(x_bounds, dtype=np.float64)

        y_scale = 1.0
        z_scale = 1.0
        if grain_side is not None:
            y_scale, z_scale = self.__grain_strain_scales(grain_side)
        strain_scales = np.array([1.0, y_scale, z_scale], dtype=np.float64)

        rotated_unit_cell_basis = self.__unit_cell.conventional @ R_grain.T

        primitive_periods = np.asarray(periodic_miller_rows[1:], dtype=np.float64)
        primitive_periods = primitive_periods @ rotated_unit_cell_basis

        # Selection and wrapping operate on strained lab-frame coordinates, so the
        # period vectors passed to those helpers must carry the same lab-frame y/z
        # strain as the atoms.
        strained_periods = primitive_periods * strain_scales

        reduced_periods = np.linalg.solve(
            rotated_unit_cell_basis.T, primitive_periods.T
        ).T
        x_direction_lattice = np.cross(reduced_periods[0], reduced_periods[1])
        rounded_direction = np.rint(x_direction_lattice)
        if np.allclose(
            x_direction_lattice, rounded_direction, atol=self.__epsilon, rtol=0.0
        ) and np.any(rounded_direction):
            x_direction_lattice = self.__reduce_integer_row(
                rounded_direction.astype(int)
            ).astype(np.float64)

        # Build the final strained selection basis, then map it back through the
        # lab-frame strain before converting to lattice coordinates. This keeps the
        # coefficient search conservative for the unstrained lattice that is enumerated
        # before atom positions are strained.
        selection_box_basis = self.__selection_basis_vectors(strained_periods).copy()
        axis_dims = (self.__y_dim, self.__z_dim)
        inplane_periodic = self.__inplane_periodic
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
        nx_range = self.__x_index_range(
            primitive_periods, rotated_unit_cell_basis, local_x_bounds
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
        lattice_coefficients = np.array(
            np.meshgrid(*coefficient_ranges, indexing="ij")
        ).reshape(3, -1).T

        basis_size = len(self.__unit_cell.asarray())
        atoms = self.get_supercell(lattice_coefficients @ self.__unit_cell.conventional)
        origin_ids = np.repeat(
            np.arange(len(lattice_coefficients), dtype=np.int64), basis_size
        )

        positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
        rotated_positions = positions @ R_grain.T
        rotated_positions *= strain_scales
        rotated_positions[:, 0] += x_bounds[0]
        atoms["x"], atoms["y"], atoms["z"] = rotated_positions.T

        if any(inplane_periodic):
            atoms, origin_ids = self.__select_complete_origins_in_box_basis(
                atoms, origin_ids, strained_periods, x_bounds, basis_size
            )

        atoms, origin_ids = self.__clip_complete_origins_to_cartesian_box(
            atoms, origin_ids, x_bounds, basis_size
        )
        atoms, origin_ids = self.__deduplicate_complete_origins(
            atoms, origin_ids, basis_size
        )

        if len(atoms) == 0:
            raise GBMakerValueError(
                f"Float grain generation removed all complete origins for the "
                f"{grain_side or 'unstrained'} grain."
            )

        return _FloatGrainBuildResult(
            atoms=atoms,
            origin_ids=origin_ids,
            basis_size=basis_size,
        )

    def __set_gb_region(self):
        """
        Identifies the atoms in the GB region based on the gb thickness.
        """
        x_gb = self.__vacuum_thickness + self.__left_x
        left_cut = x_gb - self.__gb_thickness / 2.0
        right_cut = x_gb + self.__gb_thickness / 2.0
        left_gb = self.__left_grain[self.__left_grain['x'] > left_cut]
        right_gb = self.__right_grain[self.__right_grain['x'] < right_cut]
        self.__gb_region = np.hstack((left_gb, right_gb))

    def __reduced_coordinate_tolerance(self, basis_vector: np.ndarray) -> float:
        """
        Convert the Cartesian epsilon to reduced-coordinate units for a basis vector.

        :param basis_vector: Cartesian basis vector used to define the coordinate scale.
        :return: Reduced-coordinate tolerance corresponding to ``self.__epsilon``.
        """
        basis_vector = np.asarray(basis_vector, dtype=np.float64)
        basis_length = np.linalg.norm(basis_vector)

        return self.__epsilon / basis_length

    def __scaled_periodic_basis_vector(
        self, period_vector: np.ndarray, box_length: float, axis_index: int
    ) -> np.ndarray:
        """
        Scale a periodic basis vector so one axis projection matches the box length.

        :param period_vector: Cartesian periodic basis vector.
        :param box_length: Desired box length along the selected axis.
        :param axis_index: Axis whose projection should match ``box_length``.
        :return: Scaled periodic basis vector.
        """

        period_vector = np.asarray(period_vector, dtype=np.float64)
        box_length = float(box_length)
        if box_length <= 0.0:
            raise GBMakerValueError("box_length must be strictly positive.")
        axis_index = int(axis_index)

        # We ignore overflow/invalid values because the check immediately after catches
        # those states and raises a GBMakerValueError
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            scale = box_length / period_vector[axis_index]
            scaled_vector = period_vector * scale
        if not np.all(np.isfinite(scaled_vector)):
            raise GBMakerValueError("Scaled periodic basis vector must be finite.")
        return scaled_vector

    def __box_periodic_basis(self, primitive_periods: np.ndarray) -> np.ndarray:
        """
        Build the in-plane box basis from primitive periodic vectors.

        :param primitive_periods: 2x3 array containing primitive y/z period vectors.
        :return: 2x3 array containing the box basis vectors for y and z.
        """
        primitive_periods = np.asarray(primitive_periods, dtype=np.float64)

        inplane_periodic = self.__inplane_periodic
        box_lengths = (self.__y_dim, self.__z_dim)
        box_basis = np.zeros((2, 3), dtype=np.float64)

        for row_index, (is_periodic, box_length) in enumerate(
            zip(inplane_periodic, box_lengths)
        ):
            if not is_periodic:
                continue

            axis_index = row_index + 1
            axis_projection = primitive_periods[row_index, axis_index]
            if np.isclose(axis_projection, 0.0, atol=self.__epsilon, rtol=0.0):
                raise GBMakerValueError(
                    "primitive_periods must have a non-zero projection on the "
                    "selected box axis."
                )
            box_basis[row_index] = self.__scaled_periodic_basis_vector(
                primitive_periods[row_index], box_length, axis_index
            )

        return box_basis

    def __selection_basis_vectors(self, primitive_periods: np.ndarray) -> np.ndarray:
        """
        Build the canonical in-plane selection basis for y/z box coordinates.

        Periodic axes use the box-periodic basis vectors; non-periodic axes fall back
        to the corresponding Cartesian unit vectors.

        :param primitive_periods: 2x3 array containing primitive y/z period vectors.
        :return: 2x3 array containing the y/z selection basis vectors.
        """
        selection_basis = self.__box_periodic_basis(primitive_periods)
        inplane_periodic = self.__inplane_periodic

        for row_index, is_periodic in enumerate(inplane_periodic):
            if is_periodic:
                continue
            selection_basis[row_index, row_index + 1] = 1.0

        return selection_basis

    def __x_index_range(
        self,
        primitive_periods: np.ndarray,
        rotated_unit_cell_basis: np.ndarray,
        x_bounds: np.ndarray,
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
        :return: Contiguous integer array of lattice indices along the x-period
            direction.
        """
        primitive_periods = np.asarray(primitive_periods, dtype=np.float64)
        rotated_unit_cell_basis = np.asarray(
            rotated_unit_cell_basis, dtype=np.float64
        )
        x_bounds = np.asarray(x_bounds, dtype=np.float64)

        determinant = np.linalg.det(rotated_unit_cell_basis)
        if np.isclose(determinant, 0.0, atol=self.__epsilon, rtol=0.0):
            raise GBMakerValueError(
                "rotated_unit_cell_basis must form an invertible 3x3 basis."
            )

        reduced_periods = np.linalg.solve(
            rotated_unit_cell_basis.T, primitive_periods.T
        ).T
        x_direction_lattice = np.cross(reduced_periods[0], reduced_periods[1])
        if np.linalg.norm(x_direction_lattice) <= self.__epsilon:
            raise GBMakerValueError(
                "primitive_periods must define distinct in-plane directions."
            )

        rounded_direction = np.rint(x_direction_lattice)
        if np.allclose(
            x_direction_lattice, rounded_direction, atol=self.__epsilon, rtol=0.0
        ) and np.any(rounded_direction):
            x_direction_lattice = self.__reduce_integer_row(
                rounded_direction.astype(int)
            ).astype(np.float64)

        x_period_vector = x_direction_lattice @ rotated_unit_cell_basis
        x_projection = float(x_period_vector[0])
        if np.isclose(x_projection, 0.0, atol=self.__epsilon, rtol=0.0):
            raise GBMakerValueError(
                "x-period direction must have a non-zero projection on x."
            )
        if x_projection < 0.0:
            x_projection = -x_projection

        box_basis = self.__box_periodic_basis(primitive_periods)
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

    def __reduced_box_coordinates(
        self, cartesian_coordinates: np.ndarray, box_basis: np.ndarray
    ) -> np.ndarray:
        """
        Convert Cartesian coordinates to mixed box coordinates ``[x_cart, u_y, u_z]``.

        The mixed basis is ``[e_x, A_y, A_z]`` where ``e_x`` is the Cartesian x-axis
        and ``A_y``/``A_z`` are the in-plane box basis vectors.

        :param cartesian_coordinates: Cartesian coordinates with shape ``(..., 3)``.
        :param box_basis: 2x3 array containing ``A_y`` and ``A_z``.
        :return: Mixed box coordinates with shape ``(..., 3)``.
        """
        cartesian_coordinates = np.asarray(cartesian_coordinates, dtype=np.float64)
        box_basis = np.asarray(box_basis, dtype=np.float64)

        yz_basis = box_basis[:, 1:].T
        determinant = np.linalg.det(yz_basis)
        if np.isclose(determinant, 0.0, atol=self.__epsilon, rtol=0.0):
            raise GBMakerValueError(
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

    def __cartesian_from_box_coordinates(
        self, box_coordinates: np.ndarray, box_basis: np.ndarray
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

    def __complete_origin_atom_mask(
        self,
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
        :raises GBMakerValueError: If the arrays are not one-dimensional and parallel,
            if ``origin_ids`` is not integer-valued, or if ``basis_size`` is not a
            positive integer.
        """
        atom_mask = np.asarray(atom_mask)
        origin_ids = np.asarray(origin_ids)

        if atom_mask.ndim != 1:
            raise GBMakerValueError("atom_mask must be a one-dimensional array.")
        if not np.issubdtype(atom_mask.dtype, np.bool_):
            raise GBMakerValueError("atom_mask must be a boolean array.")

        if origin_ids.ndim != 1:
            raise GBMakerValueError("origin_ids must be a one-dimensional array.")
        if not np.issubdtype(origin_ids.dtype, np.integer):
            raise GBMakerValueError("origin_ids must contain integer values.")

        if len(atom_mask) != len(origin_ids):
            raise GBMakerValueError("atom_mask and origin_ids must have equal length.")

        if isinstance(basis_size, (bool, np.bool_)) or not isinstance(
            basis_size, (int, np.integer)
        ):
            raise GBMakerValueError(
                f"basis_size must be a positive integer; got {basis_size!r}."
            )

        basis_size = int(basis_size)
        if basis_size < 1:
            raise GBMakerValueError(
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

    def __filter_complete_origins(
        self,
        atoms: np.ndarray,
        origin_ids: np.ndarray,
        atom_mask: np.ndarray,
        basis_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter atoms and origin IDs while preserving complete origin groups.

        Converts ``atom_mask`` into a complete-origin mask using
        ``__complete_origin_atom_mask``. An origin is kept only when every one of its
        ``basis_size`` atoms passes the input mask. The returned arrays are copies.

        :param atoms: Structured atom array to filter.
        :param origin_ids: Integer origin-ID array parallel to ``atoms``.
        :param atom_mask: Boolean atom-level mask parallel to ``atoms``.
        :param basis_size: Number of atoms expected in one complete origin group.
        :return: ``(filtered_atoms, filtered_origin_ids)``.
        :raises GBMakerValueError: If ``atoms`` and ``origin_ids`` are not parallel, or
            if ``__complete_origin_atom_mask`` rejects the mask, origin IDs, or basis
            size.
        """
        if len(atoms) != len(origin_ids):
            raise GBMakerValueError("atoms and origin_ids must have equal length.")

        keep_atoms = self.__complete_origin_atom_mask(
            atom_mask,
            origin_ids,
            basis_size,
        )
        return atoms[keep_atoms].copy(), origin_ids[keep_atoms].copy()

    def __clip_complete_origins_to_cartesian_box(
        self,
        atoms: np.ndarray,
        origin_ids: np.ndarray,
        x_bounds: np.ndarray,
        basis_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clip atoms to the Cartesian grain box by complete origin groups.

        Atoms are tested against the half-open x interval ``[x_bounds[0], x_bounds[1])``
        using the instance tolerance. Non-periodic in-plane axes are also clipped to
        their Cartesian box dimensions. Periodic in-plane axes are not clipped here
        because they have already been selected and wrapped by the mixed-basis selection
        path.

        Complete-origin filtering is applied after the atom-level box mask is
        constructed, so an origin is retained only when all atoms in that origin remain
        inside the requested box.

        :param atoms: Structured atom array to clip.
        :param origin_ids: Integer origin-ID array parallel to ``atoms``.
        :param x_bounds: Length-2 array containing lower and upper x bounds (Angstroms).
        :param basis_size: Number of atoms expected in one complete origin group.
        :return: ``(clipped_atoms, clipped_origin_ids)``.
        :raises GBMakerValueError: If ``x_bounds`` is not a finite increasing two-value
            interval, or if complete-origin filtering rejects the inputs.
        """
        try:
            x_bounds = np.asarray(x_bounds, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise GBMakerValueError(
                f"x_bounds must be a finite two-value interval; got {x_bounds!r}."
            ) from exc

        if x_bounds.shape != (2,):
            raise GBMakerValueError(
                f"x_bounds must be a finite two-value interval; got {x_bounds!r}."
            )
        if not np.all(np.isfinite(x_bounds)) or x_bounds[1] <= x_bounds[0]:
            raise GBMakerValueError(
                f"x_bounds must be a finite increasing interval; got {x_bounds!r}."
            )

        inside_box = (
            (atoms["x"] >= x_bounds[0] - self.__epsilon)
            & (atoms["x"] < x_bounds[1] - self.__epsilon)
        )

        axis_names = ("y", "z")
        axis_dims = (self.__y_dim, self.__z_dim)

        for axis_name, axis_dim, is_periodic in zip(
            axis_names,
            axis_dims,
            self.__inplane_periodic,
        ):
            if is_periodic:
                continue

            inside_box &= (
                (atoms[axis_name] >= -self.__epsilon)
                & (atoms[axis_name] < axis_dim)
            )

        clipped_atoms, clipped_origin_ids = self.__filter_complete_origins(
            atoms,
            origin_ids,
            inside_box,
            basis_size,
        )

        for axis_name, is_periodic in zip(axis_names, self.__inplane_periodic):
            if is_periodic:
                continue

            clipped_atoms[axis_name] = np.where(
                (clipped_atoms[axis_name] < 0.0)
                & (clipped_atoms[axis_name] >= -self.__epsilon),
                0.0,
                clipped_atoms[axis_name],
            )

        return clipped_atoms, clipped_origin_ids

    def __deduplicate_complete_origins(
        self,
        atoms: np.ndarray,
        origin_ids: np.ndarray,
        basis_size: int,
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
        :return: ``(deduplicated_atoms, deduplicated_origin_ids)``.
        :raises GBMakerValueError: If the inputs are not parallel, if
            ``origin_ids`` is not integer-valued, if ``basis_size`` is not a
            positive integer, or if the atom array cannot be reshaped into complete
            contiguous origin groups.
        """
        origin_ids = np.asarray(origin_ids)

        if origin_ids.ndim != 1:
            raise GBMakerValueError("origin_ids must be a one-dimensional array.")
        if not np.issubdtype(origin_ids.dtype, np.integer):
            raise GBMakerValueError("origin_ids must contain integer values.")

        if isinstance(basis_size, (bool, np.bool_)) or not isinstance(
            basis_size, (int, np.integer)
        ):
            raise GBMakerValueError(
                f"basis_size must be a positive integer; got {basis_size!r}."
            )

        basis_size = int(basis_size)
        if basis_size < 1:
            raise GBMakerValueError(
                f"basis_size must be a positive integer; got {basis_size!r}."
            )

        if len(atoms) != len(origin_ids):
            raise GBMakerValueError("atoms and origin_ids must have equal length.")

        if len(atoms) == 0:
            return atoms.copy(), origin_ids.copy()

        if len(atoms) % basis_size != 0:
            raise GBMakerValueError(
                "Complete-origin deduplication requires full origin groups."
            )

        n_origins = len(atoms) // basis_size
        grouped_origin_ids = origin_ids.reshape(n_origins, basis_size)
        if not np.all(grouped_origin_ids == grouped_origin_ids[:, :1]):
            raise GBMakerValueError(
                "Complete-origin deduplication requires contiguous origin groups."
            )

        positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
        quantized = np.round(positions / self.__epsilon).astype(np.int64)

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

    def __filter_float_result_complete_origins(
        self,
        result: _FloatGrainBuildResult,
        atom_mask: np.ndarray,
    ) -> _FloatGrainBuildResult:
        """Filter a float-path build result by complete origin groups.

        Applies ``atom_mask`` to ``result.atoms`` through complete-origin filtering,
        preserving only conventional-cell origins for which every atom in the origin
        group passes the mask. The returned result carries the filtered atom array,
        filtered parallel origin IDs, and the original basis size.

        :param result: Float-path grain build result to filter.
        :param atom_mask: Boolean atom-level mask parallel to ``result.atoms``.
        :return: Filtered float-path grain build result.
        :raises GBMakerValueError: If complete-origin filtering rejects the mask, origin
            IDs, or basis size.
        """
        atoms, origin_ids = self.__filter_complete_origins(
            result.atoms,
            result.origin_ids,
            atom_mask,
            result.basis_size,
        )
        return _FloatGrainBuildResult(
            atoms=atoms,
            origin_ids=origin_ids,
            basis_size=result.basis_size,
        )

    def __trim_float_result_to_upper_x(
        self,
        result: _FloatGrainBuildResult,
        upper_x: float,
    ) -> _FloatGrainBuildResult:
        """Trim a float-path grain to an upper x bound by complete origins.

        Retains only complete conventional-cell origins whose atoms all lie below
        ``upper_x`` using the same half-open upper-bound convention as the rest of the
        grain-generation pipeline.

        :param result: Float-path grain build result to trim.
        :param upper_x: Upper x bound in Angstroms.
        :return: Trimmed float-path grain build result.
        :raises GBMakerValueError: If ``upper_x`` is not finite or if complete-origin
            filtering rejects the result metadata.
        """
        try:
            upper_x = float(upper_x)
        except (TypeError, ValueError) as exc:
            raise GBMakerValueError(
                f"upper_x must be a finite number; got {upper_x!r}."
            ) from exc

        if not math.isfinite(upper_x):
            raise GBMakerValueError(
                f"upper_x must be a finite number; got {upper_x!r}."
            )

        atom_mask = result.atoms["x"] < upper_x - self.__epsilon
        return self.__filter_float_result_complete_origins(result, atom_mask)

    def __select_complete_origins_in_box_basis(
        self,
        atoms: np.ndarray,
        origin_ids: np.ndarray,
        primitive_periods: np.ndarray,
        x_bounds: np.ndarray,
        basis_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select and wrap in-plane coordinates while preserving complete origins.

        Builds the y/z selection basis from ``primitive_periods`` and filters atoms into
        the in-plane simulation box. Periodic in-plane axes are selected in reduced
        coordinates and wrapped onto the periodic box. Non-periodic in-plane axes are
        selected against their Cartesian box extents.

        When the selection basis has no x component, the method uses an axis-aligned
        fast path. Otherwise, atoms are converted into mixed box coordinates ``[x_cart,
        u_y, u_z]``, selected/wrapped there, converted back to Cartesian coordinates,
        and then re-filtered by complete origins against the x slab.

        :param atoms: Structured atom array to select and wrap.
        :param origin_ids: Integer origin-ID array parallel to ``atoms``.
        :param primitive_periods: Two-row array containing the in-plane y and z
            primitive period vectors in strained lab-frame Cartesian coordinates.
        :param x_bounds: Length-2 array containing lower and upper x bounds (Angstroms).
        :param basis_size: Number of atoms expected in one complete origin group.
        :return: ``(selected_atoms, selected_origin_ids)`` after complete-origin
            selection and periodic wrapping.
        :raises GBMakerValueError: If ``x_bounds`` is not a finite increasing interval,
            if the selection basis is singular, or if complete-origin filtering rejects
            the inputs.
        """
        try:
            x_bounds = np.asarray(x_bounds, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise GBMakerValueError(
                f"x_bounds must be a finite two-value interval; got {x_bounds!r}."
            ) from exc

        if x_bounds.shape != (2,):
            raise GBMakerValueError(
                f"x_bounds must be a finite two-value interval; got {x_bounds!r}."
            )
        if not np.all(np.isfinite(x_bounds)) or x_bounds[1] <= x_bounds[0]:
            raise GBMakerValueError(
                f"x_bounds must be a finite increasing interval; got {x_bounds!r}."
            )

        selection_basis = self.__selection_basis_vectors(primitive_periods)
        positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
        inplane_periodic = self.__inplane_periodic

        if np.allclose(selection_basis[:, 0], 0.0, atol=self.__epsilon, rtol=0.0):
            inside_box = np.ones(len(atoms), dtype=bool)
            for row_index, is_periodic in enumerate(inplane_periodic):
                axis = row_index + 1
                period = selection_basis[row_index, axis]
                coord = positions[:, axis]

                if is_periodic:
                    tol = self.__reduced_coordinate_tolerance(
                        selection_basis[row_index]
                    )
                    reduced_coord = coord / period
                    inside_box &= (
                        (reduced_coord >= -tol)
                        & (reduced_coord < 1.0 + tol)
                    )

            selected_atoms, selected_origin_ids = self.__filter_complete_origins(
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
                tol = self.__reduced_coordinate_tolerance(selection_basis[row_index])

                wrapped = np.mod(selected_atoms[axis_name], period)
                selected_atoms[axis_name] = np.where(
                    (wrapped < tol * period) | ((period - wrapped) < tol * period),
                    0.0,
                    wrapped,
                )

            inside_x = (
                (selected_atoms["x"] >= x_bounds[0] - self.__epsilon)
                & (selected_atoms["x"] < x_bounds[1] - self.__epsilon)
            )
            return self.__filter_complete_origins(
                selected_atoms,
                selected_origin_ids,
                inside_x,
                basis_size,
            )

        box_coordinates = self.__reduced_box_coordinates(positions, selection_basis)

        inside_box = np.ones(len(atoms), dtype=bool)
        axis_dims = (self.__y_dim, self.__z_dim)

        for row_index, (axis_dim, is_periodic) in enumerate(
            zip(axis_dims, inplane_periodic)
        ):
            reduced_axis = box_coordinates[:, row_index + 1]

            if is_periodic:
                tol = self.__reduced_coordinate_tolerance(selection_basis[row_index])
                inside_box &= (
                    (reduced_axis >= -tol)
                    & (reduced_axis < 1.0 + tol)
                )
            else:
                inside_box &= (
                    (reduced_axis >= -self.__epsilon)
                    & (reduced_axis < axis_dim)
                )

        selected_atoms, selected_origin_ids = self.__filter_complete_origins(
            atoms,
            origin_ids,
            inside_box,
            basis_size,
        )
        if len(selected_atoms) == 0:
            return selected_atoms, selected_origin_ids

        selected_mask = self.__complete_origin_atom_mask(
            inside_box,
            origin_ids,
            basis_size,
        )
        selected_box_coordinates = box_coordinates[selected_mask].copy()

        for row_index, is_periodic in enumerate(inplane_periodic):
            coordinate_index = row_index + 1

            if is_periodic:
                tol = self.__reduced_coordinate_tolerance(selection_basis[row_index])
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
                        >= -self.__epsilon
                    )
                ),
                0.0,
                selected_box_coordinates[:, coordinate_index],
            )

        wrapped_positions = self.__cartesian_from_box_coordinates(
            selected_box_coordinates,
            selection_basis,
        )
        selected_atoms["x"], selected_atoms["y"], selected_atoms["z"] = (
            wrapped_positions.T
        )

        inside_x = (
            (selected_atoms["x"] >= x_bounds[0] - self.__epsilon)
            & (selected_atoms["x"] < x_bounds[1] - self.__epsilon)
        )
        return self.__filter_complete_origins(
            selected_atoms,
            selected_origin_ids,
            inside_x,
            basis_size,
        )

    def __grain_strain_scales(self, grain_side: str) -> tuple[float, float]:
        """Return lab-frame in-plane strain scale factors for one grain.

        The returned scale factors are applied to the rotated lab-frame y and z
        coordinates of atoms in the selected grain. Axes without mismatch accommodation
        use scale factor ``1.0``.

        :param grain_side: Grain side, either ``"left"`` or ``"right"``.
        :return: ``(y_scale, z_scale)`` for the selected grain.
        :raises GBMakerValueError: If ``grain_side`` is not ``"left"`` or ``"right"``.
        """
        if grain_side not in {"left", "right"}:
            raise GBMakerValueError(
                f"grain_side must be 'left' or 'right'; got {grain_side!r}."
            )

        y_accommodation = self.__strain_accommodation.get("y")
        z_accommodation = self.__strain_accommodation.get("z")

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

    def __update_dims(self) -> None:
        """Updates the y_dim and z_dim parameters after a relevant parameter has been
        changed.

        In-plane strain-accommodation search, minimum in-plane dimension enforcement,
        and box-dimension assembly are a pure construction stage; see
        ``GBOpt.gbmaker.dimension.plan_dimensions``.
        """
        use_exact = (
            self.__embedding is not None
            and self.__embedding.exact
            and self.__embedding.P is not None
        )

        plan, (self.__repeat_factor[0], self.__repeat_factor[1]) = (
            self.__translate_construction_error(
                plan_dimensions,
                a0=self.__a0,
                left_periodic_miller_rows=self.__left_periodic_miller_rows,
                right_periodic_miller_rows=self.__right_periodic_miller_rows,
                spacing_y=self.__spacing["y"],
                spacing_z=self.__spacing["z"],
                repeat_factor=tuple(self.__repeat_factor),
                mismatch_tol=self.__mismatch_tol,
                mismatch_max_cells=self.__mismatch_max_cells,
                strain_grain=self.__strain_grain,
                require_exact_pair=use_exact,
                interaction_distance=self.__interaction_distance,
                x_dim=self.__x_dim,
                vacuum_thickness=self.__vacuum_thickness,
                normal_topology=self.__normal_topology,
                epsilon=self.__epsilon,
            )
        )
        self.__strain_accommodation = dict(plan.accommodation)
        self.__y_dim = float(plan.box_dims[1][1])
        self.__z_dim = float(plan.box_dims[2][1])
        self.__box_dims = np.array(plan.box_dims, dtype=float)

        self.__generate_gb()
        self.__set_gb_region()

    def __validate(
        self,
        value: Any,
        expected_types: type | tuple[type, ...],
        parameter_name: str,
        *,
        positive: bool = False,
        expected_length: int | None = None,
        strictly_positive: bool = False
    ):
        """Private method for validating the values passed in using the setters.

        Thin wrapper delegating to the single pure implementation in
        ``GBOpt.gbmaker.config._validate_scalar`` (also used by
        ``normalize_legacy_config`` for the legacy constructor), via
        ``__translate_construction_error``.

        :param value: The value to validate.
        :param expected_types: Single type or tuple containing the valid types for
            value.
        :param parameter_name: The name of the parameter.
        :param positive: Whether or not the value should be positive (>= 0), optional,
            defaults to False.
        :param expected_length: Specific to sequences or arrays. The expected length of
            the sequence or array, optional, defaults to None.
        :param strictly_positive: Supercedes ``positive`` by enforcing value > 0.
            Optional, defaults to False.
        :raises GBMakerTypeError: Exception raised if the type of the value does not
            match the expected type(s).
        :raises GBMakerValueError: Exception raised when invalid values are given for
            the specified parameter.
        :return: The validated value.
        """
        return self.__translate_construction_error(
            _validate_scalar,
            value,
            expected_types,
            parameter_name,
            nonnegative=positive,
            expected_length=expected_length,
            strictly_positive=strictly_positive,
        )

    # Public methods
    def get_supercell(self, corners: np.ndarray) -> np.ndarray:
        """Generates a supercell of lattice sites.

        :param corners: Array containing the position of the corners of the unit cells.
        :return: Structured numpy array containing the atom data (type and position) for
            the supercell.
        """
        # Unit cell as structured array
        unit_cell = self.__unit_cell.asarray()
        supercell = np.tile(unit_cell, len(corners))
        translations = np.repeat(corners, len(unit_cell), axis=0)
        supercell["x"] += translations[:, 0]
        supercell["y"] += translations[:, 1]
        supercell["z"] += translations[:, 2]
        return supercell

    def update_spacing(self, threshold: float = None) -> None:
        """Update the periodic spacing based on the rotation matrix and the optional
        threshold parameter.

        :param threshold: The maximum allowed value that any spacing can take
        """
        self.__spacing = self.__calculate_periodic_spacing(threshold)
        self.__update_dims()

    def write_lammps(
        self,
        file_name: str,
        atoms: np.ndarray = None,
        box_sizes: np.ndarray = None,
        *,
        type_as_int: bool = False,
        precision: int = 6,
        charges: dict = None,
        triclinic: bool = False
    ) -> None:
        """Writes atom positions with the given box dimensions to a LAMMPS input file.

        :param str file_name: The filename to save the data
        :param np.ndarray atoms: The numpy array containing the atom data.
        :param np.ndarray box_sizes: 3x2 array containing the min and max dimensions for
            each of the x, y, and z dimensions.
        :param type_as_int: Whether to write the atom types as a chemical name or a
            number. Keyword argument, optional, defaults to False (write as a chemical
            name).
        :param precision: The decimal precision to use when writing float values,
            optional, default = 6.
        :param charges: dict containing the charge values for each type. Keys are
            expected to be integers, values are expected to be numeric. Optional,
            default is None.
        """
        if not isinstance(file_name, str):
            raise GBMakerTypeError("file_name must be of type str")
        if atoms is None and box_sizes is None:
            atoms = self.__whole_system
            box_sizes = self.__box_dims
        elif (atoms is None and box_sizes is not None) or (
            atoms is not None and box_sizes is None
        ):
            raise GBMakerValueError(
                "'atoms' and 'box_sizes' must be specified together."
            )

        atom_names = np.unique(atoms["name"])
        if set(atom_names).issubset(self.__unit_cell.type_map.keys()):
            name_to_int = {
                name: self.__unit_cell.type_map[name]
                for name in self.__unit_cell.type_map
                if name in atom_names
            }
        else:
            name_to_int = {name: i + 1 for i, name in enumerate(atom_names)}

        if charges is not None:
            if not all(isinstance(i, (int, str)) for i in charges):
                raise GBMakerValueError(
                    "'charges' keys are required to be integers or strings.")
            if not all([isinstance(i, Number) for i in charges.values()]):
                raise GBMakerValueError(
                    "'charges' values are required to be numeric.")
            if type_as_int:
                if all([isinstance(i, str) for i in charges]):
                    for name in np.unique(atoms["name"]):
                        charges[name_to_int[name]] = charges[name]

        def format_atom_line(index, name, pos, charge=None):
            if type_as_int:
                name = name_to_int[name]
            if charge is not None:
                return (f"{index} {name} {charge:.{precision}f} " +
                        f"{pos[0]:.{precision}f} {pos[1]:.{precision}f} " +
                        f"{pos[2]:.{precision}f}\n")
            else:
                return (f"{index} {name} {pos[0]:.{precision}f} " +
                        f"{pos[1]:.{precision}f} {pos[2]:.{precision}f}\n")

        # Write LAMMPS data file
        with open(file_name, "w") as fdata:
            # First line is a comment line
            atom_names = "".join(np.unique(atoms["name"]))
            fdata.write(f"Crystalline {atom_names} atoms\n\n")

            # --- Header ---#
            # Specify number of atoms and atom types
            fdata.write(f"{len(atoms)} atoms\n")
            fdata.write("{} atom types\n".format(len(set(atoms["name"]))))
            # Specify box dimensions
            fdata.write(
                f"{box_sizes[0][0]:.{precision}f} "
                f"{box_sizes[0][1]:.{precision}f} xlo xhi\n"
            )
            fdata.write(
                f"{box_sizes[1][0]:.{precision}f} "
                f"{box_sizes[1][1]:.{precision}f} ylo yhi\n"
            )
            fdata.write(
                f"{box_sizes[2][0]:.{precision}f} "
                f"{box_sizes[2][1]:.{precision}f} zlo zhi\n"
            )
            if triclinic:
                xy, xz, yz, theta = self.__get_triclinic_params()
                fdata.write(
                    f"{xy:.{precision}f} {xz:.{precision}f} "
                    f"{yz:.{precision}f} xy xz yz\n"
                )
                ct, st = math.cos(theta), math.sin(theta)
                Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])

            if not type_as_int:
                fdata.write("\nAtom Type Labels\n\n")
                for name, value in name_to_int.items():
                    fdata.write(f"{value} {name}\n")

            # Atoms section
            fdata.write("\nAtoms\n\n")

            # Write each position.
            for i, (name, *pos) in enumerate(atoms):
                if charges is not None:
                    charge = charges[name_to_int[name]
                                     ]if type_as_int else charges[name]
                else:
                    charge = None

                if triclinic:
                    pos = Rx @ np.array(pos, dtype=float)
                fdata.write(format_atom_line(i + 1, name, pos, charge))

    # Properties with getters and setters. Automatic updates for related parameters are
    # automatically taken care of.
    @property
    def a0(self) -> float:
        return self.__a0

    @a0.setter
    def a0(self, value: Number) -> None:
        atom_types = tuple(self.__unit_cell.names())
        self.__a0 = self.__validate(value, float, "a0", positive=True)
        self.__unit_cell = self.__init_unit_cell(atom_types)
        self.update_spacing()

    @property
    def epsilon(self) -> float:
        return self.__epsilon

    @epsilon.setter
    def epsilon(self, value: Number) -> None:
        self.__epsilon = self.__validate(
            value, Number, "epsilon", strictly_positive=True)

    @property
    def gb_thickness(self) -> float:
        return self.__gb_thickness

    @gb_thickness.setter
    def gb_thickness(self, value: Number):
        self.__gb_thickness = self.__validate(
            value, Number, "gb_thickness", positive=True)
        self.__box_dims = self.__calculate_box_dimensions()

    @property
    def id(self) -> int:
        return self.__id

    @id.setter
    def id(self, value: int):
        self.__id = self.__validate(value, int, "id", positive=True)

    @property
    def interaction_distance(self) -> float:
        return self.__interaction_distance

    @interaction_distance.setter
    def interaction_distance(self, value: Number) -> None:
        self.__interaction_distance = self.__validate(
            value, Number, "interaction_distance", positive=True)
        self.__update_dims()

    @property
    def misorientation(self) -> np.ndarray:
        return np.hstack((self.__misorientation, self.__inclination))

    @misorientation.setter
    def misorientation(self, value: np.ndarray):
        misorientation = self.__validate(
            value, np.ndarray, "misorientation", expected_length=5
        )
        self.__assign_orientations(misorientation)
        # Discard any active embedding so update_spacing uses the new Euler
        # angles rather than the stale embedding-derived rotation matrices.
        self.__embedding = None
        self.update_spacing()

    @property
    def repeat_factor(self) -> int:
        return self.__repeat_factor

    @repeat_factor.setter
    def repeat_factor(self, value: int):
        self.__repeat_factor = self.__validate(
            value, (int, Sequence), "repeat_factor", positive=True)
        self.__update_dims()

    @property
    def structure(self) -> str:
        return self.__structure

    @structure.setter
    def structure(self, value: str) -> None:
        self.__structure = self.__validate(value, str, "structure")
        if {self.__structure, value}.issubset({"fluorite", "rocksalt", "zincblende"}):
            raise GBMakerValueError(
                f"Cannot estimate conversion from {self.__structure} to {value}"
            )
        else:
            atom_types = tuple(set(self.__unit_cell.names()))

        self.__unit_cell = self.__init_unit_cell(atom_types)

    @property
    def vacuum_thickness(self) -> float:
        return self.__vacuum_thickness

    @vacuum_thickness.setter
    def vacuum_thickness(self, value: Number):
        old_vacuum = self.__vacuum_thickness
        vacuum_value = self.__validate(
            value, Number, "vacuum_thickness", positive=True
        )
        self.__vacuum_thickness, self.__normal_topology = (
            _normalize_vacuum_topology(
                vacuum_value,
                tolerance=self.__epsilon,
            )
        )
        delta = self.__vacuum_thickness - old_vacuum
        self.__left_grain["x"] += delta
        self.__right_grain["x"] += delta
        self.__whole_system["x"] += delta
        self.__gb_region["x"] += delta
        self.__box_dims = self.__calculate_box_dimensions()

    @property
    def x_dim_min(self) -> np.ndarray:
        return self.__x_dim_min

    @x_dim_min.setter
    def x_dim_min(self, value: Number):
        self.__x_dim_min = self.__validate(
            value, Number, "x_dim_min", positive=True)
        self.update_spacing()
        self.__box_dims = self.__calculate_box_dimensions()

    # Additional getters for other class properties
    @property
    def inplane_periodic(self) -> tuple[bool, bool]:
        """Read-only view of the in-plane periodicity flags (y, z)."""
        return tuple(bool(v) for v in self.__inplane_periodic)

    @property
    def normal_topology(self) -> BoundaryNormalTopology:
        """Explicit physical topology along the boundary normal"""
        return self.__normal_topology

    @property
    def uses_exact_construction(self) -> bool:
        """Read-only flag indicating exact integer P/Q construction is active."""
        return bool(
            self.__embedding is not None
            and self.__embedding.exact
            and self.__embedding.P is not None
        )

    @property
    def box_dims(self) -> np.ndarray:
        return self.__box_dims

    @property
    def whole_system(self) -> np.ndarray:
        return self.__whole_system

    @property
    def left_grain(self) -> np.ndarray:
        return self.__left_grain

    @property
    def radius(self) -> float:
        return self.__radius

    @property
    def right_grain(self) -> np.ndarray:
        return self.__right_grain

    @property
    def gb_plane_x(self) -> float:
        return self.__vacuum_thickness + self.__left_x

    @property
    def spacing(self) -> dict:
        return self.__spacing

    @property
    def unit_cell(self) -> UnitCell:
        return self.__unit_cell

    @property
    def x_dim(self) -> float:
        return self.__x_dim

    @property
    def y_dim(self) -> float:
        return self.__y_dim

    @property
    def z_dim(self) -> float:
        return self.__z_dim
