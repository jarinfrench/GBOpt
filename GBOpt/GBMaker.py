# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Grain boundary builder utilities."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

import numpy as np

from GBOpt.BoundarySpec import (
    BoundaryEmbedding,
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
from GBOpt.gbmaker.assembly import assemble_bicrystal
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
from GBOpt.gbmaker.geometry import _triclinic_tilt_params
from GBOpt.gbmaker.geometry import wrap_reduced_coordinate as _wrap_reduced_coordinate
from GBOpt.gbmaker.orientation import (
    _decompose_misorientation,
    resolve_orientation,
)
from GBOpt.gbmaker.types import (
    AxisAccommodation,
    GBMakerConstructionTypeError,
    GBMakerConstructionValueError,
    MaterialState,
)
from GBOpt.io.lammps.data_writer import LammpsDataWriter
from GBOpt.io.lammps.types import LammpsWriteError
from GBOpt.io.types import StructureData, StructureValueError
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


@dataclass
class _MakerConfig:
    """Canonical validated top-level ``GBMaker`` build configuration.

    Grouping-only: every field arrives already validated by ``GBMaker``'s own
    per-field validators (``__validate`` and its siblings), which are deliberately not
    the same validation rules as ``GBOpt.gbmaker.types``'s own pipeline dataclasses
    (``MaterialState``/``GBBuildConfig``) -- e.g. ``a0`` is accepted at exactly ``0``
    here, matching the legacy setter's ``nonnegative`` check, where ``MaterialState``
    requires it strictly positive (see ``CLAUDE.md``'s R04 ``positive=True`` history
    and ``REFACTOR_CLEANUP.md``). This dataclass performs no independent validation of
    its own and is mutated in place by property setters, matching the flat instance
    attributes it replaces field-for-field.

    :param radius: Atom radius (``a0 * unit_cell.radius``), computed once at
        construction and, matching the pre-R10 flat ``self.__radius`` attribute it
        replaces, never recomputed by any property setter (including ``a0``'s and
        ``structure``'s) even though both change quantities it depends on.
    """

    a0: float
    structure: str
    unit_cell: UnitCell
    gb_thickness: float
    repeat_factor: list[int]
    x_dim_min: float
    interaction_distance: float
    gb_id: int
    epsilon: float
    mismatch_tol: float | None
    mismatch_max_cells: int
    strain_grain: str
    radius: float


@dataclass
class _BoundaryState:
    """Canonical resolved orientation, periodicity, and vacuum-topology state.

    Grouping-only, like ``_MakerConfig``: fields are mutated in place by the existing
    private orchestration methods (``__assign_orientations``,
    ``__calculate_periodic_spacing``, ``__update_dims``) exactly as the flat instance
    attributes they replace were, so every field defaults to a placeholder and is
    filled in during ``__init__`` in the same order the old flat assignments ran.
    """

    embedding: BoundaryEmbedding | None = None
    misorientation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inclination: np.ndarray = field(default_factory=lambda: np.zeros(2))
    R_mis: np.ndarray = field(default_factory=lambda: np.eye(3))
    R_incl: np.ndarray = field(default_factory=lambda: np.eye(3))
    R_left: np.ndarray = field(default_factory=lambda: np.eye(3))
    R_right: np.ndarray = field(default_factory=lambda: np.eye(3))
    left_periodic_miller_rows: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3))
    )
    right_periodic_miller_rows: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3))
    )
    inplane_periodic: tuple[bool, bool] = (True, True)
    left_x: float = 0.0
    right_x: float = 0.0
    x_dim: float = 0.0
    y_dim: float = 0.0
    z_dim: float = 0.0
    spacing: dict = field(default_factory=dict)
    # Maps axis name ("y" or "z") to commensurate repeat metadata when mismatch
    # accommodation is active; empty when mismatch_tol is None.
    strain_accommodation: dict[str, AxisAccommodation] = field(default_factory=dict)
    vacuum_thickness: float = 0.0
    normal_topology: BoundaryNormalTopology = BoundaryNormalTopology.PERIODIC_BICRYSTAL
    box_dims: np.ndarray = field(default_factory=lambda: np.empty((3, 2)))


@dataclass
class _AssembledResult:
    """Canonical cached bicrystal-assembly output.

    Grouping-only, like ``_MakerConfig``/``_BoundaryState``: mirrors the four
    structured atom arrays ``__generate_gb`` used to mirror onto separate flat
    instance attributes (``__left_grain``/``__right_grain``/``__whole_system``/
    ``__gb_region``). Not the pipeline's own ``GBOpt.gbmaker.types.BicrystalResult``
    directly: that type also carries ``box_dims``/``normal_topology``/``gb_id`` (kept
    on ``_BoundaryState``/``_MakerConfig`` here instead, since those are known before
    the first bicrystal assembly ever runs) and is frozen, which would require
    reconstructing a new instance on every ``vacuum_thickness``-setter incremental atom
    shift rather than mutating the existing arrays' contents in place as today.
    """

    atoms: np.ndarray = field(default_factory=lambda: np.empty(0))
    left_atoms: np.ndarray = field(default_factory=lambda: np.empty(0))
    right_atoms: np.ndarray = field(default_factory=lambda: np.empty(0))
    gb_region_atoms: np.ndarray = field(default_factory=lambda: np.empty(0))


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


def wrap_reduced_coordinate(reduced_coord: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """
    Wrap reduced coordinates into [0, 1) and snap both periodic faces to 0.

    Thin wrapper delegating to ``GBOpt.gbmaker.geometry.wrap_reduced_coordinate``,
    kept as a module-level compatibility alias since existing code imports
    ``GBOpt.GBMaker.wrap_reduced_coordinate`` directly.

    :param reduced_coord: Reduced coordinates to wrap.
    :param tol: Tolerance in reduced-coordinate units. Optional, defaults to 1e-8
    :return: Wrapped reduced coordinates in [0, 1).
    :raises GBMakerValueError: If ``tol`` is not finite or is negative.
    """
    try:
        return _wrap_reduced_coordinate(reduced_coord, tol)
    except GBMakerConstructionValueError as exc:
        raise GBMakerValueError(str(exc)) from exc


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

        unit_cell = config.material.unit_cell
        if unit_cell is None:
            # normalize_legacy_config always resolves a UnitCell via
            # resolve_material_state; this is an assertion against that invariant, not
            # a reachable runtime path, so a clear domain error beats a raw
            # AttributeError below if it were ever violated.
            raise GBMakerValueError("material.unit_cell was not resolved")

        self._config = _MakerConfig(
            a0=config.material.a0,
            structure=config.material.structure,
            unit_cell=unit_cell,
            gb_thickness=config.gb_thickness,
            repeat_factor=list(config.repeat_factor),
            x_dim_min=config.x_dim_min,
            interaction_distance=config.interaction_distance,
            gb_id=config.gb_id,
            epsilon=config.epsilon,
            mismatch_tol=config.mismatch_tol,
            mismatch_max_cells=config.mismatch_max_cells,
            strain_grain=config.strain_grain,
            radius=config.material.a0 * unit_cell.radius,
        )
        vacuum_thickness, normal_topology = _normalize_vacuum_topology(
            config.vacuum,
            tolerance=self._config.epsilon,
        )
        self._boundary = _BoundaryState(
            embedding=_embedding,
            vacuum_thickness=vacuum_thickness,
            normal_topology=normal_topology,
        )
        self._result = _AssembledResult()

        self.__assign_orientations(
            self.__validate(
                np.asarray(misorientation),
                np.ndarray,
                "misorientation",
                expected_length=5,
            )
        )

        self._boundary.spacing = self.__calculate_periodic_spacing()
        self.__update_dims()

        self._boundary.box_dims = self.__calculate_box_dimensions()

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
            self._boundary.misorientation,
            self._boundary.inclination,
            self._boundary.R_mis,
            self._boundary.R_incl,
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
                self._boundary.x_dim,
                self._boundary.vacuum_thickness,
                self._boundary.y_dim,
                self._boundary.z_dim,
            )
        )

    def __material_state(self) -> MaterialState:
        """Return the current crystal identity as a ``MaterialState``.

        Thin wrapper constructing a ``GBOpt.gbmaker.types.MaterialState`` from current
        instance state, for ``GBOpt.gbmaker.assembly.assemble_bicrystal``'s ``material``
        argument.

        :return: Current material identity.
        """
        return self.__translate_construction_error(
            MaterialState,
            a0=self._config.a0,
            structure=self._config.structure,
            atom_types=tuple(self._config.unit_cell.names()),
            unit_cell=self._config.unit_cell,
        )

    def __generate_gb(self) -> None:
        """Generate the left grain, right grain, combined GB atom array, and GB region.

        Thin wrapper delegating to ``GBOpt.gbmaker.assembly.assemble_bicrystal``, which
        builds each grain using the exact integer path when a coherent exact boundary
        embedding with integer P/Q matrices is available (otherwise the floating-point
        grain-generation path), equalizes the periodic x gap, concatenates the grains,
        and selects the grain-boundary-region window. Periodic x-gap equalization
        remains available only to the floating path. Exact decorated grains retain
        every enumerated site and are assembled without x-layer deletion.

        :return: ``None``. Updates ``self._result``'s ``left_atoms``, ``right_atoms``,
            ``atoms``, and ``gb_region_atoms``.
        :raises GBMakerValueError: If exact grain generation requires missing P/Q data,
            if float-path gap equalization lacks right-grain build metadata, if an
            exact grain crosses the central or periodic x boundary, or if a downstream
            grain-generation stage fails.
        """
        result = self.__translate_construction_error(
            assemble_bicrystal,
            material=self.__material_state(),
            embedding=self._boundary.embedding,
            R_left=self._boundary.R_left,
            R_right=self._boundary.R_right,
            left_periodic_miller_rows=self._boundary.left_periodic_miller_rows,
            right_periodic_miller_rows=self._boundary.right_periodic_miller_rows,
            left_x=self._boundary.left_x,
            right_x=self._boundary.right_x,
            x_dim=self._boundary.x_dim,
            vacuum_thickness=self._boundary.vacuum_thickness,
            inplane_periodic=self._boundary.inplane_periodic,
            inplane_box_lengths=(self._boundary.y_dim, self._boundary.z_dim),
            epsilon=self._config.epsilon,
            strain_accommodation=self._boundary.strain_accommodation,
            gb_thickness=self._config.gb_thickness,
            box_dims=self._boundary.box_dims,
            normal_topology=self._boundary.normal_topology,
            gb_id=self._config.gb_id,
        )

        self._result.left_atoms = result.left_atoms
        self._result.right_atoms = result.right_atoms
        self._result.atoms = result.atoms
        self._result.gb_region_atoms = result.gb_region_atoms

    def __calculate_periodic_spacing(self, threshold: float = None) -> dict:
        """
        Calculate the periodic spacing based on the rotation matrix.

        :param threshold: The maximum allowed value that any spacing can take. Default
            is 15 * a0.
        :return: Dict containing the periodic spacing along the 'x', 'y', and 'z'
            directions for the given misorientation.
        """
        if threshold is None:
            threshold = self._config.a0 * 15

        # Rotation-matrix and periodic-Miller-row assignment, and in-plane periodicity
        # determination, are a pure construction stage; see
        # ``GBOpt.gbmaker.orientation.resolve_orientation``. Exact P/Q rows are never
        # routed through integer-row approximation there.
        orientation = self.__translate_construction_error(
            resolve_orientation,
            np.hstack((self._boundary.misorientation, self._boundary.inclination)),
            embedding=self._boundary.embedding,
            a0=self._config.a0,
            threshold=threshold,
        )
        self._boundary.R_mis = orientation.R_mis
        self._boundary.R_incl = orientation.R_incl
        self._boundary.R_left = orientation.R_left
        self._boundary.R_right = orientation.R_right
        self._boundary.left_periodic_miller_rows = orientation.left_periodic_miller_rows
        self._boundary.right_periodic_miller_rows = orientation.right_periodic_miller_rows
        self._boundary.inplane_periodic = orientation.inplane_periodic

        # Periodic-spacing and boundary-normal x-extent arithmetic is a pure
        # construction stage; see ``GBOpt.gbmaker.dimension.plan_periodic_spacing``.
        # In-plane periodicity was already resolved by ``resolve_orientation`` above
        # (including, on the legacy/five-DOF path, its own threshold warning); this
        # only reapplies the resulting flags to the returned spacing values used for
        # box-dimension planning.
        spacing, self._boundary.left_x, self._boundary.right_x, self._boundary.x_dim = (
            self.__translate_construction_error(
                plan_periodic_spacing,
                a0=self._config.a0,
                left_periodic_miller_rows=self._boundary.left_periodic_miller_rows,
                right_periodic_miller_rows=self._boundary.right_periodic_miller_rows,
                x_dim_min=self._config.x_dim_min,
                epsilon=self._config.epsilon,
                inplane_periodic=self._boundary.inplane_periodic,
                threshold=threshold,
                legacy_periodicity_heuristic=(
                    self._boundary.embedding is None
                    or self._boundary.embedding.source == "five_dof"
                ),
            )
        )

        return spacing

    def __get_triclinic_params(self):
        """
        Computes the LAMMPS restricted-triclinic tilt factors.

        Thin wrapper delegating to ``GBOpt.gbmaker.geometry._triclinic_tilt_params``.

        :return: (xy, xz, yz, theta) - the three tilt scalars and the rotation angle to
                                       apply to atom coordinates
        :raises GBMakerValueError: If the y/z directions are not both periodic, or if
            the selected grain's primitive periods have a near-zero projection on
            their own box axis.
        """
        return self.__translate_construction_error(
            _triclinic_tilt_params,
            inplane_periodic=self._boundary.inplane_periodic,
            left_periodic_miller_rows=self._boundary.left_periodic_miller_rows,
            right_periodic_miller_rows=self._boundary.right_periodic_miller_rows,
            R_left=self._boundary.R_left,
            R_right=self._boundary.R_right,
            conventional_basis=self._config.unit_cell.conventional,
            y_dim=self._boundary.y_dim,
            z_dim=self._boundary.z_dim,
            epsilon=self._config.epsilon,
        )

    def __init_unit_cell(self, atom_types: str | tuple[str, ...]) -> UnitCell:
        """
        Initializes the unit cell.

        :return: The unit cell initialized by structure.
        """
        unit_cell = UnitCell()
        unit_cell.init_by_structure(self._config.structure, self._config.a0, atom_types)
        return unit_cell

    def __update_dims(self) -> None:
        """Updates the y_dim and z_dim parameters after a relevant parameter has been
        changed.

        In-plane strain-accommodation search, minimum in-plane dimension enforcement,
        and box-dimension assembly are a pure construction stage; see
        ``GBOpt.gbmaker.dimension.plan_dimensions``.
        """
        use_exact = (
            self._boundary.embedding is not None
            and self._boundary.embedding.exact
            and self._boundary.embedding.P is not None
        )

        repeat_factor = self._config.repeat_factor
        plan, (repeat_factor[0], repeat_factor[1]) = (
            self.__translate_construction_error(
                plan_dimensions,
                a0=self._config.a0,
                left_periodic_miller_rows=self._boundary.left_periodic_miller_rows,
                right_periodic_miller_rows=self._boundary.right_periodic_miller_rows,
                spacing_y=self._boundary.spacing["y"],
                spacing_z=self._boundary.spacing["z"],
                repeat_factor=tuple(repeat_factor),
                mismatch_tol=self._config.mismatch_tol,
                mismatch_max_cells=self._config.mismatch_max_cells,
                strain_grain=self._config.strain_grain,
                require_exact_pair=use_exact,
                interaction_distance=self._config.interaction_distance,
                x_dim=self._boundary.x_dim,
                vacuum_thickness=self._boundary.vacuum_thickness,
                normal_topology=self._boundary.normal_topology,
                epsilon=self._config.epsilon,
            )
        )
        self._boundary.strain_accommodation = dict(plan.accommodation)
        self._boundary.y_dim = float(plan.box_dims[1][1])
        self._boundary.z_dim = float(plan.box_dims[2][1])
        self._boundary.box_dims = np.array(plan.box_dims, dtype=float)

        self.__generate_gb()

    def __validate(
        self,
        value: Any,
        expected_types: type | tuple[type, ...],
        parameter_name: str,
        *,
        nonnegative: bool = False,
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
        :param nonnegative: Whether or not the value should be non-negative (>= 0),
            optional, defaults to False.
        :param expected_length: Specific to sequences or arrays. The expected length of
            the sequence or array, optional, defaults to None.
        :param strictly_positive: Supercedes ``nonnegative`` by enforcing value > 0.
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
            nonnegative=nonnegative,
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
        unit_cell = self._config.unit_cell.asarray()
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
        self._boundary.spacing = self.__calculate_periodic_spacing(threshold)
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
            atoms = self._result.atoms
            box_sizes = self._boundary.box_dims
        elif (atoms is None and box_sizes is not None) or (
            atoms is not None and box_sizes is None
        ):
            raise GBMakerValueError(
                "'atoms' and 'box_sizes' must be specified together."
            )

        box_sizes = np.asarray(box_sizes, dtype=float)
        cell = np.diag(box_sizes[:, 1] - box_sizes[:, 0])
        origin = box_sizes[:, 0].copy()

        if triclinic:
            # LAMMPS restricted-triclinic box vectors are rows [lx,0,0], [xy,ly,0],
            # [xz,yz,lz]; cell[1, 0]/cell[2, 0]/cell[2, 1] carry the tilt factors.
            xy, xz, yz, theta = self.__get_triclinic_params()
            cell[1, 0] = xy
            cell[2, 0] = xz
            cell[2, 1] = yz
            ct, st = math.cos(theta), math.sin(theta)
            Rx = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
            # Copy before rotating in place: 'atoms' may be the caller's own array (or
            # self._result.atoms), which write_lammps must not mutate as a side effect.
            atoms = atoms.copy()
            positions = np.column_stack((atoms["x"], atoms["y"], atoms["z"]))
            rotated_positions = (Rx @ positions.T).T
            atoms["x"], atoms["y"], atoms["z"] = rotated_positions.T

        try:
            structure = StructureData(atoms, cell, origin)
        except StructureValueError as exc:
            raise GBMakerValueError(str(exc)) from exc

        try:
            LammpsDataWriter().write(
                file_name,
                structure,
                type_as_int=type_as_int,
                precision=precision,
                charges=charges,
                type_map=self._config.unit_cell.type_map,
                triclinic=triclinic,
            )
        except LammpsWriteError as exc:
            raise GBMakerValueError(str(exc)) from exc

    # Properties with getters and setters. Automatic updates for related parameters are
    # automatically taken care of.
    @property
    def a0(self) -> float:
        return self._config.a0

    @a0.setter
    def a0(self, value: Number) -> None:
        atom_types = tuple(self._config.unit_cell.names())
        self._config.a0 = self.__validate(value, float, "a0", nonnegative=True)
        self._config.unit_cell = self.__init_unit_cell(atom_types)
        self.update_spacing()

    @property
    def epsilon(self) -> float:
        return self._config.epsilon

    @epsilon.setter
    def epsilon(self, value: Number) -> None:
        self._config.epsilon = self.__validate(
            value, Number, "epsilon", strictly_positive=True)

    @property
    def gb_thickness(self) -> float:
        return self._config.gb_thickness

    @gb_thickness.setter
    def gb_thickness(self, value: Number):
        self._config.gb_thickness = self.__validate(
            value, Number, "gb_thickness", nonnegative=True)
        self._boundary.box_dims = self.__calculate_box_dimensions()

    @property
    def id(self) -> int:
        return self._config.gb_id

    @id.setter
    def id(self, value: int):
        self._config.gb_id = self.__validate(value, int, "id", nonnegative=True)

    @property
    def interaction_distance(self) -> float:
        return self._config.interaction_distance

    @interaction_distance.setter
    def interaction_distance(self, value: Number) -> None:
        self._config.interaction_distance = self.__validate(
            value, Number, "interaction_distance", nonnegative=True)
        self.__update_dims()

    @property
    def misorientation(self) -> np.ndarray:
        return np.hstack((self._boundary.misorientation, self._boundary.inclination))

    @misorientation.setter
    def misorientation(self, value: np.ndarray):
        misorientation = self.__validate(
            value, np.ndarray, "misorientation", expected_length=5
        )
        self.__assign_orientations(misorientation)
        # Discard any active embedding so update_spacing uses the new Euler
        # angles rather than the stale embedding-derived rotation matrices.
        self._boundary.embedding = None
        self.update_spacing()

    @property
    def repeat_factor(self) -> int:
        return self._config.repeat_factor

    @repeat_factor.setter
    def repeat_factor(self, value: int):
        self._config.repeat_factor = self.__validate(
            value, (int, Sequence), "repeat_factor", nonnegative=True)
        self.__update_dims()

    @property
    def structure(self) -> str:
        return self._config.structure

    @structure.setter
    def structure(self, value: str) -> None:
        self._config.structure = self.__validate(value, str, "structure")
        if {self._config.structure, value}.issubset(
            {"fluorite", "rocksalt", "zincblende"}
        ):
            raise GBMakerValueError(
                f"Cannot estimate conversion from {self._config.structure} to {value}"
            )
        else:
            atom_types = tuple(set(self._config.unit_cell.names()))

        self._config.unit_cell = self.__init_unit_cell(atom_types)

    @property
    def vacuum_thickness(self) -> float:
        return self._boundary.vacuum_thickness

    @vacuum_thickness.setter
    def vacuum_thickness(self, value: Number):
        old_vacuum = self._boundary.vacuum_thickness
        vacuum_value = self.__validate(
            value, Number, "vacuum_thickness", nonnegative=True
        )
        self._boundary.vacuum_thickness, self._boundary.normal_topology = (
            _normalize_vacuum_topology(
                vacuum_value,
                tolerance=self._config.epsilon,
            )
        )
        delta = self._boundary.vacuum_thickness - old_vacuum
        self._result.left_atoms["x"] += delta
        self._result.right_atoms["x"] += delta
        self._result.atoms["x"] += delta
        self._result.gb_region_atoms["x"] += delta
        self._boundary.box_dims = self.__calculate_box_dimensions()

    @property
    def x_dim_min(self) -> np.ndarray:
        return self._config.x_dim_min

    @x_dim_min.setter
    def x_dim_min(self, value: Number):
        self._config.x_dim_min = self.__validate(
            value, Number, "x_dim_min", nonnegative=True)
        self.update_spacing()
        self._boundary.box_dims = self.__calculate_box_dimensions()

    # Additional getters for other class properties
    @property
    def inplane_periodic(self) -> tuple[bool, bool]:
        """Read-only view of the in-plane periodicity flags (y, z)."""
        return tuple(bool(v) for v in self._boundary.inplane_periodic)

    @property
    def normal_topology(self) -> BoundaryNormalTopology:
        """Explicit physical topology along the boundary normal"""
        return self._boundary.normal_topology

    @property
    def uses_exact_construction(self) -> bool:
        """Read-only flag indicating exact integer P/Q construction is active."""
        return bool(
            self._boundary.embedding is not None
            and self._boundary.embedding.exact
            and self._boundary.embedding.P is not None
        )

    @property
    def box_dims(self) -> np.ndarray:
        return self._boundary.box_dims

    @property
    def whole_system(self) -> np.ndarray:
        return self._result.atoms

    @property
    def left_grain(self) -> np.ndarray:
        return self._result.left_atoms

    @property
    def radius(self) -> float:
        return self._config.radius

    @property
    def right_grain(self) -> np.ndarray:
        return self._result.right_atoms

    @property
    def gb_plane_x(self) -> float:
        return self._boundary.vacuum_thickness + self._boundary.left_x

    @property
    def spacing(self) -> dict:
        return self._boundary.spacing

    @property
    def unit_cell(self) -> UnitCell:
        return self._config.unit_cell

    @property
    def x_dim(self) -> float:
        return self._boundary.x_dim

    @property
    def y_dim(self) -> float:
        return self._boundary.y_dim

    @property
    def z_dim(self) -> float:
        return self._boundary.z_dim
