# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED
"""Grain boundary builder utilities."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
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
from GBOpt.gbmaker.approximate_grain import filter_grain_result_complete_origins
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
from GBOpt.gbmaker.geometry import (
    _box_periodic_basis,
    _complete_origin_atom_mask,
    _scaled_periodic_basis_vector,
)
from GBOpt.gbmaker.geometry import wrap_reduced_coordinate as _wrap_reduced_coordinate
from GBOpt.gbmaker.orientation import (
    _decompose_misorientation,
    resolve_orientation,
)
from GBOpt.gbmaker.types import (
    AxisAccommodation,
    GBMakerConstructionTypeError,
    GBMakerConstructionValueError,
    GrainBuildResult,
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

    def __material_state(self) -> MaterialState:
        """Return the current crystal identity as a ``MaterialState``.

        Thin wrapper constructing a ``GBOpt.gbmaker.types.MaterialState`` from current
        instance state, for ``GBOpt.gbmaker.assembly.assemble_bicrystal``'s ``material``
        argument.

        :return: Current material identity.
        """
        return self.__translate_construction_error(
            MaterialState,
            a0=self.__a0,
            structure=self.__structure,
            atom_types=tuple(self.__unit_cell.names()),
            unit_cell=self.__unit_cell,
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

        :return: ``None``. Updates ``self.__left_grain``, ``self.__right_grain``,
            ``self.__whole_system``, and ``self.__gb_region``.
        :raises GBMakerValueError: If exact grain generation requires missing P/Q data,
            if float-path gap equalization lacks right-grain build metadata, if an
            exact grain crosses the central or periodic x boundary, or if a downstream
            grain-generation stage fails.
        """
        result = self.__translate_construction_error(
            assemble_bicrystal,
            material=self.__material_state(),
            embedding=self.__embedding,
            R_left=self.__R_left,
            R_right=self.__R_right,
            left_periodic_miller_rows=self.__left_periodic_miller_rows,
            right_periodic_miller_rows=self.__right_periodic_miller_rows,
            left_x=self.__left_x,
            right_x=self.__right_x,
            x_dim=self.__x_dim,
            vacuum_thickness=self.__vacuum_thickness,
            inplane_periodic=self.__inplane_periodic,
            inplane_box_lengths=(self.__y_dim, self.__z_dim),
            epsilon=self.__epsilon,
            strain_accommodation=self.__strain_accommodation,
            gb_thickness=self.__gb_thickness,
            box_dims=self.__box_dims,
            normal_topology=self.__normal_topology,
            gb_id=self.__id,
        )

        self.__left_grain = result.left_atoms
        self.__right_grain = result.right_atoms
        self.__whole_system = result.atoms
        self.__gb_region = result.gb_region_atoms

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

    def __scaled_periodic_basis_vector(
        self, period_vector: np.ndarray, box_length: float, axis_index: int
    ) -> np.ndarray:
        """
        Scale a periodic basis vector so one axis projection matches the box length.

        Thin wrapper delegating to
        ``GBOpt.gbmaker.geometry._scaled_periodic_basis_vector``.

        :param period_vector: Cartesian periodic basis vector.
        :param box_length: Desired box length along the selected axis.
        :param axis_index: Axis whose projection should match ``box_length``.
        :return: Scaled periodic basis vector.
        """
        return self.__translate_construction_error(
            _scaled_periodic_basis_vector, period_vector, box_length, axis_index
        )

    def __box_periodic_basis(self, primitive_periods: np.ndarray) -> np.ndarray:
        """
        Build the in-plane box basis from primitive periodic vectors.

        Thin wrapper delegating to ``GBOpt.gbmaker.geometry._box_periodic_basis``.

        :param primitive_periods: 2x3 array containing primitive y/z period vectors.
        :return: 2x3 array containing the box basis vectors for y and z.
        """
        return self.__translate_construction_error(
            _box_periodic_basis,
            primitive_periods,
            self.__inplane_periodic,
            (self.__y_dim, self.__z_dim),
            self.__epsilon,
        )

    def __complete_origin_atom_mask(
        self,
        atom_mask: np.ndarray,
        origin_ids: np.ndarray,
        basis_size: int,
    ) -> np.ndarray:
        """Promote an atom-level mask to a complete-origin atom mask.

        Thin wrapper delegating to
        ``GBOpt.gbmaker.geometry._complete_origin_atom_mask``.

        An origin is retained only when exactly ``basis_size`` atoms are present for
        that origin and every atom from that origin passes ``atom_mask``. The returned
        mask is parallel to ``atom_mask`` and ``origin_ids``; retained atoms are marked
        ``True``.

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
        return self.__translate_construction_error(
            _complete_origin_atom_mask, atom_mask, origin_ids, basis_size
        )

    def __filter_float_result_complete_origins(
        self,
        result: GrainBuildResult,
        atom_mask: np.ndarray,
    ) -> GrainBuildResult:
        """Filter a float-path build result by complete origin groups.

        Thin wrapper delegating to
        ``GBOpt.gbmaker.approximate_grain.filter_grain_result_complete_origins``.

        :param result: Float-path grain build result to filter.
        :param atom_mask: Boolean atom-level mask parallel to ``result.atoms``.
        :return: Filtered float-path grain build result.
        :raises GBMakerValueError: If complete-origin filtering rejects the mask, origin
            IDs, or basis size.
        """
        return self.__translate_construction_error(
            filter_grain_result_complete_origins, result, atom_mask
        )

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
            # self.__whole_system), which write_lammps must not mutate as a side effect.
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
                type_map=self.__unit_cell.type_map,
                triclinic=triclinic,
            )
        except LammpsWriteError as exc:
            raise GBMakerValueError(str(exc)) from exc

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
