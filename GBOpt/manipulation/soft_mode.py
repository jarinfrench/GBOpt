# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Single-mode soft-phonon displacement as a built-in ``Manipulation`` operation.

``SoftModeDisplacement`` wraps the post-#27 (R17) single-mode soft-displacement
calculation -- select the softest ``mode_index``-th non-acoustic phonon mode across a
small q-point mesh and displace the GB region's atoms along it -- as a one-child
operation. The physical dynamical-matrix, bond-hardness, and q-point calculations
themselves are unchanged from R17 (moved here verbatim, apart from
``_calculate_bond_hardness``'s signature: it previously took a ``Parent`` instance
directly and read ``.whole_system``/``.gb_indices``/``.box_dims``/``.gb_thickness`` off
it, an inconsistency with every other kernel here already taking plain arrays/values;
that coupling is removed, with no change to what it computes).

Like ``GBOpt.manipulation.density``, ``InterfaceCandidate`` carries no unit-cell or
GB-thickness information, so this operation requires ``unit_cell`` and ``gb_thickness``
as explicit ``context.params``, and recomputes GB-region membership from the
candidate's atoms, ``gb_plane_x``, and that supplied ``gb_thickness`` -- the same
``grain_ownership is not None``-equivalent formula used in
``GBOpt.manipulation.density._gb_region_indices``, duplicated here (not imported
across sibling modules, matching this package's established small-helper-duplication
convention) since it is a three-line utility. ``GBManipulator.displace_along_soft_modes``
continues to read ``Parent.gb_indices`` directly, unchanged.

This operation has no stochastic component (mode selection is a deterministic sort of
computed eigenvalues), so it does not use ``context.rng``.
"""

from __future__ import annotations

import warnings
from itertools import combinations_with_replacement

import numpy as np
import scipy.sparse as sps
import spglib as spg
from numba import jit, prange
from numba.typed import List

from GBOpt.Atom import Atom
from GBOpt.interface import InterfaceCandidate
from GBOpt.interface.types import (
    InterfaceCandidateTypeError,
    InterfaceCandidateValueError,
)
from GBOpt.manipulation.types import (
    ManipulationCapabilityError,
    ManipulationConfigurationError,
    ManipulationContext,
    ManipulationResult,
)


def _create_neighbor_list(rcut: float, pos: np.ndarray) -> list:
    """Create a neighbor list using a KDTree.

    Duplicated from ``GBOpt.manipulation.density`` (see that module's docstring for
    why -- a small helper shared across sibling modules here is duplicated rather than
    cross-imported, the codebase's established tradeoff).

    :param rcut: Cutoff distance for considering an atom a neighbor to another.
    :param pos: The array of atom positions.
    :return: The neighbor list for the atoms in **pos**
    """
    from scipy.spatial import KDTree

    kdtree = KDTree(pos)
    neighbor_list = kdtree.query_ball_tree(kdtree, r=rcut)
    for i, neighbor in enumerate(neighbor_list):
        neighbor.remove(i)
    return neighbor_list


def _gb_region_indices(
    atoms_x: np.ndarray, gb_plane_x: float, gb_thickness: float
) -> np.ndarray:
    """Return indices of ``atoms_x`` within ``gb_thickness / 2`` of ``gb_plane_x``.

    Duplicated from ``GBOpt.manipulation.density``; see that module's copy for the
    formula's rationale.

    :param atoms_x: x coordinates of every candidate atom.
    :param gb_plane_x: Interface-gap midpoint.
    :param gb_thickness: Full width of the GB region in angstroms.
    :return: Indices of atoms within the GB region.
    """
    half = gb_thickness / 2.0
    return np.where(
        (atoms_x >= gb_plane_x - half) & (atoms_x <= gb_plane_x + half)
    )[0]


def _soft_mode_q_points(unit_cell, mesh_size: int) -> np.ndarray:
    """Return irreducible q points in Cartesian reciprocal-space coordinates.

    A self-consistent spglib cell is constructed from the conventional lattice and
    basis. spglib is then used to reduce that cell to a primitive cell and identify the
    irreducible reciprocal-mesh representatives.

    The returned q vectors are expressed in Cartesian reciprocal coordinates with units
    of inverse Angstroms so they can be combined directly with Cartesian interatomic
    displacement vectors.

    :param unit_cell: Nominal bulk unit cell used to determine crystal symmetry.
    :param mesh_size: Uniform reciprocal-space mesh size along each primitive reciprocal
        axis.
    :return: Irreducible q vectors sorted by increasing physical magnitude.
    :raises ManipulationCapabilityError: If spglib cannot identify a primitive cell.
    """
    conventional_lattice = np.asarray(
        unit_cell.conventional,
        dtype=np.float64,
    )
    cartesian_positions = np.asarray(
        unit_cell.positions(),
        dtype=np.float64,
    )

    scaled_positions = np.linalg.solve(
        conventional_lattice.T,
        cartesian_positions.T,
    ).T
    scaled_positions = np.mod(scaled_positions, 1.0)

    close_to_zero = np.isclose(
        scaled_positions,
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
    close_to_one = np.isclose(
        scaled_positions,
        1.0,
        rtol=0.0,
        atol=1e-12,
    )
    scaled_positions[close_to_zero | close_to_one] = 0.0

    conventional_cell = (
        conventional_lattice,
        scaled_positions,
        unit_cell.types(),
    )

    primitive_cell = spg.find_primitive(conventional_cell)
    if primitive_cell is None:
        raise ManipulationCapabilityError(
            "Could not identify a primitive cell for soft-mode q-point generation."
        )

    mesh = np.full(3, mesh_size, dtype=np.intc)

    mapping, grid = spg.get_ir_reciprocal_mesh(
        mesh,
        primitive_cell,
    )

    ir_indices = np.unique(mapping)
    q_fractional = (
        np.asarray(grid[ir_indices], dtype=np.float64)
        / mesh.astype(np.float64)
    )

    primitive_lattice = np.asarray(
        primitive_cell[0],
        dtype=np.float64,
    )

    reciprocal_lattice = (
        2.0
        * np.pi
        * np.linalg.inv(primitive_lattice).T
    )
    q_cartesian = q_fractional @ reciprocal_lattice

    magnitudes = np.linalg.norm(q_cartesian, axis=1)
    order = np.argsort(magnitudes, kind="stable")
    return q_cartesian[order]


def _calculate_bond_hardness(
    *,
    atoms: np.ndarray,
    gb_indices: np.ndarray,
    box_dims: np.ndarray,
    gb_thickness: float,
    ideal_bonds: dict,
    neighbor_list: list,
) -> np.ndarray:
    """Calculate pairwise bond-hardness values for the GB region.

    :param atoms: Keyword argument, required. Structured whole-system atom rows.
    :param gb_indices: Keyword argument, required. Whole-system row indices of atoms
        in the GB region.
    :param box_dims: Keyword argument, required. 3 by 2 Cartesian box bounds.
    :param gb_thickness: Keyword argument, required. Full width of the GB region in
        angstroms.
    :param ideal_bonds: Keyword argument, required. Ideal bond lengths by type pair.
    :param neighbor_list: Keyword argument, required. Per-atom neighbor index lists.
    :return: Symmetric bond-hardness matrix, indexed by whole-system atom index.
    """
    types = Atom.as_array(atoms)[:, 0]

    atom_info = {}
    for idx, atom in enumerate(atoms):
        a = Atom(*atom)
        if a.name not in atom_info:
            atom_info[a.name] = {
                "num": types[idx],
                "r_cov": a["r_cov"],
                "valence": a["valence"],
                "valence_electrons": a["valence_electrons"]
            }

    atom_type_to_name = {info["num"]: name for name, info in atom_info.items()}
    atom_name_to_type = {name: num for num, name in atom_type_to_name.items()}
    atom_types = list(atom_info.keys())

    n_of_bond_type = {
        (atom1, atom2): 0
        for atom1 in atom_types for atom2 in atom_types
    }

    for idx in gb_indices:
        for jdx in neighbor_list[idx]:
            if jdx < idx:
                continue
            n_of_bond_type[(atoms[idx]["name"], atoms[jdx]["name"])] += 1

    Delta_k = {}
    sorted_atom_type_to_name = sorted(atom_type_to_name)
    for type1, type2 in combinations_with_replacement(sorted_atom_type_to_name, 2):
        name1 = atom_type_to_name[type1]
        name2 = atom_type_to_name[type2]
        dk_tuple = (type1, type2)
        Delta_k[dk_tuple] = 0.5 * (ideal_bonds[(type1, type2)] -
                                   atom_info[name1]["r_cov"] - atom_info[name2]["r_cov"])
    bond_valence = np.sum(np.exp(-np.asarray(list(Delta_k.values())) / 0.37))

    y_dim = box_dims[1, 1] - box_dims[1, 0]
    z_dim = box_dims[2, 1] - box_dims[2, 0]
    V = gb_thickness * y_dim * z_dim
    N = np.sum(list(n_of_bond_type.values()))
    Hij = np.zeros((len(atoms), len(atoms)))
    for i1 in gb_indices:
        atom1 = Atom(*atoms[i1])
        type1 = atom_name_to_type[atom1["name"]]
        i1_CN = atom1["valence"] / bond_valence
        for i2 in neighbor_list[i1]:
            atom2 = Atom(*atoms[i2])
            type2 = atom_name_to_type[atom2["name"]]
            dk_tuple = (type1, type2) if type1 <= type2 else (type2, type1)
            i1_electronegativity = 0.481 * \
                atom1["valence_electrons"] / \
                (atom1["r_cov"] + Delta_k[dk_tuple])
            i2_electronegativity = 0.481 * \
                atom2["valence_electrons"] / \
                (atom2["r_cov"] + Delta_k[dk_tuple])
            i2_CN = atom2["valence"] / bond_valence
            Xij = np.sqrt(i1_electronegativity / i1_CN * i2_electronegativity / i2_CN)
            fi = abs(i1_electronegativity - i2_electronegativity) / \
                (4*np.sqrt(i1_electronegativity * i2_electronegativity))
            Hij[i1, i2] = Xij / (V / N) * np.exp(-2.7 * fi)
            Hij[i2, i1] = Hij[i1, i2]

    return Hij


@jit(nopython=True, cache=True)
def _calculate_dynamical_matrix(
    hardness,
    positions,
    gb_atom_indices,
    neighbor_list,
    q_vec,
):
    num_gb_atoms = len(gb_atom_indices)
    Dij = np.zeros((3 * num_gb_atoms, 3 * num_gb_atoms), dtype=np.complex128)

    for d_i in prange(num_gb_atoms):
        id1 = gb_atom_indices[d_i]

        for id2 in neighbor_list[id1]:
            bond_hardness = hardness[id1, id2]

            for aa in range(3):
                Dij[
                    3 * d_i + aa,
                    3 * d_i + aa,
                ] += bond_hardness

            if id2 not in gb_atom_indices:
                continue

            d_j = np.where(gb_atom_indices == id2)[0][0]
            rij = positions[id2] - positions[id1]
            exp_term = np.exp(1j * np.dot(q_vec, rij))

            for aa in range(3):
                Dij[
                    3 * d_i + aa,
                    3 * d_j + aa,
                ] -= bond_hardness * exp_term

    return Dij


def soft_mode_displacement_atoms(
    *,
    structured_atoms: np.ndarray,
    unit_cell,
    gb_indices: np.ndarray,
    box_dims: np.ndarray,
    gb_thickness: float,
    mesh_size: int,
    num_q: int,
    mode_index: int,
    subtract_displacement: bool,
) -> np.ndarray:
    """Return ``structured_atoms`` displaced along its ``mode_index``-th soft mode.

    Pure computational core shared by ``GBManipulator.displace_along_soft_modes`` and
    ``SoftModeDisplacement.execute``. No unrelated change to the physical soft-mode
    calculation or q-point selection relative to R17 (#27) -- this is that method's own
    body, taking every input explicitly instead of a ``Parent``.

    :param structured_atoms: Keyword argument, required. Structured whole-system atom
        rows.
    :param unit_cell: Keyword argument, required. Unit cell supplying ideal bond
        lengths, atomic radius, and crystal symmetry.
    :param gb_indices: Keyword argument, required. Whole-system row indices of atoms
        in the GB region (the movable degrees of freedom).
    :param box_dims: Keyword argument, required. 3 by 2 Cartesian box bounds.
    :param gb_thickness: Keyword argument, required. Full width of the GB region in
        angstroms.
    :param mesh_size: Keyword argument, required. Reciprocal-space mesh size.
    :param num_q: Keyword argument, required. Number of unique q points to use.
    :param mode_index: Keyword argument, required. Selects which non-acoustic soft
        mode to displace along, ordered from softest (0) to next-softest (1), and so
        on.
    :param subtract_displacement: Keyword argument, required. Whether to subtract,
        rather than add, the eigenvector displacement.
    :return: Structured whole-system atom rows after displacement.
    :raises ManipulationCapabilityError: If ``mode_index`` is out of range for this
        system.
    """
    atoms = Atom.as_array(structured_atoms)
    positions = atoms[:, 1:]

    ideal_bonds = unit_cell.ideal_bond_lengths
    cutoff = 1.5 * max(ideal_bonds.values())
    neighbor_list = _create_neighbor_list(cutoff, positions)
    neighbor_list_typed = List()
    for neighbor in neighbor_list:
        neighbor_list_typed.append(List(neighbor))
    hardness = _calculate_bond_hardness(
        atoms=structured_atoms,
        gb_indices=gb_indices,
        box_dims=box_dims,
        gb_thickness=gb_thickness,
        ideal_bonds=ideal_bonds,
        neighbor_list=neighbor_list,
    )
    q_points = _soft_mode_q_points(unit_cell, mesh_size)

    if len(q_points) < num_q:
        warnings.warn(
            f"Fewer q_points generated than desired: {len(q_points)} < {num_q}. "
            "Recommended to increase mesh size."
        )

    n_atoms = len(gb_indices)

    sparse_threshold = 10000

    num_modes_needed = mode_index + 1
    if num_modes_needed > 3 * n_atoms:
        raise ManipulationCapabilityError(
            f"mode_index={mode_index} is out of range: at most "
            f"{3 * n_atoms} mode(s) can be computed for this system."
        )

    freqs = np.zeros((num_q, num_modes_needed))
    disps = np.zeros((num_q, num_modes_needed, 3 * n_atoms))

    for i, q_vec in enumerate(q_points[:num_q]):
        dynamical_matrix = _calculate_dynamical_matrix(
            hardness, positions, gb_indices, neighbor_list_typed, q_vec)
        if 3 * n_atoms <= sparse_threshold:
            freq_vals, disp_vals = np.linalg.eigh(dynamical_matrix)
        else:
            sparse_matrix = sps.csc_matrix(dynamical_matrix)
            if num_modes_needed >= 3 * n_atoms - 1 != num_modes_needed:
                raise ManipulationCapabilityError(
                    "Cannot generate the requested soft mode.")
            freq_vals, disp_vals = sps.linalg.eigsh(
                sparse_matrix, k=num_modes_needed, which="SA")
        freqs[i] = freq_vals[:num_modes_needed]
        disps[i, :, :] = np.real(disp_vals)[:, :num_modes_needed].T

    non_acoustic_indices = np.where(~np.isclose(freqs, 0))

    filtered_freqs = freqs[non_acoustic_indices]
    sorted_filtered_freq_indices = np.argsort(filtered_freqs)
    saved_disps = disps[non_acoustic_indices[0][sorted_filtered_freq_indices],
                        non_acoustic_indices[1][sorted_filtered_freq_indices], :]

    if mode_index >= len(saved_disps):
        raise ManipulationCapabilityError(
            f"mode_index={mode_index} is out of range: only "
            f"{len(saved_disps)} non-acoustic soft mode(s) available."
        )

    d_min = 2 * unit_cell.radius

    precomputed_distances = np.zeros(len(gb_indices))
    for i, atom_idx in enumerate(gb_indices):
        neighbors = neighbor_list[atom_idx]
        neighbor_positions = positions[neighbors]
        dists = np.linalg.norm(positions[atom_idx] - neighbor_positions, axis=1)
        precomputed_distances[i] = np.min(dists) - d_min

    pos = np.copy(positions)
    disp_vector = saved_disps[mode_index].reshape(-1, 3)
    disp_magnitude = np.linalg.norm(disp_vector, axis=1)

    if not np.any(disp_magnitude == 0):
        overlap_condition = precomputed_distances < disp_magnitude
        safe_displacements = np.ones_like(disp_magnitude)
        if np.any(overlap_condition):
            overlapped_atoms = precomputed_distances[overlap_condition]
            overlap_disps = disp_magnitude[overlap_condition]
            safe_displacements[overlap_condition] = overlapped_atoms / overlap_disps

        adjusted_displacements = disp_vector * safe_displacements[:, None]
        pos[gb_indices] = positions[gb_indices] + \
            adjusted_displacements * (-1 if subtract_displacement else 1)

    structured_pos = np.zeros((len(atoms)), dtype=Atom.atom_dtype)
    structured_pos["name"] = structured_atoms["name"]
    structured_pos["x"] = pos[:, 0]
    structured_pos["y"] = pos[:, 1]
    structured_pos["z"] = pos[:, 2]
    return structured_pos


def _require(params, key):
    """Return ``params[key]``, raising ``ManipulationConfigurationError`` if absent."""
    if key not in params:
        raise ManipulationConfigurationError(f"{key} is a required parameter")
    return params[key]


def _build_candidate(parent: InterfaceCandidate, atoms: np.ndarray) -> InterfaceCandidate:
    """Construct a candidate reusing ``parent``'s geometry with displaced ``atoms``.

    :param parent: Parent candidate supplying every geometry field but atom positions.
    :param atoms: Full structured atom rows for the produced candidate.
    :return: Complete immutable candidate.
    :raises ManipulationConfigurationError: If the resulting candidate is malformed.
    """
    try:
        return InterfaceCandidate(
            atoms=atoms,
            box_dims=parent.box_dims,
            gb_plane_x=parent.gb_plane_x,
            left_grain_x_bounds=parent.left_grain_x_bounds,
            right_grain_x_bounds=parent.right_grain_x_bounds,
            grain_labels=parent.grain_labels,
            inplane_periodic=parent.inplane_periodic,
            normal_topology=parent.normal_topology,
            coordinate_tolerance=parent.coordinate_tolerance,
            interface_separation=parent.interface_separation,
        )
    except InterfaceCandidateTypeError as exc:
        raise TypeError(str(exc)) from exc
    except InterfaceCandidateValueError as exc:
        raise ManipulationConfigurationError(str(exc)) from exc


class SoftModeDisplacement:
    """Displace a single parent's GB region along one selected soft phonon mode."""

    @property
    def name(self) -> str:
        """Stable operation name used for registry lookup and lineage metadata."""
        return "soft_mode_displacement"

    @property
    def arity(self) -> int:
        """Number of parent candidates this operation requires."""
        return 1

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        """Displace ``context.parents[0]``'s GB region along its selected soft mode.

        :param context: Validated input whose ``params`` supply ``unit_cell`` and
            ``gb_thickness`` (required), and ``mesh_size``, ``num_q``, ``mode_index``,
            ``subtract_displacement`` (all optional, matching
            ``GBManipulator.displace_along_soft_modes``'s own defaults).
        :return: A single displaced child candidate; ``parameters`` records the
            resolved ``mode_index`` (and the other resolved parameters) rather than
            embedding the displaced structure a second time.
        :raises ManipulationConfigurationError: If a required parameter is missing or
            malformed.
        :raises ManipulationCapabilityError: If ``mode_index`` is out of range for this
            system.
        """
        parent = context.parents[0]
        unit_cell = _require(context.params, "unit_cell")
        gb_thickness = float(_require(context.params, "gb_thickness"))
        mesh_size = int(context.params.get("mesh_size", 4))
        num_q = int(context.params.get("num_q", 1))
        mode_index = int(context.params.get("mode_index", 0))
        subtract_displacement = bool(context.params.get("subtract_displacement", False))

        if mesh_size < 1:
            raise ManipulationCapabilityError("mesh_size must be >= 1.")
        if num_q < 1:
            raise ManipulationCapabilityError("num_q must be >= 1.")
        if mode_index < 0:
            raise ManipulationCapabilityError("mode_index must be >= 0.")

        atoms_struct = parent.atoms
        gb_indices = _gb_region_indices(
            atoms_struct["x"].astype(float), parent.gb_plane_x, gb_thickness
        )

        displaced = soft_mode_displacement_atoms(
            structured_atoms=atoms_struct,
            unit_cell=unit_cell,
            gb_indices=gb_indices,
            box_dims=parent.box_dims,
            gb_thickness=gb_thickness,
            mesh_size=mesh_size,
            num_q=num_q,
            mode_index=mode_index,
            subtract_displacement=subtract_displacement,
        )
        child = _build_candidate(parent, displaced)
        return ManipulationResult(
            children=(child,),
            parameters={
                "mesh_size": mesh_size,
                "num_q": num_q,
                "mode_index": mode_index,
                "subtract_displacement": subtract_displacement,
            },
            lineage={"operation": self.name, "parent_count": 1},
        )


__all__ = ["SoftModeDisplacement"]
