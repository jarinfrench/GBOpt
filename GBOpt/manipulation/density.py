# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Atom insertion and removal as built-in ``Manipulation`` operations.

``AtomInsertion``/``AtomRemoval`` change the number of atoms in a single parent's
grain-boundary region: insertion adds atoms at empty lattice sites found by Delaunay
triangulation or a coarse grid; removal deletes atoms, preferring low local-order
(unstable) sites when stoichiometric ratios must be preserved. Both share their
stochastic-selection and site-generation logic with ``GBManipulator.insert_atoms``/
``remove_atoms`` through the plain-array pure functions in this module, the same
"share the pure computation, not the value-typed boundary" split used for right-grain
translation (see ``GBOpt.manipulation.translation``).

Unlike ``translate_right_grain``, the reason for the split here is not an
``InterfaceCandidate`` leniency conflict -- removal only deletes rows (never moves one
outside its already-valid labeled bounds), and insertion's own site-to-label assignment
already enforces the identical bounds check ``InterfaceCandidate`` performs, so a
produced candidate is always constructible. The reason is that ``InterfaceCandidate``
carries no GB-thickness/GB-region concept, so ``AtomInsertion``/``AtomRemoval.execute()``
must recompute GB-region membership from ``gb_plane_x`` and an explicit ``gb_thickness``
parameter (see ``_gb_region_indices`` below), which is not always identical to
``Parent.gb_atoms``/``gb_indices`` (see that function's docstring) -- so the legacy
methods keep reading ``Parent.gb_atoms``/``gb_indices`` directly, byte-for-byte as
before, and only share the stochastic/geometric selection functions, not ``execute()``
itself.

An inserted atom's grain-ownership label is decided once, at insertion time, from
which physical grain x-interval it falls into (``left_grain_x_bounds`` /
``right_grain_x_bounds``, tie-broken toward the grain boundary plane); it is then
persistent, exactly like every other explicit ownership label in this codebase, and is
never recomputed from a later position. An atom whose insertion site lies outside both
physical grain intervals is rejected (``ManipulationCapabilityError``) rather than
silently mislabeled.

``InterfaceCandidate`` carries no unit-cell or GB-thickness information, so both
operations require ``unit_cell`` and ``gb_thickness`` as explicit ``context.params``
(the legacy methods supply their parent's own values for these automatically). The
GB-region membership used here is computed directly from the candidate's atoms,
``gb_plane_x``, and the supplied ``gb_thickness`` -- matching ``Parent.__finish_init``'s
``grain_ownership is not None`` branch (whole-system indices within ``gb_thickness / 2``
of the plane), since an ``InterfaceCandidate`` always represents persistent explicit
ownership. This does not change ``GBManipulator.insert_atoms``/``remove_atoms``'s own
long-standing ``parent.gb_atoms``/``parent.gb_indices`` behavior, which those methods
continue to read directly from ``Parent`` unchanged.
"""

from __future__ import annotations

import warnings

import numpy as np
from numba import float64, jit
from scipy.spatial import ConvexHull, Delaunay, KDTree

from GBOpt.GrainOwnership import LEFT_GRAIN_LABEL, RIGHT_GRAIN_LABEL
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


@jit(float64(float64, float64), nopython=True, cache=True)
def _gaussian(x: float, sigma: float = 0.02) -> float:
    """Calculate a Gaussian-smeared delta function at ``x``.

    :param x: Where to calculate the Gaussian-smeared delta function.
    :param sigma: Standard deviation of the Gaussian-smeared delta function, optional,
        defaults to 0.02.
    :return: Value of the Gaussian-smeared delta function at x.
    """
    prefactor = 1 / (sigma * np.sqrt(2 * np.pi))
    return prefactor * np.exp(-x * x / (2 * sigma * sigma))


@jit(nopython=True, cache=True)
def _calculate_fingerprint_vector(atom, neighs, NB, V, Btype, Delta, Rmax):
    """Calculate the fingerprint for ``atom`` per Lyakhov *et al.*, Computer Phys.
    Comm. 181 (2010) 1623-1632 (Eq. 4).

    :param np.ndarray atom: The atom we are calculating the fingerprint for.
    :param np.ndarray neighs: list of Atom containing the neighbors to **atom**.
    :param int NB: The number of atoms of type B neighbor to **atom**.
    :param float V: The volume of the unit cell in angstroms**3.
    :param int Btype: The type of neighbors we are interested in.
    :param float Delta: The discretization length for Rs in angstroms.
    :param float Rmax: The maximum distance from the *atom* to another atom to
        calculate the fingerprint.
    :return: The vector containing the fingerprint for *atom*.
    """
    Rs = np.arange(0, Rmax + Delta, Delta)

    fingerprint_vector = np.zeros_like(Rs)
    for idx, R in enumerate(Rs):
        local_sum = 0
        for neigh in neighs:
            if neigh[0] == Btype:
                diff = atom[1:] - neigh[1:]
                distance = np.sqrt(np.dot(diff, diff))
                delta = _gaussian(R - distance, 0.02)
                local_sum += delta / \
                    (4 * np.pi * distance * distance * (NB / V) * Delta)
        fingerprint_vector[idx] = local_sum - 1

    return fingerprint_vector


@jit(nopython=True, cache=True, parallel=True)
def _calculate_local_order(atom, neighs, unit_cell_types, unit_cell_a0, N, Delta, Rmax):
    """Calculate the local order parameter per Lyakhov *et al.*, Computer Phys.
    Comm. 181 (2010) 1623-1632 (Eq. 5).

    :param np.ndarray atom: Atom we are calculating the local order for.
    :param np.ndarray neighs: Neighbors of *atom*.
    :param np.ndarray unit_cell_types: The types of the atoms in the unit cell.
    :param float unit_cell_a0: The lattice parameter.
    :param int N: The number of atoms in the unit cell.
    :param float Delta: Bin size to calculate the fingerprint vector.
    :param float Rmax: Maximum distance from *atom* to consider as a neighbor to
        *atom* in angstroms.
    :return: The local order parameter for *atom* based on its neighbors.
    """
    local_sum = 0
    atom_types = np.unique(neighs[:, 0])
    V = unit_cell_a0 ** 3
    prefactor = Delta / (N * (V / N) ** (1 / 3))
    for Btype in atom_types:
        NB = np.sum(unit_cell_types == Btype)
        fingerprint = _calculate_fingerprint_vector(
            atom, neighs, NB, V, Btype, Delta, Rmax)
        local_sum += NB * prefactor * np.dot(fingerprint, fingerprint)
    return np.sqrt(local_sum)


def _create_neighbor_list(rcut: float, pos: np.ndarray) -> list:
    """Create a neighbor list using a KDTree.

    :param rcut: Cutoff distance for considering an atom a neighbor to another.
    :param pos: The array of atom positions.
    :return: The neighbor list for the atoms in **pos**
    """
    kdtree = KDTree(pos)
    neighbor_list = kdtree.query_ball_tree(kdtree, r=rcut)
    for i, neighbor in enumerate(neighbor_list):
        neighbor.remove(i)
    return neighbor_list


def _get_stoichiometric_change(n_units: int, ratio: dict[int, int]) -> dict[int, int]:
    """Return the number of atoms of each type affected by changing ``n_units``
    formula units.

    :param n_units: The number of atom units (defined as the sum of the values in the
        ratio dict) that will be modified.
    :param ratio: The ratio of each atom type in the unit cell. Must be a dict where
        the keys and values are positive integers.
    :return: The number of atoms of each type that will be changed.
    """
    return {atom_type: num * n_units for atom_type, num in ratio.items()}


def _random_type_counts(total: int, num_types: int, rng) -> np.ndarray:
    """Return a random nonnegative partition of ``total`` into ``num_types`` parts.

    Shared by ``select_removal_indices``/``select_insertion_sites``'s ``keep_ratio=False``
    branch. Draws its random breakpoints from ``rng`` (never the global NumPy RNG), the
    fix for a pre-existing bug: the legacy ``remove_atoms``/``insert_atoms`` code this
    replaces called ``np.random.choice`` directly for this split, ignoring the
    manipulator's own seeded generator entirely.

    :param total: Keyword argument, required. Total count to split.
    :param num_types: Keyword argument, required. Number of parts.
    :param rng: Keyword argument, required. Random-number generator (or duck-typed
        equivalent exposing ``choice``) to draw from.
    :return: Array of ``num_types`` nonnegative integers summing to ``total``.
    """
    breaks = np.sort(
        rng.choice(range(1, total), num_types - 1, replace=False)
    )
    breaks = np.concatenate(([0], breaks, [total]))
    return np.diff(breaks)


def _gb_region_indices(
    atoms_x: np.ndarray, gb_plane_x: float, gb_thickness: float
) -> np.ndarray:
    """Return indices of ``atoms_x`` within ``gb_thickness / 2`` of ``gb_plane_x``.

    Matches ``Parent.__finish_init``'s ``grain_ownership is not None`` branch: an
    ``InterfaceCandidate`` always represents persistent explicit ownership, so this is
    the one of ``Parent``'s two historic GB-region formulas that applies here.

    :param atoms_x: x coordinates of every candidate atom.
    :param gb_plane_x: Interface-gap midpoint.
    :param gb_thickness: Full width of the GB region in angstroms.
    :return: Indices of atoms within the GB region.
    """
    half = gb_thickness / 2.0
    return np.where(
        (atoms_x >= gb_plane_x - half) & (atoms_x <= gb_plane_x + half)
    )[0]


def delaunay_insertion_sites(
    gb_atoms: np.ndarray, atom_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate interstitial insertion sites via Delaunay triangulation.

    Potential insertion sites are the circumcenters of the tetrahedra formed by
    ``gb_atoms`` (replicated in y/z to account for in-plane periodicity), filtered to
    those inside the original bounds, with a nondegenerate simplex volume, and not
    touching the convex hull.

    :param gb_atoms: x/y/z positions of atoms in the GB region.
    :param atom_radius: The radius of an atom.
    :return: ``(sites, probabilities)`` -- candidate site positions and their
        interstitial-radius-weighted selection probabilities, summing to 1.
    """
    min_bounds = np.min(gb_atoms, axis=0)
    max_bounds = np.max(gb_atoms, axis=0)
    _Lx, Ly, Lz = max_bounds - min_bounds
    tiles = [(dy, dz) for dy in [-1, 0, 1] for dz in [-1, 0, 1]]
    replicas = []
    original_indices = []
    for dy, dz in tiles:
        shift = np.zeros_like(gb_atoms)
        shift[:, 1] = dy * Ly
        shift[:, 2] = dz * Lz
        replicas.append(gb_atoms + shift)
        original_indices.extend(np.arange(len(gb_atoms)))
    tiled = np.vstack(replicas)
    original_indices = np.array(original_indices)

    tri = Delaunay(tiled)
    circumcenters = -np.einsum(
        "ijk,ik->ij",
        tri.transform[:, :3, :],
        tri.transform[:, 3, :]
    )
    circumcenters[:, 1] = np.mod(
        circumcenters[:, 1] - min_bounds[1], Ly) + min_bounds[1]
    circumcenters[:, 2] = np.mod(
        circumcenters[:, 2] - min_bounds[2], Lz) + min_bounds[2]

    original_indices_simplices = original_indices[tri.simplices]

    in_bounds = np.all((circumcenters >= min_bounds) & (
        circumcenters <= max_bounds), axis=1)
    mask = in_bounds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        simplices = gb_atoms[original_indices_simplices]
        A, B, C, D = simplices[:, 0], simplices[:, 1], simplices[:, 2], simplices[:, 3]
        volumes = np.abs(
            np.einsum("ij,ij->i", np.cross(B - A, C - A), D - A)) / 6.0
    volume_threshold = 1e-3
    volume_mask = (volumes > volume_threshold * np.median(volumes)) & ~np.isnan(
        circumcenters).any(axis=1)
    mask &= volume_mask

    hull_vertices = set(ConvexHull(gb_atoms).vertices)
    simplex_mask = ~np.any(np.isin(tri.simplices, list(hull_vertices)), axis=1)
    mask &= simplex_mask

    valid_circumcenters = circumcenters[mask]
    valid_simplices = original_indices_simplices[mask, 0]
    sphere_radii = np.linalg.norm(
        gb_atoms[valid_simplices] - valid_circumcenters, axis=1)
    interstitial_radii = sphere_radii - atom_radius
    interstitial_radii -= np.min(interstitial_radii)  # make everything >= 0
    probabilities = interstitial_radii / np.sum(interstitial_radii)
    probabilities = probabilities / np.sum(probabilities)  # normalize
    assert abs(1 - np.sum(probabilities)
               ) < 1e-8, "Probabilities are not normalized!"

    return valid_circumcenters, probabilities


def grid_insertion_sites(
    gb_atoms: np.ndarray, atom_radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return candidate interstitial insertion sites on a 1x1x1 Angstrom grid.

    Sites must be at least ``atom_radius`` away from every atom in ``gb_atoms``.

    :param gb_atoms: x/y/z positions of atoms in the GB region.
    :param atom_radius: The radius of an atom.
    :return: ``(sites, probabilities)`` -- candidate site positions and their
        nearest-neighbor-distance-weighted selection probabilities, summing to 1.
    """
    max_x, max_y, max_z = gb_atoms.max(axis=0)
    min_x, min_y, min_z = gb_atoms.min(axis=0)
    X, Y, Z = np.meshgrid(
        np.arange(np.floor(min_x), np.ceil(max_x) + 1),
        np.arange(np.floor(min_y), np.ceil(max_y) + 1),
        np.arange(np.floor(min_z), np.ceil(max_z) + 1),
        indexing="ij"
    )
    sites = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    GB_tree = KDTree(gb_atoms)
    sites_tree = KDTree(sites)
    indices_to_remove = GB_tree.query_ball_tree(sites_tree, atom_radius)
    indices_to_remove = list(set(
        [i for sublist in indices_to_remove for i in sublist]))
    filtered_sites = np.delete(sites, indices_to_remove, axis=0)

    distances, _ = GB_tree.query(filtered_sites, k=1)
    probabilities = distances / np.sum(distances)
    probabilities = probabilities / np.sum(probabilities)  # normalize
    assert abs(1 - np.sum(probabilities)
               ) < 1e-8, "Probabilities are not normalized!"

    return filtered_sites, probabilities


def select_removal_indices(
    *,
    atoms: np.ndarray,
    positions: np.ndarray,
    gb_atom_indices: np.ndarray,
    type_map: dict[int, str],
    ratio: dict[int, int],
    unit_cell,
    num_to_remove: int,
    keep_ratio: bool,
    rng,
) -> np.ndarray:
    """Select whole-system row indices to remove from the GB region.

    Shared stochastic/geometric core of ``GBManipulator.remove_atoms`` and
    ``AtomRemoval.execute``. With one atom type, or ``keep_ratio=False``, atoms are
    selected uniformly (multitype random-count split via ``rng``) without regard to
    local order. With ``keep_ratio=True`` and multiple types, removal favors atoms
    with the lowest local order parameter (Lyakhov *et al.*), preferring central-type
    atoms first and then their nearest same-formula-unit neighbors of every other type.

    :param atoms: Keyword argument, required. ``(type, x, y, z)`` float rows for every
        atom in the whole system.
    :param positions: Keyword argument, required. ``atoms[:, 1:]``.
    :param gb_atom_indices: Keyword argument, required. Whole-system row indices of
        atoms in the GB region.
    :param type_map: Keyword argument, required. Integer-type-to-name mapping.
    :param ratio: Keyword argument, required. Per-type formula-unit ratio.
    :param unit_cell: Keyword argument, required. Unit cell supplying neighbor
        distances, lattice parameter, and atom count for the local-order calculation.
    :param num_to_remove: Keyword argument, required. Total atoms to remove.
    :param keep_ratio: Keyword argument, required. Whether to preserve stoichiometry.
    :param rng: Keyword argument, required. Random-number generator (or duck-typed
        equivalent exposing ``choice``) used for every stochastic selection.
    :return: Whole-system row indices selected for removal.
    :raises ManipulationCapabilityError: If there are not enough atoms of some type
        available to remove while preserving stoichiometry, or the produced selection
        does not match ``num_to_remove``.
    """
    if len(type_map) == 1:
        num_to_remove_dict = {1: num_to_remove}
    elif keep_ratio:
        num_to_remove_dict = _get_stoichiometric_change(num_to_remove, ratio)
        num_to_remove = sum(list(num_to_remove_dict.values()))
        central_type = min(num_to_remove_dict, key=num_to_remove_dict.get)
        cutoff = (unit_cell.nn_distance(2) + unit_cell.nn_distance(1)) / 2
        neighbor_list = _create_neighbor_list(cutoff, positions)
        Delta = 0.05  # Bin size to calculate the fingerprint vector.
        Rmax = 15  # Max distance allowed to be a neighbor
        args_list = [
            (
                atoms[atom_idx],
                atoms[neighbor_list[atom_idx]],
                unit_cell.names(asint=True),
                unit_cell.a0,
                len(unit_cell.unit_cell),
                Delta,
                Rmax,
            )
            for idx, atom_idx in enumerate(gb_atom_indices)
        ]
        order = np.zeros(len(args_list))
        for i, args in enumerate(args_list):
            order[i] = _calculate_local_order(*args)

        probabilities = max(order) - order + min(order)
        probabilities = probabilities / np.sum(probabilities, dtype=float)
    else:
        counts = _random_type_counts(num_to_remove, len(type_map), rng)
        num_to_remove_dict = {i + 1: int(counts[i]) for i in range(len(type_map))}

    if keep_ratio and len(type_map) > 1:
        type_mask = atoms[gb_atom_indices][:, 0] == central_type
        central_indices = gb_atom_indices[type_mask]
        central_probabilities = probabilities[type_mask]
        central_probabilities = (
            central_probabilities / np.sum(central_probabilities)
        )

        if len(central_indices) == 0:
            raise ManipulationCapabilityError(
                f"No atoms found for type {central_type} in the grain boundary."
            )

        central_num_to_remove = num_to_remove_dict[central_type]
        selected_central_indices = rng.choice(
            central_indices,
            central_num_to_remove,
            replace=False,
            p=central_probabilities,
        )

        distances = {
            idx: np.full(len(neighbor_list[idx]), np.inf)
            for idx in selected_central_indices
        }
        for central_idx in selected_central_indices:
            neighbors = neighbor_list[central_idx]
            gb_neighbors = np.intersect1d(neighbors, gb_atom_indices)
            mask = np.isin(neighbors, gb_neighbors)
            distances[central_idx][mask] = np.linalg.norm(
                positions[gb_neighbors] - positions[central_idx], axis=1
            )

        indices_to_remove = list(distances.keys())
        for atom_type, atom_ratio in ratio.items():
            if atom_type == central_type:
                continue
            for idx, dists in distances.items():
                neighbor_indices = np.asarray(neighbor_list[idx])
                gb_neighbor_indices = np.intersect1d(
                    neighbor_indices, gb_atom_indices)
                mask = np.isin(neighbor_indices, gb_neighbor_indices)
                type_mask = atoms[gb_neighbor_indices][:, 0] == atom_type
                type_indices = neighbor_indices[mask][type_mask]
                duplicates = [
                    i for i, el in enumerate(type_indices)
                    if el in indices_to_remove
                ]

                type_indices = list(set(type_indices) - set(duplicates))
                if len(type_indices) < atom_ratio:
                    raise ManipulationCapabilityError(
                        f"Not enough neighbor atoms of type {atom_type} to remove."
                    )

                dists[dists < 1e-8] = 1e-8
                type_probabilities = 1 / dists[mask][type_mask]
                type_probabilities = type_probabilities / np.sum(type_probabilities)

                type_idx_to_remove = rng.choice(
                    type_indices, atom_ratio, replace=False, p=type_probabilities
                )

                indices_to_remove.extend(type_idx_to_remove)

    else:  # keep_ratio == False or len(type_map) == 1
        indices_to_remove = []
        for atom_type, num in num_to_remove_dict.items():
            type_indices = gb_atom_indices[
                atoms[gb_atom_indices][:, 0] == atom_type
            ]
            type_idx_to_remove = rng.choice(type_indices, num, replace=False)
            indices_to_remove.extend(type_idx_to_remove)

    if not len(indices_to_remove) == num_to_remove:
        raise ManipulationCapabilityError(
            "removal selection did not produce the requested atom count"
        )
    return np.asarray(indices_to_remove)


def select_insertion_sites(
    *,
    possible_sites: np.ndarray,
    probabilities: np.ndarray,
    type_map: dict[int, str],
    ratio: dict[int, int],
    unit_cell,
    num_to_insert: int,
    keep_ratio: bool,
    rng,
) -> dict[int, list[int]]:
    """Select which candidate site each newly inserted atom occupies, by type.

    Shared stochastic core of ``GBManipulator.insert_atoms`` and
    ``AtomInsertion.execute``. With one atom type, or ``keep_ratio=False``, sites are
    drawn independently per type (multitype random-count split via ``rng``). With
    ``keep_ratio=True`` and multiple types, central-type atoms are placed first
    (weighted by ``probabilities``) and every other type is then placed at one of that
    central atom's nearest not-yet-assigned candidate-site neighbors, per formula-unit
    ratio.

    :param possible_sites: Keyword argument, required. Candidate site positions.
    :param probabilities: Keyword argument, required. Selection probability for each
        candidate site, aligned with ``possible_sites``.
    :param type_map: Keyword argument, required. Integer-type-to-name mapping.
    :param ratio: Keyword argument, required. Per-type formula-unit ratio.
    :param unit_cell: Keyword argument, required. Unit cell supplying neighbor
        distances used to relate central- and other-type site placement.
    :param num_to_insert: Keyword argument, required. Total atoms to insert.
    :param keep_ratio: Keyword argument, required. Whether to preserve stoichiometry.
    :param rng: Keyword argument, required. Random-number generator (or duck-typed
        equivalent exposing ``choice``) used for every stochastic selection.
    :return: Mapping from integer atom type to the list of ``possible_sites`` indices
        assigned that type.
    :raises ManipulationCapabilityError: If there are not enough unassigned candidate
        sites available near a central atom to preserve stoichiometry.
    """
    if len(type_map) == 1:
        num_to_insert_dict = {1: num_to_insert}
    elif keep_ratio:
        num_to_insert_dict = _get_stoichiometric_change(num_to_insert, ratio)
        num_to_insert = sum(list(num_to_insert_dict.values()))
        central_type = min(num_to_insert_dict, key=num_to_insert_dict.get)
    else:
        counts = _random_type_counts(num_to_insert, len(type_map), rng)
        num_to_insert_dict = {i + 1: int(counts[i]) for i in range(len(type_map))}

    if keep_ratio and len(type_map) > 1:
        central_num_to_insert = num_to_insert_dict[central_type]
        selected_central_indices = rng.choice(
            list(range(len(possible_sites))),
            central_num_to_insert,
            replace=False,
            p=probabilities
        )
        cutoff = (unit_cell.nn_distance(2) + unit_cell.nn_distance(1)) / 2.0
        possible_sites_neighbor_list = _create_neighbor_list(cutoff, possible_sites)

        atoms_to_add = {
            type_map[i]: [] if type_map[i] != central_type else list(
                selected_central_indices
            )
            for i in type_map.keys()
        }
        for atom_type, atom_ratio in ratio.items():
            if atom_type == central_type:
                continue
            for idx in selected_central_indices:
                neighbors = possible_sites_neighbor_list[idx]
                already_assigned = {idx for v in atoms_to_add.values() for idx in v}
                available_neighbors = list(set(neighbors) - already_assigned)
                if len(available_neighbors) < atom_ratio:
                    raise ManipulationCapabilityError(
                        "Not enough sites to insert atoms into."
                    )
                partial_probabilities = probabilities[available_neighbors]
                partial_probabilities = partial_probabilities / \
                    np.sum(partial_probabilities)
                selected_neighbor_offsets = rng.choice(
                    list(range(len(available_neighbors))), atom_ratio, replace=False,
                    p=partial_probabilities
                )
                selected_indices = [
                    available_neighbors[offset]
                    for offset in selected_neighbor_offsets
                ]
                atoms_to_add[atom_type].extend(selected_indices)
    else:
        atoms_to_add = {}
        site_indices = list(range(len(possible_sites)))
        already_assigned: set[int] = set()
        for atom_type, num in num_to_insert_dict.items():
            available_indices = list(set(site_indices) - already_assigned)
            available_probabilities = probabilities[available_indices]
            available_probabilities = (
                available_probabilities / np.sum(available_probabilities)
            )
            type_idx_to_insert = rng.choice(
                available_indices, num, replace=False, p=available_probabilities
            )
            atoms_to_add[atom_type] = list(type_idx_to_insert)
            already_assigned.update(type_idx_to_insert)

    return atoms_to_add


def _build_candidate(
    parent: InterfaceCandidate, atoms: np.ndarray, grain_labels: np.ndarray
) -> InterfaceCandidate:
    """Construct a candidate reusing ``parent``'s geometry with new ``atoms``.

    :param parent: Parent candidate supplying every geometry field but atom positions.
    :param atoms: Full structured atom rows for the produced candidate.
    :param grain_labels: Labels aligned with ``atoms``.
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
            grain_labels=grain_labels,
            inplane_periodic=parent.inplane_periodic,
            normal_topology=parent.normal_topology,
            coordinate_tolerance=parent.coordinate_tolerance,
            interface_separation=parent.interface_separation,
        )
    except InterfaceCandidateTypeError as exc:
        raise TypeError(str(exc)) from exc
    except InterfaceCandidateValueError as exc:
        raise ManipulationConfigurationError(str(exc)) from exc


def _require(params, key):
    """Return ``params[key]``, raising ``ManipulationConfigurationError`` if absent."""
    if key not in params:
        raise ManipulationConfigurationError(f"{key} is a required parameter")
    return params[key]


class AtomRemoval:
    """Remove a fraction or count of atoms from a single parent's GB region."""

    @property
    def name(self) -> str:
        """Stable operation name used for registry lookup and lineage metadata."""
        return "atom_removal"

    @property
    def arity(self) -> int:
        """Number of parent candidates this operation requires."""
        return 1

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        """Remove atoms from ``context.parents[0]``'s GB region.

        :param context: Validated input whose ``params`` supply ``unit_cell`` and
            ``gb_thickness`` (required), one of ``gb_fraction``/``num_to_remove``
            (required), and ``keep_ratio`` (optional, defaults to ``True``).
        :return: A single child candidate with the selected atoms removed; the
            removed atoms' own rows are recorded in ``parameters["removed_atoms"]``,
            and metadata records the resolved count and indices rather than embedding
            the surviving structure a second time.
        :raises ManipulationConfigurationError: If a required parameter is missing or
            malformed.
        :raises ManipulationCapabilityError: If the requested fraction/count is out of
            range or the selection cannot preserve stoichiometry.
        """
        parent = context.parents[0]
        unit_cell = _require(context.params, "unit_cell")
        gb_thickness = float(_require(context.params, "gb_thickness"))
        gb_fraction = context.params.get("gb_fraction")
        num_to_remove = context.params.get("num_to_remove")
        keep_ratio = bool(context.params.get("keep_ratio", True))

        if not gb_fraction and not num_to_remove:
            raise ManipulationConfigurationError(
                "gb_fraction or num_to_remove must be specified."
            )

        atoms_struct = parent.atoms
        type_map = unit_cell.type_map
        name_to_type = {name: type_id for type_id, name in type_map.items()}
        atoms = np.column_stack(
            (
                np.array(
                    [name_to_type[str(name)] for name in atoms_struct["name"]],
                    dtype=float,
                ),
                atoms_struct["x"].astype(float),
                atoms_struct["y"].astype(float),
                atoms_struct["z"].astype(float),
            )
        )
        positions = atoms[:, 1:]
        gb_atom_indices = _gb_region_indices(
            atoms_struct["x"].astype(float), parent.gb_plane_x, gb_thickness
        )

        if gb_fraction is not None and (gb_fraction <= 0 or gb_fraction > 0.25):
            raise ManipulationCapabilityError(
                f"Invalid value for gb_fraction ({gb_fraction=}). Must be "
                "0 < gb_fraction <= 0.25"
            )
        if num_to_remove is not None and (
            num_to_remove < 1 or num_to_remove > int(0.25 * len(gb_atom_indices))
        ):
            raise ManipulationCapabilityError(
                "Invalid num_to_remove value. Must be >= 1, and must be less than or "
                "equal to 25% of the total number of atoms in the GB region."
            )
        if num_to_remove is None:
            num_to_remove = int(gb_fraction * len(gb_atom_indices))

        labels = parent.grain_labels
        if num_to_remove == 0:
            child = _build_candidate(parent, atoms_struct, labels)
            return ManipulationResult(
                children=(child,),
                parameters={"num_to_remove": 0, "removed_indices": ()},
                lineage={"operation": self.name, "parent_count": 1},
            )

        indices_to_remove = select_removal_indices(
            atoms=atoms,
            positions=positions,
            gb_atom_indices=gb_atom_indices,
            type_map=type_map,
            ratio=unit_cell.ratio,
            unit_cell=unit_cell,
            num_to_remove=num_to_remove,
            keep_ratio=keep_ratio,
            rng=context.rng,
        )
        removed_atom_details = tuple(
            {
                "name": str(atoms_struct["name"][index]),
                "x": float(atoms_struct["x"][index]),
                "y": float(atoms_struct["y"][index]),
                "z": float(atoms_struct["z"][index]),
                "grain_label": int(labels[index]),
            }
            for index in indices_to_remove
        )
        remaining = np.delete(atoms_struct, indices_to_remove, axis=0)
        remaining_labels = np.delete(labels, indices_to_remove, axis=0)
        child = _build_candidate(parent, remaining, remaining_labels)
        return ManipulationResult(
            children=(child,),
            parameters={
                "num_to_remove": len(indices_to_remove),
                "removed_indices": tuple(int(i) for i in indices_to_remove),
                "removed_atoms": removed_atom_details,
            },
            lineage={"operation": self.name, "parent_count": 1},
        )


class AtomInsertion:
    """Insert a fraction or count of atoms into a single parent's GB region."""

    @property
    def name(self) -> str:
        """Stable operation name used for registry lookup and lineage metadata."""
        return "atom_insertion"

    @property
    def arity(self) -> int:
        """Number of parent candidates this operation requires."""
        return 1

    def execute(self, context: ManipulationContext) -> ManipulationResult:
        """Insert atoms into ``context.parents[0]``'s GB region.

        :param context: Validated input whose ``params`` supply ``unit_cell`` and
            ``gb_thickness`` (required), one of ``fill_fraction``/``num_to_insert``
            (required), ``method`` (optional, one of ``"delaunay"``/``"grid"``,
            defaults to ``"delaunay"``), and ``keep_ratio`` (optional, defaults to
            ``True``).
        :return: A single child candidate with the selected atoms inserted; metadata
            records the count and the inserted atoms' own rows rather than embedding
            the surviving structure a second time.
        :raises ManipulationConfigurationError: If a required parameter is missing or
            malformed.
        :raises ManipulationCapabilityError: If the requested fraction/count is out of
            range, the method is unrecognized, an insertion site falls outside both
            physical grain intervals, or the selection cannot preserve stoichiometry.
        """
        parent = context.parents[0]
        unit_cell = _require(context.params, "unit_cell")
        gb_thickness = float(_require(context.params, "gb_thickness"))
        fill_fraction = context.params.get("fill_fraction")
        num_to_insert = context.params.get("num_to_insert")
        method = context.params.get("method", "delaunay")
        keep_ratio = bool(context.params.get("keep_ratio", True))

        if not fill_fraction and not num_to_insert:
            raise ManipulationConfigurationError(
                "fill_fraction or num_to_insert must be specified."
            )

        atoms_struct = parent.atoms
        type_map = unit_cell.type_map
        type_map_inverse = {v: k for k, v in type_map.items()}
        gb_atom_indices = _gb_region_indices(
            atoms_struct["x"].astype(float), parent.gb_plane_x, gb_thickness
        )
        gb_atoms_xyz = np.column_stack(
            (
                atoms_struct["x"][gb_atom_indices].astype(float),
                atoms_struct["y"][gb_atom_indices].astype(float),
                atoms_struct["z"][gb_atom_indices].astype(float),
            )
        )

        if fill_fraction is not None and (fill_fraction <= 0 or fill_fraction > 0.25):
            raise ManipulationCapabilityError(
                f"Invalid value for fill_fraction ({fill_fraction=}). Must be 0 < "
                "fill_fraction <= 0.25"
            )
        if num_to_insert is not None and (
            num_to_insert < 1 or num_to_insert > int(0.25 * len(gb_atom_indices))
        ):
            raise ManipulationCapabilityError(
                "Invalid num_to_insert value. Must be >= 1, and must be less than or "
                "equal to 25% of the total number of atoms in the GB region."
            )

        if num_to_insert is None:
            num_to_insert = int(fill_fraction * len(gb_atom_indices))

        labels = parent.grain_labels
        if num_to_insert == 0:
            child = _build_candidate(parent, atoms_struct, labels)
            return ManipulationResult(
                children=(child,),
                parameters={"num_to_insert": 0},
                lineage={"operation": self.name, "parent_count": 1},
            )

        if method == "delaunay":
            possible_sites, probabilities = delaunay_insertion_sites(
                gb_atoms_xyz, unit_cell.radius)
        elif method == "grid":
            possible_sites, probabilities = grid_insertion_sites(
                gb_atoms_xyz, unit_cell.radius)
        else:
            raise ManipulationCapabilityError(
                f"Unrecognized insert_atoms method: {method}")

        atoms_to_add = select_insertion_sites(
            possible_sites=possible_sites,
            probabilities=probabilities,
            type_map=type_map,
            ratio=unit_cell.ratio,
            unit_cell=unit_cell,
            num_to_insert=num_to_insert,
            keep_ratio=keep_ratio,
            rng=context.rng,
        )

        new_atoms = np.array(
            [
                (type_map_inverse[atom_type], *possible_sites[idx])
                for atom_type in atoms_to_add.keys()
                for idx in atoms_to_add[atom_type]
            ],
            dtype=atoms_struct.dtype,
        )

        left_bounds = parent.left_grain_x_bounds
        right_bounds = parent.right_grain_x_bounds
        tolerance = parent.coordinate_tolerance
        inserted_labels = np.empty(len(new_atoms), dtype=np.int8)
        for index, x_value in enumerate(new_atoms["x"]):
            x_coord = float(x_value)
            in_left = (
                x_coord >= left_bounds[0] - tolerance
                and x_coord < left_bounds[1]
            )
            in_right = (
                x_coord >= right_bounds[0] - tolerance
                and x_coord < right_bounds[1]
            )
            if in_left and not in_right:
                inserted_labels[index] = LEFT_GRAIN_LABEL
            elif in_right and not in_left:
                inserted_labels[index] = RIGHT_GRAIN_LABEL
            elif in_left and in_right:
                inserted_labels[index] = (
                    LEFT_GRAIN_LABEL
                    if x_coord < parent.gb_plane_x
                    else RIGHT_GRAIN_LABEL
                )
            else:
                raise ManipulationCapabilityError(
                    "inserted atom lies outside both explicit physical grain x "
                    "intervals"
                )

        candidate_atoms = np.hstack((atoms_struct, new_atoms))
        candidate_labels = np.hstack((labels, inserted_labels))
        child = _build_candidate(parent, candidate_atoms, candidate_labels)
        inserted_atom_details = tuple(
            {
                "name": str(new_atoms["name"][index]),
                "x": float(new_atoms["x"][index]),
                "y": float(new_atoms["y"][index]),
                "z": float(new_atoms["z"][index]),
                "grain_label": int(inserted_labels[index]),
            }
            for index in range(len(new_atoms))
        )
        return ManipulationResult(
            children=(child,),
            parameters={
                "num_to_insert": len(new_atoms),
                "inserted_atoms": inserted_atom_details,
            },
            lineage={"operation": self.name, "parent_count": 1},
        )


__all__ = ["AtomInsertion", "AtomRemoval"]
