import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from GBOpt.Atom import Atom

if TYPE_CHECKING:
    from GBOpt.GBMaker import GBMaker


def _validate_array(
    arr: np.ndarray,
    *,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[Union[np.dtype, type]] = None
) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if expected_shape and arr.shape != expected_shape:
        raise ValueError(f"Array shape must be {expected_shape}, but got {arr.shape}")

    if expected_dtype and arr.dtype != np.dtype(expected_dtype):
        raise ValueError(f"Array dtype must be {expected_dtype}, but got {arr.dtype}")

    return arr


def _flatten(arr: Union[List, np.ndarray]) -> np.ndarray:
    if isinstance(arr, list):
        arr = np.array(arr)

    return arr.flatten()


def write_lammps(
    file_name: str,
    gbmaker: Optional['GBMaker'] = None,
    atoms: Optional[np.ndarray] = None,
    box_sizes: Optional[np.ndarray] = None,
    *,
    type_as_int: bool = False
) -> None:
    """
    Writes the atom positions with the given box dimensions to a LAMMPS input file.

    :param str file_name: The filename to save the data
    :param gbmaker: A GBMaker instance to write the contents to a LAMMPS dump file.
    :param atoms: A numpy array containing the atom data.
    :param box_sizes: A 3x2 array containing the min and max dimensions for each of the
        x, y, and z dimensions.
    :param type_as_int: Whether to write the atom types as a chemical name or a number.
        Keyword argument, optional, defaults to False (write as a chemical name).
    """
    if not isinstance(file_name, str):
        raise TypeError("file_name must be of type str")

    if gbmaker is not None:
        atoms = gbmaker.whole_system
        box_sizes = gbmaker.box_dims
    elif (atoms is None and box_sizes is not None) or (
        atoms is not None and box_sizes is None
    ):
        raise ValueError("'atoms' and 'box_sizes' must be specified together.")

    name_to_int = {name: i+1 for i, name in enumerate(np.unique(atoms['name']))}

    # Write LAMMPS data file
    with open(file_name, 'w') as fdata:
        # First line is a comment line
        fdata.write('Crystalline Cu atoms\n\n')

        # --- Header ---#
        # Specify number of atoms and atom types
        fdata.write('{} atoms\n'.format(len(atoms)))
        fdata.write('{} atom types\n'.format(len(set(atoms['name']))))
        # Specify box dimensions
        fdata.write('{} {} xlo xhi\n'.format(
            box_sizes[0][0], box_sizes[0][1]))
        fdata.write('{} {} ylo yhi\n'.format(
            box_sizes[1][0], box_sizes[1][1]))
        fdata.write('{} {} zlo zhi\n'.format(
            box_sizes[2][0], box_sizes[2][1]))

        if not type_as_int:
            fdata.write('\nAtom Type Labels\n\n')
            for name, value in name_to_int.items():
                fdata.write(f"{value} {name}\n")

        # Atoms section
        fdata.write('\nAtoms\n\n')

        # Write each position.
        if type_as_int:
            for i, (name, *pos) in enumerate(atoms):
                fdata.write('{} {:n} {} {} {}\n'.format(
                    i+1, name_to_int[name], *pos))
        else:
            for i, (name, *pos) in enumerate(atoms):
                fdata.write('{} {} {} {} {}\n'.format(i+1, name, *pos))


def approximate_rotation_matrix_as_int(
    m: np.ndarray,
    precision: float = 5
) -> np.ndarray:
    """
    Approximate a rotation matrix in integer format given the original matrix and
    the desired precision.

    :param m: The matrix to approximate
    :param precision: Decimal precision to use during calculations, defaults to 5
    :return: Integer approximation of the rotation matrix m
    """
    # first round the matrix to the desired precision
    R0 = np.linalg.norm(Rotation.from_matrix(m).as_rotvec(degrees=True))
    min_by_row_excluding_0 = np.ma.amin(np.ma.masked_equal(np.abs(m), 0), axis=1).data
    ratio = m / min_by_row_excluding_0[:, np.newaxis]

    rounded = np.round(ratio, precision)
    scaled = (10**precision * rounded).astype(int)
    gcds = np.gcd.reduce(scaled, axis=1)
    approx_m = scaled / gcds[:, None]

    R_approx_normed = np.linalg.norm(
        Rotation.from_matrix(
            approx_m / np.linalg.norm(approx_m, axis=1)[:, None]
        ).as_rotvec(degrees=True)
    )

    if abs(R0-R_approx_normed) > 0.5:
        warnings.warn(
            "Approximated rotation matrix error is greater than 0.5 degrees.")
    return approx_m.astype(int)


def get_points_inside_box(atoms: np.ndarray, box_dim: Union[List, np.ndarray]) -> np.ndarray:
    """
    Selects the lattice points that are inside the given box dimensions.

    :param atoms: Atoms to check. 4xn array containing types and positions.
    :param box_dim: Dimensions of box (x_min, y_min, z_min, x_max, y_max, z_max).
    :return: 4xn array containing the Atom positions inside the given box.
    """
    atoms = _validate_array(atoms, expected_dtype=Atom.atom_dtype)
    box_dim = _validate_array(_flatten(box_dim), expected_shape=(6,))
    x_min, y_min, z_min, x_max, y_max, z_max = box_dim
    if x_min >= x_max:
        raise ValueError('x_min cannot be greater than or equal to x_max.')
    if y_min >= y_max:
        raise ValueError('y_min cannot be greater than or equal to y_max.')
    if z_min >= z_max:
        raise ValueError('z_min cannot be greater than or equal to z_max')

    # We use '<' for the y and z directions to not duplicate atoms across the
    # periodic boundary. For the x direction this doesn't matter as much because we
    # do not have periodic boundaries in this direction, but '<=' allows for more
    # atoms to be placed.
    inside_box = (
        (atoms['x'] >= x_min) & (atoms['x'] <= x_max) &
        (atoms['y'] >= y_min) & (atoms['y'] < y_max) &
        (atoms['z'] >= z_min) & (atoms['z'] < z_max)
    )
    return atoms[inside_box]
