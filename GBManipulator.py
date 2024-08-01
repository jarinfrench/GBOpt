import math

from GBMaker import GBMaker


class GBManipulator:
    """
    Class to manipulate atoms in the grain boundary region
    :param GBMaker GB: The GBMaker instance containing the generated GB
    """

    def __init__(self, GB: GBMaker):
        self.grain_dims = [GB.grain_xdim, GB.grain_ydim, GB.grain_zdim]
        self.vacuum_thickness = GB.vacuum_thickness
        self.ID = GB.ID
        self.a = GB.a
        self.radius = GB.radius
        self.misorientation = GB.misorientation
        self.gb_thickness = GB.gb_thickness
        self.repeat_factor = GB.repeat_factor
        self.spacing = GB.spacing
        self.right_grain = GB.right_grain
        self.left_grain = GB.left_grain
        self.write_lammps = GB.write_lammps

    def translate_right_grain(self, dy: float, dz: float):
        """
        Displace the right grain in the plane of the GB by dy, dz
        :param float dy: Displacement in y direction (angstroms)
        :param float dz: Displacement in z direction (angstroms)
        """
        d = [0, dy, dz]

        # Displace all atoms in the right grain by [0, dy, dz]. We modulo by the
        # grain dimensions so atoms do not exceed the original boundary conditions
        for i in range(1, 3):
            self.right_grain[:, i] = (
                self.right_grain[:, i] + d[i]) % self.grain_dims[i]


if __name__ == "__main__":
    theta = math.radians(36.869898)
    GB = GBMaker(lattice_parameter=3.61,        gb_thickness=0.0,
                 misorientation=[theta, 0, 0],        repeat_factor=4)
    GBManip = GBManipulator(GB)
    i = 1
    for dy in range(10):
        for dz in range(10):
            # NOTE: this translates the grain from whatever the current state is!
            GBManip.translate_right_grain(0.1, 0.1)
            GBManip.write_lammps(f"CSL_GB_{i}.dat")
            i += 1
