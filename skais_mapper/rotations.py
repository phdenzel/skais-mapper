"""
skais_mapper.cast module

@author: phdenzel
"""

from typing import Optional, Self
import numpy as np
# import healpy as hp
# import pynbody as pb
# import matplotlib.pyplot as plt
import skais_mapper


class R(object):
    """
    Class of rotation operators
    """

    def __init__(self, omega: Optional[np.ndarray] = None):
        """
        Initialize rotator object.

        Args:
            omega: Rotation angle in
        """
        self.omega = omega

    def __call__(self, arr: np.ndarray, **kwargs):
        """
        Rotate the input according to this rotation operator.

        Args:
            arr: The array to be rotated
            kwargs: Dummy keyword arguments

        Returns:
            arr: The rotated array
        """
        if self.omega is None:
            self.omega = self._x(0)
        # if isinstance(arr, (pb.snapshot.SimSnap, pb.snapshot.SubSnap, pb.array.SimArray)):
        #     for k in arr.keys():
        #         karr = arr[k]
        #         if len(karr.shape) == 2 and karr.shape[1] == 3:
        #             arr[k] = np.dot(self.omega, karr.transpose()).transpose()
        arr = np.dot(self.omega, arr.transpose()).transpose()
        return arr

    def __mul__(self, other: Self):
        """
        Combine rotation operators through multiplication.

        Args:
            other: Another rotation object to be multiplied.

        Returns:
            obj: a new instance of R
        """
        return self.__class__(self.omega * other.omega)

    @classmethod
    def x(cls, theta: float, degrees: bool = True):
        """
        Rotation operator about the current x-axis by angle theta.

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        omega = cls._x(theta, degrees=degrees)
        return cls(omega)

    @classmethod
    def y(cls, theta: float, degrees: bool = True):
        """
        Rotation operator about the current y-axis by angle theta.

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        omega = cls._y(theta, degrees=degrees)
        return cls(omega)

    @classmethod
    def z(cls, theta: float, degrees: bool = True):
        """
        Rotation operator about the current z-axis by angle theta.

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        omega = cls._z(theta, degrees=degrees)
        return cls(omega)

    @staticmethod
    def _x(theta: float, degrees: bool = True):
        """
        Rotation matrix about the current x-axis by angle theta

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        if degrees:
            theta *= np.pi / 180.0
        return np.matrix(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

    @staticmethod
    def _y(theta: float, degrees: bool = True):
        """
        Rotation matrix about the current y-axis by angle theta

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        if degrees:
            theta *= np.pi / 180.0
        return np.matrix(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    @staticmethod
    def _z(theta: float, degrees: bool = True):
        """
        Rotation matrix about the current z-axis by angle theta

        Args:
            theta: Rotation angle in degrees or radians (see below)
            degrees: Rotation in degrees, if false in radians
        """
        if degrees:
            theta *= np.pi / 180.0
        return np.matrix(
            [
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
                [0, 0, 1],
            ]
        )


# def W_kernel(kstep: float=0.5, kernel: pb.sph.Kernel=pb.sph.Kernel(), ds: np.ndarray=None):
#     """
#     Calculate integrated kernel weights
#     Args:
#         kstep <float> - step size through the kernel
#         kernel <pb.sph.Kernel instance> - kernel type
#         ds <np.ndarray> - integration points above 0
#     """
#     if ds is None:
#         ds = np.arange(kstep, kernel.max_d+kstep/2, kstep)
#     weights = np.zeros_like(ds)
#     for i, d1 in enumerate(ds):
#         d0 = d1 - kstep
#         dvals = np.arange(d0, d1, 0.05)
#         ivals = list(map(kernel.get_value, dvals))
#         ivals *= dvals
#         integ = ivals.sum() * 0.05
#         weights[i] = 2 * integ / (d1**2 - d0**2)
#     weights[:-1] -= weights[1:]
#     return ds, weights


# def calc_paths(sim, particles: list=None, boxsize=None, units='kpc'):
#     """
#     Calculate box configurations
#     """
#     if 'boxsize' not in sim.properties:
#         skais_mapper.read.dummy_boxsize(sim)
#     with s.immediate_mode:
#         pos, x, y, z, r = (s[k].view(np.ndarray) for k in ('pos', 'x', 'y', 'z', 'r'))
#     kstep = 0.2
#     ds3D, weights3D = W_kernel(kstep, kernel=pb.sph.Kernel())
#     ds2D, weights2D = W_kernel(kstep, kernel=pb.sph.Kernel2D())
#     plt.plot(weights3D, c=skais_mapper.utils.default_colors[0])
#     plt.plot(weights2D, c=skais_mapper.utils.default_colors[1])
#     plt.show()


if __name__ == "__main__":
    from skais_mapper.cosmology import CosmoModel
    from skais_mapper.simobjects import GasolineGalaxy

    gasoline2_data = (
        "/data/procomp/gasoline2/Capelo_et_al_2018_Run01_0.4Gyr/isogal_hr_z3_gf0."
        "6_0.5ZSol_phot_nometal_dustcazcool_C1_UVz3_shield.05100"
    )
    gasoline2_pars = (
        "/data/procomp/gasoline2/Capelo_et_al_2018_Run01_0.4Gyr/isogal_hr_z3_gf0."
        "6_0.5ZSol_phot_nometal_dustcazcool_C1_UVz3_shield.param"
    )
    cosmo_model = CosmoModel()
    z = 3
    a = 1.0 / (1.0 + z)

    s = GasolineGalaxy.read(
        gasoline2_data,
        properties={'a': a, 'h': cosmo_model.h},
        paramname=gasoline2_pars,
    )
    skais_mapper.simobjects.dummy_boxsize(s)

    # inspect units
    # for key, unit in [("t", "Gyr"), ("a", None), ("boxsize", None)]:
    #     if key not in s.properties:
    #         continue
    #     if unit is not None:
    #         print(f"{key}:\t {s.properties[key].to_units(unit)}")
    #     else:
    #         print(f"{key}:\t {s.properties[key]}")
    # for key in ["mass", "smooth"]:
    #     if key not in s.all_keys():
    #         continue
    #     print(f"{key} units:\t {s[key].units}")

    # cosmology test
    dz = CosmoModel.d_z(z, cosmo_model)
    dang = CosmoModel.d_z2kpc(dz)
    a2k = CosmoModel.arcsec2kpc(z, dist_z=dang, cosmo_model=cosmo_model)
    dcomov = (1 + z) * dang
    print(f"Assumed cosmology: {cosmo_model}")
    print(f'Scale @ z=3 [kpc/"]:        \t{a2k}')
    print(f"Angular dist. @ z={z} [Mpc]: \t{dang/1000}")
    print(f"Comoving dist. @ z={z} [Mpc]:\t{dcomov/1000}")

    # rotations test
    # rot_op = R.z(45)*R.y(15)
    # rot_op(s)
    # pb.plot.sph.image(s.g, qty='rho', units="Msol kpc^-2", width="50 kpc",
    #                   vmin=np.nanmin(s.g['rho']), vmax=np.nanmax(s.g['rho']),
    #                   cmap="magma", qtytitle="$\Sigma_{gas}$")
    # plt.show()

    # theta, phi, rho = np.random.randint(180, size=3)
    # print(f"Rotation angles: \t theta={theta} \t phi={phi} \t rho={rho}")
    # rot_op = R.z(rho) * R.x(theta) * R.y(phi)
    # rot_op(s)
    # pb.plot.sph.image(
    #     s.g,
    #     qty="rho",
    #     units="Msol kpc^-2",
    #     width="50 kpc",
    #     # vmin=np.nanmin(s.g['rho']), vmax=np.nanmax(s.g['rho']),
    #     cmap="magma",
    #     qtytitle=r"$\Sigma_{gas}$",
    # )
    # plt.show()

    # # edge calculations
    # calc_paths(s)

    # wrap_offsets -> [0] if boxsize not defined
    # loop through each particle
    # - get particle properties: x, y, z, smooth, qty
    # - deal with z_camera
    # - check smoothing within specified smoothing limits: skip otherwise
    # -
