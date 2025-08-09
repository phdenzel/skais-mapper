# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Utilities for handling cosmology-dependent calculations."""

from dataclasses import dataclass, field
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp
from astropy import units as au
from astropy import constants as ac
from typing import TypeVar

TCosmo = TypeVar("TCosmo", bound="CosmoModel")


@dataclass
class CosmoModel:
    """Set cosmological parameter for distance calculations, mass projections, etc.

    Args:
        omega_m (float): matter energy fraction
        omega_l (float): dark energy fraction
        omega_k (float): curvature fraction
        omega_r (float): radiation energy fraction (will be calculated
                          assuming flat universe if not given)
        h (float): 'little' H (H0 / 100 km/s/Mpc)
        c (ac.Constant): speed of light
        z (float): redshift
    """

    omega_m: float = 0.279952
    omega_l: float = 0.72
    omega_k: float = 0.0
    # omega_r: float = field(init=False, default=None, hash=False)
    _omega_r: float | None = field(init=False, default=None, hash=False)
    z: float = 0.0
    c: ac.Constant = field(default_factory=lambda: ac.c)
    h: float = 0.718
    u_H0: au.CompositeUnit = au.km / au.s / au.Mpc

    def __str__(self) -> str:
        """Instance string representation."""
        return (
            f"<CosmoModel[Ω_m:{self.omega_m:2.6f} Ω_l:{self.omega_l:2.6f} "
            f"Ω_k:{self.omega_k:2.6f} Ω_r:{self.omega_r:2.6f} "
            f"h:{self.h:2.4f} c:{self.c.to(au.km/au.s)} z:{self.z}]>"
        )

    def __repr__(self) -> str:
        """Instance string representation."""
        return self.__str__()

    @property
    def omega_r(self) -> float:
        """The radiation density parameter getter."""
        if self._omega_r is None:
            self._omega_r = (1 + self.omega_k) - self.omega_m - self.omega_l
        return self._omega_r

    @omega_r.setter
    def omega_r(self, r: float | None):
        """The radiation density parameter setter."""
        if r is None:
            r = (1 + self.omega_k) - self.omega_m - self.omega_l
        self._omega_r = r

    @property
    def a(self) -> float:
        """The scale parameter getter."""
        if self.z:
            return 1.0 / (1 + self.z)
        return 1.0  # default: z = 0

    @a.setter
    def a(self, a: float):
        """The scale parameter setter."""
        self.z = (1.0 / a) - 1

    @property
    def rho_crit(self) -> au.Quantity:
        """The critical density getter."""
        H0 = self.H0
        return (3 * H0**2 / (8 * np.pi * ac.G)).to(au.Msun * au.Mpc ** (-3))

    @property
    def H0(self) -> au.Quantity:
        """The Hubble constant getter."""
        return self.h * 1e2 * self.u_H0

    @H0.setter
    def H0(self, H0: au.Quantity):
        """The Hubble constant setter."""
        self.h = H0 / (1e2 * self.u_H0)

    def H(self, a: float | None = None) -> au.Quantity:
        """The Hubble parameter at a given scale factor `a`.

        Args:
            a: scale factor

        Returns:
            (float): value of the Hubble parameter
        """
        if a is None and self.z and self.a:
            a = self.a
        elif a is None:
            a = 1.0
        return (
            self.H0
            * (self.omega_m / a**3 + self.omega_r / a**4 + self.omega_k / a**2 + self.omega_l)
            ** 0.5
        )

    @staticmethod
    def d_comov(a: float, r: float, cosmo_model: TCosmo | None = None) -> au.Quantity:
        """Calculate the comoving distance from scale factor (for solve_ivp).

        Args:
            a: scale factor
            r: distance
            cosmo_model: cosmological parameter lookup class

        Returns:
            (float): comoving radial distance (scale-free)
        """
        if cosmo_model is None:
            cosmo_model = CosmoModel()
        return 1.0 / (a * a * cosmo_model.H(a))

    @staticmethod
    def d_z(
        z: float | None = None,
        cosmo_model: TCosmo | None = None,
        scaled: bool = False,
    ) -> au.Quantity:
        """Angular distance d_z from a redshift z within given cosmology.

        Args:
            z: redshift
            cosmo_model: cosmological parameter lookup class
            scaled: return result with c/H0 in units of kpc

        Return:
            (float): apparent distance d_z
        """
        if cosmo_model is None:
            cosmo_model = CosmoModel()
        comov = partial(CosmoModel.d_comov, cosmo_model=cosmo_model)
        if z is None and cosmo_model.a:
            a = cosmo_model.a
        else:
            a = 1.0 / (1.0 + z)
        a_lim = [a, 1]
        res = solve_ivp(comov, a_lim, [0])
        if res["success"]:
            r = res["y"][0]
            D = a * (r[-1] - r[0])
            if scaled:
                return CosmoModel.d_z2kpc(D)
            return D
        raise ValueError(f"No solution found for the inputs z={z}, " f"cosmo_model={cosmo_model}")

    @staticmethod
    def d_z2kpc(
        distance: float | np.ndarray, cosmo_model: TCosmo | None = None
    ) -> au.Quantity:
        """Given scale-less distance d, return scaled distance c/H0 * d [kpc].

        Args:
            distance: scale-less distance
            cosmo_model: cosmological parameter lookup class
        """
        if cosmo_model is None:
            cosmo_model = CosmoModel()
        c = cosmo_model.c
        H0 = cosmo_model.H0
        return (distance * c / H0).to(au.kpc)

    @staticmethod
    def arcsec2kpc(
        z: float,
        dist_z: float | None = None,
        cosmo_model: TCosmo | None = None,
    ) -> au.Quantity:
        """Angular distance d_z from a redshift z within given cosmology.

        Args:
            z: redshift
            dist_z: scaled comoving distance
            cosmo_model: cosmological parameter lookup class

        Return:
            (float): kpc/arcsec scaling within given redshift and cosmology
        """
        if dist_z is None:
            dist_z = CosmoModel.d_z(z, cosmo_model=cosmo_model, scaled=True)
        d_kpc_arcsec = (dist_z / au.rad).to(au.kpc / au.arcsec)
        return d_kpc_arcsec


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # distance calculation test z=3
    cosmo = CosmoModel(z=3)
    dz = CosmoModel.d_z(cosmo.z, cosmo)
    dang = CosmoModel.d_z2kpc(dz, cosmo)
    dcomov = (1 + cosmo.z) * dang
    a2k = CosmoModel.arcsec2kpc(cosmo.z, dang, cosmo)
    print(f"Assumed cosmology:\t{cosmo}")
    print(f"Scale @ z={cosmo.z}:         \t{a2k}")
    print(f"Angular dist. @ z={cosmo.z}: \t{dang.to(au.Mpc)}")
    print(f"Comoving dist. @ z={cosmo.z}:\t{dcomov.to(au.Mpc)}")
    # distance calculation test z=1
    cosmo = CosmoModel(z=1)
    dz = CosmoModel.d_z(cosmo.z, cosmo)
    dang = CosmoModel.d_z2kpc(dz, cosmo)
    dcomov = (1 + cosmo.z) * dang
    a2k = CosmoModel.arcsec2kpc(cosmo.z, dang, cosmo)
    print(f"Assumed cosmology:\t{cosmo}")
    print(f"Scale @ z={cosmo.z}:         \t{a2k}")
    print(f"Angular dist. @ z={cosmo.z}: \t{dang.to(au.Mpc)}")
    print(f"Comoving dist. @ z={cosmo.z}:\t{dcomov.to(au.Mpc)}")
    # distance calculation test z=0
    cosmo = CosmoModel(z=0.01)
    dz = CosmoModel.d_z(cosmo.z, cosmo)
    dang = CosmoModel.d_z2kpc(dz, cosmo)
    dcomov = (1 + cosmo.z) * dang
    a2k = CosmoModel.arcsec2kpc(cosmo.z, dang, cosmo)
    print(f"Assumed cosmology:\t{cosmo}")
    print(f"Scale @ z={cosmo.z}:         \t{a2k}")
    print(f"Angular dist. @ z={cosmo.z}: \t{dang.to(au.Mpc)}")
    print(f"Comoving dist. @ z={cosmo.z}:\t{dcomov.to(au.Mpc)}")

    # plotting angular distances [0.2-13]
    cosmo = CosmoModel()
    redshifts = np.linspace(0.2, 13, 300)
    d_zs = np.array([CosmoModel.d_z(z, cosmo) for z in redshifts])
    dangs = CosmoModel.d_z2kpc(d_zs, cosmo)
    plt.figure(figsize=(16, 9), dpi=100)
    plt.plot(redshifts, dangs.to(au.Mpc))
    plt.xlabel("redshifts")
    plt.ylabel("D$_{ang}$ [Mpc]")
    plt.show()
