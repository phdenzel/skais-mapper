# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.cosmology module."""

import pytest
import numpy as np
from astropy import units as au
from skais_mapper.cosmology import CosmoModel


def test_default_init_and_str_repr():
    """Test default initialization and string representation."""
    cosmo = CosmoModel()
    s = str(cosmo)
    r = repr(cosmo)
    assert isinstance(s, str)
    assert "<CosmoModel[" in s
    assert s == r


def test_init_with_parameters():
    """Test initialization with specific cosmological parameters."""
    cosmo = CosmoModel(omega_m=0.3, omega_l=0.65, omega_k=0.05, h=0.7, z=2.0)
    assert cosmo.omega_m == 0.3
    assert cosmo.omega_l == 0.65
    assert cosmo.omega_k == 0.05
    assert cosmo.h == 0.7
    assert cosmo.z == 2.0


def test_omega_r_property_and_setter():
    """Test the omega_r property and its setter."""
    cosmo = CosmoModel(omega_m=0.2, omega_l=0.7, omega_k=0.1)
    # default: _omega_r is None, so it gets calculated
    expected = (1 + 0.1) - 0.2 - 0.7
    assert np.isclose(cosmo.omega_r, expected)
    # manually set omega_r
    cosmo.omega_r = 0.123
    assert np.isclose(cosmo.omega_r, 0.123)
    # set omega_r to None triggers recalc
    cosmo.omega_r = None
    assert np.isclose(cosmo.omega_r, expected)


def test_a_property_and_setter():
    """Test the scale factor `a` property and its setter."""
    cosmo = CosmoModel(z=4)
    assert np.isclose(cosmo.a, 1 / 5)
    cosmo.a = 0.25
    assert np.isclose(cosmo.z, 3)
    cosmo.z = 0
    assert np.isclose(cosmo.a, 1)


def test_rho_crit_property():
    """Test the critical density property."""
    cosmo = CosmoModel()
    rho = cosmo.rho_crit
    assert isinstance(rho, au.Quantity)
    assert rho.unit.is_equivalent(au.Msun / au.Mpc**3)


def test_H0_property_and_setter():
    """Test the Hubble constant `H0` property and its setter."""
    cosmo = CosmoModel(h=0.7)
    H0 = cosmo.H0
    assert isinstance(H0, au.Quantity)
    assert au.allclose(H0, 70 * au.km / au.s / au.Mpc)
    cosmo.H0 = 65 * au.km / au.s / au.Mpc
    assert np.isclose(cosmo.h, 0.65)


def test_H_method_variants():
    """Test the Hubble parameter method with different arguments."""
    # No argument, but z not 0
    cosmo = CosmoModel(h=0.7, omega_m=0.3, omega_l=0.7, z=1)
    H = cosmo.H()
    assert H.unit.is_equivalent(au.km / au.s / au.Mpc)
    # No argument, should default to a=1
    cosmo.z = 0
    H = cosmo.H()
    assert H.unit.is_equivalent(au.km / au.s / au.Mpc)
    assert np.isclose(H.value, 70)
    # Pass a specific scale factor
    a = 0.5
    H_a = cosmo.H(a)
    assert H_a.unit.is_equivalent(au.km / au.s / au.Mpc)
    # Edge: omega_r is None triggers calculation
    cosmo._omega_r = None
    H = cosmo.H(0.9)
    assert H.unit.is_equivalent(au.km / au.s / au.Mpc)


def test_d_comov_staticmethod():
    """Test the comoving distance calculation static method."""
    cosmo = CosmoModel()
    val = CosmoModel.d_comov(1.0, 0, cosmo)
    assert isinstance(val, au.Quantity)
    # Without cosmo_model argument
    v2 = CosmoModel.d_comov(1.0, 0)
    assert isinstance(v2, au.Quantity)


def test_d_z_staticmethod_success_and_scaled():
    """Test the d_z static method for redshift distance."""
    cosmo = CosmoModel(z=1.0)
    dz = CosmoModel.d_z(cosmo.z, cosmo)
    assert isinstance(dz, float) or isinstance(dz, np.floating)
    # Test scaled=True (returns astropy quantity)
    dz_scaled = CosmoModel.d_z(cosmo.z, cosmo, scaled=True)
    assert isinstance(dz_scaled, au.Quantity)
    # Coverage: z=None, uses cosmo_model.a
    dz2 = CosmoModel.d_z(None, cosmo)
    assert isinstance(dz2, float) or isinstance(dz2, np.floating)
    # Coverage: z=1, cosmo_model=None
    dz3 = CosmoModel.d_z(1.0)
    assert isinstance(dz3, float) or isinstance(dz3, np.floating)
    # Error path: force solve_ivp failure
    class DummyCosmo(CosmoModel):
        def H(self, a=None):
            return 0  # will cause division by zero

    with pytest.raises(ZeroDivisionError):
        CosmoModel.d_z(0.5, DummyCosmo())

    # Error path: negative scale factor
    with pytest.raises(ValueError):
        with pytest.warns(RuntimeWarning):
            CosmoModel.d_z(-2.0, cosmo)


def test_d_z2kpc_staticmethod():
    """Test the d_z2kpc static method for converting redshift distance to kpc."""
    cosmo = CosmoModel()
    d = 1.0
    kpc = CosmoModel.d_z2kpc(d, cosmo)
    assert kpc.unit.is_equivalent(au.kpc)
    arr = np.array([1.0, 2.0])
    kpc2 = CosmoModel.d_z2kpc(arr, cosmo)
    assert kpc2.shape == arr.shape


def test_arcsec2kpc_staticmethod():
    """Test the arcsec2kpc static method for converting arcseconds to kpc."""
    cosmo = CosmoModel()
    z = 1.0
    dist_z = CosmoModel.d_z2kpc(CosmoModel.d_z(z, cosmo), cosmo)
    res = CosmoModel.arcsec2kpc(z, dist_z, cosmo)
    assert res.unit.is_equivalent(au.kpc / au.arcsec)
    # Test with dist_z=None (should trigger calculation)
    res2 = CosmoModel.arcsec2kpc(z, None, cosmo)
    assert res2.unit.is_equivalent(au.kpc / au.arcsec)
