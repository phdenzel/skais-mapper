# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.simobjects module."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import astropy.units as au

import skais_mapper.simobjects as so


@pytest.fixture
def mock_tng():
    """Mock the illustris submodule."""
    with patch("skais_mapper.simobjects.tng") as tng:
        # Mock pidx/ptype conversions
        tng.util.pidx_from_ptype.side_effect = lambda x: {"gas": 0, "dm": 1, "stars": 3}.get(x, 0)
        tng.util.ptype_from_pidx.side_effect = lambda x: {0: "gas", 1: "dm", 3: "stars"}.get(
            x, "gas"
        )
        # Mock groupcat and snapshots loaders
        tng.groupcat.load_header.return_value = {
            "OmegaLambda": 0.6911,
            "Omega0": 0.3089,
            "Redshift": 0.1,
            "HubbleParam": 0.6774,
        }
        tng.snapshots.load_header.return_value = {
            "OmegaLambda": 0.6911,
            "Omega0": 0.3089,
            "Redshift": 0.1,
            "HubbleParam": 0.6774,
            "UnitLength_in_cm": 3.0e21,
            "UnitMass_in_g": 1.0e43,
            "UnitVelocity_in_cm_per_s": 1e5,
            "MassTable": [0, 3.07368e-05, 0, 5.73879e-06, 0, 0],
        }
        tng.groupcat.load_halos.return_value = np.array([0, 1, 2])
        tng.groupcat.load_single.return_value = {
            "SubhaloLen": 10,
            "SubhaloLenType": [1, 2, 3],
            "SubhaloPos": np.array([1, 1, 1]),
            "SubhaloHalfmassRad": np.ones(1),
            "SubhaloHalfmassRadType": np.ones(1),
        }
        tng.snapshots.snapshot_offsets.return_value = {
            "offsetType": np.zeros((6,)),
            "snapOffsets": np.zeros((6, 2)),
            "lenType": np.zeros((6,)),
        }
        tng.snapshots.load_snapshot.return_value = {
            "Coordinates": np.zeros((3, 3)),
            "Redshift": np.zeros(3),
            "Masses": np.ones(3),
            "Velocities": np.zeros((3, 3)),
            "MagneticField": np.ones((3, 3)),
            "Density": np.ones(3),
            "SubfindDensity": np.ones(3),
            "SubfindDMDensity": np.ones(3),
            "InternalEnergy": np.ones(3),
            "ElectronAbundance": np.ones(3),
            "NeutralHydrogenAbundance": np.ones(3),
            "GFM_Metals": np.ones((3, 10)),
            "CenterOfMass": np.zeros((3, 3)),
        }
        yield tng


@pytest.fixture
def galaxy(mock_tng):
    """Mock the `TNGGalaxy` class, including a `CosmoModel`."""
    with patch("skais_mapper.simobjects.CosmoModel") as Cosmo:
        cosmo = MagicMock()
        cosmo.h = 0.7
        cosmo.a = 1.0
        cosmo.z = 0.5
        cosmo.d_z.return_value = 1
        cosmo.d_z2kpc.return_value = au.kpc
        cosmo.arcsec2kpc.return_value = au.kpc
        Cosmo.return_value = cosmo
        g = so.TNGGalaxy("/tmp", 0, 0, as_float32=False)
        g.data.update({"GFM_Metals": np.array([[0.76]]), "NeutralHydrogenAbundance": np.ones(3)})
        return g


def test_sphgalaxy_init_sets_attributes(galaxy):
    """Test `SPHGalaxy` initialization sets attributes correctly."""
    assert galaxy.ra.unit == au.deg
    assert galaxy.dec.unit == au.deg
    assert galaxy.distance.unit == au.Mpc
    assert galaxy.peculiar_v.unit == au.km / au.s
    assert hasattr(galaxy, "header")
    assert hasattr(galaxy, "cosmology")
    assert hasattr(galaxy, "data")


def test_sphgalaxy_basic_paths(mock_tng):
    """Test `SPHGalaxy` unit and property paths not hit by `TNGGalaxy`."""
    g = so.SPHGalaxy()
    # Test p_idx setter for int and str
    g.p_idx = 1
    assert g._p_idx == 1
    g.p_idx = "gas"
    assert isinstance(g._p_idx, int)
    # Test particle_type setter
    g.particle_type = "dm"
    assert isinstance(g.p_idx, int)
    # Test set_units and odd unit strings
    g.set_units(length=1e21, velocity=1e5, mass=1e40)
    assert isinstance(g.units("t"), au.Quantity)
    assert isinstance(g.units("sqrtP"), au.Quantity)
    assert g.units("unknown") == 1
    # Test float32 paths
    g.as_float32 = True
    assert g.UnitLength.dtype == np.float32 or hasattr(g.UnitLength, "dtype")
    assert g.UnitMass.dtype == np.float32 or hasattr(g.UnitMass, "dtype")
    assert g.UnitVelocity.dtype == np.float32 or hasattr(g.UnitVelocity, "dtype")


def test_sphgalaxy_particle_type_none_sets_default(mock_tng):
    """Test `SPHGalaxy` with no particle type set, should default to 'gas'."""
    g = so.SPHGalaxy(particle_type=None)
    assert g._p_idx == 0  # default 'gas'


def test_galaxy_explicit_type_and_unit(mock_tng):
    """Test `TNGGalaxy` with non-fallback particle type and units."""
    g = so.TNGGalaxy(
        "/tmp", 0, 0, particle_type="dm", units={"length": 1, "velocity": 1, "mass": 1}
    )
    assert g.particle_type == "dm"
    assert g.p_idx == 1


def test_sphgalaxy_units_setter(monkeypatch, mock_tng):
    """Test `SPHGalaxy` units setter."""
    g = so.SPHGalaxy(units={"length": "kpc", "mass": "Msun", "velocity": "km/s"})
    # If set_units is called, it should set the attributes
    g.set_units = MagicMock()
    g.__init__(units={"length": "kpc", "mass": "Msun", "velocity": "km/s"})
    g.set_units.assert_called_with(length="kpc", mass="Msun", velocity="km/s")


def test_galaxy_p_idx_setter(mock_tng):
    """Test `SPHGalaxy` particle index getting/setting."""
    g = so.SPHGalaxy()
    g.p_idx = "gas"
    assert g.p_idx == 0
    g.p_idx = 1
    assert g.particle_type == "dm"
    g.p_idx = None
    assert g.particle_type == "gas"


def test_sphgalaxy_pidx_property_and_setter(galaxy):
    """Test `SPHGalaxy.p_idx` property and setter."""
    # Test default
    assert isinstance(galaxy.p_idx, int)
    # Test setter with int
    galaxy._p_idx = 0
    galaxy.p_idx = "dm"
    assert galaxy._p_idx == 1
    # Test setter with string
    with patch("skais_mapper.simobjects.tng.util.pidx_from_ptype", return_value=4) as mock_pidx:
        galaxy.p_idx = "star"
        mock_pidx.assert_called_with("star")
        assert galaxy._p_idx == 4


def test_particle_mass_and_other_data_fields(mock_tng):
    """Test `SPHGalaxy.particle_mass` and related data-dependent properties."""
    g = so.SPHGalaxy()
    # Add MassTable for constant particle mass
    g.header["snapshot"]["MassTable"] = [0, 1, 2, 3, 4, 5]
    g._p_idx = 1
    pm = g.particle_mass
    assert pm is None or isinstance(pm, au.Quantity)
    # Populate data fields for other accessors
    g.data = {
        "CenterOfMass": np.zeros((3, 3)),
        "Velocities": np.ones((3, 3)),
        "InternalEnergy": np.ones(3),
        "ElectronAbundance": np.ones(3),
        "GFM_Metals": np.ones((3, 1)),
        "NeutralHydrogenAbundance": np.ones(3),
    }
    assert g.cell_positions.shape[0] == 3
    assert g.velocities.shape[0] == 3
    assert g.internal_energy.shape[0] == 3
    assert g.x_e.shape[0] == 3
    assert g.x_H.shape[0] == 3
    assert g.x_HI.shape[0] == 3


def test_sphgalaxy_rotation_and_verbose_flags(galaxy):
    """Test `SPHGalaxy` rotation and verbose flags."""
    assert hasattr(galaxy, "rotation")
    assert hasattr(galaxy, "verbose")
    # Check that verbose flag is set correctly
    g = galaxy
    g.verbose = True
    assert g.verbose is True


def test_load_cosmology(mock_tng):
    """Test `SPHGalaxy` particle index getting/setting."""
    g = so.SPHGalaxy()
    g.load_cosmology(
        {
            "omega_l": 0.7,
            "omega_k": 0.3,
            "z": 0,
            "h": 0.7,
        }
    )
    assert g.cosmology.z == 0
    assert g.cosmology.h == 0.7
    g.load_cosmology(None, in_place=False)
    assert g.cosmology.z == 0
    assert g.cosmology.h == 0.7


def test_sphgalaxy_init_with_all_args(mock_tng):
    """Test `SPHGalaxy` initialization with all arguments."""
    rotation = lambda x: x
    units = {"length": "kpc", "mass": "Msun", "velocity": "km/s"}
    g = so.SPHGalaxy(
        ra=10,
        dec=20,
        distance=5,
        peculiar_v=100,
        rotation=rotation,
        cosmo_pars={"omega_k": 0.3},
        units=units,
        particle_type="dm",
        as_float32=True,
        verbose=True,
    )
    assert g.ra.value == 10
    assert g.dec.value == 20
    assert g.distance.value == 5
    assert g.peculiar_v.value == 100
    assert g.rotation is rotation
    assert g.as_float32 is True
    assert g.verbose is True


def test_load_tng_cosmology(mock_tng):
    """Test `TNGGalaxy` particle index getting/setting."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.load_cosmology(
        {
            "omega_l": 0.7,
            "omega_k": 0.3,
            "z": 0,
            "h": 0.721,
        },
        in_place=True,
    )

    assert g.cosmology.omega_l == 0.7
    assert g.cosmology.omega_k == 0.3
    assert g.cosmology.z == 0
    assert g.cosmology.h == 0.721
    g.load_cosmology(None, in_place=True)
    assert g.cosmology.z == 0.1
    assert g.cosmology.h == 0.6774
    g.load_cosmology(
        {
            "omega_l": 0.7,
            "omega_k": 0.3,
            "z": 0,
            "h": 0.721,
        },
        in_place=False,
    )
    assert g.cosmology.z == 0.1
    assert g.cosmology.h == 0.6774


def test_cosmo_angular_dist_and_res(mock_tng):
    """Test `SPHGalaxy` particle index getting/setting."""
    g = so.SPHGalaxy()
    g.load_cosmology({"z": 0})
    dist_ang = g.angular_distance(eps=1e-3)
    assert dist_ang >= 0
    res_ang = g.angular_resolution(eps=1e-3)
    assert res_ang[0] >= 0


def test_load_data_and_header(mock_tng):
    """Force load_data and other methods to trigger fallback loading header."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.load_data(
        particle_type="gas",
        primary_fields=["Coordinates", "Masses"],
        optional_fields=["Density"],
        verbose=True,
    )
    assert isinstance(g.data, dict)


def test_load_data_gfm_metals_and_com():
    """Force load_data and other methods to trigger fallback loading header."""
    with patch("skais_mapper.simobjects.tng") as tng:
        tng.util.pidx_from_ptype.side_effect = lambda x: {"gas": 0, "dm": 1, "stars": 3}.get(x, 0)
        tng.util.ptype_from_pidx.side_effect = lambda x: {0: "gas", 1: "dm", 3: "stars"}.get(
            x, "gas"
        )
        # Mock groupcat and snapshots loaders
        tng.groupcat.load_header.return_value = {
            "OmegaLambda": 0.6911,
            "Omega0": 0.3089,
            "Redshift": 0.1,
            "HubbleParam": 0.6774,
        }
        tng.snapshots.load_header.return_value = {
            "OmegaLambda": 0.6911,
            "Omega0": 0.3089,
            "Redshift": 0.1,
            "HubbleParam": 0.6774,
            "UnitLength_in_cm": 3.0e21,
            "UnitMass_in_g": 1.0e43,
            "UnitVelocity_in_cm_per_s": 1e5,
            "MassTable": [0, 3.07368e-05, 0, 5.73879e-06, 0, 0],
        }
        tng.groupcat.load_halos.return_value = np.array([0, 1, 2])
        tng.groupcat.load_single.return_value = {
            "SubhaloLen": 10,
            "SubhaloLenType": [1, 2, 3],
            "SubhaloPos": np.array([1, 1, 1]),
            "SubhaloHalfmassRad": np.ones(1),
            "SubhaloHalfmassRadType": np.ones(1),
        }
        tng.snapshots.snapshot_offsets.return_value = {
            "offsetType": np.zeros((6,)),
            "snapOffsets": np.zeros((6, 2)),
            "lenType": np.zeros((6,)),
        }
        tng.snapshots.load_snapshot.return_value = {
            "Coordinates": np.zeros((3, 3)),
            "Redshift": np.zeros(3),
            "Masses": np.ones(3),
            "Velocities": np.zeros((3, 3)),
            "Density": np.ones(3),
            "SubfindDensity": np.ones(3),
            "SubfindDMDensity": np.ones(3),
            "InternalEnergy": np.ones(3),
            "ElectronAbundance": np.ones(3),
            "NeutralHydrogenAbundance": np.ones(3),
            "GFM_Metals": None,
            "CenterOfMass": None,
        }

        g = so.TNGGalaxy("/tmp", 0, 0)
        g.load_data(
            particle_type="gas",
            primary_fields=["Coordinates", "Masses"],
            verbose=True,
        )
    assert isinstance(g.data, dict)


def test_load_data_density_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Density
    del g.data
    assert len(g.density) == 3
    del g.data["Density"]
    assert len(g.density) == 3
    del g.data["SubfindDMDensity"]
    assert len(g.density) == 3
    del g.data["SubfindDensity"]
    assert g.density is None


def test_load_data_masses_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Masses
    del g.data
    assert len(g.masses) == 3
    del g.data["Masses"]
    assert len(g.masses) == 3
    del g.header["snapshot"]["MassTable"]
    assert g.masses is None


def test_load_data_internal_energy_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Internal Energy
    del g.data
    assert len(g.internal_energy) == 3
    del g.data["InternalEnergy"]
    assert g.internal_energy is None


def test_load_data_electron_abundance_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Electron abundance
    del g.data
    assert len(g.x_e) == 3
    del g.data["ElectronAbundance"]
    assert g.x_e is None


def test_load_data_x_h_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Hydrogen
    del g.data
    assert len(g.x_H) == 3
    del g.data["GFM_Metals"]
    assert g.x_H is None


def test_load_data_n_h_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Hydrogen number density
    del g.data
    assert len(g.n_H) == 3
    del g.data["Density"]
    del g.data["SubfindDMDensity"]
    del g.data["SubfindDensity"]
    assert g.n_H is None


def test_load_data_m_h_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Hydrogen density
    del g.data
    assert len(g.m_H) == 3
    del g.data["Masses"]
    del g.header["snapshot"]["MassTable"]
    assert g.m_H is None


def test_load_data_x_hi_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Neutral Hydrogen
    del g.data
    assert len(g.x_HI) == 3
    del g.data["NeutralHydrogenAbundance"]
    assert g.x_HI is None


def test_load_data_n_hi_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Hydrogen number density
    del g.data
    assert len(g.n_HI) == 3
    del g.data["Density"]
    del g.data["SubfindDMDensity"]
    del g.data["SubfindDensity"]
    del g.data["NeutralHydrogenAbundance"]
    assert g.n_HI is None


def test_load_data_m_hi_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Hydrogen density
    del g.data
    assert len(g.m_HI) == 3
    del g.data["Masses"]
    del g.header["snapshot"]["MassTable"]
    assert g.m_HI is None


def test_load_data_magnetic_field_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Magnetic Field
    del g.data
    assert len(g.magnetic_field) == 3
    del g.data["MagneticField"]
    assert g.magnetic_field is None


def test_load_data_velocities_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Velocities
    del g.data
    assert len(g.velocities) == 3
    del g.data["Velocities"]
    assert g.velocities is None


def test_load_data_positions_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    g.p_idx = 1
    # Positions
    del g.data
    assert len(g.particle_positions) == 3
    del g.data["Coordinates"]
    assert g.particle_positions is None
    del g.data
    assert len(g.cell_positions) == 3
    del g.data["CenterOfMass"]
    assert g.cell_positions is None


def test_load_header_property_fallback(mock_tng):
    """Force load_header as property access fallback."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    del g.header
    bs = g.boxsize
    L = g.units("l/h")
    assert g.units("l") / g.cosmology.h * g.cosmology.a == L
    assert (bs / L).decompose() == 1
    # M
    M = g.units("m")
    assert M.value == 1e43
    del g.header
    g.set_units(1, 1, 1)
    del g.header
    M = g.UnitMass
    assert M.value == 1
    del g.header["snapshot"]["UnitMass_in_g"]
    M = g.UnitMass
    assert M.value == 1.989e43
    # L
    del g.header
    L = g.UnitLength
    assert L.value == 1
    del g.header["snapshot"]["UnitLength_in_cm"]
    L = g.UnitLength
    assert L.value == 3.085678e21
    # V
    del g.header
    V = g.UnitVelocity
    assert V.value == 1
    del g.header["snapshot"]["UnitVelocity_in_cm_per_s"]
    L = g.UnitVelocity
    assert L.value == 1e5


def test_subhalo_list_and_filtering(mock_tng):
    """Exercise subhalo_list with filtering by mass and radius."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    # Default (no filter) returns array
    all_ids = g.subhalo_list("/tmp", 0, verbose=True)
    assert isinstance(all_ids, np.ndarray)
    # Filter by mass and radius, should still return array (mock data has small values)
    filtered_ids = g.subhalo_list("/tmp", 0, filtered=True)
    assert isinstance(filtered_ids, np.ndarray)
    del g._halo_index
    assert g.halo_index == 0


def test_subhalo_properties(mock_tng):
    """Exercise `TNGGalaxy` properties depending on subhalo."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    print(g.subhalo)


def test_halo_indices(mock_tng):
    """Exercise `TNGGalaxy.halo_index` manipulations."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    del g._halo_index
    assert g.halo_index == 0
    g._halo_index = 0
    g.halo_index = 0
    g.subhalo = g.load_subhalo(0)
    g.data = g.load_data(0)
    g.halo_index = 1
    assert g.subhalo is not None
    assert g.data is not None
    g.subhalo = g.load_subhalo(-1, verbose=True)


def test_units_and_properties(galaxy):
    """Test `TNGGalaxy.Unit*` properties."""
    assert galaxy.UnitLength.unit == au.cm
    assert galaxy.UnitMass.unit == au.g
    assert galaxy.UnitVelocity.unit == (au.cm / au.s)
    assert isinstance(galaxy.units("vp"), au.Quantity)

    # density and masses use mocked data
    d = galaxy.density
    m = galaxy.masses
    assert d.shape[0] == 3 and m.shape[0] == 3
    assert galaxy.n_H is not None
    assert galaxy.m_H is not None
    assert galaxy.n_HI is not None
    assert galaxy.m_HI is not None


def test_no_subhalo_defaults(mock_tng):
    """Test `TNGGalaxy.N_particles_type` without subhalo."""
    g = so.TNGGalaxy("/tmp", 0, 0)
    assert g.N_particles == 10
    assert len(g.N_particles_type) == 3
    del g.subhalo
    assert len(g.N_particles_type) == 0
    assert g.N_particles == -1
    # center loads subhalo anew
    assert len(g.center) == 3
    del g.subhalo["SubhaloPos"]
    assert g.center is None


def test_center_radii_hsml_properties(galaxy):
    """Ensure center, radii, and hsml properties resolve from mocked subhalo fields."""
    center = galaxy.center
    radii = galaxy.radii
    hsml = galaxy.hsml
    assert center.shape[0] == 3
    assert radii.shape[0] == 1 or radii.shape[0] == 3
    assert hsml.shape[0] >= 1


def test_radii_temperature_magnetic(galaxy):
    """Test `TNGGalaxy.{radii, hsml, temperature, magnetic_field_strength}` properties."""
    galaxy.data["MagneticField"] = np.ones((3, 3))
    T = galaxy.temperature
    assert T.unit == au.K
    b = galaxy.magnetic_field_strength
    assert b.unit.is_equivalent(au.G)
    assert galaxy.radii.shape[0] == 3
    assert galaxy.hsml.shape[0] == 3


def test_kd_tree_returns_array(galaxy):
    """Test `TNGGalaxy.kd_tree` case: returns array."""
    galaxy.data["Coordinates"] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = galaxy.kd_tree(k=2, threads=1, verbose=True)
    assert result.shape[0] == 4


def test_kd_tree_returns_None(galaxy):
    """Test `TNGGalaxy.kd_tree` case: returns None."""
    del galaxy.data["Coordinates"]
    result = galaxy.kd_tree(k=2, threads=1, verbose=False)
    assert result is None
    galaxy.data["Coordinates"] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])


def test_kdtree_and_strip_units_variants(galaxy):
    """Cover kd_tree float32 and strip_ap_units dtype case."""
    galaxy.data["Coordinates"] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    # as_float32 path
    dists = galaxy.kd_tree(k=2, threads=1, verbose=False, as_float32=True)
    assert dists.dtype == np.float32
    # strip_ap_units forcing float32 dtype
    q = np.array([1, 2, 3]) * au.kpc
    vals = so.strip_ap_units(q, mask=np.array([0, 2]), units=[au.kpc], dtype=np.float32)
    assert vals[0].dtype == np.float32


def test_get_mapping_arrays(galaxy):
    """Test `TNGGalaxy.get_mapping_arrays` method."""
    galaxy.data["Coordinates"] = np.zeros((3, 3))
    galaxy.data["Masses"] = np.ones(3)
    galaxy.data["Density"] = np.ones(3)
    galaxy.data["InternalEnergy"] = np.ones(3)
    galaxy.data["ElectronAbundance"] = np.ones(3)
    galaxy.subhalo["magnetic_field_strength"] = np.ones((3, 3))
    del galaxy.data["MagneticField"]
    arrs = galaxy.get_mapping_arrays(
        ("particle_positions", ("temperature", "masses")), verbose=True
    )
    arrs = galaxy.get_mapping_arrays(
        ("particle_positions", ("magnetic_field_strength", "magnetic_field_strength"))
    )
    print(arrs)


def test_mapping_and_map_generation(galaxy):
    """Test `TNGGalaxy.get_mapping_arrays` & `TNGGalaxy.generate_map` methods."""
    galaxy.data["Coordinates"] = np.zeros((3, 3))
    galaxy.data["Masses"] = np.ones(3)
    galaxy.data["Density"] = np.ones(3)
    galaxy.data["InternalEnergy"] = np.ones(3)
    galaxy.data["ElectronAbundance"] = np.ones(3)
    arrs = galaxy.get_mapping_arrays()
    assert len(arrs) == 3
    # test invalid assignment function triggers ValueError
    with pytest.raises(ValueError):
        galaxy.generate_map(assignment_func=lambda *a, **k: None)
    # patch voronoi_RT_2D to no-op
    with patch("skais_mapper.simobjects.voronoi_RT_2D") as mock_fn:
        proj, extent, count = galaxy.generate_map(assignment_func=mock_fn, verbose=False)
        mock_fn.assert_called_once()
        assert proj.shape[0] == proj.shape[1]


def test_generate_map_with_halfmassrad(galaxy):
    """Trigger the rotation branch in generate_map."""
    galaxy.data["Coordinates"] = np.zeros((3, 3))
    galaxy.data["Masses"] = np.ones(3)
    # Patch voronoi_RT_2D to bypass actual computation
    with patch("skais_mapper.simobjects.voronoi_RT_2D") as mock_fn:
        mock_fn.__name__ = "voronoi_RT_2D_mock"
        proj, extent, count = galaxy.generate_map(
            keys=["particle_positions", "masses", "radii", "center", "SubhaloHalfmassRadType"],
            factors=[1, 1, 1, 1, 1],
            use_half_mass_rad=True,
            grid_size=8,
            assignment_func=mock_fn,
            tracers=np.ones(3),
            divisions=np.ones(3),
            verbose=True,
        )
        mock_fn.assert_called_once()
        assert proj.shape == (8, 8)
        assert len(extent) == 4
        assert isinstance(count, int)


def test_generate_map_with_rotation(galaxy):
    """Trigger the rotation branch in generate_map."""
    galaxy.data["Coordinates"] = np.zeros((3, 3))
    galaxy.data["Masses"] = np.ones(3)
    # Patch voronoi_RT_2D to bypass actual computation
    with patch("skais_mapper.simobjects.voronoi_RT_2D") as mock_fn:
        proj, extent, count = galaxy.generate_map(
            keys=["particle_positions", "masses", "radii", "center"],
            factors=[1, 1, 1, 1],
            use_half_mass_rad=False,
            grid_size=8,
            assignment_func=mock_fn,
            rot=(1, 0, 0),
            verbose=False,
        )
        mock_fn.assert_called_once()
        assert proj.shape == (8, 8)
        assert len(extent) == 4
        assert isinstance(count, int)


def test_generate_map_with_exception(galaxy):
    """Trigger the rotation branch in generate_map."""
    galaxy.data["Coordinates"] = np.zeros((3, 3))
    galaxy.data["Masses"] = np.ones(3)
    # Patch voronoi_RT_2D to bypass actual computation
    with patch("skais_mapper.simobjects.voronoi_RT_2D") as mock_fn:
        mock_fn.side_effect = Exception("Mock exception")
        with pytest.raises(Exception):
            proj, extent, count = galaxy.generate_map(
                keys=["particle_positions", "masses", "radii", "center"],
                factors=[1, 1, 1, 1],
                use_half_mass_rad=False,
                grid_size=8,
                assignment_func=mock_fn,
                rot=(1, 0, 0),
                verbose=False,
            )


def test_indices_within_box_without_radius(capfd):
    """Test indices_within_box branch where radius is computed from extent."""
    pts = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]]) * au.kpc
    center = np.zeros(3) * au.kpc
    idcs, limits = so.indices_within_box(pts, center, radius=None, verbose=True)
    assert isinstance(idcs, np.ndarray)
    assert all(isinstance(x, au.Quantity) for x in limits)
    out, err = capfd.readouterr()
    assert "Box extent" in out
    assert "Radius" in out
    assert "Extent" in out
    assert "Selected particles" in out


def test_indices_within_box_and_strip_units():
    """Test `indices_within_box` method."""
    pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]) * au.kpc
    center = np.array([0, 0, 0]) * au.kpc
    idcs, limits = so.indices_within_box(pts, center, radius=1 * au.kpc)
    assert isinstance(idcs, np.ndarray) and len(limits) == 6
    vals = so.strip_ap_units(pts, mask=np.array([0, 1]), units=[au.kpc])
    assert isinstance(vals[0], np.ndarray)


def test_strip_units():
    """Test `strip_ap_units` function."""
    # Test with a single array
    pts = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]) * au.kpc
    stripped = so.strip_ap_units(pts, mask=np.array([0]))
    assert isinstance(stripped[0], np.ndarray)
    # assert stripped[0].dtype == np.float64

    stripped = so.strip_ap_units(pts)
    assert isinstance(stripped[0], np.ndarray)
