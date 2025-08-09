# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tools for manipulating data objects from simulations."""

import time
from pathlib import Path
from typing import Any
from collections.abc import Callable
from numpy.typing import NDArray
import numpy as np
import scipy as sp
import astropy.units as au
import astropy.constants as ac
import skais_mapper.illustris as tng
from skais_mapper.cosmology import CosmoModel
from skais_mapper.rotations import R
from skais_mapper.raytrace import voronoi_RT_2D, voronoi_NGP_2D


__all__ = ["SPHGalaxy", "TNGGalaxy", "GasolineGalaxy"]


class SPHGalaxy:
    """A generic base SPH Galaxy simulation parser."""

    primary_hdf5_fields: dict[int, list[str]] = {0: [], 1: [], 4: []}
    optional_hdf5_fields: dict[int, list[str]] = {0: [], 1: [], 4: []}

    def __init__(
        self,
        ra: float = 0.0,
        dec: float = 0.0,
        distance: float = 3.0,
        peculiar_v: float = 0.0,
        rotation: Callable | None = None,
        cosmo_pars: dict | None = None,
        units: dict | None = None,
        particle_type: str | None = None,
        as_float32: bool = False,
        verbose: bool = False,
    ):
        """Initialize an SPHGalaxy base class instance.

        Args:
            ra: The right ascension sky coordinate
            dec: The declination sky coordinate
            distance: The distance from observer
            peculiar_v: The galaxy's peculiar velocity
            rotation: Arbitrary rotation of the galaxy
            cosmo_pars: Cosmology settings
            units: unit system in the simulations
            particle_type: the particle type name e.g. 'gas', 'dm', or 'star' (see illustris.util)
            as_float32: If True, load simulation data as 32-bit floats (for efficiency)
            verbose: If True, print information to the command line
        """
        self.as_float32 = as_float32
        self.header = self.load_header()
        cosmo_pars = self.load_cosmology(cosmo_pars)
        self.cosmology = CosmoModel(**cosmo_pars)
        if particle_type is None:
            particle_type = "gas"
        self._p_idx = tng.util.pidx_from_ptype(particle_type)
        self.data = self.load_data(as_float32=self.as_float32, verbose=verbose)
        self.ra = ra * au.deg
        self.dec = dec * au.deg
        self.distance = distance * au.Mpc
        self.peculiar_v = peculiar_v * au.km / au.s
        self.rotation = rotation
        if units:
            self.set_units(**units)
        self.verbose = verbose

    @property
    def p_idx(self) -> int:
        """Particle index {0: 'gas', 1: 'dm', 2: 'tracers', 3: 'stars', 4: 'BHs'}."""
        return self._p_idx

    @p_idx.setter
    def p_idx(self, p_idx: int | str):
        """Setter for particle index."""
        if isinstance(p_idx, int):
            self._p_idx = p_idx
        elif isinstance(p_idx, str):
            self._p_idx = tng.util.pidx_from_ptype(p_idx)
        else:
            self._p_idx = 0
        # reload data
        self.load_data(as_float32=self.as_float32, verbose=self.verbose)

    @property
    def particle_type(self) -> str:
        """Particle type {0: 'gas', 1: 'dm', 2: 'tracers', 3: 'stars', 4: 'BHs'}."""
        p_idx = self.p_idx
        return tng.util.ptype_from_pidx(p_idx)

    @particle_type.setter
    def particle_type(self, ptype: int | str):
        """Setter for particle type."""
        self.p_idx = ptype

    def load_header(self) -> dict:
        """Dummy method, to be overridden in subclasses.

        Returns:
            (dict): the header dictionary which should contain
                - a 'snapshot' and 'catalog' subdictionary
                - the 'snapshot' subdictionary should contain
                    - 'UnitLength_in_cm', 'UnitVelocity_in_cm_per_s', 'UnitMass_in_g'
                    - 'OmegaLambda', 'Omega0', 'Redshift' / 'Time', 'HubbleParam'
                    - optionally: 'OmegaK'
        """
        self.header = {}
        self.header["snapshot"] = {}
        self.header["snapshot"]["UnitLength_in_cm"] = 3.085678e21
        self.header["snapshot"]["UnitVelocity_in_cm_per_s"] = 1e5
        self.header["snapshot"]["UnitMass_in_g"] = 1.989e43
        return self.header

    def load_cosmology(self, cosmo_pars: dict | None, in_place: bool = True) -> dict:
        """Dummy method, to be overridden in subclasses.

        Returns:
            (dict): a dictionary with cosmological parameters pulled from a header
                or set manually. The dictionary will be used for the CosmoModel
                dataclass.
        """
        if cosmo_pars is None:
            cosmo_pars = {}
        if in_place:
            self.cosmology = CosmoModel(**cosmo_pars)
        return cosmo_pars

    def angular_distance(self, eps: float = 1e-4) -> float:
        """Calculate the angular distance of the simulation at the given redshift.

        Args:
            eps: minimum cutoff value for the redshift
        """
        z = self.cosmology.z
        z += eps if z < eps else 0
        dz = self.cosmology.d_z(z, cosmo_model=self.cosmology, scaled=False)
        dang = self.cosmology.d_z2kpc(dz, cosmo_model=self.cosmology)
        return dang

    def angular_resolution(self, eps: float = 1e-4) -> tuple[float, float]:
        """Calculate the angular scale of the simulation at the given redshift.

        Args:
            eps: minimum cutoff value for the redshift
        """
        z = self.cosmology.z
        z += eps if z < eps else 0
        dz = self.cosmology.d_z(z, cosmo_model=self.cosmology, scaled=True)
        arcsec2kpc = self.cosmology.arcsec2kpc(z, dz)
        return 1.0 / arcsec2kpc.to(au.kpc / au.deg), z

    @property
    def boxsize(self) -> au.Quantity:
        """The virtual boxsize of the simulation."""
        if not hasattr(self, "header"):
            self.header = self.load_header()
        L = self.units("l/h")
        box = self.header["snapshot"].get("BoxSize", 1) * L
        return box.to(au.kpc)

    def units(self, u: str) -> au.Quantity:
        """Get common units from descriptor strings ('l', 'm', 'v', and variants).

        Args:
            u: the unit string describing the dimensionality;
                l -> length;
                m -> mass;
                v -> velocity;
                t -> time;
                vp -> peculiar velocity;
                sqrtP -> square root of pressure;
                c[] -> comoving quantities;
                []/h -> quantities divided by the dimensionless Hubble const.

        Returns:
            (astropy.units.Quantity): specified (scaled/composite) unit.
        """
        h = self.cosmology.h
        a = self.cosmology.a
        match u:
            case "l":
                return self.UnitLength
            case "m":
                return self.UnitMass
            case "v":
                return self.UnitVelocity
            case "t":
                return self.UnitLength / self.UnitVelocity
            case "l/h":
                return self.UnitLength * (a / h)
            case "m/h":
                return self.UnitMass / h
            case "vp":
                return self.UnitVelocity * np.sqrt(a)
            case "sqrtP":
                return (
                    h
                    / a**2
                    * (self.UnitMass / self.UnitLength) ** (1.0 / 2)
                    / (self.UnitLength / self.UnitVelocity)
                )
            case _:
                return 1

    def set_units(self, length: float, velocity: float, mass: float):
        """Set units in the header attribute.

        Args:
            length: length unit
            velocity: velocity unit (sets the time unit implicitly)
            mass: mass unit
        """
        if not hasattr(self, "header"):
            self.header = self.load_header()
        self.header["snapshot"]["UnitLength_in_cm"] = length
        self.header["snapshot"]["UnitVelocity_in_cm_per_s"] = velocity
        self.header["snapshot"]["UnitMass_in_g"] = mass

    @property
    def UnitLength(self) -> au.Quantity:
        """The simulation's length unit as astropy.units.Quantity."""
        if not hasattr(self, "header"):
            self.header = self.load_header()
        h = self.header["snapshot"]
        if "UnitLength_in_cm" in h:
            length = h["UnitLength_in_cm"] * au.cm
        else:
            length = 3.085678e21 * au.cm
        if self.as_float32:
            return length.to(au.kpc).astype(np.float32)
        return length

    @property
    def UnitMass(self) -> au.Quantity:
        """The simulation's mass unit as astropy.units.Quantity."""
        if not hasattr(self, "header"):
            self.header = self.load_header()
        h = self.header["snapshot"]
        if "UnitMass_in_g" in h:
            mass = h["UnitMass_in_g"] * au.g
        else:
            mass = 1.989e43 * au.g
        if self.as_float32:
            return mass.to(au.Msun).astype(np.float32)
        return mass

    @property
    def UnitVelocity(self) -> au.Quantity:
        """The simulation's velocity unit as astropy.units.Quantity."""
        if not hasattr(self, "header"):
            self.header = self.load_header()
        h = self.header["snapshot"]
        if "UnitVelocity_in_cm_per_s" in h:
            velocity = h["UnitVelocity_in_cm_per_s"] * au.cm / au.s
        else:
            velocity = 1e5 * au.cm / au.s
        if self.as_float32:
            return velocity.astype(np.float32)
        return velocity

    def load_data(self, **kwargs) -> dict:
        """Dummy method, to be overridden in subclasses.

        Returns:
            (dict): the data dictionary loaded from the simulations containing:
                - 'Density', 'Masses', 'Coordinates', 'Velocities', 'InternalEnergy',
                  'ElectronAbundance'
                - optionally 'CenterOfMass', 'GFM_Metals' (axis 1 should contain
                  hydrogen gas fractions), 'NeutralHydrogenAbundance'
        """
        kwargs.setdefault(
            "fields",
            self.primary_hdf5_fields[self.p_idx] + self.optional_hdf5_fields[self.p_idx],
        )
        data = {k: None for k in kwargs["fields"]}
        return data

    @property
    def density(self) -> au.Quantity | None:
        """The simulation's density data in corresponding units."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        M = self.units("m/h")
        perVol = np.power(self.units("l/h"), -3)
        if "Density" in self.data:
            return self.data["Density"][:] * M * perVol
        elif "SubfindDMDensity" in self.data:
            return self.data["SubfindDMDensity"][:] * M * perVol
        elif "SubfindDensity" in self.data:
            return self.data["SubfindDensity"][:] * M * perVol
        return None

    @property
    def masses(self) -> au.Quantity | None:
        """The simulation's mass data in corresponding units."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        M = self.units("m/h")
        if "Masses" in self.data:
            return (self.data["Masses"][:] * M).to(au.Msun)
        elif self.particle_mass is not None:
            masses = np.ones(self.data["Coordinates"].shape[0])
            return masses * self.particle_mass
        return None

    @property
    def particle_mass(self) -> au.Quantity | float | None:
        """The simulation's particle mass (for constant mass particles)."""
        h = self.load_header()["snapshot"]
        M = self.units("m/h")
        p_mass = None
        if "MassTable" in h and self.p_idx in [1, 3]:
            p_mass = (h["MassTable"][self.p_idx] * M).to(au.Msun)
        return p_mass

    @property
    def particle_positions(self) -> au.Quantity | None:
        """The simulation's particle position data in corresponding units."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        L = self.units("l/h")
        if "Coordinates" in self.data:
            return (self.data["Coordinates"][:] * L).to(au.kpc)
        return None

    @property
    def cell_positions(self) -> au.Quantity | None:
        """The simulation's cell position data in corresponding units."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        L = self.units("l/h")
        if "CenterOfMass" in self.data:
            return (self.data["CenterOfMass"][:] * L).to(au.kpc)
        return None

    @property
    def velocities(self) -> au.Quantity | None:
        """The simulation's particle velocity data in corresponding units."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        V = self.units("vp")
        if "Velocities" in self.data:
            return self.data["Velocities"][:] * V
        return None

    @property
    def internal_energy(self) -> au.Quantity | None:
        """The simulation's internal energy data in corresponding units."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        U = np.power(self.units("v"), 2)
        if "InternalEnergy" in self.data:
            return self.data["InternalEnergy"][:] * U
        return None

    @property
    def x_e(self) -> au.Quantity | None:
        """The simulation's electron abundance data in corresponding units."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        if "ElectronAbundance" in self.data:
            return self.data["ElectronAbundance"][:]
        return None

    @property
    def x_H(self) -> au.Quantity | None:
        """The simulation's neutral hydrogen abundance data in corresponding units."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        if "GFM_Metals" in self.data:
            return self.data["GFM_Metals"][:, 0][:]
        return None

    @property
    def x_HI(self) -> au.Quantity | None:
        """The simulation's ionized hydrogen abundance data in corresponding units."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        if "NeutralHydrogenAbundance" in self.data:
            return self.data["NeutralHydrogenAbundance"][:]
        return None

    @property
    def n_H(self) -> au.Quantity | None:
        """The simulation's hydrogen number density data in corresponding units."""
        if self.density is None:
            return None
        x_H = self.x_H
        perVol = au.cm ** (-3)
        # choose constants in favorable units to avoid overflows
        m_p = ac.m_p.to(au.u)
        n_H = x_H * self.density / m_p
        return n_H.to(perVol)

    @property
    def m_H(self) -> au.Quantity | None:
        """The simulation's hydrogen mass data in corresponding units."""
        if self.masses is None:
            return None
        m_H = self.x_H * self.masses
        return m_H.to(au.Msun)

    @property
    def n_HI(self) -> au.Quantity | None:
        """The simulation's ionized hydrogen number density data in corresponding units."""
        if self.x_HI is None or self.n_H is None:
            return None
        return self.x_HI * self.n_H

    @property
    def m_HI(self) -> au.Quantity | None:
        """The simulation's ionized hydrogen mass data in corresponding units."""
        if self.x_HI is None or self.m_H is None:
            return None
        m_HI = self.x_HI * self.m_H
        return m_HI.to(au.Msun)

    def kd_tree(
        self,
        k: int = 8,
        threads: int = 1,
        epsilon: float = 1.0e-4,
        as_float32: bool = False,
        verbose: bool = True,
    ) -> au.Quantity | NDArray:
        """Compute the distance of each gas particle to its k nearest neighbors.

        Args:
            k: Number of nearest neighbors.
            threads: Number of computing threads.
            epsilon: Small increase of the box size to avoid segfaults
            as_float32: If True, load simulation data as 32-bit floats (for efficiency)
            verbose: If True, print out results to the command line.

        Returns:
            (np.ndarray): The distance of each particle to its farthest neighbor
        """
        p = self.particle_positions
        if p is None:
            return None
        p = p.to(au.kpc)
        boxsize = (1 + epsilon) * self.boxsize.to(au.kpc)
        if verbose:
            print("Building KDTree...")
        t_ini = time.time()
        kdtree = sp.spatial.cKDTree(p, leafsize=16, boxsize=boxsize)
        if verbose:
            print(f"Time to build KDTree: {time.time() - t_ini:.3f} secs.")
        if verbose:
            print("Querying KDTree...")
        dist, _ = kdtree.query(p, k, workers=threads)
        if verbose:
            print(f"Time to querying KDTree: {time.time() - t_ini:.3f} secs.")
        if as_float32:
            return (dist[:, -1] * au.kpc).astype(np.float32)
        return dist[:, -1] * au.kpc


class TNGGalaxy(SPHGalaxy):
    """IllustrisTNG Galaxy simulation parser."""

    primary_hdf5_fields: dict[int, list[str]] = {
        0: [
            "Coordinates",
            "Density",
            "Masses",
            "Velocities",
            "InternalEnergy",
            "ElectronAbundance",
            "MagneticField",
        ],
        1: ["Coordinates", "Potential", "SubfindDMDensity", "Velocities"],
        4: ["Coordinates", "Masses", "Velocities", "SubfindDensity"],
    }
    optional_hdf5_fields: dict[int, list[str]] = {
        0: ["CenterOfMass", "GFM_Metals", "NeutralHydrogenAbundance"],
        1: ["SubfindDensity", "SubfindHsml", "SubfindVelDisp"],
        4: ["GFM_Metals"],
    }

    def __init__(self, base_path: str | Path, snapshot: int, halo_index: int, **kwargs):
        """Initialize a TNGGalaxy instance.

        Args:
            base_path (str): Base path to the Illustris(TNG) snapshots.
            snapshot (int): Snapshot ID {0-99}.
            halo_index (int): Halo index, index of the subhalo ID list.
            kwargs (dict): Additional keyword arguments
                - verbose (bool): If True, print information to the command line.
        """
        kwargs.setdefault("verbose", False)
        self.base_path = base_path
        self.snapshot = snapshot
        self._halo_index = halo_index
        self.subhalo_ids = self.subhalo_list(base_path, snapshot, verbose=kwargs["verbose"])
        self.subhalo = self.load_subhalo(self.halo_index, verbose=kwargs["verbose"])
        super().__init__(**kwargs)

    @property
    def halo_index(self):
        """The halo index getter."""
        if hasattr(self, "_halo_index"):
            return self._halo_index
        return 0

    @halo_index.setter
    def halo_index(self, index):
        """The halo index setter."""
        if hasattr(self, "_halo_index") and self._halo_index == index:
            return
        self._halo_index = index

        self.subhalo = self.load_subhalo(index) if hasattr(self, "subhalo") else None
        self.data = (
            self.load_data(as_float32=self.as_float32, verbose=self.verbose)
            if hasattr(self, "data")
            else None
        )

    @SPHGalaxy.p_idx.setter
    def p_idx(self, p_idx: int | str):
        """Setter for particle index."""
        SPHGalaxy.p_idx.fset(self, p_idx)

    def load_header(self) -> dict:
        """Load the header (overrides parent class method).

        Returns:
            (dict): a dictionary containing the headers from a snapshot and the
              corresponding group catalog.
        """
        header = {}
        header["catalog"] = tng.groupcat.load_header(self.base_path, self.snapshot)
        header["snapshot"] = tng.snapshots.load_header(self.base_path, self.snapshot)
        return header

    def load_cosmology(self, cosmo_pars: dict | None = None, in_place: bool = True) -> dict:
        """Load the cosmological parameters from the header dictionary.

        Note: This overrides the parent class method.

        Args:
            cosmo_pars: The default dictionary for the CosmoModel dataclass.
            in_place: Load CosmoModel into instanace property `cosmology` directly.

        Returns:
            (dict): a dictionary with cosmological parameters pulled the header.
        """
        cosmo_pars = {} if cosmo_pars is None else cosmo_pars
        cosmo_pars.setdefault("omega_l", self.header["snapshot"].get("OmegaLambda", 0.6911))
        cosmo_pars.setdefault("omega_m", self.header["snapshot"].get("Omega0", 0.3089))
        cosmo_pars.setdefault("omega_k", self.header["snapshot"].get("OmegaK", 0))
        cosmo_pars.setdefault("z", self.header["snapshot"].get("Redshift", None))
        cosmo_pars.setdefault("z", 1.0 / self.header["snapshot"].get("Time", 1) - 1)
        cosmo_pars.setdefault("h", self.header["snapshot"].get("HubbleParam", 0.6774))
        if in_place:
            self.cosmology = CosmoModel(**cosmo_pars)
        return cosmo_pars

    @staticmethod
    def subhalo_list(
        base_path: str,
        snapshot: int,
        filtered: bool = True,
        verbose: bool = False,
    ) -> NDArray:
        """Pull a list of subhalo IDs from the snapshot's FOF group catalog.

        Note: the indices of the list are halo IDs.

        Args:
            base_path: Base path to the Illustris(TNG) snapshots.
            snapshot: Snapshot ID {0-99}.
            filtered: If True, negative subhalo IDs are filtered out.
            verbose: If True, print out results to the command line.

        Returns:
            (list | np.ndarray): a list of subhalo i.e. galaxy IDs.
        """
        kwargs = {"fields": ["GroupFirstSub"]}
        if verbose:
            print(f"Searching group catalog: snapshot {snapshot}")
        galaxy_list = tng.groupcat.load_halos(base_path, snapshot, as_array=True, **kwargs)
        galaxy_list = galaxy_list[galaxy_list >= 0] if filtered else galaxy_list
        if verbose:
            N_g = galaxy_list.max()
            N_g += 0 in galaxy_list
            print(f"Found {N_g} groups in catalog: snapshot {snapshot}")
        return galaxy_list

    def load_subhalo(self, halo_index: int = 0, verbose: bool = False) -> dict:
        """Load the FOF subhalo metadata of a specified group ID.

        Args:
            halo_index: the group ID corresponding to a subhalo ID
            verbose: Print status information to the command-line.

        Returns:
            (dict): dictionary containing the FOF subhalo metadata
        """
        if halo_index < 0 or halo_index >= len(self.subhalo_ids):
            halo_index = 0
        sh_id = self.subhalo_ids[halo_index]
        if verbose:
            print(f"Loading group: {sh_id}")
        data = tng.groupcat.load_single(self.base_path, self.snapshot, subhalo_id=sh_id)
        return data

    @property
    def N_particles_type(self) -> list[int]:
        """Number of particles per type in the galaxy."""
        if hasattr(self, "subhalo"):
            return [int(i) for i in self.subhalo["SubhaloLenType"]]
        return []

    @property
    def N_particles(self) -> int:
        """Total number of particles in the galaxy."""
        if hasattr(self, "subhalo"):
            return int(self.subhalo["SubhaloLen"])
        return -1

    @property
    def center(self) -> au.Quantity:
        """Center position getter of the galaxy in units of L/h."""
        if not hasattr(self, "subhalo"):
            self.subhalo = self.load_subhalo(self.halo_index)
        L = self.units("l/h")
        if "SubhaloPos" in self.subhalo:
            return (self.subhalo["SubhaloPos"] * L).to(au.kpc)
        return None

    def load_data(
        self,
        halo_index: int | None = None,
        primary_fields: list[str] | None = None,
        optional_fields: list[str] | None = None,
        particle_type: str | None = None,
        **kwargs,
    ):
        """Load the snapshot's relevant subset data of the specified group ID.

        Args:
            halo_index: Galaxy/Halo ID in the simulation.
            primary_fields: The primary fields to load (less fields load faster)
            optional_fields: The secondary fields to load (less fields load faster)
            particle_type: Manually set the particle type.
            kwargs:
              - verbose (bool): Print information to the command-line.

        Returns:
            (dict): dictionary containing the relevant snapshot data
        """
        kwargs.setdefault("verbose", False)
        verbose = kwargs.pop("verbose")
        if particle_type is not None:
            p_idx = tng.util.pidx_from_ptype(particle_type)
        else:
            p_idx = self.p_idx
            particle_type = self.particle_type
        if primary_fields is None:
            primary_fields = self.primary_hdf5_fields[p_idx]
        if optional_fields is None:
            optional_fields = self.optional_hdf5_fields[p_idx]
        if halo_index is None or halo_index < 0 or halo_index >= len(self.subhalo_ids):
            halo_index = self.halo_index
        subset = tng.snapshots.snapshot_offsets(self.base_path, self.snapshot, halo_index, "Group")
        if verbose:
            print(f"Loading primary fields from snapshot [{particle_type}]: {self.snapshot}")
        data = tng.snapshots.load_snapshot(
            self.base_path,
            self.snapshot,
            particle_type,
            subset=subset,
            fields=primary_fields,
            **kwargs,
        )
        try:
            if verbose:
                print(f"Loading optional fields from snapshot [{particle_type}]: {self.snapshot}")
            data.update(
                tng.snapshots.load_snapshot(
                    self.base_path,
                    self.snapshot,
                    particle_type,
                    subset=subset,
                    fields=optional_fields,
                    as_array=False,
                    **kwargs,
                )
            )
        except Exception as ex:
            if ("Particle type" in ex.args[0]) and ("does not have field" in ex.args[0]):
                for f in optional_fields:
                    data[f] = None
            else:
                raise
        if particle_type == "gas" and ("GFM_Metals" not in data or data["GFM_Metals"] is None):
            data["GFM_Metals"] = np.array([[0.76]])
        if "CenterOfMass" not in data or data["CenterOfMass"] is None:
            data["CenterOfMass"] = data["Coordinates"]
        return data

    @property
    def temperature(self):
        """Temperature data getter in units of Kelvin."""
        # ~ O(1)
        x_H = self.x_H
        # choose constants in favorable units to avoid overflows
        gamma = 5.0 / 3
        m_p = ac.m_p.to(au.u)
        k_B = ac.k_B.to(au.eV / au.K)
        mu = 4 * m_p / (1 + 3 * x_H + 4 * x_H * self.x_e)
        T = (gamma - 1) * mu * self.internal_energy / k_B
        return T.to(au.K)

    @property
    def magnetic_field(self):
        """Magnetic vector field in cgs units of Gauss."""
        if not hasattr(self, "data"):
            self.data = self.load_data()
        srP = self.units("sqrtP")
        if "MagneticField" in self.data:
            return self.data["MagneticField"][:] * srP
        return None

    @property
    def magnetic_field_strength(self):
        """Magnetic field strength in units of SI Gauss."""
        bfield3D = self.magnetic_field * np.sqrt(4 * np.pi * ac.mu0)
        return np.linalg.norm(bfield3D, axis=-1).to(au.Gauss)

    @property
    def radii(self):
        """Particle radius distances assuming spherical shape."""
        # Voronoi cell volume
        vol = self.masses / self.density
        # estimate radii assuming spherical shape
        r_cell = np.power(0.75 * np.pi * vol, 1.0 / 3).to(au.kpc)
        return r_cell

    @property
    def r_cell(self):
        """Alias for radii."""
        return self.radii

    @property
    def hsml(self):
        """Average kernel smoothing length."""
        # hsml has in mind a cubic spline that =0 at h, I think
        # find_fwhm(CubicSplineKernel().kernel)
        hsml = 2.5 * self.r_cell  # * find_fwhm(CubicSplineKernel().kernel)
        return hsml

    def get_mapping_arrays(
        self,
        keys: list[str] | None = None,
        factors: list[float] = None,
        verbose: bool = False,
    ) -> list:
        """Fetch data (arrays or scalars) from a TNGGalaxy object given keys.

        Args:
            keys: Keys to fetch data.
            factors: Factors for modifying the fetched data.
            verbose: If True, print status updates to command line.

        Returns:
            (list): List of fetched data arrays or scalars.
        """
        if keys is None:
            keys = ["particle_positions", "masses", "radii"]
        if factors is None:
            factors = [1, 1, 3]
        vals = []
        for key, f in zip(keys, factors):
            if isinstance(key, tuple | list):
                try:
                    v1 = getattr(self, key[0])
                except Exception:
                    v1 = self.subhalo[key[0]]
                try:
                    v2 = getattr(self, key[1])
                except Exception:
                    v2 = self.subhalo[key[1]]
                v = v1 * v2 * f
            else:
                try:
                    v = getattr(self, key) * f
                except Exception:
                    v = self.subhalo[key] * f
            vals.append(v)
        if verbose:
            print(f"Loading arrays: {keys}...")
        return [*vals]

    def generate_map(
        self,
        keys: list[str] | None = None,
        factors: list[float] | None = None,
        use_half_mass_rad: bool = True,
        fh: float = 3,
        grid_size: int = 512,
        xaxis: int = 0,
        yaxis: int = 1,
        periodic: bool = True,
        assignment_func: Callable = voronoi_RT_2D,
        tracers: int | None = None,
        divisions: int | None = None,
        rot: list[int] | list[float] | None = None,
        verbose: bool = False,
    ) -> tuple[au.Quantity, au.Quantity, int] | Any:
        """Generate raytracing projection map.

        Args:
            keys: Keys to fetch data for projecting onto the map.
            factors: Factors for modifying the projection data.
            use_half_mass_rad:
              If True, the SubhaloHalfmassRad from the subfind catalog is used for
              selecting relevant particles. Otherwise, a fraction of the entire
              particle extent is used.
            fh: Expansion factor for the SPH particle radii.
            grid_size: The size of the maps/images. Default: 512.
            xaxis: Projection axis for x.
            yaxis: Projection axis for y.
            periodic: Use periodic boundary conditions for the projection.
            assignment_func: Mass assignment algorithm; one of
              [voronoi_RT_2D, voronoi_NGP_2D].
            tracers: Number of tracer particles to use for the Nearest Grid Point algorithm.
            divisions: Number of sphere divisions to use for the Nearest Grid Point algorithm.
            rot:
              Angles to rotate the particle positions.
            verbose: If True, print status updates to command line.

        Returns:
            (np.ndarray, np.ndarray, int): The projected map, the map extent, and
               number of particles projected.
        """
        L, M, T = au.Mpc, au.Msun, au.K
        projected = np.zeros((grid_size, grid_size), dtype=np.float64)
        if keys is None:
            keys = ["particle_positions", "masses", "radii", "center"]
        if factors is None:
            factors = [1, 1, fh, 1]
        if use_half_mass_rad:
            if "SubhaloHalfmassRadType" not in keys:
                keys += ["SubhaloHalfmassRadType"]
                factors += [self.units("l/h")]
            else:
                factors.insert(keys.index("SubhaloHalfmassRadType"), self.units("l/h"))
        if assignment_func not in [voronoi_RT_2D, voronoi_NGP_2D]:
            raise ValueError(f"Assignment function `{assignment_func.__name__}` not compatible.")
        uarrs = self.get_mapping_arrays(keys=keys, factors=factors, verbose=verbose)
        if rot is not None:
            rot_op = R.y(rot[1]) * R.x(rot[0])
            uarrs[0] = rot_op(uarrs[0]).astype(np.float32)
            uarrs[3] = rot_op(uarrs[3]).astype(np.float32)
        # Ngids = uarrs[0].shape[0]
        hmr = uarrs[4][0] if use_half_mass_rad else None  # always use gas (p_idx=0) hmr
        # hmr = (uarrs[4][obj.p_idx] if use_half_mass_rad else None)
        idcs, limits = indices_within_box(uarrs[0], uarrs[3], radius=hmr, verbose=verbose)
        hmr2 = limits[3] - limits[0]
        args = strip_ap_units(*uarrs[:3], *limits[:2], hmr2, mask=idcs, units=[L, M, T * M])
        if tracers is not None:
            args.append(tracers)
        else:
            args.append(xaxis)
        if divisions is not None:
            args.append(divisions)
        else:
            args.append(yaxis)
        if verbose:
            print(f"Raytracing particles with `{assignment_func.__name__}`...")
        try:
            assignment_func(projected, *args, periodic, verbose=True)
        except Exception as e:
            print(e)
            return projected * uarrs[1].unit / L**2
        projected = projected * uarrs[1].unit / L**2
        return projected, hmr2 / 2 * np.array([-1, 1, -1, 1]), idcs.shape[0]


class ArepoGalaxy(TNGGalaxy):
    """An Arepo Galaxy parser (alias to TNGGalaxy for now)."""


class GasolineGalaxy(SPHGalaxy):
    """A Gasoline Galaxy parser."""


def indices_within_box(
    pos: np.ndarray | au.Quantity,
    center: list | np.ndarray | au.Quantity,
    radius: float | au.Quantity = None,
    fraction: float = 1.0,
    verbose: bool = False,
) -> tuple[au.Quantity, list[au.Quantity]]:
    """Get particle indices within a box of given radius from the centre, e.g. half-mass radius.

    Args:
        pos: Particle positions to be filtered.
        center: Center position of the cube.
        radius: Radius, i.e. half-side of the cube.
        fraction: Box fraction for the default radius if not given.
        verbose: If True, print status updates to command line.

    Returns:
        (au.Quantity, list[au.Quantity]):
          Indices of the filtered particle positions and their 3D extent.
    """
    pos_extent = (
        pos[:, 0].max() - pos[:, 0].min(),
        pos[:, 1].max() - pos[:, 1].min(),
        pos[:, 2].max() - pos[:, 2].min(),
    )
    w = max(pos_extent)
    if verbose:
        print(f"Box extent:  {pos_extent[0]}   {pos_extent[1]}   {pos_extent[2]}")
    if radius is None:
        radius = 0.25 * fraction * w
    else:
        radius *= fraction
    if verbose:
        print(f"Radius: {radius}")
    xmin, xmax = center[0] - radius, center[0] + radius
    ymin, ymax = center[1] - radius, center[1] + radius
    zmin, zmax = center[2] - radius, center[2] + radius
    if verbose:
        print(f"Extent: {xmax - xmin}   {ymax - ymin}   {zmax - zmin}")
    indices = np.where(
        (pos[:, 0] > xmin)
        & (pos[:, 0] < xmax)
        & (pos[:, 1] > ymin)
        & (pos[:, 1] < ymax)
        & (pos[:, 2] > zmin)
        & (pos[:, 2] < zmax)
    )[0]
    if verbose:
        print(f"Selected particles: {indices.shape[0]:,} / {pos.shape[0]:,}")
    return indices, [xmin, ymin, zmin, xmax, ymax, zmax]


def strip_ap_units(
    *args,
    mask: list | None = None,
    units: list[au.Unit] | None = None,
    dtype: Any = np.float32,
) -> list:
    """Remove astropy units from data arrays or scalars.

    Args:
        args: Data arrays or scalars with astropy units.
        mask: Mask for the data arrays.
        units: Astropy units to be stripped.
        dtype: Data type of the stripped data array.

    Returns:
        (list[np.ndarray]): Of astropy units stripped data arrays or scalars.
    """
    arg_ls = list(args)
    for i, arg in enumerate(arg_ls):
        if units is not None:
            for u in units:
                if arg_ls[i].unit.is_equivalent(u):
                    arg_ls[i] = arg_ls[i].to(u)
        arg_ls[i] = arg_ls[i].value
        if isinstance(arg_ls[i], np.ndarray):
            if mask is not None:
                arg_ls[i] = arg_ls[i][mask].astype(dtype)
    return arg_ls


if __name__ == "__main__":
    import pprint

    # Illustris loads
    tng_path = "/scratch/data/illustris/tng50-1"
    tng_id = 99
    tng_src = TNGGalaxy(tng_path, tng_id, halo_index=10, as_float32=True)
    pprint.pprint(tng_src.header)
    pprint.pprint(tng_src.cosmology)
    pprint.pprint(tng_src.subhalo.keys())
    pprint.pprint(tng_src.data.keys())
    # dists = tng_src.kd_tree(k=8, threads=4)
    # print(dists.shape, dists)
