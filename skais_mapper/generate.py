# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate maps from simulations."""

from pathlib import Path
import gc
import datetime
import csv
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import numpy as np
from matplotlib.colors import Colormap
from astropy import units as au

# from astropy.visualization import ImageNormalize, MinMaxInterval
# from astropy.visualization import AsinhStretch
# from astropy.visualization import LogStretch
from typing import Any
from collections.abc import Iterable, Callable
import skais_mapper
from skais_mapper.utils import get_run_id, compress_encode
from skais_mapper.data import Img2H5Buffer
from skais_mapper.raytrace import voronoi_NGP_2D
from skais_mapper.simobjects import TNGGalaxy
from skais_mapper.plotting import plot_image


def map_TNG_sample(
    obj: TNGGalaxy,
    gid: int,
    group: str = "gas",
    projected_unit: au.Unit = None,
    cmap: Colormap | None = None,
    hdf5_file: str | Path | None = None,
    hdf5_save: bool = True,
    npy_save: bool = False,
    png_save: bool = False,
    subdir_save: bool = False,
    grid_size: int = 512,
    fh: float = 3,
    rot: list[float] | tuple[float, float] | None = None,
    xaxis: int = 0,
    yaxis: int = 1,
    periodic: bool = False,
    rng_seed: int = 42,
    flag_lim: float = 0,
    flag_N: int = 64,
    post_hook: Callable | None = None,
    dry_run: bool = False,
    verbose: bool = True,
):
    """Project a subfind ID from an IllstrisTNG snapshot.

    Args:
        obj: Instance at a set snapshot, pointing at set subfind ID.
        gid: Galaxy/halo index.
        group: Galaxy property of the map, one of [gas,star,gas,hi,hi/21cm,temp,bfield].
        projected_unit: Units in which the map is to be projected.
        cmap: Colormap for map plot.
        hdf5_file: Basename of the HDF5 file.
        hdf5_save: If True, save map to HDF5 file.
        npy_save: If True, save map as numpy binary files.
        png_save: If True, save map plot as PNG file.
        subdir_save: If True, saves numpy binary and PNG files in corresponding subdirectories.
        grid_size: The size of the maps/images. Default: 512.
        fh: Expansion factor for the SPH particle radii.
        rot: Angles by which the sample is rotated before projection.
        xaxis: Projection axis for x.
        yaxis: Projection axis for y.
        periodic: Use periodic boundary conditions for the projection (for metadata).
        rng_seed: Seed for the random number generation.
        flag_lim: Flag the map in the metadata if N pixel values fall below the limit.
        flag_N: The number of pixels before an image is flagged.
        post_hook: Post projection callback function to, e.g., rescale the map.
        dry_run: If True, nothing is saved and expensive computation is skipped.
        verbose: If True, print status updates to command line.
    """
    # gather settings
    kwargs: dict[str, Any] = {
        "use_half_mass_rad": True,
        "fh": fh,
        "grid_size": grid_size,
        "xaxis": xaxis,
        "yaxis": yaxis,
        "periodic": periodic,
        "rot": rot,
        "verbose": verbose,
    }
    # set up configs for group
    if group == "gas":
        kwargs["keys"] = ["particle_positions", "masses", "radii", "center"]
        if projected_unit is None:
            projected_unit = au.Msun / au.kpc**2
    elif group == "hi":
        kwargs["keys"] = ["particle_positions", "m_HI", "radii", "center"]
        if projected_unit is None:
            projected_unit = au.Msun / au.kpc**2
    elif group == "hi/21cm":
        kwargs["keys"] = ["particle_positions", "m_HI", "radii", "center"]
        kwargs["assignment_func"] = voronoi_NGP_2D
        kwargs["tracers"] = 128
        kwargs["divisions"] = 2
        pixel_size = 1.0 / grid_size
        z, h, H0, Hz = (
            obj.cosmology.z,
            obj.cosmology.h,
            obj.cosmology.H0,
            obj.cosmology.H(obj.cosmology.a),
        )
        sigma_crit = obj.cosmology.rho_crit

        def post_hook_21cm(x, y):
            return (
                189
                * au.mK
                * h
                * (1 + z) ** 2
                * (H0 / Hz)
                * x
                / ((y[1] - y[0]) * pixel_size * sigma_crit)
            )

        if post_hook is None:
            post_hook = post_hook_21cm
        if projected_unit is None:
            projected_unit = au.mK
        flag_lim, flag_N = 0, int(grid_size**2 / 10)
    elif group == "temp":
        kwargs["keys"] = [
            "particle_positions",
            ("masses", "temperature"),
            "radii",
            "center",
        ]
        if projected_unit is None:
            projected_unit = au.K
    elif group == "bfield":
        kwargs["keys"] = [
            "particle_positions",
            ("masses", "magnetic_field_strength"),
            "radii",
            "center",
        ]
        if projected_unit is None:
            projected_unit = au.Gauss
    elif group == "star":
        kwargs["keys"] = ["particle_positions", "masses", "radii", "center"]
        if projected_unit is None:
            projected_unit = au.Msun / au.kpc**2
    elif group == "dm":
        kwargs["keys"] = ["particle_positions", "masses", "radii", "center"]
        if projected_unit is None:
            projected_unit = au.Msun / au.kpc**2
    else:
        raise ValueError(f"Map group type {group} is not known.")
    if isinstance(kwargs["keys"][1], tuple | list):
        keys = kwargs.pop("keys")
        quantity, extent, N = obj.generate_map(keys=keys, **kwargs)
        keys[1] = keys[1][0]
        weight_map, _, _ = obj.generate_map(keys=keys, **kwargs)
        projected = np.zeros_like(quantity.value)
        np.place(
            projected,
            weight_map.value != 0,
            quantity.value[weight_map.value != 0] / weight_map.value[weight_map.value != 0],
        )
        projected *= quantity.unit / weight_map.unit
    else:
        # allocate arrays and raytrace
        projected, extent, N = obj.generate_map(**kwargs)
    if post_hook is not None:
        projected = post_hook(projected, extent)
    # convert to chosen units
    projected = projected.to(projected_unit)
    # check for potential problems
    flag = 0
    if np.sum(projected.value < flag_lim) > flag_N:
        print("Potential issue with projection, flagging image...")
        flag = 1
    has_bh = obj.N_particles_type[-1]
    # plot data
    rot_str = f"_rotxy.{rot[0]}.{rot[1]}" if rot is not None else ""
    bname = f"{str(Path(group).stem)}_tng50-1.{obj.snapshot:02d}.gid.{obj.halo_index:07d}{rot_str}"
    md = {
        "class": group,
        "gid": obj.halo_index,
        "snapshot": obj.snapshot,
        "units": f"{projected.unit}",
        "extent": extent.value,
        "units_extent": f"{extent.unit}",
        "name": bname,
        "num_particles": N,
        "rotxy": rot if rot is not None else (0, 0),
        "N_particle_flag": flag,
        "has_bh": has_bh,
        "rng_seed": rng_seed,
    }
    plot_image(
        projected,
        info=md,
        cbar=True,
        norm="log" if group != "hi/21cm" else None,
        savefig=png_save and not dry_run,
        path=hdf5_file.parent / group / "png" / f"{bname}.png"
        if subdir_save
        else hdf5_file.parent / f"{bname}.png",
        show=dry_run,
        close=not dry_run,
        verbose=verbose
    )
    # save data
    if npy_save:
        npname = f"{bname}.units.{projected.unit}.extent.{extent[1] - extent[0]:4.8f}.npy".replace(
            " ", ""
        ).replace("/", "_")
        np_dir = hdf5_file.parent
        if subdir_save:
            np_dir = np_dir / group / "npy"
        if not dry_run:
            if not np_dir.exists():
                np_dir.mkdir(parents=True)
            np.save(np_dir / npname, projected.value)
        if verbose:
            print(f"Saving to [npy]: {np_dir / npname}")
    if hdf5_save and hdf5_file is not None:
        if subdir_save:
            hdf5_file = hdf5_file.parent / group / "hdf5" / hdf5_file.name
            if not hdf5_file.parent.exists() and not dry_run:
                hdf5_file.parent.mkdir(parents=True)
        img2h5 = Img2H5Buffer(target=hdf5_file, size="2G")
        img_target = f"{str(hdf5_file)}"
        mdt_target = f"{str(hdf5_file)}"
        img_h5group = f"{group}/images"
        mdt_h5group = f"{group}/metadata/"
        if not dry_run:
            img_h5group = f"{group}/images"
            img2h5.inc_write(img_target, data=projected.value, group=img_h5group, verbose=verbose)
            mdt_h5group = f"{group}/metadata/{img2h5.index:04d}"
            img2h5.inc_write(mdt_target, data=md, group=mdt_h5group, verbose=verbose)
        if verbose:
            print(f"Saving to [hdf5]: {img_target}:{img_h5group}")
            print(f"Saving to [hdf5]: {mdt_target}:{mdt_h5group}")


def map_TNG_galaxies(
    snapshots: list[int],
    gids: int | Iterable[int],
    groups: list[str] | None = None,
    output: str | None = None,
    src_dir: str | None = None,
    sim_type: str = "illustris/tng50-1",
    csv_file: str | Path | None = None,
    part_max: int | None = None,
    part_min: int | None = 20_000,
    retries: int | None = None,
    subfind_limit: int | None = 15_000,
    grid_size: int = 512,
    rotations: np.ndarray | None = None,
    random_rotations: bool = True,
    rng_seed: int = 42,
    dry_run: bool = False,
    verbose: bool = True,
):
    """Generate any number of maps from an IllustrisTNG snapshot(s).

    Args:
        snapshots: Snapshots number of the IllustrisTNG run.
        gids: Subfind IDs, i.e. galaxies, from which to generate maps.
        groups: Galaxy properties to map, e.g. star, gas, or dm.
        output: Output filename. Can have format fields '{}' for group and snapshot.
        src_dir: Path to the root of the simulation snapshots.
        sim_type: Simulation type (should correspond to the subpath in `src_dir`).
        csv_file: Path the the CSV file used for logging supplemental information.
        part_max: Maximum number of particles to use for map generation.
        part_min: Minimum number of particles to use for map generation.
        retries: If not None, sets the maximum number of replacement candidates for skipped groups.
        subfind_limit: If not None, sets the maximum subfind ID allowed as replacement.
        grid_size: The size of the maps/images. Default: 512.
        rotations: List of angle pairs (theta, phi) per rotation for each subfind ID;
          e.g. for 4 separate rotations per subfind ID, its shape is (len(gids), 4, 2).
        random_rotations: If True, use random rotations (2 per subfind ID) to
          augment the dataset.
        rng_seed: Random number seed.
        dry_run: If True, nothing is saved and expensive computation is skipped.
        verbose: If True, print status updates to command line.
    """
    snapshots = list(snapshots)
    gids = list(range(gids)) if isinstance(gids, int) else list(gids)
    # list of groups to generate
    if groups is None:
        groups = ["gas"]
    Ng = len(gids) * len(groups)
    skip_count = [0] * (Ng // len(groups))
    # gather paths
    src_path = Path(src_dir) if src_dir is not None else Path("./simulations")
    tng_path = src_path / sim_type
    if output is None:
        hdf5_file = Path(
            str(datetime.datetime.now().date()).replace("-", "")
            + f"_{tng_path.name}.{{}}.2D.{{}}.hdf5"
        )
    else:
        hdf5_file = Path(output)
    if csv_file is None:
        csv_file = hdf5_file.parent / f"{get_run_id()}.group_particles.csv"
    if not csv_file.exists() and not dry_run:
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        with csv_file.open(mode="w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(
                [
                    "snapshot",
                    "gid",
                    "N_particles_gas",
                    "N_particles_dm",
                    "N_particles_stars",
                    "N_particles_bh",
                ]
            )

    # Resolve paths and verify existance
    print("Resolved paths:")
    print("Source path:", src_path.resolve())
    print(f"{src_path.resolve()}: exists", src_path.exists())
    print("tng50-1:    ", tng_path.resolve())
    print(f"{tng_path.resolve()}: exists", tng_path.exists())
    print("Output:     ", hdf5_file)

    # Precompute all rotations
    if random_rotations:
        rng = np.random.default_rng(rng_seed)
        N_rot = Ng
        rotations = np.stack(
            (
                rng.integers(25, 180, size=N_rot),
                rng.integers(25, 90, size=N_rot),
                rng.integers(90, 270, size=N_rot),
            )
        )
        rotations = np.vstack(
            (
                rotations,
                rotations[0] + rng.integers(20, 40, size=N_rot),
                rotations[1] + rng.integers(70, 110, size=N_rot),
                rotations[2] + rng.integers(40, 90, size=N_rot),
            )
        ).T.reshape(Ng, 3, 2)

    # Generation loop
    for snap_id in snapshots:
        # run through galaxies
        for i, gid in enumerate(gids):
            if Ng < 0:
                break
            angles = [] if rotations is None else rotations[i]
            for j, group in enumerate(groups):
                p_group = group if group in ["gas", "star", "dm"] else "gas"
                tng_src = TNGGalaxy(
                    tng_path,
                    snap_id,
                    halo_index=gids[0],
                    particle_type=p_group,
                    as_float32=True,
                    verbose=True,
                )
                if gid != tng_src.halo_index:
                    tng_src.halo_index = gid
                if p_group != tng_src.particle_type:
                    tng_src.particle_type = group
                print(f"\n# Snapshot {snap_id}, subhalo {gid}, {group}")
                # check if number of particles in halo is within accepted range
                if (part_max is not None and part_max < tng_src.N_particles_type[0]) or (
                    part_min is not None and tng_src.N_particles_type[0] < part_min
                ):
                    if verbose:
                        print(
                            "Skipping candidate due to low particle number"
                            f" {tng_src.N_particles_type[0]}..."
                        )
                    # add another group candidate below the limit
                    if retries is not None and skip_count[-1] >= retries:
                        Ng -= 1
                    elif subfind_limit is not None:
                        while gid in gids and gid <= subfind_limit:
                            gid += 1
                        gids.append(gid)
                        if rotations is not None:
                            rotations = np.concatenate(
                                (rotations, rotations[i][np.newaxis, ...]), axis=0
                            )
                        skip_count.append(skip_count[-1] + 1)
                    break
                Ng -= 1

                # construct actual hdf5 filename
                if str(hdf5_file).count("{") == 2:
                    out_hdf5 = Path(str(hdf5_file).format(snap_id, group.replace("/", ".")))
                elif str(hdf5_file).count("{") == 1:
                    out_hdf5 = Path(str(hdf5_file).format(group.replace("/", ".")))
                else:
                    out_hdf5 = hdf5_file
                # generate maps, plots, and save to files
                map_TNG_sample(
                    tng_src,
                    gid,
                    group=group,
                    hdf5_file=out_hdf5,
                    grid_size=grid_size,
                    fh=3 if group == "dm" else 2 if group == "star" else 1,
                    rng_seed=rng_seed,
                    rot=None,
                    hdf5_save=True,
                    npy_save=True,
                    png_save=True,
                    subdir_save=True,
                    dry_run=dry_run,
                    verbose=verbose,
                )
                for theta, phi in angles:
                    map_TNG_sample(
                        tng_src,
                        gid,
                        group=group,
                        hdf5_file=out_hdf5,
                        grid_size=grid_size,
                        fh=3 if group == "dm" else 2 if group == "star" else 1,
                        rng_seed=rng_seed,
                        rot=(theta, phi),
                        hdf5_save=True,
                        npy_save=True,
                        png_save=True,
                        subdir_save=True,
                        dry_run=dry_run,
                        verbose=verbose,
                    )
            if csv_file.exists():
                with open(csv_file, "a", newline="") as fcsv:
                    writer = csv.writer(fcsv)
                    writer.writerow(
                        [snap_id]
                        + [gid]
                        + tng_src.N_particles_type[0:2]
                        + tng_src.N_particles_type[4:],
                    )
            if verbose:
                print(
                    f"Number of particles in group: {tng_src.N_particles_type[0]} [gas]"
                    f" | {tng_src.N_particles_type[tng_src.p_idx]} [{group}]"
                )
            gc.collect()


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(cfg: DictConfig | dict):
    """Main routine for generating any number of maps from a simulation snapshot(s).."""
    log = logging.getLogger(__name__)
    output_dir = HydraConfig.get().runtime.output_dir
    opt = instantiate(cfg, _convert_="all")
    if not cfg.exclude_git_state:
        log.info(f"Git state: {skais_mapper.GIT_STATE}")
    if cfg.include_git_diff:
        for d in skais_mapper.GIT_DIFF:
            log.info(f"Git diff: {compress_encode(d)}")
    log.info(f"Job id: {opt['run_id']}")
    log.info(f"Output directory: {output_dir}")
    if cfg.verbose:
        print("Configuration:")
        print(OmegaConf.to_yaml(cfg))

    if cfg.save_configs:
        hydra_subdir = Path(HydraConfig.get().output_subdir)
        src_file = hydra_subdir / "config.yaml"
        dst_file = Path(opt["output"])
        if dst_file.suffix not in [".yaml", ".yml"]:
            dst_file = Path(f"./{opt['run_id']}.yaml")
        dst_file.write_bytes(src_file.read_bytes())

    if "tng" in opt["simulation_type"].lower():
        np.random.seed(opt["random_seed"])
        map_TNG_galaxies(
            opt["snapshots"],
            range(*opt["num_samples"]),
            groups=opt["groups"],
            sim_type=opt["simulation_type"],
            output=opt["output"],
            src_dir=opt["source"],
            part_min=opt["part_min"],
            retries=opt["retries"],
            subfind_limit=opt["subfind_limit"],
            rng_seed=opt["random_seed"],
            grid_size=opt["grid_size"],
            dry_run=opt["dry_run"],
            verbose=opt["verbose"],
        )
