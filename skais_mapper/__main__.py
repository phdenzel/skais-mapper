#!/usr/bin/env python
"""
skais_mapper.__main__ entry points

@author: phdenzel
"""

import sys
import gc
import datetime
import csv
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from typing import Optional, Any, Iterable
import numpy as np
from matplotlib.colors import Colormap  # , LogNorm
from astropy import units as au

# from astropy.visualization import ImageNormalize, MinMaxInterval
# from astropy.visualization import AsinhStretch
# from astropy.visualization import LogStretch
import skais_mapper
from skais_mapper.utils import SkaisColorMaps
from skais_mapper.data import Img2H5Buffer
from skais_mapper.raytrace import voronoi_NGP_2D
from skais_mapper.simobjects import TNGGalaxy, plot_map


def parse_args(return_parser: bool = False, **kwargs) -> dict:
    """Parse arguments.

    Args:
        return_parser: If True, returns parser instead of parsed argument dictionary.
        kwargs: Additional keyword arguments for compatibility.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, add_help=False)
    subparsers = parser.add_subparsers()
    # General arguments
    parser.add_argument(
        "-i", "--config-file", "--config", type=Path, help="Configuration file to be loaded."
    )
    parser.add_argument(
        "--exclude-git-state",
        action="store_true",
        help="Do not include git state in the configuration.",
    )
    parser.add_argument(
        "--include-git-diff", action="store_true", help="Include git diff in the configuration."
    )
    parser.add_argument("-o", "--output", type=Path, help="Output file or directory.")
    parser.add_argument(
        "-s",
        "--src-dir",
        "--src",
        type=Path,
        help="Source (root) directory where simulations are stored.",
    )
    parser.add_argument(
        "--config-dir",
        "--save-config",
        type=Path,
        default=Path("./"),
        help="Directory where the configuration is saved.",
    )
    parser.add_argument(
        "-n", "--dry-run", action="store_true", help="For testing (nothing is saved if True)."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Set verbosity.")

    # Generate arguments
    parser_generate = subparsers.add_parser(
        "generate", aliases=["g", "gen", "generate-maps"], parents=[parser]
    )
    parser_generate.set_defaults(func=generate)
    parser_generate.add_argument(
        "-t",
        "--simulation-type",
        "--sim-type",
        choices=["illustris/tng50-1", "illustris/tng100-1", "illustris/tng300-1"],
        default="illustris/tng50-1",
        help="Simulation type to be mapped (so far only 'illustris/tng*' has been implemented).",
    )
    parser_generate.add_argument(
        "--snapshots",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[99],
        help="Simulations snapshot to be mapped (comma-separated list of integers).",
    )
    parser_generate.add_argument(
        "--num-samples",
        type=lambda n: [int(item) for item in n.split(",")],
        default=[1000],
        help="Number range of samples to be mapped (comma-separated list of integers).",
    )
    parser_generate.add_argument(
        "-g",
        "--groups",
        type=lambda s: [item for item in s.split(",")],
        default=["gas"],
        help=(
            "Group/quantity to be mapped; any of ['dm','star','gas','hi','hi/21cm','temp','bfield']"
        ),
    )
    parser_generate.add_argument(
        "--grid-size", "--gs", type=int, default=512, help="Grid size of the raytracing maps."
    )
    parser_generate.add_argument(
        "--retries",
        type=int,
        default=100,
        help="Number of retries (after minimum particle skips) before the job is terminated.",
    )
    parser_generate.add_argument(
        "--part-max",
        "--particle-upper-limit",
        type=int,
        help="Maximum number of particles allowed to use for map generation.",
    )
    parser_generate.add_argument(
        "--part-min",
        "--particle-limit",
        "--particle-lower-limit",
        type=int,
        default=10_000,
        help="Minimum number of particles to use for map generation.",
    )
    parser_generate.add_argument(
        "--subfind-limit",
        type=int,
        default=11_000,
        help="Minimum number of particles to use for map generation.",
    )

    # Configuration arguments
    parser_config = subparsers.add_parser(
        "configure",
        aliases=["c", "conf", "config", "configuration"],
        parents=[parser_generate],
        add_help=False
    )
    parser_config.set_defaults(func=configure)

    if return_parser:
        return parser
    # Convert to dict
    args, _ = parser.parse_known_args()
    configs = vars(args)
    if "config_file" in configs \
       and configs["config_file"] is not None \
       and configs["config_file"].exists():
        configs = skais_mapper.utils.load_config(configs["config_file"])
    return configs


def map_TNG_sample(
    obj: TNGGalaxy,
    gid: int,
    group: str = "gas",
    projected_unit: au.Unit = None,
    cmap: Optional[Colormap] = None,
    hdf5_file: Optional[str | Path] = None,
    hdf5_save: bool = True,
    npy_save: bool = False,
    png_save: bool = False,
    subdir_save: bool = False,
    grid_size: int = 512,
    fh: float = 3,
    rot: Optional[list[float] | tuple[float, float]] = None,
    xaxis: int = 0,
    yaxis: int = 1,
    periodic: bool = False,
    rng_seed: int = 42,
    flag_lim: float = 0,
    flag_N: int = 64,
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
    no_log = False
    post_hook = None
    # set up configs for group
    if group == "gas":
        kwargs["keys"] = ["particle_positions", "masses", "radii", "center"]
        if projected_unit is None:
            projected_unit = au.Msun / au.kpc**2
        if cmap is None:
            cmap = getattr(SkaisColorMaps, "gaseous")
        cbar_label = "log " + "\u03a3" + r"$_{\mathrm{gas}}$ "
    elif group == "hi":
        kwargs["keys"] = ["particle_positions", "m_HI", "radii", "center"]
        if projected_unit is None:
            projected_unit = au.Msun / au.kpc**2
        if cmap is None:
            cmap = getattr(SkaisColorMaps, "gaseous")
        cbar_label = "log " + "\u03a3" + r"$_{\mathrm{HI}}$ "
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

        def post_hook(x, y):
            return (
                189
                * au.mK
                * h
                * (1 + z) ** 2
                * (H0 / Hz)
                * x
                / ((y[1] - y[0]) * pixel_size * sigma_crit)
            )

        if projected_unit is None:
            projected_unit = au.mK
        if cmap is None:
            cmap = getattr(SkaisColorMaps, "nava")
        cbar_label = r"T$_{\mathrm{b}}$ "
        flag_lim, flag_N = 0, int(grid_size**2 / 10)
        no_log = True
    elif group == "temp":
        kwargs["keys"] = [
            "particle_positions",
            ("masses", "temperature"),
            "radii",
            "center",
        ]
        if projected_unit is None:
            projected_unit = au.K
        if cmap is None:
            cmap = getattr(SkaisColorMaps, "phoenix")
        cbar_label = "log T "
    elif group == "bfield":
        kwargs["keys"] = [
            "particle_positions",
            ("masses", "magnetic_field_strength"),
            "radii",
            "center",
        ]
        if projected_unit is None:
            projected_unit = au.Gauss
        if cmap is None:
            cmap = getattr(SkaisColorMaps, "gravic")
        cbar_label = "log |B| "
    elif group == "star":
        kwargs["keys"] = ["particle_positions", "masses", "radii", "center"]
        if projected_unit is None:
            projected_unit = au.Msun / au.kpc**2
        if cmap is None:
            cmap = getattr(SkaisColorMaps, "hertzsprung")
        cbar_label = "log " + "\u03a3" + r"$_{\mathrm{star}}$ "
    elif group == "dm":
        kwargs["keys"] = ["particle_positions", "masses", "radii", "center"]
        if projected_unit is None:
            projected_unit = au.Msun / au.kpc**2
        if cmap is None:
            cmap = getattr(SkaisColorMaps, "obscura")
        cbar_label = "log " + "\u03a3" + r"$_{\mathrm{dm}}$ "
    if isinstance(kwargs["keys"][1], (tuple, list)):
        keys = kwargs.pop("keys")
        quantity, extent, N = obj.generate_map(keys=keys, **kwargs)
        keys[1] = keys[1][0]
        weight_map, _, _ = obj.generate_map(keys=keys, **kwargs)
        projected = np.zeros_like(quantity.value)
        np.place(
            projected,
            weight_map.value != 0,
            quantity.value[weight_map.value != 0] / weight_map.value[weight_map.value != 0])
        projected *= quantity.unit / weight_map.unit
    else:
    # allocate arrays and raytrace
        projected, extent, N = obj.generate_map(**kwargs)
    if post_hook is not None:
        projected = post_hook(projected, extent)
    # convert to chosen units
    projected = projected.to(projected_unit)
    cbar_label += f"[{projected.unit}]"
    # check for potential problems
    flag = 0
    if np.sum(projected.value < flag_lim) > flag_N:
        print("Potential issue with projection, flagging image...")
        flag = 1
    has_bh = obj.N_particles_type[-1]
    # plot data
    rot_str = f"_rotxy.{rot[0]}.{rot[1]}" if rot is not None else ""
    bname = f"{str(Path(group).stem)}_tng50-1.{obj.snapshot:02d}.gid.{obj.halo_index:07d}{rot_str}"
    plot_map(
        projected,
        extent,
        group=group,
        out_path=hdf5_file.parent,
        subdir_save=subdir_save,
        basename=bname,
        cbar_label=cbar_label,
        cmap=cmap,
        no_log=no_log,
        savefig=png_save and not dry_run,
        show=dry_run,
        verbose=verbose,
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
            print(f"Saving to [npy]: {npname}")
    if hdf5_save and hdf5_file is not None:
        if subdir_save:
            hdf5_file = hdf5_file.parent / group / "hdf5" / hdf5_file.name
            if not hdf5_file.parent.exists() and not dry_run:
                hdf5_file.parent.mkdir(parents=True)
        img2h5 = Img2H5Buffer(target=hdf5_file, size="2G")
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
    groups: Optional[list[str]] = None,
    output: Optional[str] = None,
    src_dir: Optional[str] = None,
    sim_type: str = "illustris/tng50-1",
    part_max: Optional[int] = None,
    part_min: Optional[int] = 20_000,
    retries: Optional[int] = None,
    subfind_limit: Optional[int] = 15_000,
    grid_size: int = 512,
    rotations: Optional[np.ndarray] = None,
    random_rotations: bool = True,
    rng_seed: int = 42,
    dry_run: bool = False,
    verbose: bool = True,
):
    """
    Generate any number of maps from an IllustrisTNG snapshot(s).

    Args:
        snapshots: Snapshots number of the IllustrisTNG run.
        gids: Subfind IDs, i.e. galaxies, from which to generate maps.
        groups: Galaxy properties to map, e.g. star, gas, or dm.
        output: Output filename. Can have format fields '{}' for group and snapshot.
        src_dir: Path to the root of the simulation snapshots.
        sim_type: Simulation type (should correspond to the subpath in `src_dir`).
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
    skip_count = [0]*(Ng//len(groups))
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
    csv_file = hdf5_file.parent / f"{skais_mapper.utils.get_run_id()}.group_particles.csv"
    if not csv_file.exists() and not dry_run:
        with open(csv_file, "w", newline="") as fcsv:
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


def generate():
    """CLI entry point for generating any number of maps from a simulation snapshot(s)."""
    rng_seed = 42
    np.random.seed(rng_seed)

    # skais-mapper-generate
    if sys.argv[0].endswith("skais-mapper-generate"):
        sys.argv.insert(1, "generate")
        sys.argv[0] = sys.argv[0].replace("-generate", "")

    # Options
    configs = configure()

    # Loop over all samples in all snapshots
    if "tng" in configs["simulation_type"].lower():
        map_TNG_galaxies(
            configs["snapshots"],
            range(*configs["num_samples"]),
            groups=configs["groups"],
            sim_type=configs["simulation_type"],
            output=configs["output"],
            src_dir=configs["src_dir"],
            retries=configs["retries"],
            part_min=configs["part_min"],
            subfind_limit=configs["subfind_limit"],
            rng_seed=rng_seed,
            grid_size=configs["grid_size"],
            dry_run=configs["dry_run"],
            verbose=configs["verbose"],
        )


def configure():
    """CLI entry point for creating configuration files."""
    if sys.argv[0].endswith("skais-mapper-configure"):
        sys.argv.insert(1, "configure")
        sys.argv[0] = sys.argv[0].replace("-configure", "")

    configs = parse_args()
    skais_mapper.utils.print_config(configs, **configs)
    if not configs["dry_run"]:
        if "config_file" in configs \
           and configs["config_file"] is not None:
            skais_mapper.utils.save_config(configs, configs["config_file"])
        else:
            skais_mapper.utils.save_config(configs, configs["config_dir"])
    return configs


def main():
    """Main CLI skais-mapper entry point."""
    parser = parse_args(return_parser=True)
    args, _ = parser.parse_known_args()
    configs = vars(args)
    if "func" in configs:
        configs["func"]()
    else:
        parser.print_help()


if __name__ == "__main__":
    generate()
