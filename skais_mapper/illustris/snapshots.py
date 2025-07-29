# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Illustris simulation snapshot file i/o.

Adapted from: https://github.com/illustristng/illustris_python
"""

import os
import numpy as np
from tqdm import trange
from skais_mapper.illustris.util import IllustrisH5File, pidx_from_ptype
from numpy.typing import NDArray
import h5py
from skais_mapper.illustris.groupcat import get_path as get_cat_path
from skais_mapper.illustris.groupcat import get_offset_path


def get_path(base_path: str, snapshot: int, partition: int = 0) -> str:
    """Get absolute path to a snapshot HDF5 file (modify as needed).

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        partition: Subfile partition ID {0-600+}.

    Returns:
        (str): Absolute path to a snapshot HDF5 file
    """
    snap_dir = os.path.join(base_path, f"snapshots/{snapshot:03d}")
    filepath = os.path.join(snap_dir, f"snap_{snapshot:03d}.{partition:d}.hdf5")
    filepath_alt = filepath.replace("snap_", "snapshot_")
    if os.path.isfile(filepath):
        return filepath
    return filepath_alt


def particle_numbers(header: h5py.Group, n_types: int = 6) -> NDArray[np.int64]:
    """Calculate the number of particles of all types given a snapshot header.

    Args:
        header: Header of the snapshot HDF5 file.
        n_types: Number of particle types (almost always 6).

    Returns:
        (np.ndarray[np.int64]): Number of particles for each type.
    """
    if "NumPart_Total_HighWord" not in header:
        return header["NumPart_Total"]  # new u64 convention
    n_part = np.zeros(n_types, dtype=np.int64)
    for j in range(n_types):
        n_part[j] = header["NumPart_Total"][j] | (header["NumPart_Total_HighWord"][j] << 32)
    return n_part


def load_snapshot(
    base_path: str,
    snapshot: int,
    ptype: str,
    fields: list[str] | str | None = None,
    mdi: list[int] | int | None = None,
    subset: dict | None = None,
    as_float32: bool = False,
    as_array: bool = True,
    with_pbar: bool = True,
) -> dict | NDArray | None:
    """Load a subset of fields of a snapshot for all particles of a given type.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        ptype: Particle type description string;
            e.g. 'gas', 'dm', 'stars', '1', '2', etc.
        fields: Fields to be loaded for the corresponding ptype,
            e.g. ['Coordinates', 'Masses'] or 'NeutralHydrogenAbundance'.
        mdi: Multi-dimensional indices for fields; must be the
            same length as fields. E.g. fields = ['Coordinates', 'Masses'] and
            mdi = [1, None] returns a 1D array of y-coordinates and masses instead
            of a 3D array of coordinates with masses.
        subset: Subset specification dictionary;
            see return of <skais_mapper.illustris.snapshots.snapshot_offsets>.
        as_float32: Load float64 data types as float32 (to save memory).
        as_array: return a numpy array instead of a dictionary; takes
            effect only if a single field was requested.
        with_pbar: If True, a progress bar will show the current status.

    Returns:
        (dict | numpy.ndarray): A dictionary of the loaded data
    """
    data = {}
    # process arguments
    p_idx = pidx_from_ptype(ptype)
    key = "PartType" + str(p_idx)
    if fields is None:
        fields = []
    elif isinstance(fields, str | bytes):
        fields = [fields]
    if mdi is None:
        mdi = []
    elif isinstance(mdi, int):
        mdi = [mdi]
    # load header from first partition
    IllustrisH5File.path_func = get_path
    header = load_header(base_path, snapshot, as_dict=True)
    p_numbers = particle_numbers(header)
    # decide global read size, starting partition number and offset
    if subset:
        offset_ptype = subset["offsetType"][p_idx] - subset["snapOffsets"][p_idx, :]
        file_id = np.max(np.where(offset_ptype >= 0))
        file_off = offset_ptype[file_id]
        p_n = subset["lenType"][p_idx]
    else:
        file_id = 0
        file_off = 0
        p_n = p_numbers[p_idx]
    # save total count of requested particle type
    data["count"] = p_n
    if not p_n:  # if any, otherwise we're done here
        return data
    # find the first partition with this particle type
    f = find_group_in_partitions(base_path, snapshot, key)
    if not f:
        return None
    # if fields not specified, load everything
    if not fields:
        fields = list(f[key].keys())
    # loop over all requested fields
    for i, field in enumerate(fields):
        if field not in f[key].keys():
            raise KeyError(f"Particle type [{p_idx}] does not have field [{field}]")
        # replace local length with global
        shape = list(f[key][field].shape)
        shape[0] = p_n
        # or use multi-dimensional index slice
        if mdi and mdi[i] is not None:
            if len(shape) != 2:
                raise IndexError("Read error: mdi requested on non-2D field [{field}]")
            shape = [shape[0]]
        # allocate data arrays
        dtype = f[key][field].dtype
        if dtype == np.float64 and as_float32:
            dtype = np.float32
        data[field] = np.zeros(shape, dtype=dtype)
    f.close()
    # loop over partitions
    arr_offset = 0
    p_n_all = p_n
    n_files = header["NumFilesPerSnapshot"] - file_id
    if with_pbar:
        print(f"Reading relevant files [{n_files}]...")
        partition_iterator = trange(file_id, header["NumFilesPerSnapshot"])
    else:
        partition_iterator = range(file_id, header["NumFilesPerSnapshot"])
    for file_id in partition_iterator:
        f = IllustrisH5File(base_path, snapshot, file_id)
        # if no particles of requested type in partition, update and continue
        if key not in f:
            f.close()
            file_off = 0
            continue
        # set local read length for this partition
        p_n_file = f["Header"].attrs["NumPart_ThisFile"][p_idx]
        p_n_local = p_n
        # truncate local size
        if file_off + p_n_local > p_n_file:
            p_n_local = p_n_file - file_off
        # fetch all requested fields from partition
        for i, field in enumerate(fields):
            if mdi and mdi[i] is not None:
                data[field][arr_offset : arr_offset + p_n_local] = f[key][field][
                    file_off : file_off + p_n_local, mdi[i]
                ]
            else:
                data[field][arr_offset : arr_offset + p_n_local] = f[key][field][
                    file_off : file_off + p_n_local
                ]
        # reset for the next partition
        arr_offset += p_n_local
        p_n -= p_n_local
        file_off = 0
        f.close()
        if p_n <= 0:
            partition_iterator.update(header["NumFilesPerSnapshot"] - file_id)
            partition_iterator.refresh()
            partition_iterator.close()
            break
    # verify we read the correct number
    if p_n_all != arr_offset:
        raise RuntimeError(f"Read [{arr_offset}] particles, but was expecting [{p_n_all}]")
    if as_array and len(fields) == 1:
        return data[fields[0]]
    return data


def snapshot_offsets(
    base_path: str,
    snapshot: int,
    group_id: int,
    gtype: str,
) -> dict:
    """Compute offsets within snapshot for a particular HDF5 group/subgroup.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        group_id: Group ID, i.e. a halo or subhalo ID value from the FOF catalog
        gtype: Group type, i.e. 'Group' or 'Subhalo'.

    Returns:
        (dict): Subset of snapshot data to be loaded with <load_snapshot>.
    """
    subset = {}
    # old/new format
    IllustrisH5File.path_func = get_cat_path
    if "fof_subhalo" in get_cat_path(base_path, snapshot):
        # use separate 'offsets_nnn.hdf5' files
        with IllustrisH5File(base_path, snapshot, path_func=get_offset_path) as f:
            file_offsets = f["FileOffsets/" + gtype][()]
            # for consistency
            subset["snapOffsets"] = np.transpose(f["FileOffsets/SnapByType"][()])
    else:
        # load groupcat partition offsets from header of the first file
        with IllustrisH5File(base_path, snapshot) as f:
            file_offsets = f["Header"].attrs["FileOffsets_" + gtype]
            subset["snapOffsets"] = f["Header"].attrs["FileOffsets_Snap"]
    # get target groups partition which contains this group_id
    file_offsets = int(group_id) - file_offsets
    file_id = np.max(np.where(file_offsets >= 0))
    group_offset = file_offsets[file_id]
    # load the length (by type) of this group/subgroup from the group catalog
    with IllustrisH5File(base_path, snapshot, file_id) as f:
        subset["lenType"] = f[gtype][gtype + "LenType"][group_offset, :]
    # old/new format: load the offset (by type) of this group in the snapshot
    if "fof_subhalo" in get_cat_path(base_path, snapshot):
        with IllustrisH5File(base_path, snapshot, path_func=get_offset_path) as f:
            subset["offsetType"] = f[gtype + "/SnapByType"][group_id, :]
    else:
        with IllustrisH5File(base_path, snapshot, file_id) as f:
            subset["offsetType"] = f["Offsets"][gtype + "_SnapByType"][group_offset, :]
    return subset


def load_subhalo(
    base_path: str, snapshot: int, subhalo_id: int, p_type: str, **kwargs
) -> dict | NDArray:
    """Load all particles of a type for a specific subhalo (optionally limited to a subset fields).

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        subhalo_id: Group ID, i.e. subhalo ID value from the FOF catalog
        p_type: Particle type description string;
            e.g. 'gas', 'dm', 'stars', '1', '2', etc.
        kwargs:
            fields: Fields to be loaded for the corresponding ptype,
                e.g. ['Coordinates', 'Masses'] or 'NeutralHydrogenAbundance'.
            mdi: Multi-dimensional indices for fields; must be the
                same length as fields. E.g. fields = ['Coordinates', 'Masses'] and
                mdi = [1, None] returns a 1D array of y-coordinates and masses instead
                of a 3D array of coordinates with masses.
            as_float32 (bool): Load float64 data types as float32 (to save memory).
            as_array (bool): return a numpy array instead of a dictionary; takes
              effect only if a single field was requested.

    Returns:
        (dict | numpy.ndarray): A dictionary of the loaded data.
    """
    subset = snapshot_offsets(base_path, snapshot, subhalo_id, "Subhalo")
    return load_snapshot(base_path, snapshot, p_type, subset=subset, **kwargs)


def load_halo(base_path: str, snapshot: int, halo_id: int, p_type: str, **kwargs) -> dict | NDArray:
    """Load all particles of a type for a specific halo (optionally restricted to a subset fields).

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        halo_id: Group ID, i.e. halo ID value from the FOF catalog
        p_type: Particle type description string;
          e.g. 'gas', 'dm', 'stars', '1', '2', etc.
        kwargs:
            fields: Fields to be loaded for the corresponding ptype,
                e.g. ['Coordinates', 'Masses'] or 'NeutralHydrogenAbundance'.
            mdi: Multi-dimensional indices for fields; must be the
                same length as fields. E.g. fields = ['Coordinates', 'Masses'] and
                mdi = [1, None] returns a 1D array of y-coordinates and masses instead
                of a 3D array of coordinates with masses.
            as_float32: Load float64 data types as float32 (to save memory).
            as_array: return a numpy array instead of a dictionary; takes
                effect only if a single field was requested.

    Returns:
        (dict | numpy.ndarray): A dictionary of the loaded data.
    """
    subset = snapshot_offsets(base_path, snapshot, halo_id, "Group")
    return load_snapshot(base_path, snapshot, p_type, subset=subset, **kwargs)


def load_header(
    base_path: str,
    snapshot: int,
    as_dict: bool = True,
) -> dict | h5py.Group:
    """Load the header of a snapshot.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        as_dict: If True, a dictionary is returned, otherwise as h5py.Group object.

    Returns:
        (dict): The header of the snapshot HDF5 file as a dictionary
    """
    with IllustrisH5File(base_path, snapshot) as f:
        if as_dict:
            header = dict(f["Header"].attrs.items())
        else:
            header = f["Header"]
    return header


def find_group_in_partitions(base_path: str, snapshot: int, key: str) -> h5py.File:
    """Find the first occurrence of a group name in a set of HDF5 file partitions.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        key: The group name to find in the partitions, e.g. PartType5

    Returns:
        (h5py.File): A HDF5 file partition containing the group name.

    Note: Remember to close the HDF5 file afterwards.
    """
    partition = 0
    while True:
        f = IllustrisH5File(base_path, snapshot, partition)
        if not f.exists:
            return None
        if key in f:
            break
        partition += 1
        f.close()
    return f


if __name__ == "__main__":
    # TODO: unittest
    IllustrisH5File.path_func = get_path
    tng_dir = "/scratch/data/illustris/tng50-1"
    snap_id = 99

    # test getting a file path
    print("# Test getting a file path")
    filename_0 = get_path(tng_dir, snap_id)
    filename = get_path(tng_dir, snap_id, 10)
    print(filename_0)
    print(filename)

    # test opening a HDF5 file manually
    print("# Test opening a HDF5 file manually")
    with IllustrisH5File(tng_dir, snap_id) as f:
        print(f, f.keys())

    # test loading the header of a HDF5 file
    print("# Test loading the header of a HDF5 file")
    hdr = load_header(tng_dir, snap_id)
    print(hdr)

    # test loading snapshots (unless you have 200 GB RAM this fails)
    print("# Test loading snapshots (unless you have 200 GB RAM this fails)")
    try:
        gas_pos = load_snapshot(tng_dir, snap_id, "gas", fields=["Coordinates"], as_float32=False)
        print(gas_pos.shape, gas_pos)
    except np.core._exceptions._ArrayMemoryError as e:
        print(str(e))

    # test loading a subset for a halo 100
    print("# Test loading a subset for a halo 100")
    subset_halo100 = snapshot_offsets(tng_dir, snap_id, 100, "Group")
    print(subset_halo100)

    # test loading a subset for a subhalo 1
    print("# Test loading a subset for a subhalo 1")
    subset_subhalo1 = snapshot_offsets(tng_dir, snap_id, 1, "Subhalo")
    print(subset_subhalo1)

    # test loading a halo from snapshot
    print("# Test loading a halo from snapshot")
    halo1 = load_halo(tng_dir, snap_id, 1, "gas", fields=["Coordinates", "Masses"])
    print(halo1)

    # test loading a halo from snapshot (using <load_snapshot> w/ kwargs(subset))
    print("# test loading a halo from snapshot (using <load_snapshot> w/ kwargs(subset))")
    subset_halo1 = snapshot_offsets(tng_dir, snap_id, 1, "Group")
    halo1_manually = load_snapshot(
        tng_dir, snap_id, "gas", fields=["Coordinates", "Masses"], subset=subset_halo1
    )
    print("Equal count?", halo1["count"] == halo1_manually["count"], halo1["count"])
    print("Equal field?", np.all(halo1["Coordinates"] == halo1_manually["Coordinates"]))
    print("Equal field?", np.all(halo1["Masses"] == halo1_manually["Masses"]))

    # test find_group_in_partitions on existing/non-existing group
    print("# Test find_group_in_partitions on existing/non-existing group")
    f1 = find_group_in_partitions(tng_dir, snap_id, "PartType5")
    print(f1)
    f1.close()
    print("# Test find_group_in_partitions on non-existing group")
    f2 = find_group_in_partitions(tng_dir, snap_id, "PartType6")
    print(f2)
