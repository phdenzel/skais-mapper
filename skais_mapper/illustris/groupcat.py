# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Illustris file i/o for FoF and Subfind group catalog.

Adapted from: https://github.com/illustristng/illustris_python
"""

import os
import numpy as np
from tqdm import trange
import h5py
from numpy.typing import NDArray
from skais_mapper.illustris.util import IllustrisH5File


def get_path(base_path: str, snapshot: int, partition: int = 0) -> str:
    """Get absolute path to a group catalog HDF5 file (modify as needed).

    Args:
        base_path: Base path to the Illustris(TNG) snapshots
        snapshot: Snapshot ID {0-99}
        partition: Subfile partition ID {0-600+}

    Returns:
        (str): Absolute path to a group catalog HDF5 file
    """
    gc_dir = os.path.join(base_path, f"groupcats/{snapshot:03d}")
    filepath = os.path.join(gc_dir, f"groups_{snapshot:03d}.{partition:d}.hdf5")
    filepath_alt = filepath.replace("groups_", "fof_subhalo_tab_")
    if os.path.isfile(filepath):
        return filepath
    return filepath_alt


def get_offset_path(base_path: str, snapshot: int, *args) -> str:
    """Get absolute path to a separate offset file (modify as needed).

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        *args: Dummy arguments for compatibility.

    Returns:
        (str): Absolute path to a group catalog's offsets HDF5 file.
    """
    basename = f"offsets/offsets_{snapshot:03d}.hdf5"
    offset_path = os.path.join(base_path, basename)
    return offset_path


def load_catalog(
    base_path: str,
    snapshot: int,
    key: str,
    key_ref: str,
    fields: list = None,
    as_float32: bool = False,
    as_array: bool = True,
    with_pbar: bool = True,
) -> dict | NDArray:
    """Load either halo or subhalo information from the group catalog.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        key: Group name from the HDF5 group catalog, e.g. 'Group' or 'Subhalo'.
        key_ref: Group name reference string in the HDF5 group catalog's
          header keys, e.g. 'groups' or 'subgroups'/'subhalos'
        fields: Fields to be loaded for the corresponding group,
          e.g. ['GroupPos', 'GroupMass'] or 'SubhaloGasMetalFractions'.
        as_float32: Load float64 data types as float32 (to save memory).
        as_array: Return a numpy array instead of a dictionary; takes
          effect only if a single field was requested.
        with_pbar: If True, a progress bar will show the current status.

    Returns:
        (dict | numpy.ndarray): A dictionary of the loaded data
    """
    data = {}
    if fields is None:
        fields = []
    elif isinstance(fields, str | bytes):
        fields = [fields]
    # load header from first partition
    IllustrisH5File.path_func = get_path
    header = load_header(base_path, snapshot)
    group = load_group(base_path, snapshot, key)
    if f"N{key_ref}_Total" not in header and key_ref == "subgroups":
        key_ref = "subhalos"
    data["count"] = header.get(f"N{key_ref}_Total", None)
    if not data["count"]:
        print(f"Warning: zero groups, empty return (snap='{snapshot}').")
        return data
    if not fields:
        fields = list(group.keys())
    for field in fields:
        if field not in group.keys():
            raise KeyError(f"Group catalog does not have requested field [{field}]!")
        # replace local length with global
        shape = list(group[field].shape)
        shape[0] = data["count"]
        dtype = group[field].dtype
        if dtype == np.float64 and as_float32:
            dtype = np.float32
        # allocate data arrays
        data[field] = np.zeros(shape, dtype=dtype)
    group._id.close()
    # loop over partitions
    arr_offset = 0
    if with_pbar:
        partition_iterator = trange(header["NumFiles"])
    else:
        partition_iterator = range(header["NumFiles"])
    for i in partition_iterator:
        f = IllustrisH5File(base_path, snapshot, i)
        # if partition is empty
        if not f["Header"].attrs[f"N{key_ref}_ThisFile"]:
            continue
        # loop over each field
        for field in fields:
            if field not in f[key].keys():
                raise KeyError(f"Group catalog does not have requested field [{field}]!")
            # shape and type
            shape = f[key][field].shape
            if len(shape) == 1:
                data[field][arr_offset : arr_offset + shape[0]] = f[key][field][0 : shape[0]]
            else:
                data[field][arr_offset : arr_offset + shape[0], :] = f[key][field][0 : shape[0], :]
        arr_offset += shape[0]
        f.close()
    if as_array and len(fields) == 1:
        return data[fields[0]]
    return data


def load_subhalos(base_path: str, snapshot: int, **kwargs) -> dict | NDArray:
    """Load all subhalo information from the entire group catalog for one snapshot.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        **kwargs: Additional keywords such as
          fields: Fields to be loaded for the corresponding group,
            e.g. ['GroupPos', 'GroupMass'] or 'SubhaloGasMetalFractions'.
          as_float32: Load float64 data types as float32 (to save memory).
          as_array: Return a numpy array instead of a dictionary; takes
            effect only if a single field was requested.
          with_pbar: If True, a progress bar will show the current status.

    Returns:
        (dict | numpy.ndarray): A dictionary of the loaded data
    """
    return load_catalog(base_path, snapshot, "Subhalo", "subgroups", **kwargs)


def load_halos(base_path: str, snapshot: int, **kwargs) -> dict | NDArray:
    """Load all halo information from the entire group catalog for one snapshot.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        fields: Fields to be loaded for the corresponding group,
          e.g. ['GroupPos', 'GroupMass'] or 'SubhaloGasMetalFractions'.
        as_float32: Load float64 data types as float32 (to save memory).
        as_array: Return a numpy array instead of a dictionary; takes
          effect only if a single field was requested.
        with_pbar: If True, a progress bar will show the current status.
        **kwargs: Additonal keywords for `load_catalog`

    Returns:
        (dict | numpy.ndarray): A dictionary of the loaded data
    """
    return load_catalog(base_path, snapshot, "Group", "groups", **kwargs)


def load_header(
    base_path: str,
    snapshot: int,
    as_dict: bool = True,
) -> dict | h5py.Group:
    """Load the header of a group catalog.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        as_dict: If True, a dictionary is returned, otherwise as
          h5py.Group object.

    Returns:
        (dict): The header of the group catalog HDF5 file as a dictionary
    """
    with IllustrisH5File(base_path, snapshot, path_func=get_path) as f:
        if as_dict:
            header = dict(f["Header"].attrs.items())
        else:
            header = f["Header"]
    return header


def load_group(base_path: str, snapshot: int, key: str, as_dict: bool = False) -> dict | h5py.Group:
    """Load a specified HDF5 group from a group catalog.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        key: Group descriptor, i.e. a group from the HDF5 catalog
          e.g. 'Group' or 'Subhalo'.
        as_dict: If True, a dictionary is returned, otherwise as
          h5py.Group object.

    Returns:
        (dict): The HDF5 group from the group catalog HDF5 file as a dictionary

    Note: Remember to close the HDF5 file afterwards (use <group>._id.close()).
    """
    f = IllustrisH5File(base_path, snapshot, path_func=get_path)
    if key in f:
        if as_dict:
            group = dict(f[key])
        else:
            group = f[key]
    else:
        f.close()
        return None
    return group


def load(
    base_path,
    snapshot,
    subhalos_kwargs: dict = None,
    halos_kwargs: dict = None,
    header_kwargs: dict = None,
    **kwargs,
) -> dict:
    """Load complete group catalog all at once.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        subhalos_kwargs: Keyword arguments for loading subhalos.
        halos_kwargs: Keyword arguments for loading halos.
        header_kwargs: Keyword arguments for loading the header.
        **kwargs: Additional 

    Returns:
        (dict): A dictionary of the loaded data
    """
    data = kwargs
    if subhalos_kwargs is None:
        subhalos_kwargs = {}
    if halos_kwargs is None:
        halos_kwargs = {}
    if header_kwargs is None:
        header_kwargs = {}
    data["subhalos"] = load_subhalos(base_path, snapshot, **subhalos_kwargs)
    data["halos"] = load_halos(base_path, snapshot, **halos_kwargs)
    data["header"] = load_header(base_path, snapshot, **header_kwargs)
    return data


def load_single(base_path, snapshot, halo_id=-1, subhalo_id=-1) -> dict:
    """Fetch the complete group catalog information for a single halo or subhalo.

    Args:
        base_path: Base path to the Illustris(TNG) snapshots.
        snapshot: Snapshot ID {0-99}.
        halo_id: Group ID, i.e. halo ID value from the FOF catalog.
        subhalo_id: Group ID, i.e. subhalo ID value from the FOF catalog.

    Returns:
        (dict): A dictionary of the loaded data
    """
    if (halo_id < 0 and subhalo_id < 0) or (halo_id >= 0 and subhalo_id >= 0):
        raise ValueError("Must specify either halo_id or subhalo_id (not both).")
    key = "Subhalo" if subhalo_id >= 0 else "Group"
    group_id = subhalo_id if subhalo_id >= 0 else halo_id
    # old or new format
    if "fof_subhalo" in get_path(base_path, snapshot):
        # use separate 'offsets_nnn.hdf5' files
        with IllustrisH5File(base_path, snapshot, path_func=get_offset_path) as f:
            offsets = f["FileOffsets/" + key][()]
    else:
        # use header of group catalog
        with IllustrisH5File(base_path, snapshot, path_func=get_path) as f:
            offsets = f["Header"].attrs["FileOffsets_" + key]
    offsets = group_id - offsets
    file_id = np.max(np.where(offsets >= 0))
    group_offset = offsets[file_id]
    # load halo/subhalo fields into a dict
    data = {}
    with IllustrisH5File(base_path, snapshot, file_id, path_func=get_path) as f:
        for field in f[key].keys():
            data[field] = f[key][field][group_offset]
    return data


if __name__ == "__main__":
    # TODO: unittest
    IllustrisH5File.path_func = get_path
    tng_dir = "/scratch/data/illustris/tng50-1"
    snap_id = 99

    # test getting a file path
    filename_0 = get_path(tng_dir, snap_id)
    filename = get_path(tng_dir, snap_id, 10)
    print(filename_0)
    print(filename)

    # test opening a HDF5 file manually
    print("# Test opening a HDF5 file manually")
    with IllustrisH5File(tng_dir, snap_id) as f:
        print(f.keys())
        print(f["Header"])
        print(f["Config"])
        print(f["IDs"])
        print(f["Parameters"])
        print(f["Group"])
        print(f["Subhalo"])

    print("# Test loading header from HDF5 file")
    cat_header = load_header(tng_dir, snap_id, True)
    print(cat_header)

    # test loading main group from HDF5 file
    print("# Test loading main group from HDF5 file")
    group = load_group(tng_dir, snap_id, "Group")
    print(group)

    # test loading catalog from HDF5 file (w/ progress bar)
    print("# Test loading catalog from HDF5 file w/ progress bar")
    cat = load_catalog(tng_dir, snap_id, "Group", "groups", fields=["GroupPos"])
    print(cat)

    # test loading catalog from HDF5 file (w/o progress bar)
    print("# Test loading catalog from HDF5 file w/o progress bar")
    cat = load_catalog(tng_dir, snap_id, "Group", "groups", fields=["GroupPos"], with_pbar=False)
    print(cat)

    # test loading all data from HDF5 file
    print("# Test loading all data from HDF5 file")
    data = load(tng_dir, snap_id, with_pbar=True)
