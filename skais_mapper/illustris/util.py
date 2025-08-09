# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Illustris simulation snapshot utilities.

Adapted from: https://github.com/illustristng/illustris_python
"""

import os
import warnings
import re
from pathlib import Path
from collections.abc import Callable
import h5py
from skais_mapper.utils import nbytes

__all__ = ["IllustrisH5File", "pidx_from_ptype", "ptype_from_pidx"]


class IllustrisH5File(h5py.File):
    """Represents an Illustris HDF5 file. Wrapper for the <h5py.File> class."""

    path_func: Callable | None = None

    def __init__(
        self,
        base_path: str | Path,
        snapshot: int,
        partition: int = 0,
        path_func: Callable | None = None,
        mode: str = "r",
        driver: str | None = None,
        cache_size: int | float | str = "2G",
        **kwargs,
    ):
        """Initialize a IllustrisH5File instance.

        Args:
            base_path: Base path to the Illustris(TNG) snapshots.
            snapshot: Snapshot ID {0-99}.
            partition: Subfile partition ID {0-600+}.
            path_func:
                A function fetching the filename of the HDF5 file. The function
                should accept <base_path>, <snapshot>, <partition> as arguments.
            mode:
                'r'          Readonly, file must exist (default)
                'r+'         Read/write, file must exist
                'w'          Create file, truncate if exists
                'w-' or 'x'  Create file, fail if exists
                'a'          Read/write if exists, create otherwise.
            driver: Name of the driver to use; valid values are
                None (default), 'core', 'sec2', 'direct', 'stdio', 'mpio', 'ros3'.
            cache_size: Chunk cache size in bytes or passed as string.
            **kwargs: More keyword arguments
        """
        self.exists = True
        self.chunk_cache_size = nbytes(cache_size)
        kwargs.setdefault("rdcc_nbytes", int(self.chunk_cache_size))
        kwargs.setdefault("rdcc_w0", 1.0)
        if os.path.exists(self.filename):
            super().__init__(self.filename, mode=mode, driver=driver, **kwargs)
        else:
            self.exists = False

    def __new__(
        cls,
        base_path: str | Path,
        snapshot: int,
        partition: int = 0,
        path_func: Callable | None = None,
        mode: str = "r",
        driver: str | None = None,
        cache_size: int | float | str = "1G",
        **kwargs,
    ):
        """Construct a IllustrisH5File instance.

        Args:
            base_path: Base path to the Illustris(TNG) snapshots.
            snapshot: Snapshot ID {0-99}.
            partition: Subfile partition ID {0-600+}.
            path_func:
                A function fetching the filename of the HDF5 file. The function
                should accept <base_path>, <snapshot>, <partition> as arguments.
            mode:
                'r'          Readonly, file must exist (default)
                'r+'         Read/write, file must exist
                'w'          Create file, truncate if exists
                'w-' or 'x'  Create file, fail if exists
                'a'          Read/write if exists, create otherwise.
            driver: Name of the driver to use; valid values are
                None (default), 'core', 'sec2', 'direct', 'stdio', 'mpio', 'ros3'.
            cache_size: Chunk cache size in bytes or passed as string.
            **kwargs: More keyword arguments
        """
        if path_func is None and cls.path_func is not None:
            path_func = cls.path_func
        if path_func is None:
            module = __import__("skais_mapper.illustris.snapshots", fromlist=["get_path"])
            path_func = getattr(module, "get_path")
        cls.filename = path_func(base_path, snapshot, partition)
        return super().__new__(cls, **kwargs)

    def __enter__(self, *args):
        """Overwrite __enter__ to accept arbitrary arguments."""
        return super().__enter__()

    def __exit__(self, *args):
        """Overwrite __exit__ to accept arbitrary arguments."""
        if self.exists:
            return super().__exit__()
        return None

    def __str__(self):
        """String representation."""
        if self.exists:
            return super().__str__()
        return f"<HDF5 File: exists={self.exists}>"


def pidx_from_ptype(ptype: int | str) -> int:
    """Mapping common names to indices of particle types.

    Args:
        ptype: particle type description string

    Returns:
        (int) particle type index
          0 -> gas particles
          1 -> dark-matter particles
          2 -> lowres dark-matter / stellar disc particles (in zoom simulations)
          3 -> tracer / stellar buldge particles
          4 -> star / wind / stellar particles
          5 -> blackhole / sink particles

    Raises:
        (ValueError): If the name doesn't match any particle type.
    """
    if str(ptype).isdigit():
        return int(ptype)
    if str(ptype).lower() in ["gas", "cells"]:
        return 0
    if str(ptype).lower() in ["dm", "darkmatter"]:
        return 1
    if str(ptype).lower() in ["dmlowres"]:
        return 2  # only zoom simulations, not present in full periodic boxes
    if str(ptype).lower() in ["tracer", "tracers", "tracermc", "trmc"]:
        return 3
    if str(ptype).lower() in ["star", "stars", "stellar"]:
        return 4  # only those with GFM_StellarFormationTime > 0
    if str(ptype).lower() in ["wind"]:
        return 4  # only those with GFM_StellarFormationTime < 0
    if str(ptype).lower() in ["bh", "bhs", "blackhole", "blackholes"]:
        return 5
    raise ValueError(f"Unknown particle type name {ptype}.")


def ptype_from_pidx(pidx: int | str) -> str:
    """Mapping indices to names of particle types.

    Args:
        pidx: particle type index

    Returns:
        (int) particle type index
          gas particles -> 0
          dark-matter particles -> 1
          lowres dark-matter / stellar disc particles (in zoom simulations) -> 2
          tracer / stellar buldge particles -> 3
          star / wind / stellar particles -> 4
          blackhole / sink particles -> 5

    Raises:
        (ValueError): If the index is not
    """
    if isinstance(pidx, str):
        return pidx
    if pidx == 0:
        return "gas"
    if pidx == 1:
        return "dm"
    if pidx == 2:
        return "dmlowres"  # only zoom simulations, not present in full periodic boxes
    if pidx == 3:
        return "tracer"
    if pidx == 4:
        return "star"  # only those with GFM_StellarFormationTime > 0
    if pidx == 5:
        return "bh"
    raise ValueError(f"Unknown particle type name {pidx}.")


# def parse_name(
#     filename: str | Path,
#     root: str | None = None,
#     groups: str | list[str] | None = None,
#     mask: str | None = None,
#     formatters: dict | None = None,
#     **kwargs,
# ) -> dict | None:
#     r"""Parse a filename of Illustris preprocessed image files (see scripts).

#     Args:
#         filename:
#             Path (string) of the file
#         root:
#             Root directory where the file should be stored.
#         groups:
#             Selection of subdirectories indicating the image class.
#             Optional, but should be specified to speed up file search.
#         mask:
#             Format string mask for matching regular expressions.
#             Fields are split between ':' as field_name:regex_pattern:type, i.e.
#             '{test:(?P=<test>.*?):s}_{flight::s}_no.{number::d}.{extension::s}'
#         formatters:
#             Format map from strings to types, i.e. 'f' -> float
#             Default: {'s': str, 'd': int, 'f': float, '': str}
#             Note: add an empty key string for defaulting unknown cases
#         **kwargs: Additional keyword arguments.

#     Returns:
#         (dict): Keyword map with matches

#     Warns:
#         (Warning): if no matches can be found.
#     """
#     filename = Path(filename)
#     # set defaults
#     if root is None:
#         root = "."
#     if groups is None:
#         groups = "*"
#     if mask is None:
#         mask = (
#             "{base_name:(?P<base_name>.*):s}."
#             "{snapshot::d}.groupID{gid::d}_units_{unit::s}@{scale::f}_"
#             r"{scale_unit:(?P<scale_unit>\S+?):s}[[@\.](.\D+)?]"
#             r"{res:(?P<res>.*\d+)?:f}[(\w)?]"
#             r"{res_unit:(?P<res_unit>.\w+)?:s}"
#             r"[(\.)?]{extension:(?P<extension>.*):s}"
#         )
#     if formatters is None:
#         formatters = {"s": str, "d": int, "f": float, "": str}
#     # find file
#     if filename.exists():
#         filepath = filename
#     else:
#         rootpath = Path(root)
#         filepath = None
#         for group in groups:
#             files = list(rootpath.rglob(f"**/{group}/**/{filename}"))
#             if files:
#                 filepath = files[0]
#                 break
#     if filepath is None:
#         return None
#     # parse mask
#     tokens = re.split(r"\{(.*?)\}", mask)
#     keywords = [s.split(":")[0] for s in tokens[1::2]]
#     formats = []
#     for s in tokens[1::2]:
#         k_and_f = s.split(":")
#         fkey = k_and_f[-1][-1] if len(k_and_f) > 1 else ""
#         formats.append(formatters[fkey])
#     # match keywords in filename
#     tokens[1::2] = [
#         f"(?P<{k.split(':')[0]}>.*?)" if not k.split(":")[1] else f"{k.split(':')[1]}"
#         for k in tokens[1::2]
#     ]
#     tokens[0::2] = [
#         re.escape(t) if "[" not in t and "]" not in t else t[1:-1] for t in tokens[0::2]
#     ]
#     pattern = "".join(tokens)
#     matches = re.match(pattern, str(filepath))
#     match_dict = {}
#     if not matches:
#         warnings.warn(f"No matches found with format string [mask]: {mask}", Warning)
#     else:
#         match_dict = {
#             x: (f(matches.group(x)) if matches.group(x) else matches.group(x))
#             for x, f in zip(keywords, formats)
#         }
#     match_dict |= kwargs
#     return match_dict
