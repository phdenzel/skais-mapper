# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Image/map data readers and (HDF5) writers."""

from pathlib import Path
import warnings
import numpy as np
import h5py
from h5py import Dataset as H5Dataset
from tqdm import tqdm
from PIL import Image
from skais_mapper.utils import nbytes, current_time, next_prime


class ImgRead:
    """Flexible image reader for multiple formats."""

    def __call__(
        self,
        paths: str | Path | list[str | Path] | None = None,
        squash: bool = True,
        pad_val: int | float = 0,
        **kwargs,
    ) -> np.ndarray:
        """Automatically determine file type and read data appropriately.

        Args:
            paths: File path to the image to be read.
            squash: If multiple paths are passed, merge and squash arrays.
            pad_val: Padding value to be used for shape expansion if multiple
              paths are passed and images have different shape (default: 0).
            **kwargs: Additional keyword arguments for parser functions:
              `_read_npy`, `_read_png`, or `_read_jpg`.

        Returns:
            Numpy ndarray of the image data.
        """
        if isinstance(paths, list | tuple):
            data = [self(p, **kwargs) for p in tqdm(paths, desc="ImgRead")]
            if data and squash:
                try:
                    data = np.concatenate(data)
                except ValueError:
                    data = self._stack_max_expand(data, pad_val=0)
            return data
        filepath = Path(paths) if paths is not None else Path("")
        match filepath.suffix:
            case ".npy":
                return self._read_npy(filepath, **kwargs)
            case ".jpg":
                return self._read_jpg(filepath, **kwargs)
            case ".png":
                return self._read_png(filepath, **kwargs)
            case _:
                return None

    @staticmethod
    def _stack_max_expand(data: list[np.ndarray], pad_val: int | float = 0):
        """Expand image arrays to maximum shape by padding and stack them.

        Args:
            data: list of arrays to be padded.
            pad_val: Padding value to be used (default: 0).

        Returns:
            The stacked numpy array.
        """
        max_shape = np.max([d.shape for d in data], axis=0)
        padded = []
        for arr in data:
            pad_width = [(0, maxd - currd) for currd, maxd in zip(arr.shape, max_shape)]
            parr = np.pad(arr, pad_width, mode="constant", constant_values=pad_val)
            padded.append(parr)
        return np.concatenate(padded)

    @staticmethod
    def _read_npy(
        path: str | Path,
        dtype: str | np.dtype | None = None,
        expand_dim: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Numpy file parser.

        Args:
            path: Filepath to the numpy file to be read
            dtype: Typecode or data-type to which the array is cast
            expand_dim: Expand array dimensions with new axis for stacking.
            **kwargs: Additional keyword arguments for `np.load`.

        Returns:
            Numpy ndarray read from numpy file

        Raises:
            (FileNotFoundError): if the input file path does not exist
        """
        filepath = Path(path)
        kwargs.setdefault("mmap_mode", None)
        if not filepath.exists():
            raise FileNotFoundError(f"File {str(filepath)} does not exist.")
        data_array = np.load(filepath, **kwargs)
        if dtype is not None:
            data_array = data_array.astype(dtype)
        if expand_dim and len(data_array.shape) <= 3:
            data_array = data_array[np.newaxis, ...]
        return data_array

    @staticmethod
    def _read_png(
        path: str | Path,
        dtype: str | np.dtype | None = None,
        expand_dim: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """PNG file parser.

        Args:
            path: Filepath to the PNG file to be read.
            dtype: Typecode or data-type to which the array is cast.
            expand_dim: Expand array dimensions with new axis for stacking.
            **kwargs: Additional keyword arguments for `PIL.Image.open`.

        Returns:
            Numpy ndarray read from PNG file

        Raises:
            (FileNotFoundError): if the input file path does not exist
        """
        filepath = Path(path)
        kwargs.setdefault("mode", "r")
        if not filepath.exists():
            raise FileNotFoundError(f"File {str(filepath)} does not exist.")
        with Image.open(filepath, **kwargs) as img:
            data_array = np.asarray(img)
        if dtype is not None:
            data_array = data_array.astype(dtype)
        if expand_dim and len(data_array.shape) <= 3:
            data_array = data_array[np.newaxis, ...]
        return data_array

    @staticmethod
    def _read_jpg(
        path: str | Path,
        dtype: str | np.dtype | None = None,
        expand_dim: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """JPG file parser.

        Args:
            path: Filepath to the JPG file to be read.
            dtype: Typecode or data-type to which the array is cast.
            expand_dim: Expand array dimensions with new axis for stacking.
            **kwargs: Additional keyword arguments for `PIL.Image.open`.

        Returns:
            Numpy ndarray read from JPG file

        Raises:
            (FileNotFoundError): if the input file path does not exist
        """
        return ImgRead._read_png(path, dtype=dtype, expand_dim=expand_dim, **kwargs)


class Img2H5Buffer:
    """Parse images (incrementally or all at once) and write to HDF5 files.

    The directory structure of a dataset should be as follows:
      - <data_dir>: /path/to/dataset/root
      - <file_groups>: image class as a subdirectory in the dataset
      - image file <extensions>: {npy | jpg | png | etc.}
    E.g. file paths of the following structure:
      /path/to/dataset/root/**/image_class/**/423120.npy
    HDF5 files end up being:
      /image_class/dataset

    Note: by default the entire dataset is loaded into cache
    """

    default_target_name = "{}.hdf5"
    extensions = [".jpg", ".png", ".npy"]

    def __init__(
        self,
        path: str | Path = None,
        target: str | Path = None,
        data: np.ndarray | dict = None,
        size: int | float | str = "1G",
    ):
        """Constructor.

        Args:
            path: Path to a data directory where the source files are located.
            target: Filename of the HDF5 file to be written.
            data: Alternative input format to `path` for adding data arrays or
              dictionaries directly to the buffer queue.
            size: Buffer cache size in bytes or passed as string.
        """
        self.files = self.glob_path(path)
        self.queue: list[np.ndarray | dict] = []
        if isinstance(data, np.ndarray | dict):
            self.queue.append(data)
        if target is None:
            target = self.default_target_name.format(current_time()[2:])
        self.target = Path(target)
        self.cache_size = nbytes(size)
        self.index = -1

    @staticmethod
    def _split_glob(
        path: str | list[str], relative: bool = False
    ) -> tuple[list[str], list[str | None]]:
        """Split path containing wildcard into root and wildcard expression."""
        root: list[str]
        file_key: list[str | None]
        if isinstance(path, str) and "*" in path:
            if relative:
                path = path[1:] if path.startswith("/") else path
            components = Path(path).parts
            iwc = [i for i, c in enumerate(components) if "*" in c][0]
            root = [str(Path().joinpath(*components[:iwc]))]
            file_key = [str(Path().joinpath(*components[iwc:]))]
        elif isinstance(path, list | tuple):
            root = []
            file_key = []
            for p in path:
                f, k = Img2H5Buffer._split_glob(p)
                root += f
                file_key += k
        else:
            root = [str(path)]
            file_key = [None]
        return root, file_key

    @staticmethod
    def glob_path(
        path: str | Path | list[str] | list[Path], extensions: str | list[str] = None
    ) -> list[Path]:
        """Glob path recursively for files.

        Args:
            path: Filename, path or list, can contain wildcards `*` or `**`.
            extensions: File extension to look fo
        """
        files: list[Path] = []
        extensions = [extensions] if isinstance(extensions, str) else extensions
        root, file_key = Img2H5Buffer._split_glob(path)
        for p, k in zip(root, file_key):
            if k is None:
                files.append(Path(p))
            else:
                path_files = [
                    f
                    for f in Path(p).rglob(k)
                    if f.is_file() and f.suffix in Img2H5Buffer.extensions
                ]
                files += sorted(path_files)
        files = [p for p in files if p.exists()]
        return files

    @property
    def n_files(self) -> int:
        """Number of files to be parsed."""
        if self.files is not None:
            return len(self.files)
        return 0

    @property
    def nbytes(self) -> list["nbytes"]:
        """List of the number of bytes for each buffer file."""
        if self.files:
            return [nbytes(f.stat().st_size) for f in self.files]
        return [nbytes(0)]

    @property
    def total_nbytes(self) -> list["nbytes"]:
        """Total number of bytes for buffer."""
        return sum(self.nbytes)

    def __str__(self):
        """String representation of the instance."""
        return (
            f"{self.__class__.__name__}("
            f"#b{len(self.queue)}#f{self.n_files}"
            f":{self.total_nbytes.as_str()}"
            f"@{self.cache_size.as_str()}->{self.target.name}"
            ")"
        )

    def __repr__(self):
        """String representation of the instance."""
        return self.__str__()

    def configure_rdcc(
        self,
        cache_size: int | float | str | None = None,
        f: int = 10,
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        """Automatically configure HDF5 data chunking for optimal writing.

        Args:
            cache_size: Cache of the entire buffer.
            f: Factor with which to increase the number of slots.
            verbose: Print additional information to stdout.
            **kwargs: Additional keyword arguments such as
              - `rdcc_nbytes`: See h5py.File or below for details.
              - `rdcc_w0`: See h5py.File or below for details.
              - `rdcc_nslots`: See h5py.File or below for details.

        Returns:
            `rdcc_nbytes`: sets the total size (measured in bytes) of the raw data
              chunk cache for each dataset. This should be set to the size of each
              chunk times the number of chunks that are likely to be needed in cache.
            `rdcc_w0`: sets the eviction policy for chunks from the cache when more
              space is needed. 0 is always last used chunk, 1 the last used chunk fully
              read or written, and inbetween values hybrid policies.
            `rdcc_nslots`: is the number of chunk slots in the cache for each dataset.
              In order to allow the chunks to be looked up quickly in cache, each chunk
              is hashed. Thus, it should be large enough to minimize the number of hash
              value collisions. At minimum 10, for maximum performance about 100 times
              larger as the number of chunks which fit in cache, ideally a prime number.
        """
        if cache_size is None:
            cache_size = self.cache_size
        else:
            cache_size = nbytes(cache_size)
        slots_size = max(self.nbytes) if self.nbytes else nbytes("2M")
        n_slots = int(cache_size / slots_size) if self.n_files else 100_000
        # avoid calculating prime numbers if previous configuration looks similar
        if (
            hasattr(self, "_rdcc")
            and self._rdcc["rdcc_nbytes"] == int(cache_size)
            and self._rdcc["rdcc_w0"] == kwargs.get("rdcc_w0", 1.0)
        ):
            return self._rdcc
        kwargs.setdefault("rdcc_nbytes", int(cache_size))
        kwargs.setdefault("rdcc_w0", 1.0)
        kwargs.setdefault("rdcc_nslots", next_prime(int(n_slots * f)))
        if verbose:
            sample_size = max(self.nbytes)
            print(f"Sample size: {sample_size}")
            print(f"Slot size: {nbytes(kwargs['rdcc_nbytes'] / kwargs['rdcc_nslots'])}")
            print(f"Slots: {kwargs['rdcc_nslots']}({n_slots})")
            print(f"Cache size: {nbytes(kwargs['rdcc_nbytes'])}")
            print(f"Eviction policy: {kwargs['rdcc_w0']}")
        self._rdcc = kwargs
        return kwargs

    @staticmethod
    def _h5py_file_kwargs(kwargs, defaults: dict = {}, in_place: bool = True):
        """Extract or filter out relevant keyword arguments for `h5py.File`."""
        file_kw = {}
        for key in defaults:
            file_kw.setdefault(key, defaults[key])
        for key in [
            "driver",
            "userblock_size",
            "track_order",
            "fs_strategy",
            "fs_persist",
            "fs_threshold",
            "fs_page_size",
            "page_buf_size",
            "min_meta_keep",
            "min_raw_keep",
            "locking",
            "alignment_threshold",
            "alignment_interval",
        ]:
            if key in kwargs:
                if in_place:
                    file_kw[key] = kwargs.pop(key)
                else:
                    file_kw[key] = kwargs.get(key)
        if in_place:
            return file_kw, kwargs
        return file_kw

    @staticmethod
    def _h5py_create_dataset_kwargs(kwargs, defaults: dict = {}, in_place: bool = True):
        """Extract or filter out relevant keyword arguments for `h5py.Group.create_dataset`."""
        create_dataset_kw = {}
        for key in defaults:
            create_dataset_kw.setdefault(key, defaults[key])
        for key in [
            "shape",
            "dtype",
            "chunks",
            "maxshape",
            "compression_opts",
            "scaleoffset",
            "shuffle",
            "fletcher32",
            "fillvalue",
            "fill_time",
            "track_times",
            "track_order",
            "external",
            "allow_unknown_filter",
        ]:
            if key in kwargs:
                if in_place:
                    create_dataset_kw[key] = kwargs.pop(key)
                else:
                    create_dataset_kw[key] = kwargs.get(key)
        if in_place:
            return create_dataset_kw, kwargs
        return create_dataset_kw

    @property
    def page(self) -> np.ndarray | dict | None:
        """Buffer page ready to be written to file."""
        if self.queue:
            return self.queue[0]
        elif self.files:
            return ImgRead()(self.files[0])
        return None

    def store(
        self, data: np.ndarray | dict | str | Path | list[str | Path], squash: bool = True
    ) -> "Img2H5Buffer":
        """Insert data into the buffer queue.

        Args:
            data: Data to be stored in buffer.
            squash: Squash data dimensions if buffer data is compatible.
        """
        if (
            squash
            and isinstance(data, np.ndarray)
            and self.queue
            and isinstance(self.queue[-1], np.ndarray)
        ):
            try:
                self.queue[-1] = np.concatenate((self.queue[-1], data))
            except ValueError:
                self.queue[-1] = ImgRead._stack_max_expand([self.queue[-1], data])
        elif isinstance(data, np.ndarray | dict):
            self.queue.append(data)
        elif isinstance(data, str | Path) or (
            isinstance(data, list) and isinstance(data[0], str | Path)
        ):
            return self.store(ImgRead()(data), squash=squash)
        return self

    def send(self, clear: bool = True) -> np.ndarray | dict | None:
        """Grab first data page from the buffer queue."""
        if self.queue:
            if clear:
                return self.queue.pop(0)
            return self.page
        elif self.files:
            if clear:
                self.store(self.files.pop(0))
            else:
                self.store(self.files[0])
            return self.send(clear=True)
        return None

    def flush(self) -> np.ndarray | dict | list[np.ndarray | dict] | None:
        """Send all data pages from the buffer queue."""
        if self.files:
            self.store(self.files)
            self.files = []
        if self.queue:
            data = self.queue
            self.queue = []
            if len(data) == 1:
                return data[0]
            return data
        return None

    def inc_write(
        self,
        path: str | Path | None = None,
        group: str = "images",
        data: np.ndarray | dict | None = None,
        expand_dim: bool = True,
        axis: int = 0,
        overwrite: bool | int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Incrementally (append mode) write the buffer to HDF5 file.

        Args:
            path: Filename of the HDF5 file and optionally the path of the HDF5
              group where the dataset is saved.
            group: HDF5 group where to save the dataset. If it does not exist,
              it is created.
            data: Data to be written to the hdf5 file. If None, all files in
              the buffer are written to HDF5 file.
            expand_dim: Expand dimension of data array for stacking.
            axis: Axis of the n-dimensional array where to append
            overwrite: If data should overwrite indices in a pre-existing HDF5 dataset,
              set to the index.
            verbose: Print additional information to stdout.
            kwargs: Additional keyword arguments for `Img2H5Buffer.configure_rdcc`,
              h5py.File`, and/or `h5py.Group.create_dataset`.
        """
        if path is None:
            path = self.target
        else:
            path = Path(path)
        if ":" in str(path):
            group = str(path).split(":")[1]
            path = Path(str(path).split(":")[0])
        if group.startswith("/"):
            group = group[1:]
        if data is None:
            data = self.flush()
        if isinstance(data, np.ndarray) and expand_dim:
            data = data[np.newaxis, ...]
        if isinstance(overwrite, bool):
            overwrite = 0 if overwrite else None
        # configure HDF5 chunk caching
        rdcc = {"cache_size": self.cache_size, "verbose": verbose}
        for key in ["f", "rdcc_nbytes", "rdcc_w0", "rdcc_nslots"]:
            if key in kwargs:
                rdcc[key] = kwargs.pop(key)
        rdcc = self.configure_rdcc(**rdcc)
        # write metadata
        if isinstance(data, dict):
            file_kwargs, kwargs = self._h5py_file_kwargs(
                kwargs,
                defaults={"mode": "a", "libver": "latest"},
            )
            with h5py.File(path, **file_kwargs, **rdcc) as h5:
                g = h5.create_group(f"/{group}") if group not in h5.keys() else h5[group]
                for key in data:
                    g.attrs[key] = data[key]
                if verbose:
                    print(
                        f"Data attribute(s) {tuple(data.keys())} have been "
                        f"written to HDF5 file@[/{group}]"
                    )
        # write data
        elif isinstance(data, np.ndarray):
            file_kwargs, kwargs = self._h5py_file_kwargs(
                kwargs,
                defaults={
                    "mode": "a",
                    "libver": "latest",
                },
            )
            kwargs, _ = self._h5py_create_dataset_kwargs(
                kwargs | file_kwargs,
                defaults={
                    "compression": "gzip",
                    "shuffle": True,
                    "track_times": True,
                    "dtype": data.dtype,
                    "shape": data.shape,
                    "maxshape": data.shape[:axis] + (None,) + data.shape[axis + 1 :],
                    "chunks": data.shape[:axis] + (1,) + data.shape[axis + 1 :],
                },
            )
            with h5py.File(path, **file_kwargs, **rdcc) as h5:
                ds_existed = isinstance(h5[group], H5Dataset) if group in h5 else False
                ds = h5.require_dataset(group, **kwargs, **rdcc)
                ds_samples = ds.shape[axis]
                data_samples = data.shape[axis]
                if not ds_existed:
                    self.index = 0
                elif overwrite is None:
                    self.index = ds_samples
                    ds.resize(self.index + data_samples, axis=axis)
                else:
                    self.index = overwrite
                    if data_samples > ds_samples:
                        ds.resize(self.index + data_samples, axis=axis)
                slc = [slice(None)] * len(ds.shape)
                slc[axis] = slice(self.index, self.index + data_samples)
                ds[tuple(slc)] = data
                if verbose:
                    print(
                        f"Data {data.shape} have been written to HDF5 dataset "
                        f"{ds.shape}@({self.index}:{self.index + data_samples})"
                    )
        else:
            warnings.warn(
                "Img2H5Buffer did not write data to file (either "
                "because the buffer was empty or data was incompatible)."
            )

    def write(
        self,
        path: str | Path | None = None,
        group: str = "images",
        data: np.ndarray | dict | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Write all files in buffer to a new HDF5 file.

        Args:
            path: Filename of the HDF5 file and optionally the path of the HDF5
              group where the dataset is saved separated by a colon,
              e.g. `'/path/to/file.hdf5:/path/to/group'`.
            group: HDF5 group where to save the dataset. If it does not exist,
              it is created.
            data: Data to be written to the HDF5 file.
            verbose: Print additional information to stdout.
            kwargs: Additional keyword arguments for `h5py.Group.create_dataset`.
        """
        self.inc_write(path, group=group, data=data, verbose=verbose, **kwargs)


if __name__ == "__main__":
    # test Img2H5Buffer
    for img2h5 in [
        Img2H5Buffer(
            target="/scratch/data/illustris/tng50-1.2D/test_250601_empty.hdf5",
            data=np.random.rand(100, 512, 512),
        ),
        Img2H5Buffer(
            "/scratch/data/illustris/tng50-1.2D/099/dm/dm_tng50-1.99.gid.0003*.png",
            "/scratch/data/illustris/tng50-1.2D/test_250601_png.hdf5",
        ),
        Img2H5Buffer(
            "/scratch/data/illustris/tng50-1.2D/099/dm/**/dm_tng50-1.99.gid.0003*.npy",
            "/scratch/data/illustris/tng50-1.2D/test_250601_npy.hdf5",
            data=np.random.rand(1, 512, 512),
        ),
    ]:
        print()
        print(img2h5)
        g = f"{img2h5.target}:/dm/images"
        img2h5.inc_write(g, f=100, verbose=True, track_order=True)
        # pprint.pprint(img2h5.files[:12])
    print()
    img2h5.inc_write(
        "/scratch/data/illustris/tng50-1.2D/test_250601_npy.hdf5:/dm/images",
        data=np.ones((1, 512, 512)),
        verbose=True,
    )
    img2h5.inc_write(
        "/scratch/data/illustris/tng50-1.2D/test_250601_npy.hdf5:/dm/attrs",
        data={"metadata": 100, "method": "inc_write"},
        verbose=True,
    )

    # clean up
    if 0:
        for f in [
            "/scratch/data/illustris/tng50-1.2D/test_250601_empty.hdf5",
            "/scratch/data/illustris/tng50-1.2D/test_250601_png.hdf5",
            "/scratch/data/illustris/tng50-1.2D/test_250601_npy.hdf5",
        ]:
            Path(f).unlink(missing_ok=True)
