# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.data module."""

import pytest
import numpy as np
from unittest.mock import patch
from PIL import Image
from skais_mapper.data import ImgRead, Img2H5Buffer


def make_files(tmp_path, suffixes=(".jpg", ".png", ".npy")):
    """Create temporary files with given suffixes."""
    files = []
    for ext in suffixes:
        f = tmp_path / f"file{ext}"
        f.write_bytes(b"content")
        files.append(f)
    return files


@pytest.fixture
def _buffer(tmp_path):
    """Fixture to create a temporary Img2H5Buffer."""
    arr = np.ones((2, 2))
    return Img2H5Buffer(data=arr, target=tmp_path / "test.hdf5")


def test_read_png_and_jpg(tmp_path):
    """Test reading PNG and JPG images using ImgRead."""
    # Create a simple grayscale PNG image using Pillow
    img_path = tmp_path / "test.png"
    arr = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)
    out = ImgRead._read_png(str(img_path))
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 10, 10)  # expand_dim=True by default

    # Now test dtype conversion
    out16 = ImgRead._read_png(str(img_path), dtype="float32")
    assert out16.dtype == np.float32

    # Test expand_dim=False
    out2 = ImgRead._read_png(str(img_path), expand_dim=False)
    assert out2.shape == (10, 10)

    # Test JPG reading (should delegate to PNG logic)
    img_jpg = tmp_path / "test.jpg"
    Image.fromarray(arr).save(img_jpg)
    out_jpg = ImgRead._read_jpg(str(img_jpg))
    assert out_jpg.shape == (1, 10, 10)


def test_read_png_file_not_found(tmp_path):
    """Test reading PNG and JPG images with a missing file."""
    missing_file = tmp_path / "doesnotexist.png"
    with pytest.raises(FileNotFoundError):
        ImgRead._read_png(str(missing_file))
    with pytest.raises(FileNotFoundError):
        ImgRead._read_jpg(str(missing_file))


def test_read_png_multichannel(tmp_path):
    """Test reading multi-channel PNG images."""
    # RGB image
    arr = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
    img_path = tmp_path / "rgb.png"
    Image.fromarray(arr).save(img_path)
    out = ImgRead._read_png(str(img_path))
    assert out.shape == (1, 5, 5, 3)


def test_read_png_multiple_paths(tmp_path):
    """Test reading multi-channel PNG images."""
    # RGB image
    arr = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
    img_path = tmp_path / "rgb.png"
    img_path2 = tmp_path / "rgb2.png"
    Image.fromarray(arr).save(img_path)
    Image.fromarray(arr).save(img_path2)
    img_reader = ImgRead()
    out = img_reader([img_path, img_path2])
    assert out.shape == (2, 5, 5, 3)


def test_read_jpg_multiple_paths_no_squash(tmp_path):
    """Test reading multi-channel jpg images."""
    # RGB image
    arr = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
    img_path = tmp_path / "rgb.jpg"
    img_path2 = tmp_path / "rgb2.jpg"
    Image.fromarray(arr).save(img_path)
    Image.fromarray(arr).save(img_path2)
    img_reader = ImgRead()
    out = img_reader([img_path, img_path2], squash=False)
    assert len(out) == 2


def test_read_png_multiple_paths_invalid_squash(tmp_path):
    """Test reading multi-channel PNG images."""
    # RGB image
    arr = np.random.randint(0, 255, (5, 5, 3), dtype=np.uint8)
    img_path = tmp_path / "rgb.png"
    img_path2 = tmp_path / "rgb2.png"
    Image.fromarray(arr).save(img_path)
    Image.fromarray(arr[:, 1, 1:]).save(img_path2)
    img_reader = ImgRead()
    with pytest.raises(ValueError):
        img_reader([img_path, img_path2], squash=True)


def test_read_npy_default(tmp_path):
    """Test reading NPY with additional keyword arguments."""
    arr = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
    img_path = tmp_path / "test_option.npy"
    np.save(img_path, arr)
    # Should work with extra kwargs
    img_reader = ImgRead()
    out = img_reader(str(img_path))
    assert out.shape == (1, 8, 8)


def test_read_npy_dtype(tmp_path):
    """Test reading NPY with additional keyword arguments."""
    arr = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
    img_path = tmp_path / "test_option.npy"
    np.save(img_path, arr)
    # Should work with extra kwargs
    img_reader = ImgRead()
    out = img_reader(str(img_path), dtype=np.float32, expand_dim=False)
    assert out.shape == (8, 8)
    assert out.dtype == np.float32


def test_read_unknown(tmp_path):
    """Test reading unknown file with additional keyword arguments."""
    arr = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
    img_path = tmp_path / "test_option.npz"
    np.save(img_path, arr)
    # Should work with extra kwargs
    img_reader = ImgRead()
    out = img_reader(str(img_path))
    assert out is None


def test_imgread_file_not_found(tmp_path):
    """Test ImgRead with a file that does not exist."""
    with pytest.raises(FileNotFoundError):
        ImgRead._read_npy(str(tmp_path / "nonexistent.png"))


def test_imgread_stack_max_expand(tmp_path):
    """Test ImgRead._stack_max_expand."""
    arr = np.random.randint(0, 255, (1, 10, 10), dtype=np.uint8)
    arr2 = np.random.randint(0, 255, (1, 9, 9), dtype=np.uint8)
    out = ImgRead._stack_max_expand([arr, arr2], pad_val=0)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 10, 10)  # Stacked with padding


def test_init_with_ndarray_and_dict(tmp_path):
    """Test initialization with ndarray and dict."""
    arr = np.arange(10)
    d = {"x": np.zeros((2, 2))}
    buf1 = Img2H5Buffer(data=arr, target=str(tmp_path / "a.hdf5"))
    buf2 = Img2H5Buffer(data=d, target=str(tmp_path / "b.hdf5"))
    assert isinstance(buf1.queue[0], np.ndarray)
    assert isinstance(buf2.queue[0], dict)


def test_init_with_path_none(tmp_path):
    """Test initialization with path=None."""
    buf = Img2H5Buffer(path=None, target=str(tmp_path / "c.hdf5"))
    assert isinstance(buf.files, list)


def test_default_target_name_extension():
    """Test default target name and extension."""
    buf = Img2H5Buffer(target=None)
    assert str(buf.target).endswith(".hdf5")


def test_split_glob_single_path_and_wildcard():
    """Test splitting a single path with wildcard."""
    root, key = Img2H5Buffer._split_glob("foo/bar/*.jpg")
    assert root[0].endswith("foo/bar")
    assert key[0] == "*.jpg"


def test_split_glob_relative_handling():
    """Test splitting a relative path with wildcard."""
    root, key = Img2H5Buffer._split_glob("/foo/bar/*.npy", relative=True)
    # Should not start with leading slash
    assert not root[0].startswith("/")


def test_split_glob_list_and_tuple():
    """Test splitting a list and tuple of paths with wildcards."""
    roots, keys = Img2H5Buffer._split_glob(["a/*.jpg", "b/*.png"])
    """Test splitting a list of paths with wildcards."""
    assert len(roots) == 2 and len(keys) == 2


def test_split_glob_no_wildcard():
    """Test splitting a path without wildcard."""
    root, key = Img2H5Buffer._split_glob("foo/bar/image.png")
    assert key[0] is None
    assert root[0].endswith("image.png")


def test_glob_path_finds_files(tmp_path):
    """Test glob_path finds files with specific extensions."""
    make_files(tmp_path)
    found = Img2H5Buffer.glob_path(str(tmp_path / "*.jpg"))
    assert any(str(f).endswith(".jpg") for f in found)
    found_all = Img2H5Buffer.glob_path([str(tmp_path / "*.jpg"), str(tmp_path / "*.png")])
    assert len(found_all) >= 2


def test_glob_path_filters_extensions(tmp_path):
    """Test glob_path filters files by extensions."""
    f1 = tmp_path / "foo.txt"
    f1.write_text("not an image")
    files = Img2H5Buffer.glob_path(str(tmp_path / "*.txt"))
    assert files == []  # Should not include non-image extension


def test_glob_path_nonexistent(tmp_path):
    """Test glob_path with a nonexistent path."""
    files = Img2H5Buffer.glob_path(str(tmp_path / "nonexistent/*.jpg"))
    assert files == []


def test_n_files_property(tmp_path):
    """Test n_files property."""
    make_files(tmp_path)
    buf = Img2H5Buffer(path=f"{tmp_path}/file*")
    assert buf.n_files == 3


def test_n_files_none():
    """Test n_files property when files is None."""
    buf = Img2H5Buffer(path=None)
    # If self.files is None, should return 0
    buf.files = None
    assert buf.n_files == 0
    assert buf.total_nbytes == 0


def test_nbytes_property(tmp_path):
    """Test nbytes property."""
    make_files(tmp_path)
    buf = Img2H5Buffer(path=f"{tmp_path}/file*")
    print(buf.nbytes)
    assert sum(buf.nbytes) == 3 * len(b"content")


def test_queue_accepts_various(tmp_path):
    """Test queue accepts various data types."""
    buf = Img2H5Buffer(data=np.zeros(2), target=tmp_path / "t1.hdf5")
    assert isinstance(buf.queue, list)
    buf2 = Img2H5Buffer(data={"foo": 1}, target=tmp_path / "t2.hdf5")
    assert isinstance(buf2.queue[0], dict)


def test_img2h5buffer_repr():
    """Test extensions and properties of Img2H5Buffer."""
    s = str(Img2H5Buffer())
    r = repr(Img2H5Buffer())
    assert isinstance(s, str)
    assert s == r
    print(s)


def test_queue_accepts_dict_and_ndarray(tmp_path):
    """Test queue accepts dict and ndarray."""
    d = {"bar": np.arange(4)}
    arr = np.arange(4)
    buf = Img2H5Buffer(data=d, target=tmp_path / "c.hdf5")
    assert isinstance(buf.queue[0], dict)
    buf2 = Img2H5Buffer(data=arr, target=tmp_path / "d.hdf5")
    assert isinstance(buf2.queue[0], np.ndarray)


def test_configure_rdcc(_buffer):
    """Test configure_rdcc method."""
    # Should return a dict with rdcc settings
    conf = _buffer.configure_rdcc(1234)
    assert isinstance(conf, dict)
    assert "rdcc_nbytes" in conf and conf["rdcc_nbytes"] == 1234


def test_configure_rdcc_cachen_size_None(_buffer):
    """Test configure_rdcc method."""
    # Should return a dict with rdcc settings
    conf = _buffer.configure_rdcc(verbose=True)
    assert isinstance(conf, dict)
    assert "rdcc_nbytes" in conf and conf["rdcc_nbytes"] == 1073741824  # Default 1 GiB
    # use cached rdcc
    conf2 = _buffer.configure_rdcc()
    assert conf == conf2


def test_h5py_file_kwargs(_buffer):
    """Test _h5py_file_kwargs method."""
    kwargs = {
        "some_other_key": "value",
        "driver": "core",
        "userblock_size": 512,
        "track_order": True,
        "fs_strategy": "default",
        "fs_persist": True,
        "fs_threshold": 1000,
        "fs_page_size": 4096,
        "page_buf_size": 4096,
        "min_meta_keep": 1000,
        "min_raw_keep": 1000,
        "locking": True,
        "alignment_threshold": 4096,
        "alignment_interval": 4096,
    }
    kwargs = _buffer._h5py_file_kwargs(kwargs, defaults={"default": 0}, in_place=False)
    assert "some_other_key" not in kwargs
    assert "default" in kwargs and kwargs["default"] == 0


def test_h5py_file_kwargs_in_place(_buffer):
    """Test _h5py_file_kwargs method."""
    kwargs = {
        "some_other_key": "value",
        "driver": "core",
        "userblock_size": 512,
        "track_order": True,
        "fs_strategy": "default",
        "fs_persist": True,
        "fs_threshold": 1000,
        "fs_page_size": 4096,
        "page_buf_size": 4096,
        "min_meta_keep": 1000,
        "min_raw_keep": 1000,
        "locking": True,
        "alignment_threshold": 4096,
        "alignment_interval": 4096,
    }
    kwargs, _ = _buffer._h5py_file_kwargs(kwargs, defaults={"default": 0}, in_place=True)
    assert "some_other_key" not in kwargs
    assert "default" in kwargs and kwargs["default"] == 0


def test_h5py_file_kwargs_empty(_buffer):
    """Test _h5py_file_kwargs method: case empty."""
    kwargs = {}
    kwargs, _ = _buffer._h5py_file_kwargs(kwargs)
    assert isinstance(kwargs, dict)
    assert kwargs == {}


def test_h5py_create_dataset_kwargs(_buffer):
    """Test _h5py_create_dataset_kwargs method."""
    kwargs = {
        "some_other_key": "value",
        "shape": (2, 2),
        "dtype": np.float32,
        "chunks": (1, 1),
        "maxshape": (None, None),
        "compression_opts": "gzip",
        "scaleoffset": 0,
        "shuffle": True,
        "fletcher32": True,
        "fillvalue": 0,
        "fill_time": "always",
        "track_times": True,
        "track_order": True,
        "external": None,
        "allow_unknown_filter": False,
    }
    kwargs = _buffer._h5py_create_dataset_kwargs(kwargs, in_place=False)
    assert "shape" in kwargs and "dtype" in kwargs
    assert kwargs["shape"] == (2, 2)
    assert kwargs["dtype"] == np.float32
    assert "some_other_key" not in kwargs


def test_h5py_create_dataset_kwargs_defaults(_buffer):
    """Test _h5py_create_dataset_kwargs method."""
    kwargs = {
        "some_other_key": "value",
        "shape": (2, 2),
        "dtype": np.float32,
        "chunks": (1, 1),
        "maxshape": (None, None),
        "compression_opts": "gzip",
        "scaleoffset": 0,
        "shuffle": True,
        "fletcher32": True,
        "fillvalue": 0,
        "fill_time": "always",
        "track_times": True,
        "track_order": True,
        "external": None,
        "allow_unknown_filter": False,
    }
    kwargs, _ = _buffer._h5py_create_dataset_kwargs(kwargs, defaults={"default": 0}, in_place=True)
    assert "shape" in kwargs and "dtype" in kwargs
    assert kwargs["shape"] == (2, 2)
    assert kwargs["dtype"] == np.float32
    assert "some_other_key" not in kwargs
    assert "default" in kwargs and kwargs["default"] == 0


def test_h5py_create_dataset_kwargs_empty(_buffer):
    """Test _h5py_create_dataset_kwargs method: case empty."""
    kwargs = {}
    kwargs = _buffer._h5py_create_dataset_kwargs(kwargs, in_place=False)
    assert isinstance(kwargs, dict)
    assert kwargs == {}


def test_page_returns_data(_buffer):
    """Test page method returns data."""
    # With one item, should return it and advance index
    _buffer.queue = [np.arange(4)]
    page = _buffer.page
    assert np.all(page == np.arange(4))


def test_page_from_file(_buffer, tmp_path):
    """Test page method reads from file."""
    files = make_files(tmp_path, suffixes=(".npy"))
    _buffer.queue = []
    _buffer.files = files
    _buffer.page


def test_page_empty_queue(_buffer, tmp_path):
    """Test page method raises IndexError on empty buffer."""
    _buffer.queue = []
    assert _buffer.page is None


@patch("skais_mapper.data.h5py.File", autospec=True)
def test_store_and_send_and_flush(mock_h5file, _buffer, tmp_path):
    """Test store, send, and flush methods of Img2H5Buffer."""
    # Prepare data and queue
    arr = np.arange(6).reshape(2, 3)

    # test store
    _buffer.store(arr)
    assert len(_buffer.queue) == 1

    # test store (no squash)
    _buffer.store(arr, squash=False)
    assert len(_buffer.queue) == 2

    # test store file
    np.save(tmp_path / "file.npy", arr)
    _buffer.store(tmp_path / "file.npy", squash=False)
    assert len(_buffer.queue) == 3

    # test store without effect
    _buffer.store(1, squash=False)
    assert len(_buffer.queue) == 3

    # test send multiple times
    page = _buffer.send()
    assert isinstance(page, np.ndarray)
    assert len(_buffer.queue) == 2
    page = _buffer.send(clear=False)
    assert isinstance(page, np.ndarray)
    assert len(_buffer.queue) == 2
    page = _buffer.send()
    assert len(_buffer.queue) == 1
    page = _buffer.send()
    assert len(_buffer.queue) == 0
    _buffer.files = [tmp_path / "file.npy"]
    page = _buffer.send(clear=False)
    assert len(_buffer.queue) == 0
    assert page is not None
    page = _buffer.send(clear=True)
    assert page is not None
    page = _buffer.send()
    assert page is None

    # test flush
    _buffer.files = [tmp_path / "file.npy"]
    data = _buffer.flush()
    assert len(_buffer.queue) == 0
    assert len(_buffer.files) == 0
    assert isinstance(data, np.ndarray)
    _buffer.queue = [arr, arr]
    data = _buffer.flush()
    assert len(data) == 2
    assert isinstance(data, list)
    data = _buffer.flush()
    assert data is None


@patch("skais_mapper.data.h5py.File", autospec=True)
def test_img2h5buffer_inc_write_metadata(mock_h5file, _buffer, tmp_path):
    """Test Img2H5Buffer incremental write."""
    _buffer.target = tmp_path / "file.hdf5"
    _buffer.inc_write(path=f"{tmp_path}/file.hdf5:/metadata", data={"metadata": 0})
    _buffer.inc_write(path=f"{tmp_path}/file.hdf5:metadata", data={"metadata": 1})
    _buffer.inc_write(path=f"{tmp_path}/file.hdf5", group="metadata", data={"metadata": 2})
    _buffer.inc_write(group="metadata", data={"metadata": 3}, rdcc_nbytes=4000, verbose=True)


def test_img2h5buffer_inc_write_data(_buffer, tmp_path):
    """Test Img2H5Buffer incremental write."""
    arr = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
    _buffer.target = tmp_path / "file.hdf5"
    _buffer.flush()
    _buffer.store(arr)
    _buffer.inc_write(path=f"{tmp_path}/file.hdf5:/images", data=arr, expand_dim=True)
    _buffer.inc_write(path=f"{tmp_path}/file.hdf5:images", expand_dim=True)
    _buffer.inc_write(
        path=f"{tmp_path}/file.hdf5:/images", data=arr, expand_dim=True, overwrite=True
    )
    large_arr = np.random.randint(0, 255, (12, 8, 8), dtype=np.uint8)
    _buffer.inc_write(
        path=f"{tmp_path}/file.hdf5:/images",
        data=large_arr,
        expand_dim=False,
        overwrite=True,
        verbose=True,
    )


def test_img2h5buffer_write_data(_buffer, tmp_path):
    """Test Img2H5Buffer incremental write."""
    _buffer.target = tmp_path / "file.hdf5"
    _buffer.flush()
    large_arr = np.random.randint(0, 255, (12, 8, 8), dtype=np.uint8)
    _buffer.write(
        path=f"{tmp_path}/file.hdf5:/images", data=large_arr, expand_dim=False, verbose=True
    )


def test_img2h5buffer_inc_write_invalid(_buffer, tmp_path):
    """Test Img2H5Buffer incremental write."""
    _buffer.target = tmp_path / "file.hdf5"
    _buffer.flush()
    with pytest.warns():
        _buffer.inc_write(path=f"{tmp_path}/file.hdf5:/images", data=1)
