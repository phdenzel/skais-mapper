# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.plotting module."""

import pytest
from pathlib import Path
import numpy as np
from unittest.mock import patch, MagicMock

import skais_mapper.plotting as plotting


@pytest.fixture
def fake_plt(monkeypatch):
    """Fixture to mock matplotlib.pyplot."""
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    monkeypatch.setattr(plotting, "plt", mock_plt)
    return mock_plt


@pytest.fixture
def fake_cmap(monkeypatch):
    """Fixture to mock the colormap retrieval."""

    class FakeCmap:
        pass

    monkeypatch.setattr(plotting, "_get_cmap", lambda *a, **k: FakeCmap())
    return FakeCmap


@pytest.mark.parametrize(
    "batch,batch_idx,expected_data_idx,expected_meta_idx",
    [
        (np.arange(12).reshape(3, 4), None, None, None),  # No batch_idx: original array
        (np.arange(12).reshape(3, 4), 1, 1, None),  # Single int index
        (np.arange(24).reshape(2, 3, 4), (1, 2), (1, 2), None),  # Tuple/sequence index
    ],
)
def test_from_batch_numpy(batch, batch_idx, expected_data_idx, expected_meta_idx):
    """Test the _from_batch function with numpy arrays."""
    meta = {"foo": np.arange(3)} if batch.shape[0] == 3 else {}
    data, meta_out = plotting._from_batch(batch, meta, batch_idx=batch_idx)
    if expected_data_idx is not None:
        # The data should be indexed
        idx = expected_data_idx if isinstance(expected_data_idx, tuple) else (expected_data_idx,)
        expected = batch
        for i in idx:
            expected = expected[i]
        np.testing.assert_array_equal(data, expected.squeeze())
    else:
        np.testing.assert_array_equal(data, batch.squeeze())
    # Test that metadata is returned and unchanged or indexed as expected
    assert isinstance(meta_out, dict)


@pytest.mark.skipif(not plotting.TORCH_AVAILABLE, reason="torch not installed")
def test_from_batch_torch_tensor():
    """Test the _from_batch function with torch tensors."""
    import torch

    batch = torch.arange(12).reshape(3, 4)
    meta = {"foo": torch.arange(3)}
    data, meta_out = plotting._from_batch(batch, meta, batch_idx=2)
    # Should index batch and meta['foo']
    assert data.shape == (4,)
    assert meta_out["foo"].shape == torch.Size([])
    assert int(meta_out["foo"]) == 2


def test_from_batch_metadata_sequence():
    """Test the _from_batch function with metadata as a sequence."""
    batch = np.arange(24).reshape(2, 3, 4)
    meta = [{"foo": i} for i in range(2)]
    data, meta_out = plotting._from_batch(batch, meta, batch_idx=1)
    np.testing.assert_array_equal(data, batch[1].squeeze())
    assert meta_out == meta[1]


def test_from_batch_empty_metadata():
    """Test the _from_batch function with empty metadata."""
    arr = np.ones((2, 2))
    data, meta = plotting._from_batch(arr, {}, batch_idx=None)
    assert isinstance(meta, dict)
    np.testing.assert_array_equal(data, arr.squeeze())


def test_from_batch_batch_idx_none():
    """Test the _from_batch function with batch_idx=None."""
    arr = np.ones((2, 2))
    data, meta = plotting._from_batch(arr, {}, batch_idx=None)
    assert isinstance(meta, dict)
    np.testing.assert_array_equal(data, arr.squeeze())


def test_from_batch_batch_idx_out_of_range():
    """Test the _from_batch function with batch_idx out of range."""
    arr = np.ones((3, 3))
    meta = {"foo": np.arange(3)}
    with pytest.raises(IndexError):
        plotting._from_batch(arr, meta, batch_idx=10)


@patch("skais_mapper.plotting.SkaisColorMaps", autospec=True)
@patch("skais_mapper.plotting.colormaps", new_callable=lambda: ["viridis", "plasma"])
@patch("skais_mapper.plotting.plt")
def test_get_cmap_type_(mock_plt, mock_colormaps, mock_skaiscms):
    """Test the _get_cmap function with type_ argument."""
    types = {
        "gas": "gaseous",
        "dm": "obscura",
        "star": "hertzsprung",
        "bfield": "gravic",
        "temp": "phoenix",
        "hi/21cm": "nava",
        "unknown": "gaseous",
        None: "gaseous",
    }
    for type_, expected in types.items():
        setattr(mock_skaiscms, expected, MagicMock())
        cmap = plotting._get_cmap(type_=type_)
        assert cmap == getattr(mock_skaiscms, expected)


@patch("skais_mapper.plotting.SkaisColorMaps", autospec=True)
@patch("skais_mapper.plotting.colormaps", new_callable=lambda: ["viridis", "plasma"])
@patch("skais_mapper.plotting.plt")
def test_get_cmap_name(mock_plt, mock_colormaps, mock_skaiscms):
    """Test the _get_cmap function with name argument."""
    # name in SkaisColorMaps
    setattr(mock_skaiscms, "special", MagicMock())
    cmap = plotting._get_cmap(name="special")
    assert cmap == getattr(mock_skaiscms, "special")

    # name in colormaps
    mock_plt.get_cmap.return_value = "mpl_cmap"
    cmap = plotting._get_cmap(name="viridis")
    assert cmap == "mpl_cmap"

    # name not found: fallback to gaseous
    setattr(mock_skaiscms, "gaseous", MagicMock())
    cmap = plotting._get_cmap(name="notarealmap")
    assert cmap == getattr(mock_skaiscms, "gaseous")


def test_get_cmap_both_none(monkeypatch):
    """Test the _get_cmap function with both type_ and name as None."""

    class FakeCmap:
        pass

    monkeypatch.setattr(plotting, "SkaisColorMaps", type("CMS", (), {"gaseous": FakeCmap()}))
    cmap = plotting._get_cmap()
    assert isinstance(cmap, FakeCmap)


def test_get_cmap_special_colors(monkeypatch):
    """Test the _get_cmap function with special colors."""

    class FakeCmap:
        def __init__(self):
            self.under = self.over = self.bad = None

        def set_under(self, val):
            self.under = val

        def set_over(self, val):
            self.over = val

        def set_bad(self, val):
            self.bad = val

    monkeypatch.setattr(plotting, "SkaisColorMaps", type("CMS", (), {"gaseous": FakeCmap()}))
    cmap = plotting._get_cmap(type_="unknown", under="a", over="b", bad="c")
    assert cmap.under == "a"
    assert cmap.over == "b"
    assert cmap.bad == "c"


def test_get_cmap_setters_called(monkeypatch):
    """Test that the _get_cmap function calls set_under, set_over, and set_bad."""

    class FakeCmap:
        def __init__(self):
            self.set_under_called = self.set_over_called = self.set_bad_called = False

        def set_under(self, val):
            self.set_under_called = True

        def set_over(self, val):
            self.set_over_called = True

        def set_bad(self, val):
            self.set_bad_called = True

    monkeypatch.setattr(plotting, "SkaisColorMaps", type("CMS", (), {"gaseous": FakeCmap()}))
    cmap = plotting._get_cmap(type_="unknown", under="a", over="b", bad="c")
    assert cmap.set_under_called
    assert cmap.set_bad_called


def test_get_cmap_fallback_to_matplotlib(monkeypatch):
    """Test the _get_cmap function fallback to matplotlib when SkaisColorMaps is not available."""

    class Dummy:
        pass

    monkeypatch.setattr(plotting, "SkaisColorMaps", Dummy)

    class FakePlt:
        def get_cmap(self, name):
            self.called = name
            return f"mpl_{name}"

    fake_plt = FakePlt()
    monkeypatch.setattr(plotting, "plt", fake_plt)
    cmap = plotting._get_cmap(name="viridis")
    assert cmap == "mpl_viridis"
    assert fake_plt.called == "viridis"


@pytest.mark.parametrize(
    "class_type,expected",
    [
        ("dm", r"Σ$_{\mathrm{dm}}$"),
        ("star", r"Σ$_{\mathrm{star}}$"),
        ("gas", r"Σ$_{\mathrm{gas}}$"),
        ("hi", r"Σ$_{\mathrm{HI}}$"),
        ("hi/21cm", r"T$_{\mathrm{b}}$"),
        ("temp", r"T"),
        ("bfield", r"|B|"),
        ("foobar", r"Σ"),
        (None, r"Σ"),
    ],
)
def test_symbol_from_class(class_type, expected):
    """Test the _symbol_from_class function."""
    assert plotting._symbol_from_class(class_type) == expected


def test_plot_data_basic(fake_cmap, fake_plt):
    """Test the _plot_data function with basic parameters."""
    arr = np.ones((4, 4))
    info = {"class": "gas"}
    plotting._plot_data(arr, info, show=False, colorbar=True)
    plotting.plt.subplots.assert_called()


def test_plot_data_with_extent_and_labels(fake_cmap, fake_plt):
    """Test the _plot_data function with extent and labels."""
    arr = np.ones((4, 4))
    info = {"class": "gas", "extent": [-1, 1, -1, 1], "units_extent": "kpc", "units": "solMass"}
    plotting._plot_data(arr, info, extent=None, xlabel="X", ylabel="Y", colorbar=True)
    plotting.plt.subplots.assert_called()


def test_plot_data_with_extent_and_labels_from_info(fake_cmap, fake_plt):
    """Test the _plot_data function with extent and labels."""
    arr = np.ones((4, 4))
    info = {"class": "gas", "extent": [-1, 1, -1, 1], "units_extent": "kpc", "units": "solMass"}
    plotting._plot_data(arr, info, extent=None, colorbar=True)
    plotting.plt.subplots.assert_called()


def test_plot_data_with_colorbar_label(fake_cmap, fake_plt):
    """Test the _plot_data function with colorbar label."""
    arr = np.ones((4, 4))
    info = {"class": "gas", "units": "solMass"}
    plotting._plot_data(arr, info, colorbar=True, colorbar_label="mylabel2")
    assert plotting.plt.subplots()[0].colorbar.called
    label = plotting.plt.subplots()[0].colorbar.call_args[1]["label"]
    assert "M" in label or "2" in label or "mylabel" in label


def test_plot_data_savefig_creates_path(tmp_path, fake_cmap, fake_plt):
    """Test the _plot_data function with savefig creating the path."""
    arr = np.ones((4, 4))
    info = {"class": "gas"}
    img_path = tmp_path / "foo" / "bar.png"
    # Patch path.parent.exists to False to trigger directory creation
    with patch.object(Path, "exists", return_value=False), patch.object(Path, "mkdir") as mkdir:
        plotting._plot_data(arr, info, savefig=True, path=img_path)
        mkdir.assert_called()


def test_plot_data_savefig_creates_path_default_path(tmp_path, fake_cmap, fake_plt):
    """Test the _plot_data function with savefig creating the default path."""
    arr = np.ones((4, 4))
    info = {"class": "gas"}
    # Patch path.parent.exists to False to trigger directory creation
    with patch.object(Path, "exists", return_value=False), patch.object(Path, "mkdir") as mkdir:
        plotting._plot_data(arr, info, savefig=True, verbose=True, close=True)
        mkdir.assert_called()


def test_plot_data_show_and_close(fake_cmap, fake_plt):
    """Test the _plot_data function with show and close."""
    arr = np.ones((4, 4))
    info = {"class": "gas"}
    plotting._plot_data(arr, info, show=True, close=True)
    assert plotting.plt.show.called


def test_plot_image_not_implemented():
    """Test that plot_image raises NotImplementedError for non-array inputs."""
    with pytest.raises(NotImplementedError):
        plotting.plot_image(object())


def test_plot_image_array_dispatch(monkeypatch, fake_cmap, fake_plt):
    """Test that plot_image dispatches to _plot_data for numpy arrays."""
    arr = np.ones((4, 4))
    monkeypatch.setattr(plotting, "_from_batch", lambda d, metadata, batch_idx=None: (d, metadata))
    monkeypatch.setattr(plotting, "_plot_data", MagicMock())
    plotting.plot_image(arr)
    plotting._plot_data.assert_called()


@pytest.mark.skipif(not plotting.TORCH_AVAILABLE, reason="torch not installed")
def test_plot_image_tensor_dispatch(monkeypatch, fake_cmap, fake_plt):
    """Test that plot_image dispatches to _plot_data for torch tensors."""
    import torch

    arr = torch.ones(4, 4)
    monkeypatch.setattr(
        plotting, "_from_batch", lambda d, metadata, batch_idx=None: (d.numpy(), metadata)
    )
    monkeypatch.setattr(plotting, "_plot_data", MagicMock())
    plotting.plot_image(arr)
    plotting._plot_data.assert_called()


def test_plot_image_quantity_dispatch(monkeypatch, fake_cmap, fake_plt):
    """Test that plot_image dispatches to _plot_data for astropy Quantity arrays."""
    import astropy.units as u

    arr = np.ones((4, 4)) * u.kpc
    monkeypatch.setattr(plotting, "_from_batch", lambda d, metadata, batch_idx=None: (d, metadata))
    monkeypatch.setattr(plotting, "_plot_data", MagicMock())
    plotting.plot_image(arr, info={"units": u.mT})
    plotting.plot_image(arr)
    plotting._plot_data.assert_called()


def test_plot_image_sequence_dispatch(monkeypatch, fake_cmap, fake_plt):
    """Test that plot_image dispatches to _plot_data for sequences of arrays."""
    arrs = [np.ones((4, 4)), np.ones((4, 4))]
    monkeypatch.setattr(plotting, "_from_batch", lambda d, metadata, batch_idx=None: (d[0], {}))
    monkeypatch.setattr(plotting, "_plot_data", MagicMock())
    plotting.plot_image(arrs)
    plotting._plot_data.assert_called()


def test_plot_image_sequence_tuple_dispatch(monkeypatch, fake_cmap, fake_plt):
    """Test that plot_image dispatches to _plot_data for tuple of arrays and metadata."""
    arr = np.ones((4, 4))
    meta = {"class": "gas"}
    monkeypatch.setattr(plotting, "_from_batch", lambda d, metadata, batch_idx=None: (d, metadata))
    monkeypatch.setattr(plotting, "_plot_data", MagicMock())
    plotting.plot_image((arr, meta))
    plotting._plot_data.assert_called()


def test_plot_image_sequence_tuple_batch(monkeypatch, fake_cmap, fake_plt):
    """Test that plot_image dispatches to _plot_data for tuple of arrays and metadata with batch_idx."""
    arrs = [np.ones((4, 4)), np.ones((4, 4))]
    metas = [{"class": "gas"}, {"class": "dm"}]
    monkeypatch.setattr(
        plotting, "_from_batch", lambda d, metadata, batch_idx=None: (d[1], metadata[1])
    )
    monkeypatch.setattr(plotting, "_plot_data", MagicMock())
    plotting.plot_image((arrs, metas), batch_idx=1)
    plotting._plot_data.assert_called()
