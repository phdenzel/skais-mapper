# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.illustris.util module."""

import pytest
import types
import skais_mapper.illustris.util as util


# @pytest.fixture
# def dummy_path_func():
#     """A dummy path function that returns a fixed path for testing."""
#     return lambda base, snap, part: f"/fake/path/snap_{snap}_part_{part}.hdf5"


# @pytest.fixture
# def mock_nbytes(monkeypatch):
#     """Mock the nbytes function to return a fixed size for testing."""
#     monkeypatch.setattr(
#         util, "nbytes", lambda x: 16 if x == "16" else int(str(x).replace("G", "1000000000"))
#     )


# @pytest.fixture
# def fake_file(monkeypatch):
#     """A fake h5py File class to simulate the h5py.File context manager."""

#     class DummyH5File:
#         def __init__(self, *args, **kwargs):
#             self.args = args
#             self.kwargs = kwargs

#         def __enter__(self):
#             return self

#         def __exit__(self, *a, **k):
#             return None

#         def __str__(self):
#             return "<FakeH5File>"

#     monkeypatch.setattr(util, "h5py", types.SimpleNamespace(File=DummyH5File))
#     return DummyH5File


@pytest.mark.parametrize(
    "ptype,expected",
    [
        (0, 0),
        (1, 1),
        ("dm", 1),
        ("dmlowres", 2),
        ("gas", 0),
        ("star", 4),
        ("bh", 5),
        ("tracer", 3),
        ("wind", 4),
        ("star", 4),
    ],
)
def test_pidx_from_ptype(ptype, expected):
    """Test the pidx_from_ptype function with various particle types."""
    assert util.pidx_from_ptype(ptype) == expected


def test_pidx_from_ptype_invalid():
    """Test the pidx_from_ptype function with an invalid particle type."""
    with pytest.raises(ValueError):
        util.pidx_from_ptype("invalid")


@pytest.mark.parametrize(
    "idx,expected",
    [(0, "gas"), (1, "dm"), (2, "dmlowres"), (3, "tracer"), (4, "star"), (5, "bh"), ("dm", "dm")],
)
def test_ptype_from_pidx(idx, expected):
    """Test the ptype_from_pidx function with various particle indices."""
    assert util.ptype_from_pidx(idx) == expected


def test_ptype_from_pidx_invalid():
    """Test the ptype_from_pidx function with an invalid particle index."""
    with pytest.raises(ValueError):
        util.ptype_from_pidx(99)
