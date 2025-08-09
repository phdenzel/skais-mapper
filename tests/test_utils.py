# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.utils module."""

import pytest
import re
import uuid
import datetime
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import pyplot as plt
import skais_mapper
from skais_mapper.utils import (
    get_run_id,
    set_run_id,
    current_time,
    compress_encode,
    extract_decode,
    alias_kw,
    primes,
    colors,
)


def test_get_set_run_id(monkeypatch):
    """Test the get_run_id and set_run_id functions."""
    # Save original
    orig_uid = getattr(skais_mapper, "RUN_UID", None)
    # Test auto-set
    set_run_id()
    uid1 = skais_mapper.RUN_UID
    assert isinstance(uid1, uuid.UUID)
    # Test set by uuid
    new_id = uuid.uuid4()
    set_run_id(new_id)
    assert skais_mapper.RUN_UID == new_id
    # Test set by string
    set_run_id(str(new_id))
    assert skais_mapper.RUN_UID == new_id
    # Test get_run_id returns correct length and hex chars
    rid = get_run_id(12)
    assert isinstance(rid, str)
    assert len(rid) == 12
    assert re.fullmatch(r"[0-9a-fA-F]+", rid)
    # Restore
    skais_mapper.RUN_UID = orig_uid


def test_current_time_variants():
    """Test the current_time function with various parameters."""
    t = current_time()
    assert isinstance(t, str)
    assert len(t) == 8
    # Test with delimiters and datetime
    t2 = current_time(date_only=True, as_str=True, no_delim=False)
    assert "-" in t2
    t3 = current_time(date_only=False, as_str=False)
    assert isinstance(t3, (datetime.date, datetime.datetime))


def test_compress_encode_and_extract_decode():
    """Test the compress_encode and extract_decode functions."""
    s = "hello world 123"
    enc = compress_encode(s)
    assert isinstance(enc, str)
    dec = extract_decode(enc)
    assert dec == s


def test_alias_kw_decorator():
    """Test the alias_kw decorator."""

    @alias_kw("foo", "bar")
    def f(foo=0):
        return foo

    assert f(foo=1) == 1
    assert f(bar=2) == 2  # Should alias bar->foo
    assert f(foo=3, bar=4) == 4  # bar should override foo


def test_legendre_basic():
    """Test the Legendre symbol calculation."""
    # (2|7) == 1, since 2 is a quadratic residue mod 7
    assert primes._legendre(2, 7) in (1, 6)  # 6 instead of -1 per doc


def test_is_sprp_true_false():
    """Test the _is_sprp function for known primes and composites."""
    # 7 is prime, 9 is not
    assert primes._is_sprp(7, 2)
    assert not primes._is_sprp(9, 2)
    assert primes._is_sprp(2503, 2)
    assert not primes._is_sprp(2509, 5)
    assert not primes._is_sprp(7801, 9)
    assert not primes._is_sprp(341, 2)
    assert primes._is_sprp(461, 2)


def test_is_lucas_prp_edge():
    """Test the _is_lucas_prp function for edge cases."""
    # For known prime, should return True
    assert primes._is_lucas_prp(7, 5)
    assert primes._is_lucas_prp(388, 2)


def test_is_prime_true_false():
    """Test the _is_sprp function for known primes and composites."""
    # 7 is prime, 9 is not
    assert primes.is_prime(7)
    assert not primes.is_prime(9)
    assert primes.is_prime(2503)
    assert not primes.is_prime(2509)
    assert not primes.is_prime(7801)
    assert primes.is_prime(1997)
    assert primes.is_prime(4211)
    assert not primes.is_prime(49729)
    assert not primes.is_prime(2147483671)
    assert primes.is_prime(2147483869)
    assert not primes.is_prime(2147483661)
    assert primes.is_prime(2147483647)


def test_next_prime():
    """Test the next_prime function."""
    assert primes.next_prime(1) == 2
    assert primes.next_prime(7) == 11
    assert primes.next_prime(40) == 41
    assert primes.next_prime(2508) == 2521
    assert primes.next_prime(2509) == 2521
    assert primes.next_prime(2511) == 2521


def test_color_variant_hex_and_shift():
    """Test the color_variant function with hex colors and shift."""
    orig = "#0080ff"
    v = colors.color_variant(orig, shift=10)
    assert v.startswith("#") and len(v) == 6
    # Should raise if not hex
    with pytest.raises(ValueError):
        colors.color_variant("0080ff")
    with pytest.raises(ValueError):
        colors.color_variant("#0080f")  # Too short


def test_color_variant_limits():
    """Test the color_variant function with limits."""
    # Lower bound
    assert colors.color_variant("#000000", shift=-10) == "#000000"
    # Upper bound
    assert colors.color_variant("#ffffff", shift=10) == "#ffffff"
    # Middle value
    assert colors.color_variant("#808080", shift=10).startswith("#")


def test_renorm_colormap_adaptor_basic():
    """Test the ReNormColormapAdaptor class."""
    cmap = plt.get_cmap('viridis')
    scmap = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    adaptor = colors.ReNormColormapAdaptor(
        scmap, lambda x: x - x.min() / (x.max() - x.min()))
    x = np.linspace(0, 1, 10)
    result = adaptor(x)
    assert isinstance(result, np.ndarray)
    # __getattr__ dispatch (grab any attribute from base)
    assert hasattr(adaptor, "N")
    # Abstract methods: just call to check present (no-op)
    adaptor._init()
    adaptor.reversed()


def test_renorm_colormap_adaptor_orig_norm():
    """Test the ReNormColormapAdaptor class."""
    cmap = plt.get_cmap('viridis')
    norm = Normalize(0, 1)
    adaptor = colors.ReNormColormapAdaptor(cmap, Normalize)
    adaptor = colors.ReNormColormapAdaptor(cmap, norm, orig_norm=Normalize(0, 0.8))
    x = np.linspace(0, 1, 10)
    result = adaptor(x)
    assert isinstance(result, np.ndarray)
    # __getattr__ dispatch (grab any attribute from base)
    assert hasattr(adaptor, "N")
    # Abstract methods: just call to check present (no-op)
    adaptor._init()
    adaptor.reversed()


def test_skaiscolors_shades_exist():
    """Test that the SkaisColors class has valid color attributes."""
    for name in ("white", "gray", "grey", "darkish"):
        val = getattr(colors.SkaisColors, name)
        assert isinstance(val, str) and val.startswith("#") and len(val) == 7


def test_skaiscolors_palette_and_random():
    """Test the SkaisColors palette and random color generation."""
    # Palette
    palette = colors.SkaisColors.palettes["hertzsprung"]
    assert isinstance(palette, list) and all(x.startswith("#") for x in palette)
    assert palette == colors.SkaisColors.hertzsprung_palette
    # Random
    cmap = colors.SkaisColors.cmap_from_color(palette[0], palette[-1])
    assert isinstance(cmap, LinearSegmentedColormap)
    cmap_native = colors.SkaisColors.cmap_from_color("cyan", "blue")
    assert isinstance(cmap_native, LinearSegmentedColormap)
    cmap_single = colors.SkaisColors.cmap_from_color("red")
    assert isinstance(cmap_single, LinearSegmentedColormap)


def test_skaiscolormaps_reverse_and_bad_values():
    """Test the reverse function and bad/over/under color settings."""
    orig = colors.SkaisColorMaps.random()
    reversed_cmap = colors.SkaisColorMaps.reverse(
        orig, set_bad="#000000", set_under="#111111", set_over="#222222"
    )
    assert isinstance(reversed_cmap, LinearSegmentedColormap)
    # Check bad/over/under colors set
    assert tuple(reversed_cmap.get_bad()) == (0.0, 0.0, 0.0, 1.0)
    assert tuple(reversed_cmap.get_over()) == (
        0.13333333333333333,
        0.13333333333333333,
        0.13333333333333333,
        1.0,
    )


def test_skaiscolormaps_reverse_and_no_bad_values():
    """Test the reverse function and bad/over/under color settings."""
    orig = colors.SkaisColorMaps.random()
    reversed_cmap = colors.SkaisColorMaps.reverse(orig)
    assert isinstance(reversed_cmap, LinearSegmentedColormap)


def test_skaiscolormaps_palette():
    """Test the SkaisColorMaps palette function."""
    interp = colors.SkaisColorMaps.palette("hertzsprung", 50)
    assert isinstance(interp, list)
    assert len(interp) == 50


def test_skaiscolormaps_as_generator():
    """Test the reverse function and bad/over/under color settings."""
    cmap = next(colors.SkaisColorMaps.gen())
    assert isinstance(cmap, LinearSegmentedColormap)


def test_register_all_runs():
    """Test the SkaisColorMaps.register_all function."""
    colors.SkaisColorMaps.register_all(verbose=True)
    cmap = plt.get_cmap("hertzsprung")
    assert isinstance(cmap, LinearSegmentedColormap)


def test_register_all_runs_non_verbose():
    """Test the SkaisColorMaps.register_all function."""
    colors.SkaisColorMaps.register_all(verbose=False)
    cmap = plt.get_cmap("hertzsprung")
    assert isinstance(cmap, LinearSegmentedColormap)


def test_plot_gradients_runs(tmp_path, monkeypatch):
    """Test the SkaisColors.plot_gradients function."""
    # Patch plt.savefig to avoid actual file creation
    import matplotlib.pyplot as plt

    called = {}

    def fake_savefig(fname):
        called["saved"] = fname

    monkeypatch.setattr(plt, "savefig", fake_savefig)
    colors.SkaisColorMaps.plot_gradients(savefig=True)
    assert "saved" in called
    called = {}
    colors.SkaisColorMaps.plot_gradients(savefig=False)
    assert "saved" not in called
