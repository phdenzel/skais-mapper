# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.metrics module."""

from __future__ import annotations
from typing import TYPE_CHECKING
from skais_mapper._compat import TORCH_AVAILABLE
import pytest
import matplotlib.pyplot as plt

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
    import skais_mapper.metrics as metrics
    from skais_mapper.profile import _make_grid, radial_pdf, cumulative_radial_histogram
else:
    from skais_mapper import _torch_stub as _stub  # noqa
    from skais_mapper._torch_stub import *  # noqa: F401,F403


def generate_gaussian2d_batch(
    means: torch.Tensor,
    sigmas: torch.Tensor,
    rho: torch.Tensor | None,
    size: tuple[int, int],
    normalized: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Generate a batch of 2D Gaussian distributions.

    Args:
        means: Mean values for the Gaussians of shape (B, 2).
        sigmas: Standard deviations for the Gaussians of shape (B, 2).
        rho: Correlation coefficients for the Gaussians of shape (B,).
        size: Size of the output grid (W, H).
        normalized: Whether to normalize the Gaussians.
        device: Device to place the output tensor on.
        dtype: Data type of the output tensor.
    """
    B = means.shape[0]
    W, H = size
    device = torch.device(device) if device is not None else means.device
    dtype = dtype if dtype is not None else means.dtype
    means = means.to(device=device, dtype=dtype)
    sigmas = sigmas.to(device=device, dtype=dtype)
    rho = (
        rho.to(device=device, dtype=dtype)
        if rho is not None
        else torch.zeros(B, device=device, dtype=dtype)
    )
    # Grid
    xx, yy = _make_grid(W, H, device=device, dtype=dtype)
    mx, my = means[:, 0][:, None, None], means[:, 1][:, None, None]
    sx, sy = sigmas[:, 0][:, None, None], sigmas[:, 1][:, None, None]
    r = rho[:, None, None].clamp(-0.999, 0.999)
    dx = xx[None, ...] - mx
    dy = yy[None, ...] - my
    xs = dx / sx
    ys = dy / sy
    denom = 2.0 * (1.0 - r**2).clamp_min(1e-12)
    exponent = -(xs**2 + ys**2 - 2.0 * r * xs * ys) / denom
    g = torch.exp(exponent)
    if normalized:
        norm = 1.0 / (2.0 * torch.pi * sx * sy * torch.sqrt((1.0 - r**2).clamp_min(1e-12)))
        g = g * norm
    return g.unsqueeze(1)


@pytest.fixture
def gaussians() -> torch.Tensor:
    """Fixture to generate a default batch of 2D Gaussians."""
    W, H = 512, 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    means = torch.tensor(
        [
            [256.0, 256.0],
            [256.0, 256.0],
            [256.0, 256.0],
            [256.0, 256.0],
            [256.0, 256.0],
            [128.0, 128.0],
        ],
        device=device,
        dtype=dtype,
    )

    sigmas = torch.tensor(
        [
            [40.0, 60.0],
            [32.0, 32.0],
            [20.0, 10.0],
            [20.0, 24.0],
            [30.0, 8.0],
            [24.0, 20.0],
        ],
        device=device,
        dtype=dtype,
    )
    rho = torch.tensor([0.0, 0.5, -0.3, 0.0, -0.8, 0.1], device=device, dtype=dtype)
    return generate_gaussian2d_batch(means, sigmas, rho, size=(W, H), device=device, dtype=dtype)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_CenterOffsetError_half_mass_radius(gaussians, show_plots):
    """Test `CenterOffsetError.half_mass_radius`."""
    gaussians = gaussians
    coe = metrics.CenterOffsetError()
    hmr = coe._half_mass_radius(gaussians)
    hist, edges, counts = cumulative_radial_histogram(gaussians, nbins=256)
    r = 0.5 * (edges[:, :-1] + edges[:, 1:])
    assert torch.all(hmr > 0)
    if show_plots:
        for i in range(gaussians.shape[0]):
            plt.plot(r[i].cpu().numpy(), hist[i].cpu().numpy())
            plt.vlines(hmr[i].cpu().numpy(), 0, 1, cmap="gray")
            plt.show()

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_CenterOffsetError_update(gaussians, show_plots):
    """Test `CenterOffsetError.half_mass_radius`."""
    gaussians = gaussians
    coe = metrics.CenterOffsetError()
    coe.update(gaussians[0:3], gaussians[3:6])
    assert coe.min_aggregate >= 0
    assert coe.max_aggregate > 0
    assert coe.n_observations == 3
    assert coe.aggregate > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_CenterOffsetError_compute(gaussians, show_plots):
    """Test `CenterOffsetError.half_mass_radius`."""
    gaussians = gaussians
    coe = metrics.CenterOffsetError()
    coe.update(gaussians[0:3], gaussians[3:6])
    val = coe.compute()
    assert val > 0
    assert val < coe.max_aggregate
    assert coe.min_aggregate < val
    print(val)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_RadialProfileCurveError_update(gaussians):
    """Test `RadialProfileCurveError.update`."""
    gaussians = gaussians * 1e12
    gaussians_noise = gaussians + torch.rand(*gaussians.shape) * 1e12
    rpce = metrics.RadialProfileCurveError(nbins=50)
    rpce.update(gaussians_noise, gaussians)
    mean_bins = rpce.mean_per_bin
    var_bins = rpce.var_per_bin
    max_bins = rpce.max_aggregate
    min_bins = rpce.min_aggregate
    assert rpce.n_observations == gaussians.shape[0]
    assert mean_bins.shape == var_bins.shape
    assert max_bins.shape == min_bins.shape
    assert mean_bins.shape == max_bins.shape
    assert mean_bins.shape[0] == 50


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_RadialProfileCurveError_compute(gaussians, show_plots):
    """Test `RadialProfileCurveError.compute`."""
    gaussians = gaussians * 1e12
    gaussians_noise = gaussians + torch.rand(*gaussians.shape) * 1e12
    rpce = metrics.RadialProfileCurveError(nbins=50)
    rpce.update(gaussians_noise, gaussians)
    rpce.update(gaussians_noise, gaussians)
    val = rpce.compute()
    assert rpce.n_observations == gaussians.shape[0] * 2
    assert val > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_RadialProfileCurveError_compute_cumulative(gaussians, show_plots):
    """Test `RadialProfileCurveError.compute`."""
    gaussians = gaussians * 1e12
    gaussians_noise = gaussians + torch.rand(*gaussians.shape) * 1e12
    rpce = metrics.RadialProfileCurveError(nbins=50, cumulative=True)
    rpce.update(gaussians_noise, gaussians)
    rpce.update(gaussians_noise, gaussians)
    val = rpce.compute()
    assert rpce.n_observations == gaussians.shape[0] * 2
    assert val > 0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_RadialProfileCurveError_plot(gaussians, show_plots):
    """Test `RadialProfileCurveError` and plot."""
    gaussians = gaussians * 1e12
    gaussians_noise = gaussians + torch.rand(*gaussians.shape) * 1e12
    rpce = metrics.RadialProfileCurveError(nbins=50)
    rpce.update(gaussians, gaussians_noise)
    pdf, edges = radial_pdf(gaussians, nbins=50)
    pdf_noise, _ = radial_pdf(gaussians_noise, nbins=50)
    bin_centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
    if show_plots:
        plt.plot(bin_centers[0], rpce.mean_per_bin)
        plt.fill_between(
            bin_centers[0], rpce.mean_per_bin, rpce.max_aggregate, color="b", alpha=0.2
        )
        plt.fill_between(
            bin_centers[0], rpce.mean_per_bin, rpce.min_aggregate, color="b", alpha=0.2
        )
        plt.show()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_MapTotalError_update(gaussians):
    """Test `MapTotalError.update`."""
    gaussians = gaussians * 1e12
    rel_noise = torch.rand(*gaussians.shape)
    gaussians_noise = gaussians * rel_noise
    mte = metrics.MapTotalError(relative=False)
    mte.update(gaussians_noise, gaussians)
    assert mte.n_observations == gaussians.shape[0]
    assert mte.max_aggregate > mte.min_aggregate
    assert mte.aggregate > 0
