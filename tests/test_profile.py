# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Testing the skais_mapper.profile module."""

from __future__ import annotations
from typing import TYPE_CHECKING
from skais_mapper._compat import TORCH_AVAILABLE
import pytest
import matplotlib.pyplot as plt
if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
    import skais_mapper.profile as profile
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
    xx, yy = profile._make_grid(W, H, device=device, dtype=dtype)
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
    means = torch.tensor([
        [256.0, 256.0],
        [256.0, 256.0],
        [256.0, 256.0],
        [256.0, 256.0],
        [256.0, 256.0],
        [128.0, 128.0],
    ], device=device, dtype=dtype)

    sigmas = torch.tensor([
        [40.0, 60.0],
        [32.0, 32.0],
        [20.0, 10.0],
        [20.0, 24.0],
        [30.0, 8.0],
        [24.0, 20.0],
    ], device=device, dtype=dtype)
    rho = torch.tensor([0.0, 0.5, -0.3, 0.0, -0.8, 0.1], device=device, dtype=dtype)
    return generate_gaussian2d_batch(
        means, sigmas, rho, size=(W, H), device=device, dtype=dtype
    )

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
@pytest.mark.parametrize(
    "x",
    [
        torch.rand((10, 10)),
        torch.rand((1, 10, 10)),
        torch.rand((1, 1, 10, 10)),
    ],
)
def test_sanitize_ndim(x):
    """Test `sanitize_ndim` function."""
    # Test 2/3/4D input tensor
    result = profile._sanitize_ndim(x)
    assert result.shape == (1, 10, 10)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_sanitize_ndim_invalid():
    """Test `sanitize_ndim` function: case ValueError."""
    # Test invalid dimensions
    x = torch.rand((1, 1, 10, 10, 10))
    with pytest.raises(ValueError):
        profile._sanitize_ndim(x)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test__make_grid():
    """Test the _make_grid function."""
    X, Y = profile._make_grid(10, 10)
    assert X.shape == (10, 10)
    assert Y.shape == (10, 10)


# @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
# def test_gaussian2d_batch(gaussians, show_plots):
#     """Inspect the test batch of 2D Gaussians."""
#     gaussians = gaussians * 1e15
#     for gaussian in gaussians:
#         gaussian = gaussian.squeeze().cpu().numpy()
#         plt.imshow(gaussian, interpolation='nearest')
#         if show_plots:
#             plt.show()
#     plt.close("all")

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
@pytest.mark.parametrize(
    "mode,fixed_center",
    [
        ("centroid", None),
        ("image_center", None),
        ("fixed", (0.5, 0.5)),
    ],
)
def test_compute_centers(mode, fixed_center, gaussians, show_plots):
    """Test the compute_centers function."""
    X, Y = profile._make_grid(gaussians.shape[-1], gaussians.shape[-2])
    centers = profile.compute_centers(
        gaussians,
        X,
        Y,
        mode=mode,
        fixed_center=(0.5, 0.5),
    )
    assert centers.shape[0] == gaussians.shape[0]
    assert centers.shape[1] == 2
    if show_plots:
        for gaussian, center in zip(gaussians, centers):
            plt.imshow(gaussian.squeeze().cpu().numpy(), interpolation='nearest')
            plt.scatter(center[0].cpu().numpy(), center[1].cpu().numpy(), c='cyan')
            plt.show()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_compute_centers_invalid_mode(gaussians):
    """Test compute_centers with an invalid mode."""
    X, Y = profile._make_grid(gaussians.shape[-1], gaussians.shape[-2])
    with pytest.raises(ValueError):
        profile.compute_centers(
            gaussians,
            X,
            Y,
            mode="invalid_mode",
        )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_radial_histogram(gaussians, show_plots):
    """Test the radial histogram function."""
    gaussians = gaussians * 1e12
    hist, edges, counts = profile.radial_histogram(gaussians, nbins=256, average=True)
    for ih, h, c in zip(edges, hist, counts):
        h = h.cpu().numpy()
        ih = ih.cpu().numpy()
        if show_plots:
            plt.plot(0.5*(ih[:-1] + ih[1:]), h)
            plt.show()
    assert hist.shape[0] == gaussians.shape[0]
    assert hist.shape[1] == 256
    assert hist.shape == counts.shape
    assert edges.shape[1] == hist.shape[1] + 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_radial_histogram_invalid(gaussians):
    """Test radial_histogram with invalid inputs."""
    with pytest.raises(ValueError):
        profile.radial_histogram(gaussians, nbins=0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_cumulative_radial_histogram(gaussians, show_plots):
    """Test the cumulative radial histogram function."""
    gaussians = gaussians * 1e12
    hist, edges, counts = profile.cumulative_radial_histogram(gaussians, nbins=256)
    for ih, h, c in zip(edges, hist, counts):
        print(h[-1], torch.sum(c))
        h = h.cpu().numpy()
        ih = ih.cpu().numpy()
        if h.shape[0] > 0 and show_plots:
            plt.plot(0.5*(ih[:-1] + ih[1:]), h)
            plt.show()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_radial_pdf(gaussians, show_plots):
    """Test the radial probability density function."""
    gaussians = gaussians * 1e12
    pdf, edges = profile.radial_pdf(gaussians, nbins=256)
    for ip, p in zip(edges, pdf):
        p = p.cpu().numpy()
        ip = ip.cpu().numpy()
        if p.shape[0] > 0 and show_plots:
            plt.plot(0.5*(ip[:-1] + ip[1:]), p)
            plt.show()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_radial_pdf_invalid(gaussians):
    """Test the radial probability density function with invalid inputs."""
    with pytest.raises(ValueError):
        profile.radial_pdf(gaussians, nbins=0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_radial_cdf(gaussians, show_plots):
    """Test the radial cumulative density function."""
    gaussians = gaussians * 1e12
    cdf, edges = profile.radial_cdf(gaussians, nbins=256)
    for ic, c in zip(edges, cdf):
        print(c[-1])
        c = c.cpu().numpy()
        ic = ic.cpu().numpy()
        if c.shape[0] > 0 and show_plots:
            plt.plot(0.5*(ic[:-1] + ic[1:]), c)
            plt.show()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_RadialProfile(gaussians, show_plots):
    """Test the RadialProfile class."""
    gaussians = gaussians * 1e12
    rel_noise = torch.rand(*gaussians.shape)
    gaussians_noise = gaussians * rel_noise
    prof = profile.RadialProfile(nbins=100)
    prof.update(gaussians, gaussians_noise)
    prof.update(gaussians, gaussians_noise)
    assert prof.aggregate.shape[0] == gaussians.shape[0] * 2
    assert prof.aggregate.shape[1] == 100
    assert prof.aggregate.shape == prof.target_aggregrate.shape
    profile_summaries = prof.compute()
    assert profile_summaries[0].shape[0] == 100
    assert profile_summaries[0].shape == profile_summaries[1].shape
    if show_plots:
        for i in range(prof.aggregate.shape[0]):
            plt.plot(prof.edges[:-1], prof.aggregate[i].cpu().numpy(), label=f"Profile {i}")
        plt.show()
