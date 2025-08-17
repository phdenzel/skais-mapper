# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Profile routines for 2D tensor maps and images."""

from __future__ import annotations
from skais_mapper._compat import TORCH_AVAILABLE
from typing import Literal, TYPE_CHECKING

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
else:
    from skais_mapper import _torch_stub as __stub  # noqa
    from skais_mapper._torch_stub import *  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]


def _sanitize_ndim(x: torch.Tensor):
    """Standardize image dimensionality to (B, H, W)."""
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 4:
        x = x.sum(dim=1)
    if x.ndim != 3:
        raise ValueError("Require input of shape (B, C, H, W), (B, H, W), or (H, W).")
    return x


def _make_grid(
    w: int,
    h: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
):
    """Generate X & Y coordinate grids of shape (H, W) with pixel coordinates."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32
    y = torch.arange(h, device=device, dtype=dtype) + 0.5
    x = torch.arange(w, device=device, dtype=dtype) + 0.5
    Y, X = torch.meshgrid(y, x, indexing="ij")
    return X, Y


def compute_centers(
    maps: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    mode: Literal["centroid", "image_center", "fixed"] = "centroid",
    fixed_center: tuple[int, int] | tuple[float, float] = None,
    norm: bool = False,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute per-image centers.

    Args:
        maps: Maps of shape (B, H, W), (B, C, H, W), or (H, W) on which to compute centers.
        X: X coordinate grid of shape (H, W).
        Y: Y coordinate grid of shape (H, W).
        mode: Computation mode; one of ["centroid", "image_center", "fixed"].
        fixed_center: Fixed center coordinates for "fixed" mode. If `float`,
          treated as normalized with respect to image size.
        norm: Whether to normalize the coordinates to [0, 1] range.
        eps: Small value to avoid division by zero in centroid computation.

    Returns:
        Center coordinates of shape (B, 2).
    """
    maps = _sanitize_ndim(maps)
    B, H, W = maps.shape
    if mode == "fixed":
        if fixed_center is None:
            raise ValueError("Fixed center requested, but fixed_center was None.")
        if isinstance(fixed_center[0], float) or isinstance(fixed_center[1], float):
            fixed_center = (int(W * fixed_center[0]), int(H * fixed_center[1]))
        cx = torch.full((B,), float(fixed_center[0]), device=maps.device, dtype=maps.dtype)
        cy = torch.full((B,), float(fixed_center[1]), device=maps.device, dtype=maps.dtype)
        if norm:
            cx = cx / (W - 1)
            cy = cy / (H - 1)
        return torch.stack((cx, cy), dim=1)
    if mode == "image_center":
        cx = torch.full((B,), (W - 1) * 0.5, device=maps.device, dtype=maps.dtype)
        cy = torch.full((B,), (H - 1) * 0.5, device=maps.device, dtype=maps.dtype)
        if norm:
            cx = cx / (W - 1)
            cy = cy / (H - 1)
        return torch.stack((cx, cy), dim=1)
    if mode == "centroid":
        intensity = maps
        denom = intensity.flatten(1).sum(dim=1).clamp_min(eps)
        Xb = (intensity * X[None]).flatten(1).sum(dim=1)
        Yb = (intensity * Y[None]).flatten(1).sum(dim=1)
        cx = (Xb / denom).to(intensity.dtype)
        cy = (Yb / denom).to(intensity.dtype)
        if norm:
            cx = cx / (W - 1)
            cy = cy / (H - 1)
        return torch.stack((cx, cy), dim=1)
    raise ValueError(f"Unknown center mode: {mode}")


def radial_histogram(
    maps: torch.Tensor,
    r: torch.Tensor | None = None,
    bin_edges: torch.Tensor | None = None,
    nbins: int = 100,
    center_mode: Literal["centroid", "image_center", "fixed"] = "image_center",
    average: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 1D histograms of 2D maps over radial bins.

    Args:
        maps: Maps representing weights with which to sum in each bin of shape ([B, C,] H, W).
        r: Radius values of shape ([B,] H, W).
        bin_edges: Edges of the bins of shape (K+1,).
        nbins: Number of bins (K) to use if `bin_edges` is not provided.
        center_mode: Computation mode; one of ["centroid", "image_center", "fixed"].
        average: Whether to average the histogram by pixel count.

    Returns:
        hist: (B, K) with sum over pixels in each bin.
    """
    maps = _sanitize_ndim(maps)
    B, H, W = maps.shape
    if r is None:
        X, Y = _make_grid(W, H, device=maps.device, dtype=maps.dtype)
        XX = X.expand(B, -1, -1)
        YY = Y.expand(B, -1, -1)
        centers = compute_centers(maps, X, Y, mode=center_mode)
        cx = centers[:, 0].view(B, 1, 1)
        cy = centers[:, 1].view(B, 1, 1)
        r = torch.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    if r.ndim == 2:
        r = r.unsqueeze(0)
    if r.ndim != 3:
        raise ValueError(f"Expected radius tensor of shape (B, H, W), got {r.shape}.")
    if not nbins:
        raise ValueError("Number of bins must be greater than 0.")
    elif bin_edges is None:
        bin_edges = torch.linspace(0, r.max(), nbins + 1, device=maps.device, dtype=maps.dtype)
    else:
        nbins = bin_edges.numel() - 1
    m_flat = maps.reshape(B, -1)
    c_flat = torch.ones_like(m_flat, device=maps.device, dtype=maps.dtype)
    r_flat = r.reshape(B, -1)
    idx = torch.bucketize(r_flat, bin_edges, right=False) - 1
    idx = idx.clamp_(0, nbins - 1)
    hist = torch.zeros((B, nbins), device=maps.device, dtype=maps.dtype)
    counts = torch.zeros((B, nbins), device=maps.device, dtype=maps.dtype)
    hist.scatter_add_(dim=1, index=idx, src=m_flat)
    counts.scatter_add_(dim=1, index=idx, src=c_flat)
    if average:
        hist = hist / counts.clamp_min(1)
    return hist, bin_edges.repeat(B, 1), counts


def cumulative_radial_histogram(
    maps: torch.Tensor,
    r: torch.Tensor | None = None,
    bin_edges: torch.Tensor | None = None,
    nbins: int = 100,
    center_mode: Literal["centroid", "image_center", "fixed"] = "image_center",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute cumulative 1D histograms of 2D maps over radial bins.

    Args:
        maps: Maps representing weights with which to sum in each bin of shape ([B, C,] H, W).
        r: Radius values of shape ([B,] H, W).
        bin_edges: Edges of the bins of shape (K+1,).
        nbins: Number of bins (K) to use if `bin_edges` is not provided.
        center_mode: Computation mode; one of ["centroid", "image_center", "fixed"].

    Returns:
        hist: (B, K) with sum over pixels in each bin.
        edges: (B, K+1) with bin edges.
        counts: (B, K) with pixel counts in each bin.
    """
    hist, edges, counts = radial_histogram(
        maps,
        r=r,
        bin_edges=bin_edges,
        nbins=nbins,
        center_mode=center_mode,
        average=False,
    )
    hist = torch.cumsum(hist, dim=1)
    counts = torch.cumsum(counts, dim=1)
    return hist, edges, counts


def radial_pdf(
    maps: torch.Tensor,
    r: torch.Tensor | None = None,
    bin_edges: torch.Tensor | None = None,
    nbins: int = 100,
    center_mode: Literal["centroid", "image_center", "fixed"] = "image_center",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute radial probability density function (PDF) of 2D maps.

    The returned PDF satisfies (per batch item b):
        sum_i pdf[b, i] * Δr[i] = 1
    where Δr[i] = bin_edges[i+1] - bin_edges[i].

    Args:
        maps: Maps representing weights with which to sum in each bin of shape ([B, C,] H, W).
        r: Radius values of shape ([B,] H, W).
        bin_edges: Edges of the bins of shape (K+1,).
        nbins: Number of bins (K) to use if `bin_edges` is not provided.
        center_mode: Computation mode; one of ["centroid", "image_center", "fixed"].

    Returns:
        pdf: (B, K) with PDF values for each bin.
        edges: (B, K+1) with bin edges.
    """
    mass_per_bin, edges, _ = radial_histogram(
        maps,
        r=r,
        bin_edges=bin_edges,
        nbins=nbins,
        center_mode=center_mode,
        average=True,
    )
    total_mass = mass_per_bin.sum(dim=1, keepdim=True).clamp_min(1e-12)
    pmf = mass_per_bin / total_mass
    dr = edges[:, 1:] - edges[:, :-1]
    pdf = pmf / dr
    return pdf, edges

def radial_cdf(
    maps: torch.Tensor,
    r: torch.Tensor | None = None,
    bin_edges: torch.Tensor | None = None,
    nbins: int = 100,
    center_mode: Literal["centroid", "image_center", "fixed"] = "image_center",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute radial cumulative density function (CDF) of 2D maps.

    The returned CDF monotically increases and satisfies:
        cdf[b, -1] = 1

    Args:
        maps: Maps representing weights with which to sum in each bin of shape ([B, C,] H, W).
        r: Radius values of shape ([B,] H, W).
        bin_edges: Edges of the bins of shape (K+1,).
        nbins: Number of bins (K) to use if `bin_edges` is not provided.
        center_mode: Computation mode; one of ["centroid", "image_center", "fixed"].

    Returns:
        cdf: (B, K) with CDF values for each bin.
        edges: (B, K+1) with bin edges.
    """
    pdf, edges = radial_pdf(
        maps,
        r=r,
        bin_edges=bin_edges,
        nbins=nbins,
        center_mode=center_mode,
    )
    dr = edges[:, 1:] - edges[:, :-1]
    cdf = (pdf * dr).cumsum(dim=1)
    return cdf, edges
