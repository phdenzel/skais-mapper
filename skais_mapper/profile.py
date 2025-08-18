# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Profile routines for 2D tensor maps and images."""

from __future__ import annotations
from skais_mapper._compat import TORCH_AVAILABLE
from typing import Literal, Callable, TYPE_CHECKING

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


def half_mass_radius(maps: torch.Tensor, eps: float = 1e-12, **kwargs) -> torch.Tensor:
    """Compute half-mass radius for a batch of images.

    Args:
        maps: Input maps of shape (B, H, W), (B, C, H, W), or (H, W).
        eps: Numerical stability for divisions.
        kwargs: Additional keyword arguments for `cumulative_radial_histogram`.
    """
    maps_ = _sanitize_ndim(maps)
    B, H, W = maps_.shape
    chist, edges, _ = cumulative_radial_histogram(maps_, nbins=200, **kwargs)
    m_half = 0.5 * chist[:, -1]
    ge = chist >= m_half.view(B, 1)
    r = 0.5 * (edges[:, 1:] + edges[:, :-1]).clamp_min(eps)
    r50 = torch.gather(r, 1, ge.float().argmax(dim=1, keepdim=True)).squeeze(1)
    return r50


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


class RadialProfile:
    """Radial profile metric for recording and aggregating map profiles."""

    def __init__(
        self,
        nbins: int = 100,
        center_mode: Literal["centroid", "image_center", "fixed"] = "image_center",
        cumulative: bool = False,
        eps: float = 1e-12,
        device: torch.device | None = None,
        reduction: Callable | None = torch.mean,
    ) -> None:
        """Constructor:

        Args:
            nbins: Number of radial bins.
            center_mode: Centering mode for radial profiles; one of
              `["centroid", "image_center", "fixed"]`.
            cumulative: Whether to compare cumulative radial profiles.
            eps: Numerical stability for divisions.
            device: Tensor allocation/computation device.
            reduction: Reduction function to be used when computing profile summary.
        """
        self.device = torch.get_default_device() if device is None else device
        self.nbins = int(max(1, nbins))
        self.center_mode = center_mode
        self.cumulative = bool(cumulative)
        self.eps = float(eps)
        self.target_aggregate = None
        self.aggregate = None
        self.edges = None
        self.reduction = reduction

    def to(self, device: torch.device):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        self.device = device
        self.aggregate = (
            self.aggregate.to(device=self.device) if self.aggregate is not None else None
        )

    def reset(self, n_observations: int = 0, device: torch.device | None = None) -> None:
        """Reset internal metrics state."""
        self.device = device if device is not None else self.device
        self.target_aggregate = None
        self.aggregate = None

    @torch.no_grad()
    def update(self, data: torch.Tensor, prediction: torch.Tensor) -> None:
        """Accumulate profile batches.

        Args:
            data: Target maps of shape (B, H, W), (B, C, H, W) or (H, W).
            prediction: Predicted maps of matching shape.
        """
        targ_ = _sanitize_ndim(data)
        pred_ = _sanitize_ndim(prediction)
        if targ_.shape != pred_.shape:
            raise ValueError(f"Input shapes must match, got {targ_.shape} vs {pred_.shape}")
        B, H, W = pred_.shape
        # Compute radial PDFs for both maps using identical binning.
        if self.cumulative:
            prf_p, edges = radial_cdf(pred_, nbins=self.nbins, center_mode=self.center_mode)
            prf_t, _ = radial_cdf(targ_, bin_edges=edges[0], center_mode=self.center_mode)
        else:
            prf_p, edges = radial_pdf(pred_, nbins=self.nbins, center_mode=self.center_mode)
            prf_t, _ = radial_pdf(targ_, bin_edges=edges[0], center_mode=self.center_mode)
        if self.edges is None:
            self.edges = edges[0]
        if self.target_aggregate is None:
            self.target_aggregate = prf_t
        else:
            self.target_aggregrate = torch.cat((self.target_aggregate, prf_t), dim=0)
        if self.aggregate is None:
            self.aggregate = prf_p
        else:
            self.aggregate = torch.cat((self.aggregate, prf_p), dim=0)

    @torch.no_grad()
    def compute(self, reduction: Callable | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a radial profile summary (reduction across batches)."""
        if reduction is None:
            reduction = self.reduction
        return reduction(self.aggregate, dim=0), reduction(self.target_aggregate, dim=0)
