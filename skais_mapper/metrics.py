# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical metrics for tensor maps and images."""

from __future__ import annotations
from skais_mapper.profile import (
    _sanitize_ndim,
    _make_grid,
    compute_centers,
    cumulative_radial_histogram,
    radial_pdf,
    radial_cdf,
)
from skais_mapper._compat import TORCH_AVAILABLE
from typing import Literal, Callable, TYPE_CHECKING

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
else:
    from skais_mapper import _torch_stub as __stub  # noqa
    from skais_mapper._torch_stub import *  # noqa: F401,F403


__all__ = [
    "CenterOffsetError",
    "RadialProfileCurveError",
    "MapTotalError",
]


class CenterOffsetError:
    """Center offset between two maps: center-of-mass or peak position distance.

    Distances can be reported in pixels (scaled by pixel_size) or normalized by:
      - image_radius: half the image diagonal
      - half-mass radius: with respect to a reference map
    """

    def __init__(
        self,
        center: Literal["com", "peak"] = "com",
        normalize: Literal["image_radius", "r50"] = "image_radius",
        pixel_size: float = 1.0,
        eps: float = 1e-12,
        n_observations: int = 0,
        device: torch.device | None = None,
        reduction: Callable | None = torch.mean,
    ):
        """Constructor.

        Args:
            center: Centering mode; one of ["com", "peak"].
            normalize: Normalization mode; one of ["image_radius", "r50"].
            pixel_size: Physical size per pixel (multiplies distances).
            eps: Numerical stability for divisions.
            n_observations: Number of observations (bins) seen by the internal state.
            device: Tensor allocation/computation device.
            reduction: Reduction function to be used when computing metric scalar.
        """
        self.reduction = reduction
        self.device = torch.get_default_device() if device is None else device
        if center not in ("com", "peak"):
            raise ValueError("Input center must be 'com' or 'peak'")
        if normalize not in ("image_radius", "r50"):
            raise ValueError("Input normalize must be 'image_radius' or 'r50'")
        self.center = center
        self.normalize = normalize
        self.pixel_size = float(pixel_size)
        self.eps = float(eps)
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = torch.zeros(1, device=self.device)
        self.max_aggregate = -torch.inf * torch.ones(1, device=self.device)
        self.min_aggregate = torch.inf * torch.ones(1, device=self.device)

    def to(self, device: torch.device):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        self.device = device
        self.n_observations = self.n_observations.to(device=self.device)
        self.aggregate = self.aggregate.to(device=self.device)
        self.max_aggregate = self.max_aggregate.to(device=self.device)
        self.min_aggregate = self.min_aggregate.to(device=self.device)

    def reset(self, n_observations: int = 0, device: torch.device | None = None) -> None:
        """Reset internal metrics state."""
        self.device = device if device is not None else self.device
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = torch.zeros(self.nbins, device=self.device)
        self.max_aggregate = -torch.inf * torch.ones(self.nbins, device=self.device)
        self.min_aggregate = torch.inf * torch.ones(self.nbins, device=self.device)

    @staticmethod
    def _com_xy(
        maps: torch.Tensor,
        X: torch.Tensor | None = None,
        Y: torch.Tensor | None = None,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Compute center-of-mass coordinates for a batch of images.

        Args:
            maps: Input maps of shape (B, H, W), (B, C, H, W), or (H, W).
            X: X coordinate grid of shape (H, W).
            Y: Y coordinate grid of shape (H, W).
            eps: Numerical stability for divisions.
        """
        maps_ = _sanitize_ndim(maps)
        if X is None or Y is None:
            B, H, W = maps_.shape
            X, Y = _make_grid(W, H, device=maps.device, dtype=maps.dtype)
        return compute_centers(maps_, X, Y, mode="centroid", eps=eps)

    @staticmethod
    def _peak_xy(maps: torch.Tensor) -> torch.Tensor:
        """Compute peak coordinates for a batch of images.

        Args:
            maps: Input maps of shape (B, H, W), (B, C, H, W), or (H, W).
        """
        maps_ = _sanitize_ndim(maps)
        B, H, W = maps_.shape
        peaks = torch.argmax(maps_, dim=(-2, -1))
        return peaks

    @staticmethod
    def _half_mass_radius(maps: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Compute half-mass radius for a batch of images.

        Args:
            maps: Input maps of shape (B, H, W), (B, C, H, W), or (H, W).
            eps: Numerical stability for divisions.
        """
        maps_ = _sanitize_ndim(maps)
        B, H, W = maps_.shape
        chist, edges, _ = cumulative_radial_histogram(maps_, nbins=200)
        m_half = 0.5 * chist[:, -1]
        ge = chist >= m_half.view(B, 1)
        r = 0.5 * (edges[:, 1:] + edges[:, :-1]).clamp_min(eps)
        r50 = torch.gather(r, 1, ge.float().argmax(dim=1, keepdim=True)).squeeze(1)
        return r50

    @torch.no_grad()
    def update(self, data: torch.Tensor, prediction: torch.Tensor) -> None:
        """Aggregate batch center offsets.

        Args:
            data: Target maps of shape (B, H, W), (B, C, H, W) or (H, W).
            prediction: Predicted maps of matching shape.
        """
        targ_ = _sanitize_ndim(data)
        pred_ = _sanitize_ndim(prediction)
        if targ_.shape != pred_.shape:
            raise ValueError(f"Input shapes must match, got {targ_.shape} vs {pred_.shape}.")
        B, H, W = pred_.shape
        if self.center == "com":
            centers_t = self._com_xy(targ_)
            centers_p = self._com_xy(pred_)
        else:
            centers_t = self._peak_xy(targ_)
            centers_p = self._peak_xy(pred_)
        delta2 = (centers_p - centers_t) ** 2
        dist = torch.sqrt(delta2.sum(dim=1)) * self.pixel_size
        if self.normalize == "image_radius":
            denom = (
                0.5
                * torch.sqrt(torch.tensor(H**2 + W**2, dtype=dist.dtype, device=dist.device))
                * self.pixel_size
            )
        elif self.normalize == "r50":
            denom = self._half_mass_radius(targ_) * self.pixel_size
        else:
            denom = 1
        dist = dist / denom.clamp_min(self.eps)
        self.aggregate += dist.sum(dim=0)
        self.min_aggregate = torch.min(self.min_aggregate, dist.amin())
        self.max_aggregate = torch.max(self.max_aggregate, dist.amax())
        self.n_observations += B

    @torch.no_grad()
    def compute(self, reduction: Callable | None = None) -> torch.Tensor:
        """Return the center offset error over all seen samples."""
        if self.n_observations == 0:
            return self.aggregate
        if reduction is None:
            reduction = self.reduction
        return reduction(self.aggregate / float(self.n_observations))


class RadialProfileCurveError:
    """Shape error between radial density profiles of predicted and target maps.

    This metric compares the azimuthally-averaged radial distribution shapes by
    computing a radial probability density function (PDF) for each input map and
    integrating the difference across radius.

    Note: This metric is insensitive to global normalization/flux differences.
    Use `MapTotalError` to capture amplitude discrepancies.
    """

    def __init__(
        self,
        nbins: int = 100,
        center_mode: Literal["centroid", "image_center", "fixed"] = "image_center",
        cumulative: bool = False,
        eps: float = 1e-12,
        n_observations: int = 0,
        device: torch.device | None = None,
        reduction: Callable | None = torch.sum,
    ) -> None:
        """Constructor:

        Args:
            nbins: Number of radial bins.
            center_mode: Centering mode for radial profiles; one of
              `["centroid", "image_center", "fixed"]`.
            cumulative: Whether to compare cumulative radial profiles.
            eps: Numerical stability for divisions.
            n_observations: Number of observations (bins) seen by the internal state.
            device: Tensor allocation/computation device.
            reduction: Reduction function to be used when computing metric scalar.
        """
        self.reduction = reduction
        self.device = torch.get_default_device() if device is None else device
        self.nbins = int(max(1, nbins))
        self.center_mode = center_mode
        self.cumulative = bool(cumulative)
        self.eps = float(eps)
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = torch.zeros(self.nbins, device=self.device)
        self.lsq_aggregate = torch.zeros(self.nbins, device=self.device)
        self.max_aggregate = -torch.inf * torch.ones(nbins, device=self.device)
        self.min_aggregate = torch.inf * torch.ones(nbins, device=self.device)

    def to(self, device: torch.device):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        self.device = device
        self.n_observations = self.n_observations.to(device=self.device)
        self.aggregate = self.aggregate.to(device=self.device)
        self.lsq_aggregate = self.lsq_aggregate.to(device=self.device)
        self.max_aggregate = self.max_aggregate.to(device=self.device)
        self.min_aggregate = self.min_aggregate.to(device=self.device)

    def reset(self, n_observations: int = 0, device: torch.device | None = None) -> None:
        """Reset internal metrics state."""
        self.device = device if device is not None else self.device
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = torch.zeros(self.nbins, device=self.device)
        self.lsq_aggregate = torch.zeros(self.nbins, device=self.device)
        self.max_aggregate = -torch.inf * torch.ones(self.nbins, device=self.device)
        self.min_aggregate = torch.inf * torch.ones(self.nbins, device=self.device)

    @torch.no_grad()
    def update(self, data: torch.Tensor, prediction: torch.Tensor) -> None:
        """Accumulate batch errors.

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
            pdf_p, edges = radial_cdf(pred_, nbins=self.nbins, center_mode=self.center_mode)
            pdf_t, _ = radial_cdf(targ_, bin_edges=edges[0], center_mode=self.center_mode)
        else:
            pdf_p, edges = radial_pdf(pred_, nbins=self.nbins, center_mode=self.center_mode)
            pdf_t, _ = radial_pdf(targ_, bin_edges=edges[0], center_mode=self.center_mode)
        dr = (edges[:, 1:] - edges[:, :-1]).clamp_min(self.eps)
        diff = (pdf_p - pdf_t).abs()
        per_bin_err = diff * dr
        self.n_observations += B
        self.aggregate += torch.sum(per_bin_err, dim=0)
        self.lsq_aggregate += torch.sum(per_bin_err.pow(2), dim=0)
        self.min_aggregate = torch.min(self.min_aggregate, per_bin_err.amin(dim=0))
        self.max_aggregate = torch.max(self.max_aggregate, per_bin_err.amax(dim=0))

    @property
    def mean_per_bin(self) -> torch.Tensor:
        """Mean error per bin."""
        return self.aggregate / self.n_observations

    @property
    def var_per_bin(self) -> torch.Tensor:
        """Error variance per bin."""
        return (self.lsq_aggregate / self.n_observations) - self.mean_per_bin.pow(2)

    @property
    def std_per_bin(self) -> torch.Tensor:
        """Error variance per bin."""
        return self.var_per_bin.sqrt() / self.n_observations

    @torch.no_grad()
    def compute(self, reduction: Callable | None = None) -> torch.Tensor:
        """Return the radial profile curve error reduced to a scalar."""
        if reduction is None:
            reduction = self.reduction
        return reduction(self.mean_per_bin)


class MapTotalError:
    """Absolute or relative error in total integrated map quantity (e.g., flux/mass)."""

    def __init__(
        self,
        relative: bool = True,
        eps: float = 1e-12,
        n_observations: int = 0,
        device: torch.device | None = None,
        reduction: Callable | None = torch.sum,
    ) -> None:
        """Constructor.

        Args:
            relative: If True, returns absolute fractional error:
              |sum(pred) - sum(target)| / (|sum(target)| + eps).
              If False, returns absolute error |sum(pred) - sum(target)|.
            eps: Numerical stability for relative error denominator.
            n_observations: Number of observations (bins) seen by the internal state.
            device: Tensor allocation/computation device.
            reduction: Reduction function to be used when computing metric scalar.
        """
        self.relative = bool(relative)
        self.eps = float(eps)
        self.device = torch.get_default_device() if device is None else device
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = torch.zeros(1, device=self.device)
        self.max_aggregate = -torch.inf * torch.ones(1, device=self.device)
        self.min_aggregate = torch.inf * torch.ones(1, device=self.device)

    def to(self, device: torch.device):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        self.device = device
        self.n_observations = self.n_observations.to(device=self.device)
        self.aggregate = self.aggregate.to(device=self.device)
        self.max_aggregate = self.max_aggregate.to(device=self.device)
        self.min_aggregate = self.min_aggregate.to(device=self.device)

    def reset(self, n_observations: int = 0, device: torch.device | None = None) -> None:
        """Reset internal metrics state."""
        self.device = device if device is not None else self.device
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = torch.zeros(1, device=self.device)
        self.max_aggregate = -torch.inf * torch.ones(1, device=self.device)
        self.min_aggregate = torch.inf * torch.ones(1, device=self.device)

    @torch.no_grad()
    def update(self, data: torch.Tensor, prediction: torch.Tensor) -> None:
        """Accumulate batch errors.

        Args:
            data: Target maps of shape (B, H, W), (B, C, H, W) or (H, W).
            prediction: Predicted maps of matching shape.
        """
        targ_ = _sanitize_ndim(data)
        pred_ = _sanitize_ndim(prediction)
        if pred_.shape != targ_.shape:
            raise ValueError(f"Input shapes must match, got {targ_.shape} vs {pred_.shape}.")
        B, H, W = pred_.shape
        sum_p = pred_.flatten(1).sum(dim=1)
        sum_t = targ_.flatten(1).sum(dim=1)
        if self.relative:
            per_sample_err = (sum_p - sum_t).abs() / (sum_t.abs() + self.eps)
        else:
            per_sample_err = (sum_p - sum_t).abs()
        self.aggregate += per_sample_err.sum(dim=0)
        self.min_aggregate = torch.min(self.min_aggregate, per_sample_err.amin())
        self.max_aggregate = torch.max(self.max_aggregate, per_sample_err.amax())
        self.n_observations += B

    def compute(self) -> torch.Tensor:
        """Return the mean total quantity error over all seen samples."""
        if self.n_observations == 0:
            return self.aggregate
        return self.aggregate / float(self.n_observations)
