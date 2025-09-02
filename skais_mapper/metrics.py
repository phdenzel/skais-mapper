# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Physical metrics for tensor maps and images."""

from __future__ import annotations
import numpy as np
from skais_mapper.profile import (
    _sanitize_ndim,
    _make_grid,
    compute_centers,
    half_mass_radius,
    radial_pdf,
    radial_cdf,
)
from skais_mapper._compat import TORCH_AVAILABLE
from typing import Literal, Callable, TYPE_CHECKING

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch
    import torch.nn.functional as F
else:
    from skais_mapper import _torch_stub as __stub  # noqa
    from skais_mapper._torch_stub import *  # noqa: F401,F403


__all__ = [
    "CenterOffsetError",
    "RadialProfileCurveError",
    "MapTotalError",
    "AsymmetryError",
    "ClumpinessError",
    "PowerSpectrumError",
]


def _aperture_mask(
    B: int,
    H: int,
    W: int,
    centers: torch.Tensor,
    r_in: torch.Tensor,
    r_out: torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Annular mask between r_in and r_out (inclusive) per sample of shape (B, H, W).

    Args:
        B: Batch size.
        H: Mask height.
        W: Mask width.
        centers: Center coordinates of shape (B, 2).
        r_in: Inner radius of shape (B,).
        r_out: Outer radius of shape (B,).
        device: Tensor allocation/computation device.
        dtype: Tensor data type.
    """
    device = torch.device(device) if device is not None else torch.device("cpu")
    dtype = dtype if dtype is not None else torch.float32
    XX, YY = _make_grid(W, H, device=device, dtype=dtype)
    dx = XX - centers[:, 0].view(B, 1, 1)
    dy = YY - centers[:, 1].view(B, 1, 1)
    r = torch.sqrt(dx * dx + dy * dy)
    mask = (r >= r_in.view(B, 1, 1)) & (r <= r_out.view(B, 1, 1))
    return mask.to(dtype)


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
            n_observations: Number of observations seen by the internal state.
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
        self.aggregate = None

    def to(self, device: torch.device):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        self.device = device
        self.n_observations = self.n_observations.to(device=self.device)
        self.aggregate = (
            self.aggregate.to(device=self.device) if self.aggregate is not None else None
        )

    def reset(self, n_observations: int = 0, device: torch.device | None = None) -> None:
        """Reset internal metrics state."""
        self.device = device if device is not None else self.device
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = None

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
            X, Y = _make_grid(W, H, device=maps_.device, dtype=maps_.dtype)
        return compute_centers(maps_, X, Y, mode="centroid", eps=eps)

    @staticmethod
    def _peak_xy(maps: torch.Tensor) -> torch.Tensor:
        """Compute peak coordinates for a batch of images.

        Args:
            maps: Input maps of shape (B, H, W), (B, C, H, W), or (H, W).
        """
        maps_ = _sanitize_ndim(maps)
        B, H, W = maps_.shape
        peak_idcs = torch.argmax(maps_.flatten(1), dim=1)
        y = peak_idcs // W
        x = peak_idcs % W
        peaks = torch.stack([x, y], dim=1)
        return peaks

    @staticmethod
    def _half_mass_radius(maps: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """Compute half-mass radius for a batch of images.

        Args:
            maps: Input maps of shape (B, H, W), (B, C, H, W), or (H, W).
            eps: Numerical stability for divisions.
        """
        return half_mass_radius(maps, eps=eps)

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
        delta = (centers_p - centers_t) * self.pixel_size
        if self.normalize == "image_radius":
            img_r = 0.5 * torch.sqrt(
                torch.tensor(float(H**2 + W**2), device=delta.device, dtype=delta.dtype)
            )
            denom = torch.full(
                (B, 1), self.pixel_size * img_r, device=delta.device, dtype=delta.dtype
            )
        elif self.normalize == "r50":
            denom = self._half_mass_radius(targ_) * self.pixel_size
            denom = denom.view(B, 1)
        else:
            denom = torch.ones((B, 1), dtype=delta.dtype, device=delta.device)
        delta = delta / denom.clamp_min(self.eps)
        if self.aggregate is None:
            self.aggregate = delta
        else:
            self.aggregate = torch.cat([self.aggregate, delta])
        self.n_observations += B

    @torch.no_grad()
    def compute(self, reduction: Callable | None = None) -> torch.Tensor:
        """Return the center offset error over all seen samples."""
        if self.n_observations == 0 or self.aggregate is None:
            return torch.tensor(0., device=self.device)
        dist = self.aggregate.pow(2).sum(dim=1).sqrt()
        if reduction is None:
            reduction = self.reduction
        return reduction(dist)

    def dump(self) -> dict[str, np.ndarray]:
        """Dump non-reduced metric and aggregate data as numpy array."""
        raw = self.aggregate.detach().clone().cpu().numpy()
        dists = self.compute(reduction=torch.nn.Identity()).detach().cpu().numpy()
        return {"dists": dists, "aggregate": raw}


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
        log_bins: bool = False,
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
            log_bins: Use logarithmic binning.
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
        self.log_bins = log_bins
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
            pdf_p, edges = radial_cdf(
                pred_, nbins=self.nbins, log_bins=self.log_bins, center_mode=self.center_mode
            )
            pdf_t, _ = radial_cdf(
                targ_, bin_edges=edges[0], log_bins=self.log_bins, center_mode=self.center_mode
            )
        else:
            pdf_p, edges = radial_pdf(
                pred_, nbins=self.nbins, log_bins=self.log_bins, center_mode=self.center_mode
            )
            pdf_t, _ = radial_pdf(
                targ_, bin_edges=edges[0], log_bins=self.log_bins, center_mode=self.center_mode
            )
        dr = (edges[:, 1:] - edges[:, :-1]).clamp_min(self.eps)
        diff = (pdf_p - pdf_t).abs()
        per_bin_err = diff * dr
        self._r_edges = edges[0]
        self.n_observations += B
        self.aggregate += torch.sum(per_bin_err, dim=0)
        self.lsq_aggregate += torch.sum(per_bin_err.pow(2), dim=0)
        self.min_aggregate = torch.minimum(self.min_aggregate, per_bin_err.amin(dim=0))
        self.max_aggregate = torch.maximum(self.max_aggregate, per_bin_err.amax(dim=0))

    @property
    def mean_per_bin(self) -> torch.Tensor:
        """Mean error per bin."""
        return self.aggregate / max(self.n_observations.item(), 1)

    @property
    def var_per_bin(self) -> torch.Tensor:
        """Error variance per bin."""
        return (self.lsq_aggregate / max(self.n_observations.item(), 1)) - self.mean_per_bin.pow(2)

    @property
    def std_per_bin(self) -> torch.Tensor:
        """Error variance per bin."""
        return self.var_per_bin.sqrt()

    @torch.no_grad()
    def compute(self, reduction: Callable | None = None) -> torch.Tensor:
        """Return the radial profile curve error reduced to a scalar."""
        if reduction is None:
            reduction = self.reduction
        return reduction(self.mean_per_bin)

    def dump(self) -> dict[str, np.ndarray]:
        """Dump non-reduced metric and aggregate data as numpy array."""
        if self.n_observations == 0 or self.aggregate is None or not hasattr(self, "_r_edges"):
            return {}
        raw = self.aggregate.detach().clone().cpu().numpy()
        mean = self.mean_per_bin.detach().clone().cpu().numpy()
        std = self.std_per_bin.detach().clone().cpu().numpy()
        edges = self._r_edges.detach().clone().cpu().numpy()
        return {"aggregate": raw, "mean_per_bin": mean, "std_per_bin": std, "edges": edges}


class MapTotalError:
    """Absolute or relative (mean) error in total integrated map quantity (e.g., flux/mass)."""

    def __init__(
        self,
        relative: bool = True,
        eps: float = 1e-12,
        n_observations: int = 0,
        device: torch.device | None = None,
    ) -> None:
        """Constructor.

        Args:
            relative: If True, returns absolute fractional error:
              |sum(pred) - sum(target)| / (|sum(target)| + eps).
              If False, returns absolute error |sum(pred) - sum(target)|.
            eps: Numerical stability for relative error denominator.
            n_observations: Number of observations seen by the internal state.
            device: Tensor allocation/computation device.
        """
        self.relative = bool(relative)
        self.eps = float(eps)
        self.device = torch.get_default_device() if device is None else device
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = None
        self.total_pred = None
        self.total_target = None

    def to(self, device: torch.device):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        self.device = device
        self.n_observations = self.n_observations.to(device=self.device)
        self.aggregate = (
            self.aggregate.to(device=self.device) if self.aggregate is not None else None
        )
        self.total_pred = (
            self.total_pred.to(device=self.device) if self.total_pred is not None else None
        )
        self.total_target = (
            self.total_target.to(device=self.device) if self.total_target is not None else None
        )

    def reset(self, n_observations: int = 0, device: torch.device | None = None) -> None:
        """Reset internal metrics state."""
        self.device = device if device is not None else self.device
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = None
        self.total_pred = None
        self.total_target = None

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
        if self.aggregate is None:
            self.aggregate = per_sample_err
        else:
            self.aggregate = torch.cat([self.aggregate, per_sample_err], dim=0)
        if self.total_pred is None:
            self.total_pred = sum_p
        else:
            self.total_pred = torch.cat([self.total_pred, sum_p], dim=0)
        if self.total_target is None:
            self.total_target = sum_t
        else:
            self.total_target = torch.cat([self.total_target, sum_t], dim=0)
        self.n_observations += B

    def compute(self) -> torch.Tensor:
        """Return the mean total quantity error over all seen samples."""
        if self.aggregate is None or self.n_observations == 0:
            return torch.tensor(0.0, device=self.device)
        return self.aggregate.sum() / float(self.n_observations)

    def dump(self) -> dict[str, np.ndarray]:
        """Dump non-reduced metric as numpy array."""
        if self.n_observations == 0 or self.aggregate is None:
            return {}
        raw = self.aggregate.detach().clone().cpu().numpy()
        targ = self.total_target.detach().clone().cpu().numpy()
        pred = self.total_pred.detach().clone().cpu().numpy()
        return {"aggregate": raw, "target_total": targ, "pred_total": pred}


class AsymmetryError:
    """Asymmetry (A) metric difference between predicted and target maps.

    A = sum(|I - I_180| within aperture) / (sum(|I| within aperture) + eps)

    The 180-degree rotation is performed about a chosen center:
    - center_mode="image_center": uses geometric image center and fast flip.
    - center_mode="centroid": uses center-of-mass per map; rotation via bilinear sampling.

    Aperture:
    - aperture="full": use the entire image.
    - aperture="r_factor": use a circular aperture with radius r_factor * R50,
      where R50 is the half-mass radius measured on the same map and center.
    """

    def __init__(
        self,
        center_mode: Literal["image_center", "centroid"] = "image_center",
        aperture: Literal["full", "r_factor"] = "r_factor",
        aperture_from: Literal["each", "target", "prediction"] = "each",
        r_factor: float = 1.5,
        eps: float = 1e-12,
        n_observations: int = 0,
        device: torch.device | None = None,
        reduction: Callable | None = torch.mean,
    ) -> None:
        """Constructor.

        Args:
            center_mode: Centering mode for rotation; one of `["centroid", "image_center"]`.
            aperture: Aperture for error calculation; one of `["full", "r_factor"]`.
            aperture_from: Aperture calculation basis; one of `["each", "target", "prediction"]`.
            r_factor: Multiplier for R50 if `aperture="r_factor"`.
            eps: Numerical stability.
            n_observations: Number of observations seen by the internal state.
            device: Tensor allocation/computation device.
            reduction: Reduction function to be used when computing metric scalar.
        """
        if center_mode not in ("image_center", "centroid"):
            raise ValueError("center_mode must be 'image_center' or 'centroid'")
        if aperture not in ("full", "r_factor"):
            raise ValueError("aperture must be 'full' or 'r_factor'")
        if aperture_from not in ("each", "prediction", "target"):
            raise ValueError("aperture_from must be 'target', 'prediction', or 'each'")
        self.device = torch.get_default_device() if device is None else device
        self.reduction = reduction
        self.center_mode = center_mode
        self.aperture = aperture
        self.aperture_from = aperture_from
        self.r_factor = float(r_factor)
        self.eps = float(eps)
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = None
        self.map_aggregate = None
        self._grid = {}

    def to(self, device: torch.device):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        self.device = device
        self.n_observations = self.n_observations.to(device=self.device)
        for k in self._grid:
            XX, YY = self._grid[k]
            self._grid[k] = (XX.to(device=self.device), YY.to(device=self.device))
        self.aggregate = (
            self.aggregate.to(device=self.device) if self.aggregate is not None else None
        )
        self.map_aggregate = (
            self.map_aggregate.to(device=self.device) if self.map_aggregate is not None else None
        )

    def reset(self, n_observations: int = 0, device: torch.device | None = None) -> None:
        """Reset internal metrics state."""
        self.device = device if device is not None else self.device
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = None
        self.map_aggregate = None
        self._grid = {}

    @staticmethod
    def _rotate_180_about_center(
        maps: torch.Tensor,
        center: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Rotate maps by 180 degrees about a given center.

        Args:
            maps: Input maps of shape (B, H, W), (B, C, H, W), or (H, W).
            center: Center coordinates of shape (B, 2).
            eps: Numerical stability for divisions.
        """
        maps_ = _sanitize_ndim(maps)
        B, H, W = maps_.shape
        XX, YY = _make_grid(W, H, device=maps_.device, dtype=maps_.dtype)
        x_prime = 2 * center[:, 0].view(B, 1, 1) - XX
        y_prime = 2 * center[:, 1].view(B, 1, 1) - YY
        x_norm = 2 * x_prime / max(W - 1, 1) - 1
        y_norm = 2 * y_prime / max(H - 1, 1) - 1
        grid = torch.stack((x_norm, y_norm), dim=-1)
        maps_ = maps_.unsqueeze(1)
        rot = F.grid_sample(maps_, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return rot[:, 0, :, :]

    @torch.no_grad()
    def update(self, data: torch.Tensor, prediction: torch.Tensor) -> None:
        """Accumulate batch asymmetry errors.

        Args:
            data: Target maps of shape (B, H, W), (B, C, H, W) or (H, W).
            prediction: Predicted maps of matching shape.
        """
        targ_ = _sanitize_ndim(data)
        pred_ = _sanitize_ndim(prediction)
        if pred_.shape != targ_.shape:
            raise ValueError(f"Input shapes must match, got {targ_.shape} vs {pred_.shape}.")
        B, H, W = pred_.shape
        # Centers
        if (W, H) in self._grid:
            XX, YY = self._grid[(W, H)]
        else:
            XX, YY = _make_grid(W, H, device=targ_.device, dtype=targ_.dtype)
            self._grid[(W, H)] = XX, YY
        centers_t = compute_centers(targ_, XX, YY, eps=self.eps, mode=self.center_mode)
        centers_p = compute_centers(pred_, XX, YY, eps=self.eps, mode=self.center_mode)
        # Rotate maps by 180 degrees about their centers
        if self.center_mode == "image_center":
            targ_rot = torch.flip(targ_, dims=(-2, -1))
            pred_rot = torch.flip(pred_, dims=(-2, -1))
        elif self.center_mode == "centroid":
            targ_rot = self._rotate_180_about_center(targ_, centers_t, eps=self.eps)
            pred_rot = self._rotate_180_about_center(pred_, centers_p, eps=self.eps)
        # Aperture masks
        if self.aperture == "full":
            mask_t = torch.ones_like(targ_)
            mask_p = mask_t
        elif self.aperture == "r_factor":
            # r_factor * R50 using the chosen map's COM
            if self.aperture_from == "target":
                centers = centers_t
                r50 = half_mass_radius(targ_, eps=self.eps, center_mode=self.center_mode)
                r_out = (self.r_factor * r50).clamp_min(1.0)
                zero = torch.zeros_like(r_out)
                mask_t = _aperture_mask(
                    B, H, W, centers, zero, r_out, device=targ_.device, dtype=targ_.dtype
                )
                mask_p = mask_t
            elif self.aperture_from == "prediction":
                centers = centers_p
                r50 = half_mass_radius(pred_, eps=self.eps, center_mode=self.center_mode)
                r_out = (self.r_factor * r50).clamp_min(1.0)
                zero = torch.zeros_like(r_out)
                mask_p = _aperture_mask(
                    B, H, W, centers, zero, r_out, device=pred_.device, dtype=pred_.dtype
                )
                mask_t = mask_p
            else:
                r50_p = half_mass_radius(pred_, eps=self.eps, center_mode=self.center_mode)
                r50_t = half_mass_radius(targ_, eps=self.eps, center_mode=self.center_mode)
                r_out_p = (self.r_factor * r50_p).clamp_min(1.0)
                r_out_t = (self.r_factor * r50_t).clamp_min(1.0)
                zero = torch.zeros_like(r_out_t)
                mask_p = _aperture_mask(
                    B, H, W, centers_p, zero, r_out_p, device=targ_.device, dtype=targ_.dtype
                )
                mask_t = _aperture_mask(
                    B, H, W, centers_t, zero, r_out_t, device=targ_.device, dtype=targ_.dtype
                )
        # Asymmetry maps
        resid_p = mask_p * (pred_ - pred_rot).abs()
        resid_t = mask_t * (targ_ - targ_rot).abs()
        _norm_p = (mask_p * pred_.abs()).sum(dim=(1, 2)).clamp_min(self.eps).reciprocal()
        _norm_t = (mask_t * targ_.abs()).sum(dim=(1, 2)).clamp_min(self.eps).reciprocal()
        A_p = resid_p * _norm_p[:, None, None]
        A_t = resid_t * _norm_t[:, None, None]
        A_p_scalar = A_p.sum(dim=(1, 2))
        A_t_scalar = A_t.sum(dim=(1, 2))
        # Aggregate per sample asymmetries
        if self.aggregate is None:
            self.aggregate = (A_p_scalar - A_t_scalar).abs()
        else:
            self.aggregate = torch.cat([self.aggregate, (A_p_scalar - A_t_scalar).abs()], dim=0)
        # Aggregate map asymmetries
        if self.map_aggregate is None:
            self.map_aggregate = (A_p - A_t).abs().sum(dim=0)
        else:
            self.map_aggregate += (A_p - A_t).abs().sum(dim=0)
        self.n_observations += B

    def compute(self, reduction: Callable | None = None) -> torch.Tensor:
        """Return the reduced asymmetry error over all seen samples."""
        if reduction is None:
            reduction = self.reduction
        if self.n_observations == 0:
            return torch.tensor(0.0, device=self.device)
        return reduction(self.aggregate)

    def dump(self) -> dict[str, np.ndarray]:
        """Dump non-reduced metric components as numpy arrays."""
        if self.n_observations == 0 or (self.aggregate is None and self.map_aggregate is None):
            return {}
        raw = self.aggregate.detach().clone().cpu().numpy()
        map_ = self.map_aggregate.detach().clone().cpu().numpy() / float(self.n_observations)
        return {
            "aggregate": raw,
            "map_aggregate": map_,
        }


class ClumpinessError:
    """Smoothness/Clumpiness difference between two different 2D maps.

    S = sum(max(I - I_s, 0) within annulus) / (sum(I within annulus) + eps)

    The smoothed map I_s is computed via Gaussian blur with a configurable sigma:
    - sigma_mode="pixels": sigma is given in pixels (constant for all samples).
    - sigma_mode="r_factor": sigma = r_factor * R50 (per sample), where R50 is
      measured around the chosen center (image center or centroid).

    The annulus is defined by [r_inner, r_outer]:
    - r_inner = inner_excision * R50  (commonly ~0.2)
    - r_outer = aperture_factor * R50 (commonly ~1.5)
      If aperture_factor <= inner_excision, the annulus collapses to empty.
    """

    def __init__(
        self,
        center_mode: Literal["image_center", "centroid"] = "image_center",
        sigma_mode: Literal["pixels", "r_factor"] = "pixels",
        aperture_from: Literal["each", "target", "prediction"] = "target",
        sigma_pixels: float = 1.0,
        r_factor: float = 1.5,
        inner_excision: float = 0.2,
        eps: float = 1e-12,
        n_observations: int = 0,
        device: torch.device | None = None,
        reduction: Callable | None = torch.mean,
    ) -> None:
        """Constructor.

        Args:
            center_mode: Centering mode for defining R50 and annulus center;
              one of `["centroid", "image_center"]`..
            sigma_mode: "pixels" or "r_factor" for the smoothing scale.
            sigma_pixels: Gaussian sigma in pixels when sigma_mode="pixels".
            aperture_from: Aperture calculation basis; one of `["each", "target", "prediction"]`.
            r_factor: Outer radius multiple of R50 for the annulus (e.g., 1.5).
            inner_excision: Inner exclusion multiple of R50 (e.g., 0.2).
            eps: Numerical stability.
            n_observations: Number of observations seen by the internal state.
            device: Tensor allocation/computation device.
            reduction: Reduction function to be used when computing metric scalar.
        """
        if center_mode not in ("image_center", "centroid"):
            raise ValueError("center_mode must be 'image_center' or 'centroid'")
        if sigma_mode not in ("pixels", "r_factor"):
            raise ValueError("sigma_mode must be 'pixels' or 'r_factor'")
        self.device = torch.get_default_device() if device is None else device
        self.reduction = reduction
        self.center_mode = center_mode
        self.sigma_mode = sigma_mode
        self.sigma_pixels = float(sigma_pixels)
        self.aperture_from = aperture_from
        self.r_factor = float(r_factor)
        self.inner_excision = float(inner_excision)
        self.eps = float(eps)
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = None
        self.map_aggregate = None
        self._grid = {}

    def to(self, device: torch.device):
        """Perform tensor device conversion for all internal tensors.

        Args:
            device: Tensor allocation/computation device.
        """
        self.device = device
        self.n_observations = self.n_observations.to(device=self.device)
        for k in self._grid:
            XX, YY = self._grid[k]
            self._grid[k] = (XX.to(device=self.device), YY.to(device=self.device))
        self.aggregate = (
            self.aggregate.to(device=self.device) if self.aggregate is not None else None
        )
        self.map_aggregate = (
            self.map_aggregate.to(device=self.device)
            if self.map_aggregate is not None
            else None
        )

    def reset(self, n_observations: int = 0, device: torch.device | None = None) -> None:
        """Reset internal metrics state."""
        self.device = device if device is not None else self.device
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = None
        self.map_aggregate = None

    def sigma(
        self,
        batch_size: int,
        r50: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Get the Gaussian kernel sigma tensor for blurring."""
        device = torch.device(device) if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        if self.sigma_mode == "pixels":
            return torch.full((batch_size,), self.sigma_pixels, device=device, dtype=dtype)
        elif self.sigma_mode == "r_factor":
            if r50 is None:
                raise ValueError("r50 must be provided when sigma_mode='r_factor'.")
            return (self.r_factor * r50).clamp_min(0.1).to(device=device, dtype=dtype)

    @staticmethod
    def _gaussian_kernel2d(
        sigma: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create a 2D Gaussian kernel normalized to sum=1 of size = 2*ceil(3*sigma)+1."""
        device = torch.device(device) if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        if sigma <= 0:
            # Degenerate kernel -> identity in convolution; we'll skip conv in that case.
            k = torch.zeros((1, 1), dtype=dtype, device=device)
            k[0, 0] = 1.0
            return k
        radius = int(torch.ceil(torch.tensor(3.0 * sigma)).item())
        ax = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel

    @staticmethod
    def _gaussian_blur(
        maps: torch.Tensor,
        sigma: torch.Tensor,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Blur a map with a Gaussian kernel.

        Args:
            maps: Input maps of shape (B, H, W), (B, C, H, W), or (H, W).
            sigma: Gaussian kernel sigmas (B,).
            eps: Numerical stability for divisions.
        """
        maps_ = _sanitize_ndim(maps)
        B, H, W = maps_.shape
        blurred = []
        for i in range(B):
            if sigma[i] <= 0:
                blurred.append(maps_[i : i + 1, :, :])
                continue
            k = ClumpinessError._gaussian_kernel2d(
                sigma=sigma[i].item(), dtype=maps_.dtype, device=maps_.device
            )
            k = k.view(1, 1, *k.shape)
            pad_h = (k.shape[2] - 1) // 2
            pad_w = (k.shape[3] - 1) // 2
            x = maps_[i].unsqueeze(0).unsqueeze(0)
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
            y = F.conv2d(x, k)
            blurred.append(y[0, 0])
        return torch.stack(blurred, dim=0)

    @torch.no_grad()
    def update(self, data: torch.Tensor, prediction: torch.Tensor) -> None:
        """Accumulate batch clumpiness errors.

        Args:
            data: Target maps of shape (B, H, W), (B, C, H, W) or (H, W).
            prediction: Predicted maps of matching shape.
        """
        targ_ = _sanitize_ndim(data)
        pred_ = _sanitize_ndim(prediction)
        if pred_.shape != targ_.shape:
            raise ValueError(f"Input shapes must match, got {targ_.shape} vs {pred_.shape}.")
        B, H, W = targ_.shape
        device = targ_.device
        dtype = targ_.dtype
        # Aperture mask
        if (W, H) in self._grid:
            XX, YY = self._grid[(W, H)]
        else:
            XX, YY = _make_grid(W, H, device=device, dtype=dtype)
            self._grid[(W, H)] = XX, YY
        if self.aperture_from == "target":
            # Aperture mask
            centers = compute_centers(targ_, XX, YY, eps=self.eps, mode=self.center_mode)
            r50 = half_mass_radius(targ_, eps=self.eps, center_mode=self.center_mode)
            r_in = (self.inner_excision * r50).clamp_min(0.0)
            r_out = (self.r_factor * r50).clamp_min(1.0)
            mask_t = _aperture_mask(
                B, H, W, centers, r_in, r_out, device=device, dtype=dtype
            )
            mask_p = mask_t
            # Smoothing sigmas
            sigmas_t = self.sigma(B, r50=r50, device=device, dtype=dtype)
            sigmas_p = sigmas_t
        elif self.aperture_from == "prediction":
            # Aperture mask
            centers = compute_centers(pred_, XX, YY, eps=self.eps, mode=self.center_mode)
            r50 = half_mass_radius(pred_, eps=self.eps, center_mode=self.center_mode)
            r_in = (self.inner_excision * r50).clamp_min(0.0)
            r_out = (self.r_factor * r50).clamp_min(1.0)
            mask_p = _aperture_mask(
                B, H, W, centers, r_in, r_out, device=device, dtype=dtype
            )
            mask_t = mask_p
            # Smoothing sigmas
            sigmas_p = self.sigma(B, r50=r50, device=device, dtype=dtype)
            sigmas_t = sigmas_p
        else:
            # Aperture masks
            centers_t = compute_centers(targ_, XX, YY, eps=self.eps, mode=self.center_mode)
            centers_p = compute_centers(pred_, XX, YY, eps=self.eps, mode=self.center_mode)
            r50_t = half_mass_radius(targ_, eps=self.eps, center_mode=self.center_mode)
            r50_p = half_mass_radius(pred_, eps=self.eps, center_mode=self.center_mode)
            r_in_t = (self.inner_excision * r50_t).clamp_min(0.0)
            r_in_p = (self.inner_excision * r50_p).clamp_min(0.0)
            r_out_t = (self.r_factor * r50_t).clamp_min(1.0)
            r_out_p = (self.r_factor * r50_p).clamp_min(1.0)
            mask_t = _aperture_mask(
                B, H, W, centers_t, r_in_t, r_out_t, device=device, dtype=dtype
            )
            mask_p = _aperture_mask(
                B, H, W, centers_p, r_in_p, r_out_p, device=device, dtype=dtype
            )
            # Smoothing sigmas
            sigmas_t = self.sigma(B, r50=r50_t, device=device, dtype=dtype)
            sigmas_p = self.sigma(B, r50=r50_p, device=device, dtype=dtype)
        
        # Blur each sample (per-sample sigma)
        blur_t = self._gaussian_blur(targ_, sigmas_t, self.eps)
        blur_p = self._gaussian_blur(pred_, sigmas_p, self.eps)
        # Smoothness values per sample (positive residuals only)
        resid_p = mask_p * (pred_ - blur_p).clamp_min(0.0)
        resid_t = mask_t * (targ_ - blur_t).clamp_min(0.0)
        _norm_p = (mask_p * pred_.abs()).sum(dim=(1, 2)).clamp_min(self.eps).reciprocal()
        _norm_t = (mask_t * targ_.abs()).sum(dim=(1, 2)).clamp_min(self.eps).reciprocal()
        S_p = resid_p * _norm_p[:, None, None]
        S_t = resid_t * _norm_t[:, None, None]
        S_p_scalar = S_p.sum(dim=(1, 2))
        S_t_scalar = S_t.sum(dim=(1, 2))
        # Aggregate per sample asymmetries
        if self.aggregate is None:
            self.aggregate = (S_p_scalar - S_t_scalar).abs()
        else:
            self.aggregate = torch.cat([self.aggregate, (S_p_scalar - S_t_scalar).abs()], dim=0)
        # Aggregate map asymmetries
        if self.map_aggregate is None:
            self.map_aggregate = (S_p - S_t).abs().sum(dim=0)
        else:
            self.map_aggregate += (S_p - S_t).abs().sum(dim=0)
        self.n_observations += B

    def compute(self, reduction: Callable | None = None) -> torch.Tensor | None:
        """Return the mean total quantity error over all seen samples."""
        if reduction is None:
            reduction = self.reduction
        if self.n_observations == 0 or self.aggregate is None:
            return torch.tensor(0.0, device=self.device)
        return reduction(self.aggregate)

    def dump(self) -> dict[str, np.ndarray]:
        """Dump non-reduced metric components as numpy arrays."""
        if self.n_observations == 0 or (self.aggregate is None and self.map_aggregate is None):
            return {}
        raw = self.aggregate.detach().clone().cpu().numpy()
        map_ = self.map_aggregate.detach().clone().cpu().numpy() / float(self.n_observations)
        return {
            "aggregate": raw,
            "map_aggregate": map_,
        }


class PowerSpectrumError:
    """Error between radially averaged 2D power spectra.

    This metric can optionally detrend, apply windows (to avoid edge leakage),
    and computes the power spectrum of 2D maps via FFT with orthogonal norm.
    The power spectra are radially averaged to obtain P(k) curves between kmin and kmax.
    """

    def __init__(
        self,
        nbins: int = 64,
        kmin: float | None = None,
        kmax: float | None = None,
        log_bins: bool = False,
        k_mode: Literal["cycles", "angular"] = "cycles",
        pixel_size: float = 1.0,
        detrend: bool = True,
        window: Literal["hann"] | None = "hann",
        log_power: bool = False,
        per_bin_weighted: bool = True,
        freeze_edges: bool = True,
        eps: float = 1e-12,
        n_observations: int = 0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        reduction: Callable | None = torch.sum,
    ) -> None:
        """Constructor.

        Args:
            nbins: Number of radial k-bins.
            kmin: Minimum k (inclusive). If `None`, uses smallest positive k on the grid.
            kmax: Maximum k (exclusive upper edge). If `None`, uses Nyquist.
            log_bins: If `True`, use logarithmically spaced bin edges in `[kmin, kmax]`.
              `kmin` must be > 0 for log bins; will be clamped to min positive k.
            k_mode: "cycles" for cycles per unit length (default), or
              "angular" for 2π times cycles.
            pixel_size: Physical size per pixel. Units set k-units.
            detrend: Subtract per-image mean before FFT.
            window: "hann" or None.
            log_power: If `True`, compare log10(P + eps) curves instead of linear power.
            per_bin_weighted: If `True`, integrate curve error using bin width dk.
            freeze_edges: If `True`, bin edges and k-bin assignment are fixed
              after first batch.
            eps: Numerical stability constant.
            n_observations: Number of observations seen by the internal state.
            device: Tensor allocation/computation device.
            dtype: Tensor data type for internal tensors.
            reduction: Reduction function to be used when computing metric scalar.
        """
        if window not in ("hann", None):
            raise ValueError("window must be 'hann' or None")
        if k_mode not in ("cycles", "angular"):
            raise ValueError("k_mode must be 'cycles' or 'angular'")
        self.nbins = int(max(1, nbins))
        self.kmin = kmin
        self.kmax = kmax
        self.log_bins = bool(log_bins)
        self.k_mode = k_mode
        self.pixel_size = float(pixel_size)
        self.detrend = bool(detrend)
        self.window = window
        self.log_power = bool(log_power)
        self.per_bin_weighted = bool(per_bin_weighted)
        self.freeze_edges = bool(freeze_edges)
        self.device = torch.get_default_device() if device is None else device
        self.dtype = dtype if dtype is not None else torch.float32
        self.reduction = reduction
        self.eps = float(eps)
        self.n_observations = torch.tensor(n_observations, device=self.device)
        self.aggregate = torch.zeros(self.nbins, device=self.device, dtype=self.dtype)
        self.lsq_aggregate = torch.zeros(self.nbins, device=self.device, dtype=self.dtype)
        self.max_aggregate = -torch.inf * torch.ones(
            self.nbins, device=self.device, dtype=self.dtype
        )
        self.min_aggregate = torch.inf * torch.ones(
            self.nbins, device=self.device, dtype=self.dtype
        )

        self._W: int = None  # Width of the last seen image.
        self._H: int = None  # Height of the last seen image.
        self._k_edges: torch.Tensor = None  # Radial k-bin edges.
        self._dk: torch.Tensor = None  # Width of each k-bin.
        self._bin_index: torch.Tensor = None  # Pixel-to-bin assignment.
        self._hann2d: torch.Tensor = None  # Hann window for the last seen image.

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
        self.aggregate = torch.zeros(self.nbins, device=self.device, dtype=self.dtype)
        self.lsq_aggregate = torch.zeros(self.nbins, device=self.device, dtype=self.dtype)
        self.max_aggregate = -torch.inf * torch.ones(
            self.nbins, device=self.device, dtype=self.dtype
        )
        self.min_aggregate = torch.inf * torch.ones(
            self.nbins, device=self.device, dtype=self.dtype
        )
        self._W = None
        self._H = None
        self._k_edges = None
        self._dk = None
        self._bin_index = None
        self._hann2d = None

    @staticmethod
    def _hann_window_1d(
        N: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create a 1D Hann window."""
        device = torch.device(device) if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        if N <= 1:
            return torch.ones((N,), dtype=dtype, device=device)
        try:
            return torch.hann_window(N, periodic=False, dtype=dtype, device=device)
        except AttributeError:  # Fallback for older PyTorch versions
            n = torch.arange(0, N, dtype=dtype, device=device)
            return 0.5 * (1 - torch.cos(2 * torch.pi * n / (N - 1)))

    @staticmethod
    def _make_hann2d(
        W: int,
        H: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Create a 2D Hann window."""
        device = torch.device(device) if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        hann_x = PowerSpectrumError._hann_window_1d(W, device=device, dtype=dtype)
        hann_y = PowerSpectrumError._hann_window_1d(H, device=device, dtype=dtype)
        return hann_y.view(H, 1) * hann_x.view(1, W)

    def _ensure_kgrid_and_bins(
        self,
        W: int,
        H: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Prepare frequency grid, bin edges, and pixel-to-bin-assignment."""
        if (
            self.freeze_edges
            and self._k_edges is not None
            and self._bin_index is not None
            and (self._W, self._H) == (H, W)
        ):
            return

        device = torch.device(device) if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype

        # Frequency axis arrays
        fx = torch.fft.fftfreq(W, d=self.pixel_size, device=device, dtype=dtype)
        fy = torch.fft.fftfreq(H, d=self.pixel_size, device=device, dtype=dtype)
        if self.k_mode == "angular":
            fx = 2.0 * torch.pi * fx
            fy = 2.0 * torch.pi * fy

        # Default kmin and kmax
        kx = fx.view(1, W).expand(H, W)
        ky = fy.view(H, 1).expand(H, W)
        krad = torch.sqrt(kx * kx + ky * ky)
        kflat = krad.flatten()
        kpos = kflat[kflat > 0]
        kmin_eff = (
            self.kmin if self.kmin is not None else kpos.min().item() if kpos.numel() > 0 else 0.0
        )
        kmax_grid = float(
            torch.max(
                torch.sqrt(fx.max().abs() ** 2 + fy.max().abs() ** 2),
                torch.sqrt(fx.min().abs() ** 2 + fy.min().abs() ** 2),
            )
        )
        kmax_eff = self.kmax if self.kmax is not None else kmax_grid
        if self.log_bins:
            kmin_eff = max(kmin_eff, self.eps)

        # Bin edges
        if self.log_bins:
            edges = torch.logspace(
                torch.log10(torch.tensor(kmin_eff, device=device, dtype=dtype)),
                torch.log10(torch.tensor(kmax_eff, device=device, dtype=dtype)),
                steps=self.nbins + 1,
                base=10.0,
                device=device,
                dtype=dtype,
            )
        else:
            edges = torch.linspace(
                torch.tensor(kmin_eff, device=device, dtype=dtype),
                torch.tensor(kmax_eff, device=device, dtype=dtype),
                steps=self.nbins + 1,
                device=device,
                dtype=dtype,
            )

        # Assign each pixel to a bin: idx in [0, nbins-1], or -1 if out of range
        idx = torch.bucketize(kflat, edges) - 1
        valid = (idx >= 0) & (idx < self.nbins)
        idx = torch.where(valid, idx, torch.full_like(idx, -1))

        # Precompute counts per bin for mean calculation
        counts = torch.bincount(idx[idx >= 0], minlength=self.nbins).to(dtype)
        counts = torch.where(counts > 0, counts, torch.ones_like(counts))

        # Save caches
        self._H, self._W = H, W
        self._k_edges = edges
        self._dk = (edges[1:] - edges[:-1]).clamp_min(self.eps)
        self._bin_index = idx
        self._counts = counts

        # Prepare Hann window if requested
        if self.window == "hann":
            self._hann2d = self._make_hann2d(W, H, device=device, dtype=dtype)
        else:
            self._hann2d = None

    def _radial_mean_power(self, power: torch.Tensor) -> torch.Tensor:
        """Compute radial mean of power given precomputed bin assignments.

        Args:
            power: power map for a single sample of shape (H, W).

        Returns:
            curve: mean power per radial bin of shape (nbins,).
        """
        assert self._bin_index is not None and self._counts is not None
        p_flat = power.flatten()
        idx = self._bin_index
        valid_mask = idx >= 0
        # Accumulate sums per bin
        sums = torch.bincount(idx[valid_mask], weights=p_flat[valid_mask], minlength=self.nbins).to(
            power.dtype
        )
        means = sums / self._counts
        return means

    @torch.no_grad()
    def update(self, data: torch.Tensor, prediction: torch.Tensor) -> None:
        """Accumulate batch power spectra errors.

        Args:
            data: Target maps of shape (B, H, W), (B, C, H, W) or (H, W).
            prediction: Predicted maps of matching shape.
        """
        targ_ = _sanitize_ndim(data)
        pred_ = _sanitize_ndim(prediction)
        if pred_.shape != targ_.shape:
            raise ValueError(f"Input shapes must match, got {targ_.shape} vs {pred_.shape}.")
        B, H, W = targ_.shape

        self._ensure_kgrid_and_bins(W, H, device=targ_.device, dtype=targ_.dtype)

        # Process each sample in the batch
        for i in range(B):
            Ip = pred_[i]
            It = targ_[i]
            if self.detrend:
                Ip = Ip - Ip.mean()
                It = It - It.mean()
            # Windowing
            if self._hann2d is not None:
                Ip = Ip * self._hann2d
                It = It * self._hann2d
            # FFT
            Fp = torch.fft.fft2(Ip, norm="ortho")
            Ft = torch.fft.fft2(It, norm="ortho")
            Pp = Fp.real.pow(2) + Fp.imag.pow(2)
            Pt = Ft.real.pow(2) + Ft.imag.pow(2)
            # Radial mean curves
            curve_p = self._radial_mean_power(Pp)
            curve_t = self._radial_mean_power(Pt)
            if self.log_power:
                curve_p = torch.log10(curve_p + self.eps)
                curve_t = torch.log10(curve_t + self.eps)
            if self.per_bin_weighted:
                _norm_p = (curve_p * self._dk).sum().clamp_min(self.eps).reciprocal()
                _norm_t = (curve_t * self._dk).sum().clamp_min(self.eps).reciprocal()
            else:
                _norm_p = curve_p.sum().clamp_min(self.eps).reciprocal()
                _norm_t = curve_t.sum().clamp_min(self.eps).reciprocal()
            curve_p = curve_p * _norm_p
            curve_t = curve_t * _norm_t
            per_bin_err = (curve_p - curve_t).abs()
            if self.per_bin_weighted:
                per_bin_err = per_bin_err * self._dk
            self.aggregate += per_bin_err
            self.lsq_aggregate += per_bin_err.pow(2)
            self.min_aggregate = torch.minimum(self.min_aggregate, per_bin_err.amin(dim=0))
            self.max_aggregate = torch.maximum(self.max_aggregate, per_bin_err.amax(dim=0))
            self.n_observations += 1

    @property
    def mean_per_bin(self) -> torch.Tensor:
        """Return the mean per-bin error curve."""
        if self.n_observations == 0:
            return self.aggregate
        return self.aggregate / float(self.n_observations)

    @property
    def var_per_bin(self) -> torch.Tensor:
        """Error variance per bin."""
        return (self.lsq_aggregate / self.n_observations) - self.mean_per_bin.pow(2)

    @property
    def std_per_bin(self) -> torch.Tensor:
        """Error variance per bin."""
        return self.var_per_bin.sqrt()

    @torch.no_grad()
    def compute(self, reduction: Callable | None = None) -> torch.Tensor:
        """Return the power spectrum curve error reduced to a scalar."""
        if reduction is None:
            reduction = self.reduction
        return reduction(self.mean_per_bin)

    def dump(self) -> dict[str, np.ndarray]:
        """Dump non-reduced metric and aggregate data as numpy array."""
        if self.n_observations == 0:
            return {}
        raw = self.aggregate.detach().clone().cpu().numpy()
        mean = self.mean_per_bin.detach().clone().cpu().numpy()
        std = self.std_per_bin.detach().clone().cpu().numpy()
        edges = self._k_edges.detach().clone().cpu().numpy()
        return {"aggregate": raw, "mean_per_bin": mean, "std_per_bin": std, "edges": edges}
