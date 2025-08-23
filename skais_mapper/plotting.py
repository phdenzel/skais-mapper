# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Plotting routines for maps and images."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import astropy.units as au
from skais_mapper.utils import SkaisColorMaps, get_run_id, alias_kw
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colors import Colormap
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from functools import singledispatch
from typing import Any, Literal, TYPE_CHECKING
from collections.abc import Sequence
from skais_mapper._compat import TORCH_AVAILABLE

if TORCH_AVAILABLE or TYPE_CHECKING:
    import torch


def _from_batch(
    batch: torch.Tensor | np.ndarray | Sequence[torch.Tensor | np.ndarray],
    metadata: dict | Sequence[dict] = {},
    batch_idx: int | Sequence[int] | None = None,
) -> tuple[torch.Tensor, dict | None]:
    """Select data from batch using appropriate batch indexing.

    Args:
        batch: Data batch to be indexed.
        metadata: Corresponding metadata dictionaries.
        batch_idx: If not `None`, data is a single or multiple batches of images
    """
    data = batch
    if batch_idx is not None:
        batch_idx = (batch_idx,) if isinstance(batch_idx, int) else batch_idx
        for idx in batch_idx:
            data = data[idx]
            if isinstance(metadata, Sequence):
                metadata = metadata[idx]
        if isinstance(metadata, dict):
            for key in metadata:
                if isinstance(metadata[key], torch.Tensor):
                    metadata[key] = metadata[key].detach().clone()[batch_idx[-1]]
                if isinstance(metadata[key], np.ndarray | list | tuple):
                    metadata[key] = metadata[key][batch_idx[-1]]
    if len(data.shape) >= 2:
        data = data.squeeze()
    return data, metadata


def _get_cmap(
    name: str | None = None,
    type_: str | None = None,
    under: str | None = None,
    over: str | None = None,
    bad: str | None = None,
) -> Colormap:
    """Get colormap by data map type or by name directly.

    Args:
        name: Name of the colormap (names from SkaisColorMaps or matplotlib allowed).
        type_: Type of the map.
        under: If not `None`, sets the colors below the colormap.
        over: If not `None`, sets the colors above the colormap.
        bad: If not `None`, sets the colormap colors for bad values.
    """
    if type_ is not None:
        match type_:
            case "gas":
                cmap = "gaseous"
            case "dm":
                cmap = "obscura"
            case "star":
                cmap = "hertzsprung"
            case "bfield":
                cmap = "gravic"
            case "temp":
                cmap = "phoenix"
            case "hi/21cm":
                cmap = "nava"
            case _:
                cmap = "gaseous"
    elif name is not None:
        if hasattr(SkaisColorMaps, name):
            cmap = name
        elif name in list(colormaps):
            cmap = name
        else:
            cmap = "gaseous"
    else:
        cmap = "gaseous"
    cmap = getattr(SkaisColorMaps, cmap) if hasattr(SkaisColorMaps, cmap) else plt.get_cmap(cmap)
    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)
    if bad is not None:
        cmap.set_bad(bad)
    return cmap


def _symbol_from_class(class_type: str | None = None):
    """Get physical/mathematical symbol (LaTeX notation) from class string.

    Args:
        class_type: Quantity name; one of
          ["dm", "star", "gas", "hi", "hi/21cm", "temp", "bfield"]
    """
    match class_type:
        case "gas":
            symbol = "\u03a3" + r"$_{\mathrm{gas}}$"
        case "hi":
            symbol = "\u03a3" + r"$_{\mathrm{HI}}$"
        case "dm":
            symbol = "\u03a3" + r"$_{\mathrm{dm}}$"
        case "star":
            symbol = "\u03a3" + r"$_{\mathrm{star}}$"
        case "bfield":
            symbol = "|B|"
        case "temp":
            symbol = "T"
        case "hi/21cm":
            symbol = r"T$_{\mathrm{b}}$"
        case _:
            symbol = "\u03a3"
    return symbol


@alias_kw("colormap", "cmap")
@alias_kw("colorbar", "cbar")
@alias_kw("colorbar_label", "cbar_label")
@alias_kw("colorbar_ax", "cbar_ax")
@alias_kw("colorbar_pad", "cbar_pad")
@alias_kw("colorbar_loc", "cbar_loc")
def _plot_data(
    data: np.ndarray,
    info: dict = {},
    ax: Axes | None = None,
    extent: Sequence[float] | None = None,
    colormap: None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    colorbar_ax: Axes | None = None,
    colorbar_pad: float | None = None,
    colorbar_loc: Literal["left", "right"] = "right",
    xlabel: str | None = None,
    ylabel: str | None = None,
    scalebar: bool = False,
    scalebar_label: str | None = None,
    scalebar_fontsize: float | None = None,
    scalebar_r: float = 0.25,
    scalebar_h: float = 0.075,
    scalebar_pad: float = 0.3,
    scalebar_loc: Literal[
        "upper left", "upper right", "lower left", "lower right", "center"
    ] = "lower right",
    scalebar_color: str = "black",
    savefig: bool = False,
    path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    verbose: bool = False,
    **kwargs,
) -> AxesImage:
    """Plot image data array.

    Args:
        data: Array(s) of a map (or maps).
        info: Supplementary information about the data.
        ax: Optional matplotlib Axes to draw into. If `None`, a new figure/axes is created.
        extent: Extent of the map in physical units.
        colormap: Name of the colormap to use for plotting.
        colorbar: If `True`, include the colorbar in the plot.
        colorbar_label: Label of the colorbar (if enabled).
        colorbar_ax: Optional matplotlib Axes to draw the colorbar into.
          If `None`, defaults to `ax`.
        colorbar_pad: Colorbar padding.
        colorbar_loc: Colorbar location.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        scalebar: Use the a scalebar (usually makes x/yticks unnecessary).
        scalebar_label: Scale bar label; if `None`, physical extent unit is used.
        scalebar_fontsize: Font size of the scalebar label.
        scalebar_r: Scale bar length fraction of the image data.
        scalebar_h: Scale bar vertical size (in units of transform coordinate system).
        scalebar_pad: Padding around the label and size bar (in fraction of the font size).
        scalebar_loc: Location of the size bar.
        scalebar_color: Color of the scale bar.
        savefig: If `True`, save the plot to file.
        path: Filename or filepath where the figure is saved.
        show: If `True`, show the plot.
        close: If `True`, closes the figure.
        verbose: Print results to stdout.
        kwargs: Additional keyword arguments for `matplotlib.pyplot.imshow`.
    """
    figsize = kwargs.pop("figsize", [6.4, 4.8])
    dpi = kwargs.pop("dpi", 200)
    transparent = kwargs.pop("transparent", True)
    bbox_inches = kwargs.pop("bbox_inches", "tight")
    pad_inches = kwargs.pop("pad_inches", 0)
    kwargs.setdefault("interpolation", "bicubic")
    kwargs.setdefault("origin", "lower")
    colormap = _get_cmap(colormap, info["class"] if "class" in info else None, under="k", bad="k")
    if "extent" in info and extent is None:
        extent = info["extent"]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    img = ax.imshow(data, cmap=colormap, extent=extent, **kwargs)
    if colorbar:
        cbar_kw = {"location": colorbar_loc}
        if colorbar_label is None:
            colorbar_label = _symbol_from_class(info.get("class", None))
            if "units" in info:
                colorbar_label += f" [{info['units']}]"
        colorbar_label = colorbar_label.replace("solMass", "M" + r"$_{\odot}$").replace(
            "2", "\u00b2"
        )
        cbar_kw["label"] = colorbar_label
        if colorbar_pad is not None:
            cbar_kw["pad"] = colorbar_pad
        if colorbar_ax is not None:
            cbar_kw["cax"] = colorbar_ax
        else:
            cbar_kw["ax"] = ax
        try:
            fig.colorbar(img, **cbar_kw)
        except ValueError:
            pass
    if "units_extent" in info and xlabel is None:
        xlabel = f"[{info['units_extent']}]"
    if "units_extent" in info and ylabel is None:
        ylabel = f"[{info['units_extent']}]"
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if scalebar:
        if scalebar_label is None and "units_extent" in info:
            scalebar_label = f"{info['units_extent']}"
        elif scalebar_label is None:
            scalebar_label = "px"
        scalebar_len = round(
            float(extent[1] - extent[0]) * scalebar_r if extent is not None
            else scalebar_r * data.shape[1]
        )
        fp = FontProperties(size=scalebar_fontsize) if scalebar_fontsize else None
        bar = AnchoredSizeBar(
            transform=ax.transData,
            size=scalebar_len,
            label=f"{scalebar_len:g} {scalebar_label}",
            label_top=True,
            loc=scalebar_loc,
            pad=scalebar_pad,
            color=scalebar_color,
            frameon=False,
            size_vertical=scalebar_h,
            fontproperties=fp,
        )
        ax.add_artist(bar)
    if savefig:
        if path is None:
            path = f"./{get_run_id()}_image.png"
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        fig.savefig(
            path, dpi=dpi, bbox_inches=bbox_inches, pad_inches=pad_inches, transparent=transparent
        )
        if verbose:
            print(f"Saving to [png]: {path}")
        if close:
            plt.close(fig)
    if show:
        plt.show()
    return img


@singledispatch
@alias_kw("colormap", "cmap")
@alias_kw("colorbar", "cbar")
@alias_kw("colorbar_label", "cbar_label")
@alias_kw("colorbar_ax", "cbar_ax")
@alias_kw("colorbar_pad", "cbar_pad")
@alias_kw("colorbar_loc", "cbar_loc")
def plot_image(
    data: Any,
    batch_idx: int | None = None,
    extent: Sequence[float] | None = None,
    colormap: None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    colorbar_ax: Axes | None = None,
    colorbar_pad: float | None = None,
    colorbar_loc: Literal["left", "right"] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    scalebar: bool = False,
    scalebar_label: str | None = None,
    scalebar_fontsize: float | None = None,
    scalebar_r: float = 0.25,
    scalebar_h: float = 0.075,
    scalebar_pad: float = 0.3,
    scalebar_loc: Literal[
        "upper left", "upper right", "lower left", "lower right", "center"
    ] = "lower right",
    scalebar_color: str = "black",
    savefig: bool = False,
    path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Plot map from an array, tensor, or batch(es).

    Args:
        data: Array(s) of a map (or maps).
        batch_idx: If not `None`, data is a single or multiple batches of images.
        info: Supplementary information about the data.
        ax: Optional matplotlib Axes to draw into. If `None`, a new figure/axes is created.
        extent: Extent of the map in physical units.
        colormap: Name of the colormap to use for plotting.
        colorbar: If `True`, include the colorbar in the plot.
        colorbar_label: Label of the colorbar (if enabled).
        colorbar_ax: Optional matplotlib Axes to draw the colorbar into.
          If `None`, defaults to `ax`.
        colorbar_pad: Colorbar padding.
        colorbar_loc: Colorbar location.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        scalebar: Use the a scalebar (usually makes x/yticks unnecessary).
        scalebar_label: Scale bar label; if `None`, physical extent unit is used.
        scalebar_fontsize: Font size of the scalebar label.
        scalebar_r: Scale bar length fraction of the image data.
        scalebar_h: Scale bar vertical size (in units of transform coordinate system).
        scalebar_pad: Padding around the label and size bar (in fraction of the font size).
        scalebar_loc: Location of the size bar.
        scalebar_color: Color of the scale bar.
        savefig: If `True`, save the plot to file.
        path: Filename or filepath where the figure is saved.
        show: If `True`, show the plot.
        close: If `True`, closes the figure.
        verbose: Print results to stdout.
        kwargs: Additional keyword arguments for `matplotlib.pyplot.imshow`.
    """
    raise NotImplementedError(f"Invalid data type {type(data)}.")


@plot_image.register(torch.Tensor)
@alias_kw("colormap", "cmap")
@alias_kw("colorbar", "cbar")
@alias_kw("colorbar_label", "cbar_label")
@alias_kw("colorbar_ax", "cbar_ax")
@alias_kw("colorbar_pad", "cbar_pad")
@alias_kw("colorbar_loc", "cbar_loc")
def plot_image_tensor(
    data: torch.Tensor,
    batch_idx: int | Sequence[int] | None = None,
    ax: Axes | None = None,
    extent: Sequence[float] | None = None,
    colormap: Colormap | None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    colorbar_ax: Axes | None = None,
    colorbar_pad: float | None = None,
    colorbar_loc: Literal["left", "right"] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    scalebar: bool = False,
    scalebar_label: str | None = None,
    scalebar_fontsize: float | None = None,
    scalebar_r: float = 0.25,
    scalebar_h: float = 0.075,
    scalebar_pad: float = 0.3,
    scalebar_loc: Literal[
        "upper left", "upper right", "lower left", "lower right", "center"
    ] = "lower right",
    scalebar_color: str = "black",
    savefig: bool = False,
    path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Plot map from a tensor, or batch(es).

    Args:
        data: Array(s) of a map (or maps)
        batch_idx: If not `None`, data is a single or multiple batches of images.
        info: Supplementary information about the data.
        ax: Optional matplotlib Axes to draw into. If `None`, a new figure/axes is created.
        extent: Extent of the map in physical units.
        colormap: Name of the colormap to use for plotting.
        colorbar: If `True`, include the colorbar in the plot.
        colorbar_label: Label of the colorbar (if enabled).
        colorbar_ax: Optional matplotlib Axes to draw the colorbar into.
          If `None`, defaults to `ax`.
        colorbar_pad: Colorbar padding.
        colorbar_loc: Colorbar location.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        scalebar: Use the a scalebar (usually makes x/yticks unnecessary).
        scalebar_label: Scale bar label; if `None`, physical extent unit is used.
        scalebar_fontsize: Font size of the scalebar label.
        scalebar_r: Scale bar length fraction of the image data.
        scalebar_h: Scale bar vertical size (in units of transform coordinate system).
        scalebar_pad: Padding around the label and size bar (in fraction of the font size).
        scalebar_loc: Location of the size bar.
        scalebar_color: Color of the scale bar.
        savefig: If `True`, save the plot to file.
        path: Filename or filepath where the figure is saved.
        show: If `True`, show the plot.
        close: If `True`, closes the figure.
        verbose: Print results to stdout.
        kwargs: Additional keyword arguments for `matplotlib.pyplot.imshow`.
    """
    metadata = kwargs.pop("info", {})
    data, metadata = _from_batch(data, metadata=metadata, batch_idx=batch_idx)
    _plot_data(
        data,
        metadata,
        ax=ax,
        extent=extent,
        colormap=colormap,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        colorbar_ax=colorbar_ax,
        colorbar_pad=colorbar_pad,
        colorbar_loc=colorbar_loc,
        xlabel=xlabel,
        ylabel=ylabel,
        scalebar=scalebar,
        scalebar_label=scalebar_label,
        scalebar_fontsize=scalebar_fontsize,
        scalebar_r=scalebar_r,
        scalebar_h=scalebar_h,
        scalebar_pad=scalebar_pad,
        scalebar_loc=scalebar_loc,
        scalebar_color=scalebar_color,
        savefig=savefig,
        path=path,
        show=show,
        close=close,
        verbose=verbose,
        **kwargs,
    )


@plot_image.register(np.ndarray)
@alias_kw("colormap", "cmap")
@alias_kw("colorbar", "cbar")
@alias_kw("cbar_label", "colorbar_label")
@alias_kw("colorbar_ax", "cbar_ax")
@alias_kw("colorbar_pad", "cbar_pad")
@alias_kw("colorbar_loc", "cbar_loc")
def plot_image_array(
    data: np.ndarray,
    batch_idx: int | Sequence[int] | None = None,
    ax: Axes | None = None,
    extent: Sequence[float] | None = None,
    colormap: Colormap | None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    colorbar_ax: Axes | None = None,
    colorbar_pad: float | None = None,
    colorbar_loc: Literal["left", "right"] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    scalebar: bool = False,
    scalebar_label: str | None = None,
    scalebar_fontsize: float | None = None,
    scalebar_r: float = 0.25,
    scalebar_h: float = 0.075,
    scalebar_pad: float = 0.3,
    scalebar_loc: Literal[
        "upper left", "upper right", "lower left", "lower right", "center"
    ] = "lower right",
    scalebar_color: str = "black",
    savefig: bool = False,
    path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Plot map from an array or batch(es).

    Args:
        data: Array(s) of a map (or maps)
        batch_idx: If not `None`, data is a single or multiple batches of images.
        info: Supplementary information about the data.
        ax: Optional matplotlib Axes to draw into. If `None`, a new figure/axes is created.
        extent: Extent of the map in physical units.
        colormap: Name of the colormap to use for plotting.
        colorbar: If `True`, include the colorbar in the plot.
        colorbar_label: Label of the colorbar (if enabled).
        colorbar_ax: Optional matplotlib Axes to draw the colorbar into.
          If `None`, defaults to `ax`.
        colorbar_pad: Colorbar padding.
        colorbar_loc: Colorbar location.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        scalebar: Use the a scalebar (usually makes x/yticks unnecessary).
        scalebar_label: Scale bar label; if `None`, physical extent unit is used.
        scalebar_fontsize: Font size of the scalebar label.
        scalebar_r: Scale bar length fraction of the image data.
        scalebar_h: Scale bar vertical size (in units of transform coordinate system).
        scalebar_pad: Padding around the label and size bar (in fraction of the font size).
        scalebar_loc: Location of the size bar.
        scalebar_color: Color of the scale bar.
        savefig: If `True`, save the plot to file.
        path: Filename or filepath where the figure is saved.
        show: If `True`, show the plot.
        close: If `True`, closes the figure.
        verbose: Print results to stdout.
        kwargs: Additional keyword arguments for `matplotlib.pyplot.imshow`.
    """
    metadata = kwargs.pop("info", {})
    data, metadata = _from_batch(data, metadata=metadata, batch_idx=batch_idx)
    _plot_data(
        data,
        metadata,
        ax=ax,
        extent=extent,
        colormap=colormap,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        colorbar_ax=colorbar_ax,
        colorbar_pad=colorbar_pad,
        colorbar_loc=colorbar_loc,
        xlabel=xlabel,
        ylabel=ylabel,
        scalebar=scalebar,
        scalebar_label=scalebar_label,
        scalebar_fontsize=scalebar_fontsize,
        scalebar_r=scalebar_r,
        scalebar_h=scalebar_h,
        scalebar_pad=scalebar_pad,
        scalebar_loc=scalebar_loc,
        scalebar_color=scalebar_color,
        savefig=savefig,
        path=path,
        show=show,
        close=close,
        verbose=verbose,
        **kwargs,
    )


@plot_image.register(au.Quantity)
@alias_kw("colormap", "cmap")
@alias_kw("colorbar", "cbar")
@alias_kw("cbar_label", "colorbar_label")
@alias_kw("colorbar_ax", "cbar_ax")
@alias_kw("colorbar_pad", "cbar_pad")
@alias_kw("colorbar_loc", "cbar_loc")
def plot_image_quantity(
    data: au.Quantity,
    batch_idx: int | Sequence[int] | None = None,
    ax: Axes | None = None,
    extent: Sequence[float] | None = None,
    colormap: Colormap | None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    colorbar_ax: Axes | None = None,
    colorbar_pad: float | None = None,
    colorbar_loc: Literal["left", "right"] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    scalebar: bool = False,
    scalebar_label: str | None = None,
    scalebar_fontsize: float | None = None,
    scalebar_r: float = 0.25,
    scalebar_h: float = 0.075,
    scalebar_pad: float = 0.3,
    scalebar_loc: Literal[
        "upper left", "upper right", "lower left", "lower right", "center"
    ] = "lower right",
    scalebar_color: str = "black",
    savefig: bool = False,
    path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Plot map from an astropy.units.Quantity array or batch(es).

    Args:
        data: Array(s) of a map (or maps)
        batch_idx: If not `None`, data is a single or multiple batches of images.
        info: Supplementary information about the data.
        ax: Optional matplotlib Axes to draw into. If `None`, a new figure/axes is created.
        extent: Extent of the map in physical units.
        colormap: Name of the colormap to use for plotting.
        colorbar: If `True`, include the colorbar in the plot.
        colorbar_label: Label of the colorbar (if enabled).
        colorbar_ax: Optional matplotlib Axes to draw the colorbar into.
          If `None`, defaults to `ax`.
        colorbar_pad: Colorbar padding.
        colorbar_loc: Colorbar location.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        scalebar: Use the a scalebar (usually makes x/yticks unnecessary).
        scalebar_label: Scale bar label; if `None`, physical extent unit is used.
        scalebar_fontsize: Font size of the scalebar label.
        scalebar_r: Scale bar length fraction of the image data.
        scalebar_h: Scale bar vertical size (in units of transform coordinate system).
        scalebar_pad: Padding around the label and size bar (in fraction of the font size).
        scalebar_loc: Location of the size bar.
        scalebar_color: Color of the scale bar.
        savefig: If `True`, save the plot to file.
        path: Filename or filepath where the figure is saved.
        show: If `True`, show the plot.
        close: If `True`, closes the figure.
        verbose: Print results to stdout.
        kwargs: Additional keyword arguments for `matplotlib.pyplot.imshow`.
    """
    metadata = kwargs.pop("info", {})
    data, metadata = _from_batch(data, metadata=metadata, batch_idx=batch_idx)
    data, units = data.value, data.unit
    if "units" not in metadata:
        metadata["units"] = f"{units:latex}"
    _plot_data(
        data,
        metadata,
        ax=ax,
        extent=extent,
        colormap=colormap,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        colorbar_ax=colorbar_ax,
        colorbar_pad=colorbar_pad,
        colorbar_loc=colorbar_loc,
        xlabel=xlabel,
        ylabel=ylabel,
        scalebar=scalebar,
        scalebar_label=scalebar_label,
        scalebar_fontsize=scalebar_fontsize,
        scalebar_r=scalebar_r,
        scalebar_h=scalebar_h,
        scalebar_pad=scalebar_pad,
        scalebar_loc=scalebar_loc,
        scalebar_color=scalebar_color,
        savefig=savefig,
        path=path,
        show=show,
        close=close,
        verbose=verbose,
        **kwargs,
    )


@plot_image.register(Sequence)
@alias_kw("colormap", "cmap")
@alias_kw("colorbar", "cbar")
@alias_kw("cbar_label", "colorbar_label")
@alias_kw("colorbar_ax", "cbar_ax")
@alias_kw("colorbar_pad", "cbar_pad")
@alias_kw("colorbar_loc", "cbar_loc")
def plot_image_from_batch(
    data: Sequence[torch.Tensor | np.ndarray | dict],
    batch_idx: int | Sequence[int] | None = None,
    ax: Axes | None = None,
    extent: Sequence[float] | None = None,
    colormap: None = None,
    colorbar: bool = False,
    colorbar_label: str | None = None,
    colorbar_ax: Axes | None = None,
    colorbar_pad: float | None = None,
    colorbar_loc: Literal["left", "right"] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    scalebar: bool = False,
    scalebar_label: str | None = None,
    scalebar_fontsize: float | None = None,
    scalebar_r: float = 0.25,
    scalebar_h: float = 0.075,
    scalebar_pad: float = 0.3,
    scalebar_loc: Literal[
        "upper left", "upper right", "lower left", "lower right", "center"
    ] = "lower right",
    scalebar_color: str = "black",
    savefig: bool = False,
    path: str | Path | None = None,
    show: bool = False,
    close: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Plot map from an array/tensor batch (optionally including metadata).

    Args:
        data: Array(s) of a map (or maps)
        batch_idx: If not `None`, data is a single or multiple batches of images.
        info: Supplementary information about the data.
        ax: Optional matplotlib Axes to draw into. If `None`, a new figure/axes is created.
        extent: Extent of the map in physical units.
        colormap: Name of the colormap to use for plotting.
        colorbar: If `True`, include the colorbar in the plot.
        colorbar_label: Label of the colorbar (if enabled).
        colorbar_ax: Optional matplotlib Axes to draw the colorbar into.
          If `None`, defaults to `ax`.
        colorbar_pad: Colorbar padding.
        colorbar_loc: Colorbar location.
        xlabel: Label of the x-axis.
        ylabel: Label of the y-axis.
        scalebar: Use the a scalebar (usually makes x/yticks unnecessary).
        scalebar_label: Scale bar label; if `None`, physical extent unit is used.
        scalebar_fontsize: Font size of the scalebar label.
        scalebar_r: Scale bar length fraction of the image data.
        scalebar_h: Scale bar vertical size (in units of transform coordinate system).
        scalebar_pad: Padding around the label and size bar (in fraction of the font size).
        scalebar_loc: Location of the size bar.
        scalebar_color: Color of the scale bar.
        savefig: If `True`, save the plot to file.
        path: Filename or filepath where the figure is saved.
        show: If `True`, show the plot.
        close: If `True`, closes the figure.
        verbose: Print results to stdout.
        kwargs: Additional keyword arguments for `matplotlib.pyplot.imshow`.
    """
    if all([isinstance(d, torch.Tensor) or isinstance(d, np.ndarray) for d in data]):
        data = data
        metadata = {}
    else:
        data, metadata = data
    metadata = kwargs.pop("info", metadata)
    data, metadata = _from_batch(data, metadata, batch_idx=batch_idx)
    _plot_data(
        data,
        metadata,
        ax=ax,
        extent=extent,
        colormap=colormap,
        colorbar=colorbar,
        colorbar_label=colorbar_label,
        colorbar_ax=colorbar_ax,
        colorbar_pad=colorbar_pad,
        colorbar_loc=colorbar_loc,
        xlabel=xlabel,
        ylabel=ylabel,
        scalebar=scalebar,
        scalebar_label=scalebar_label,
        scalebar_fontsize=scalebar_fontsize,
        scalebar_r=scalebar_r,
        scalebar_h=scalebar_h,
        scalebar_pad=scalebar_pad,
        scalebar_loc=scalebar_loc,
        scalebar_color=scalebar_color,
        savefig=savefig,
        path=path,
        show=show,
        close=close,
        verbose=verbose,
        **kwargs,
    )


if __name__ == "__main__":
    data_img = torch.rand((1, 512, 512))
    data_imgs = torch.rand((5, 1, 512, 512))
    data_arr = np.random.rand(5, 1, 512, 512)
    data_qty = au.Quantity(data_arr, "m/s")
    data_batch = (torch.rand((5, 1, 512, 512)), torch.rand((5, 1, 512, 512)))
    batch = (
        (torch.rand((5, 1, 512, 512)), torch.rand((5, 1, 512, 512))),
        (
            {
                "class": ["gas"],
                "extent": [[-42.0, 42.0, -42.0, 42.0]],
                "units_extent": ["kpc"],
            },
            {
                "class": ["dm"],
                "extent": [[-91.0, 91.0, -91.0, 91.0]],
                "units_extent": ["kpc"],
            },
        ),
    )
    plot_image(data_img)
    plot_image(data_imgs, 0)
    plot_image(data_arr, 2)
    plot_image(data_qty, 2)
    plot_image(data_batch, (1, 3))
    plot_image(batch, (1, 0))
