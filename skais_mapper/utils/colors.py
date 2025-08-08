# SPDX-FileCopyrightText: 2025-present Philipp Denzel <phdenzel@gmail.com>
# SPDX-FileNotice: Part of skais-mapper
# SPDX-License-Identifier: GPL-3.0-or-later
"""Color module for more beautiful plots."""

import random
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, to_hex
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib import pyplot as plt

__all__ = ["color_variant", "SkaisColors", "SkaisColorMaps"]


def color_variant(hex_color: str, shift: int = 10) -> str:
    """Takes a color in hex code and produces a lighter or darker shift variant.

    Args:
        hex_color (str): formatted as '#' + rgb hex string of length 6
        shift (int): decimal shift of the rgb hex string

    Returns:
        variant (str): formatted as '#' + rgb hex string of length 6
    """
    if len(hex_color) != 7:
        message = "Passed {} to color_variant(), needs to be in hex format."
        raise ValueError(message.format(hex_color))
    rgb_hex = [hex_color[x : x + 2] for x in [1, 3, 5]]
    new_rgb_int = [int(hex_value, 16) + shift for hex_value in rgb_hex]
    # limit to interval 0 and 255
    new_rgb_int = [min([255, max([0, i])]) for i in new_rgb_int]
    # hex() produces "0x88", we want the last two digits
    return "#" + "".join([hex(i)[2:] if i else "00" for i in new_rgb_int])


class ReNormColormapAdaptor(Colormap):
    """Adaptor for re-normalizing color mappable."""

    def __init__(self, base, cmap_norm: Normalize, orig_norm: Normalize | None = None):
        if orig_norm is None:
            if isinstance(base, ScalarMappable):
                orig_norm = base.norm
                base = base.cmap
            else:
                orig_norm = Normalize(0, 1)
        self._base = base
        if isinstance(cmap_norm, type(Normalize)) and issubclass(cmap_norm, Normalize):
            # a class was provided instead of an instance. create an instance
            # with the same limits.
            cmap_norm = cmap_norm(orig_norm.vmin, orig_norm.vmax)
        self._cmap_norm = cmap_norm
        self._orig_norm = orig_norm

    def __call__(self, X, **kwargs):
        """Re-normalise the values before applying the colormap."""
        return self._base(self._cmap_norm(self._orig_norm.inverse(X)), **kwargs)

    def __getattr__(self, attr):
        """Any other attribute, we simply dispatch to the underlying cmap."""
        return getattr(self._base, attr)

    def _init(self, *args, **kwargs):
        """Abstract dummy method."""

    def reversed(self, *args, **kwargs):
        """Abstract dummy method."""


class SkaisColors:
    """An assortment of colors and palettes."""

    # Shades
    white = "#DDDEE1"  # rba(221, 222, 225)
    gray = "#98989D"  # rba(152, 152, 157)
    grey = "#98989D"  # rba(152, 152, 157)
    darkish = "#666769"  # rba(102, 103, 105)
    dark = "#3d3e41"  # rba( 61,  62,  65)
    darker = "#333437"  # rba( 51,  52,  55)
    darkest = "#212225"  # rba( 33,  34,  37)
    black = "#090F0F"  # rba(  9,  15,  15)
    textcolor = "#dddee1"  # rba(221, 222, 225)
    # Primary colors
    red = "#FF6767"
    pink = "#FF375F"  # rba(255,  55,  95)
    orange = "#FF9F0A"  # rba(255, 159,  10)
    yellow = "#FFD60A"  # rba(155, 214,  10)
    purple = "#603DD0"  # rba( 96,  61, 208)
    green = "#32D74B"  # rba( 50, 215,  75)
    cyan = "#5BC1AE"
    blue = "#6767FF"
    brown = "#D88C4E"  # rba(172, 142, 104)
    # Other
    golden = "#feb125"  # rba(256, 177,  37)
    purpleblue = "#7d7de1"  # rba(125, 125, 225)
    turquoise = "#00d1a4"  # rba( 10, 210, 165)
    # Light colors
    cyan_light = "#A0DED2"
    light_cloors = [cyan_light]
    # Dark colors
    cyan_dark = "#24A38B"
    blue_marguerite = "#756BB1"
    dark_colors = [cyan_dark]
    misc_colors = [blue_marguerite]

    colors = [
        red,
        pink,
        orange,
        yellow,
        purple,
        green,
        cyan,
        blue,
        brown,
        white,
        grey,
        cyan_light,
        cyan_dark,
    ]

    # plotting palettes
    plt5 = ["#ffa600", "#ff6361", "#bc5090", "#58508d", "#668B9d"]
    plt8 = ["#006999", "#4d6db1", "#8b6bbb", "#c464b2", "#f15e99", "#ff6773", "#ff8246", "#ffa600"]
    # Scheme colors
    palettes = {
        "arcus": ["#FC354C", "#29221F", "#13747D", "#0ABFBC", "#FCF7C5"],
        "aquaria": ["#00207F", "#A992FA", "#EA55B1", "#FEC763"],
        "geometric": ["#08F7FE", "#09FBD3", "#FE53BB", "#F5D300"],
        "neon": ["#560A86", "#7122FA", "#F148FB", "#FFACFC"],
        "psychedelic": ["#011FFD", "#FF2281", "#B76CFD", "#75D5FD"],
        "vivid": ["#FDC7D7", "#FF9DE6", "#A5D8F3", "#E8E500"],
        "abstract": ["#7B61F8", "#FF85EA", "#FDF200", "#00FECA"],
        "phoenix": ["#33135C", "#652EC7", "#DE38C8", "#FFC300"],
        "cosmicnova": ["#3B27BA", "#E847AE", "#13CA91", "#FF9472"],
        "pinkpop": ["#35212A", "#3B55CE", "#FF61BE", "#FFDEF3"],
        "agaveglitch": ["#01535F", "#02B8A2", "#FDB232", "#FDD400"],
        "coralglow": ["#F85125", "#FF8B8B", "#FEA0FE", "#79FFFE"],
        "luxuria": ["#037A90", "#00C2BA", "#FF8FCF", "#CE96FB"],
        "luminos": ["#9D72FF", "#FFB3FD", "#01FFFF", "#01FFC3"],
        "stationary": ["#FE6B25", "#28CF75", "#EBF875", "#A0EDFF"],
        "prism": ["#C24CF6", "#FF1493", "#FC6E22", "#FFFF66"],
        "retro": ["#CE0000", "#FF5F01", "#FE1C80", "#FFE3F1"],
        "terrean": ["#E8DDCB", "#CDB380", "#036564", "#033649", "#031634"],
        "cozytime": ["#FFEEB2", "#F4CE61", "#FFBF20", "#7D9CA1", "#4D8D97"],
        "gravic": ["#0b0b13", "#272e59", "#4d5693", "#ffbf20", "#f8f5ec"],
        "marsian": ["#e8ddcb", "#cd967e", "#63031d", "#490303", "#1d0202"],
        "twofourone": ["#D1313D", "#E5625C", "#F9BF76", "#8EB2C5", "#615375"],
        "ioba": ["#FF3366", "#C74066", "#8F4D65", "#575A65", "#1F6764"],
        "mintrock": ["#595B5A", "#14C3A2", "#0DE5A8"],
        "pukedrainbow": ["#482344", "#2B5166", "#429867", "#FAB243", "#E02130"],
        "acryliq": ["#F21A1D", "#FF822E", "#03DDDC", "#FEF900"],
        "hibokeh": ["#0310EA", "#FB33DB", "#7FFF00", "#FCF340"],
        "flashy": ["#04005E", "#440BD4", "#FF2079", "#E92EFB"],
        "cyber": ["#00218A", "#535EEB", "#BC75F9", "#BDBDFD"],
        "cyberfade": ["#00218A", "#535EEB", "#BC75F9", "#BDBDFD", "#DEDEFE", "#FFFFFF"],
        "zoas": ["#001437", "#7898FB", "#5CE5D5", "#B8FB3C"],
        "vilux": ["#001437", "#85B2FF", "#17E7B6", "#D4FD87", "#FDEB87", "#FDD74C", "#FCAA43"],
        "graphiq": ["#48ADF1", "#C6BDEA", "#FDCBFC", "#8AF7E4"],
        "vibes": ["#027A9F", "#12B296", "#FFAA01", "#E1EF7E"],
        "purplerain": ["#120052", "#8A2BE2", "#B537F2", "#3CB9FC"],
        "gaseous": [
            "#101016",
            "#1B366A",
            "#2F4F85",
            "#385B8F",
            "#79695d",
            "#D98647",
            "#EBC79A",
            "#FFFFFF",
        ],
        "nava": [
            "#0A001B",
            "#3A2B62",
            "#72678F",
            "#5C93A7",
            "#749DBB",
            "#84D7E7",
            "#ABF2EC",
            "#EEF5FF",
            "#FFFFFF",
            "#FCF9FF",
            "#F19B8A",
            "#D93E44",
            "#C7204C",
            "#18001a",
        ],
        "obscura": [
            "#101016",
            "#103F39",
            "#27524C",
            "#52746F",
            "#6F795D",
            "#D98647",
            "#EBC79A",
            "#FFFFFF",
        ],
        "hertzsprung": [
            "#101016",
            "#6A1B36",
            "#852F4F",
            "#8F385B",
            "#D98647",
            "#EBC79A",
            "#FFFFFF",
        ],
        "shapely": [
            (30.0 / 255, 136.0 / 255, 229.0 / 255, alpha) for alpha in np.linspace(1, 0, 100)
        ]
        + [(255.0 / 255, 13.0 / 255, 87.0 / 255, alpha) for alpha in np.linspace(0, 1, 100)],
    }

    for k, p in palettes.items():
        locals()[k + "_palette"] = p
        for i, pi in enumerate(p):
            locals()[k + f"{i}"] = pi
    locals().pop("k")
    locals().pop("p")
    locals().pop("i")
    locals().pop("pi")

    @classmethod
    def cmap_from_color(
        cls,
        color_str: str,
        secondary_color: str | None = None,
    ) -> LinearSegmentedColormap:
        """Create a colormap from a single color.

        Args:
            color_str: color string of the class color
            secondary_color: color into which the color changes in the colormap

        Returns:
            (mpl.colors.LinearSegmentedColormap object): reversed colormap
        """
        if color_str in cls.__dict__:
            color = cls.__dict__[color_str]
        else:
            color = color_str
        if secondary_color in cls.__dict__:
            secondary_color = cls.__dict__[secondary_color]
        elif secondary_color is None:
            secondary_color = color_variant(color, shift=125)
        cmap = LinearSegmentedColormap.from_list("Skais" + color_str, [secondary_color, color])
        return cmap


class SkaisColorMaps:
    """An assortment of linearly interpolated colormaps based on 4+ colors each."""

    for k, p in SkaisColors.palettes.items():
        locals()[k] = LinearSegmentedColormap.from_list(k, p)
    aslist = []
    for k in SkaisColors.palettes:
        aslist.append(locals()[k])
    asarray = np.asarray(aslist)
    N = len(aslist)
    locals().pop("k")
    locals().pop("p")

    @classmethod
    def random(cls) -> LinearSegmentedColormap:
        """Choose a random color map.

        Returns:
            cmap: random colormap from custom list
        """
        return random.choice(cls.aslist)

    @classmethod
    def gen(cls):
        """Generate colormaps.

        Returns:
            (mpl.colors.LinearSegmentedColormap object): colormap generated from custom list
        """
        yield from cls.aslist

    @classmethod
    def reverse(
        cls, cmap: Colormap, set_bad: str = None, set_under: str = None, set_over: str = None
    ) -> LinearSegmentedColormap:
        """Reverse the specified colormap.

        Args:
            cmap (mpl.colors.LinearSegmentedColormap object): colormap to be reversed
            set_bad (str): set colormaps bad values to a different color
            set_under (str): set colormaps under values to a different color
            set_over (str): set colormaps over values to a different color

        Returns:
            (mpl.colors.LinearSegmentedColormap object): reversed colormap
        """
        reverse = []
        k = []

        for key in cmap._segmentdata:
            k.append(key)

            channel = cmap._segmentdata[key]
            data = []

            for t in channel:
                data.append((1 - t[0], t[2], t[1]))
            reverse.append(sorted(data))

        linear_l = dict(zip(k, reverse))
        rcmap = LinearSegmentedColormap(f"{cmap.name}_r", linear_l)
        if set_bad is not None:
            rcmap.set_bad(set_bad)
        if set_under is not None:
            rcmap.set_over(set_under)
        if set_over is not None:
            rcmap.set_over(set_over)
        return rcmap

    @classmethod
    def palette(cls, cmap_name: str, N: int):
        """Return a palette of a colormap with N linearly interpolated color points.

        Args:
            cmap_name: name of the colormap
            N: number of colors in the palette
        """
        vals = np.linspace(0, 1, N)
        cmap = cls.__dict__[cmap_name]
        rgbas = cmap(vals)
        return [to_hex(rgba, keep_alpha=False) for rgba in rgbas]

    @classmethod
    def plot_gradients(cls, savefig: bool = False):
        """Plot all color-map gradients.

        Args:
            savefig (bool): save figure as palettes.png
        """
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        fig, axes = plt.subplots(nrows=SkaisColorMaps.N)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        for ax, cmap in zip(axes, cls.aslist):
            ax.imshow(gradient, aspect="auto", cmap=cmap)
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.0
            fig.text(x_text, y_text, cmap.name, va="center", ha="right", fontsize=10)
        for ax in axes:
            ax.set_axis_off()
        if savefig:
            plt.savefig("palette.png")

    @staticmethod
    def register_all(verbose: bool = False):
        """Register colormaps with matplotlib.

        Args:
            verbose (bool): If True, print information to command line
        """
        for g in SkaisColorMaps.aslist:
            if verbose:
                print(g.name)
            if g.name not in mpl.colormaps:
                mpl.colormaps.register(name=g.name, cmap=g)
            if f"skais_{g.name}" not in mpl.colormaps:
                mpl.colormaps.register(name=f"skais_{g.name}", cmap=g)
            
            


if __name__ == "__main__":
    SkaisColorMaps.plot_gradients()
    plt.show()
