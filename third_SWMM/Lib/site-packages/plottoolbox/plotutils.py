# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import itertools
import os
import warnings

import mando
import typic
from mando.rst_text_formatter import RSTHelpFormatter
from tstoolbox import tsutils

warnings.filterwarnings("ignore")

ldocstrings = tsutils.docstrings
ldocstrings[
    "xydata"
] = """If the input 'x,y' dataset(s) is organized as
            'index,x1,y1,x2,y2,x3,y3,...,xN,yN' then the 'index' is ignored.
            If there is one 'x,y' dataset then it can be organized as 'index,y'
            where 'index' is used for 'x'.  The "columns" keyword can be used
            to duplicate or change the order of all the data columns."""
ldocstrings[
    "ydata"
] = """Data must be organized as 'index,y1,y2,y3,...,yN'.  The 'index' is
            ignored and all data columns are plotted.  The "columns" keyword
            can be used to duplicate or change the order of all the data
            columns."""
ldocstrings[
    "yone"
] = """Data must be organized as 'index,y1'.  Can only plot one series."""
ldocstrings[
    "ofilename"
] = """ofilename : str
        [optional, defaults to 'plot.png']

        Output filename for the plot.  Extension defines
        the type, for example 'filename.png' will create a PNG file.

        If used within Python, and `ofilename` is None will return the
        Matplotlib figure that can then be changed or added to as
        needed."""
ldocstrings[
    "xtitle"
] = """xtitle : str
        [optional, default depends on type]

        Title of x-axis."""
ldocstrings[
    "ytitle"
] = """ytitle : str
        [optional, default depends on type]

        Title of y-axis."""
ldocstrings[
    "title"
] = """title : str
        [optional, defaults to '']

        Title of chart."""
ldocstrings[
    "figsize"
] = """figsize : str
        [optional, defaults to '10,6.5']

        The 'width,height' of plot in inches."""
ldocstrings[
    "subplots"
] = """subplots
        [optional, defaults to False]

        Make separate subplots for each time series."""
ldocstrings[
    "sharex"
] = """sharex
        [optional, default to True]

        In case subplots=True, share x axis."""
ldocstrings[
    "sharey"
] = """sharey
        [optional, default to False]

        In case subplots=True, share y axis."""
ldocstrings[
    "colors"
] = """colors
        [optional, default is 'auto']

        The default 'auto' will cycle through matplotlib colors in the chosen
        style.

        At the command line supply a comma separated matplotlib
        color codes, or within Python a list of color code strings.

        Can identify colors in four different ways.

        1. Use 'CN' where N is a number from 0 to 9 that gets the Nth color
        from the current style.

        2. Single character code from the table
        below.

            +------+---------+
            | Code | Color   |
            +======+=========+
            | b    | blue    |
            +------+---------+
            | g    | green   |
            +------+---------+
            | r    | red     |
            +------+---------+
            | c    | cyan    |
            +------+---------+
            | m    | magenta |
            +------+---------+
            | y    | yellow  |
            +------+---------+
            | k    | black   |
            +------+---------+

        3. Number between 0 and 1 that represents the level of gray, where 0 is
        white an 1 is black.

        4. Any of the HTML color
        names.

            +------------------+
            | HTML Color Names |
            +==================+
            | red              |
            +------------------+
            | burlywood        |
            +------------------+
            | chartreuse       |
            +------------------+
            | ...etc.          |
            +------------------+

        Color reference:
        http://matplotlib.org/api/colors_api.html"""
ldocstrings[
    "linestyles"
] = """linestyles
        [optional, default to 'auto']

        If 'auto' will iterate through the available matplotlib line types.
        Otherwise on the command line a comma separated list, or a list of
        strings if using the Python API.

        To not display lines use a space (' ') as the linestyle code.

        Separated 'colors', 'linestyles', and 'markerstyles' instead of using
        the 'style' keyword.

        +---------+--------------+
        | Code    | Lines        |
        +=========+==============+
        | ``-``   | solid        |
        +---------+--------------+
        | --      | dashed       |
        +---------+--------------+
        | -.      | dash_dot     |
        +---------+--------------+
        | :       | dotted       |
        +---------+--------------+
        | None    | draw nothing |
        +---------+--------------+
        | ' '     | draw nothing |
        +---------+--------------+
        | ''      | draw nothing |
        +---------+--------------+

        Line reference:
        http://matplotlib.org/api/artist_api.html"""
ldocstrings[
    "markerstyles"
] = """markerstyles
        [optional, default to ' ']

        The default ' ' will not plot a marker.  If 'auto' will iterate through
        the available matplotlib marker types.  Otherwise on the command line
        a comma separated list, or a list of strings if using the Python API.

        Separated 'colors', 'linestyles', and 'markerstyles' instead of using
        the 'style' keyword.

        +-------+----------------+
        | Code  | Markers        |
        +=======+================+
        | .     | point          |
        +-------+----------------+
        | o     | circle         |
        +-------+----------------+
        | v     | triangle down  |
        +-------+----------------+
        | ^     | triangle up    |
        +-------+----------------+
        | <     | triangle left  |
        +-------+----------------+
        | >     | triangle right |
        +-------+----------------+
        | 1     | tri_down       |
        +-------+----------------+
        | 2     | tri_up         |
        +-------+----------------+
        | 3     | tri_left       |
        +-------+----------------+
        | 4     | tri_right      |
        +-------+----------------+
        | 8     | octagon        |
        +-------+----------------+
        | s     | square         |
        +-------+----------------+
        | p     | pentagon       |
        +-------+----------------+
        | ``*`` | star           |
        +-------+----------------+
        | h     | hexagon1       |
        +-------+----------------+
        | H     | hexagon2       |
        +-------+----------------+
        | ``+`` | plus           |
        +-------+----------------+
        | x     | x              |
        +-------+----------------+
        | D     | diamond        |
        +-------+----------------+
        | d     | thin diamond   |
        +-------+----------------+
        | _     | hlines_y       |
        +-------+----------------+
        | None  | nothing        |
        +-------+----------------+
        | ' '   | nothing        |
        +-------+----------------+
        | ''    | nothing        |
        +-------+----------------+

        Marker reference:
        http://matplotlib.org/api/markers_api.html"""
ldocstrings[
    "style"
] = """style
        [optional, default is None]

        Still available, but if None is replaced by 'colors', 'linestyles', and
        'markerstyles' options.  Currently the 'style' option will override the
        others.

        Comma separated matplotlib style strings per time-series.  Just
        combine codes in 'ColorMarkerLine' order, for example 'r*--' is
        a red dashed line with star marker."""
ldocstrings[
    "xlim"
] = """xlim
        [optional, default is based on range of x values]

        Comma separated lower and upper limits for the x-axis of the
        plot.  For example, '--xlim 1,1000' would limit the plot from
        1 to 1000, where '--xlim ,1000' would base the lower limit on
        the data and set the upper limit to 1000."""
ldocstrings[
    "ylim"
] = """ylim
        [optional, default is based on range of y values]

        Comma separated lower and upper limits for the y-axis of the
        plot.  See `xlim` for examples."""
ldocstrings[
    "secondary"
] = """[optional, default is False]

        * list/tuple: Give the column numbers or names to plot on secondary
          y-axis.
        * (string, string): The first string is the units of the primary axis,
          the second string is the units of the secondary axis if you want just
          unit conversion.  Use any units or combination thereof from the
          "pint" library.
        * (callable, callable): Functions relating relationship between
          primary and secondary axis.  First function will be given the values
          on primary axis and returns values on secondary axis.  Second function
          will be do the inverse.  Python API only.
        * string: One of pre-built (callable, callable) combinations.  Can be
          one of "period"."""
ldocstrings[
    "mark_right"
] = """mark_right
        [optional, default is True]

        When using a secondary_y axis, should the legend label the axis of the
        various time-series automatically."""
ldocstrings[
    "grid"
] = """grid
        [optional, default is False]

        Whether to plot grid lines on the major ticks."""
ldocstrings[
    "label_rotation"
] = """label_rotation : int
        [optional]

        Rotation for major labels for bar plots."""
ldocstrings[
    "label_skip"
] = """label_skip : int
        [optional]

        Skip for major labels for bar plots."""
ldocstrings[
    "xlabel_rotation"
] = """xlabel_rotation : int
        [optional]

        Rotation for major x-axis labels for plots."""
ldocstrings[
    "xlabel_skip"
] = """xlabel_skip : int
        [optional]

        Skip for major x-axis labels for plots."""
ldocstrings[
    "ylabel_rotation"
] = """ylabel_rotation : int
        [optional]

        Rotation for major y-axis labels for plots."""
ldocstrings[
    "ylabel_skip"
] = """ylabel_skip : int
        [optional]

        Skip for major y-axis labels for plots."""
ldocstrings[
    "drawstyle"
] = """drawstyle : str
        [optional, default is 'default']

        'default' connects the points with lines. The
        steps variants produce step-plots. 'steps' is equivalent to 'steps-pre'
        and is maintained for backward-compatibility.

        ACCEPTS::

         ['default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post']"""
ldocstrings[
    "por"
] = """por
        [optional]

        Plot from first good value to last good value.  Strips NANs
        from beginning and end."""
ldocstrings[
    "plot_styles"
] = """plot_styles: str
        [optional, default is "default"]

        Set the style of the plot.  One or more of Matplotlib styles "classic",
        "Solarize_Light2", "bmh", "dark_background", "fast", "fivethirtyeight",
        "ggplot", "grayscale", "seaborn", "seaborn-bright",
        "seaborn-colorblind", "seaborn-dark", "seaborn-dark-palette",
        "seaborn-darkgrid", "seaborn-deep", "seaborn-muted",
        "seaborn-notebook", "seaborn-paper", "seaborn-pastel",
        "seaborn-poster", "seaborn-talk", "seaborn-ticks", "seaborn-white",
        "seaborn-whitegrid", "tableau-colorblind10", and

        The main SciencePlots styles are "science", "grid", "ieee", "scatter",
        "notebook", "high-vis", "bright", "vibrant", "muted", and "retro".

        Other SciencPlots styles that are less common or intended to modify
        other styles are, "cjk-jp-font.mplstyle", "cjk-kr-font.mplstyle",
        "cjk-sc-font.mplstyle", "cjk-tc-font.mplstyle",
        "high-contrast.mplstyle", "latex-sans.mplstyle", "light.mplstyle",
        "nature.mplstyle", "no-latex.mplstyle", "pgf.mplstyle",
        "russian-font.mplstyle", and "std-colors.mplstyle".

        If multiple styles then each over rides some or all of the
        characteristics of the previous.

        Color Blind Appropriate Styles

        The styles "seaborn-colorblind", "tableau-colorblind10", "bright",
        "vibrant", and "muted" are all styles that are setup to be able to be
        distinguished by someone with color blindness.

        Black, White, and Gray Styles

        The "ieee" style is appropriate for black, white, and gray, however the
        "ieee" also will change the chart size to fit in a column of the "IEEE"
        journal.

        The "grayscale" is another style useful for photo-copyable black,
        white, nd gray.

        Matplotlib styles:
            https://matplotlib.org/3.3.1/gallery/style_sheets/style_sheets_reference.html

        SciencePlots styles:
            https://github.com/garrettj403/SciencePlots"""
ldocstrings[
    "legend"
] = """legend: bool
        [optional, default is True]

        Whether to create a legend or not."""
ldocstrings[
    "legend_names"
] = """legend_names:
        [optional, default is None]

        If the default of None will take legend names from columns tiles in the
        input dataset.  Otherwise will take names from the `legend_names`
        list."""
ldocstrings[
    "hlines_y"
] = """hlines_y:
        [optional, defaults to None]

        Number or list of y values where to place a horizontal line."""
ldocstrings[
    "hlines_xmin"
] = """hlines_xmin:
        [optional, defaults to None]

        List of minimum x values to start the horizontal line.  If a list must
        be same length as `hlines_y`.  If a single number will be used as the
        minimum x values for all horizontal lines.  A missing value or None
        will start at the minimum x value for the entire plot."""
ldocstrings[
    "hlines_xmax"
] = """hlines_xmax:
        [optional, defaults to None]

        List of maximum x values to end each horizontal line.  If a list must
        be same length as `hlines_y`.  If a single number will be the maximum
        x value for all horizontal lines.  A missing value or None will end at
        the maximum x value for the entire plot."""
ldocstrings[
    "hlines_colors"
] = """hlines_colors:
        [optional, defaults to None]

        List of colors for the horizontal lines.  If a single color then will
        be used as the color for all horizontal lines.  If a list must be same
        length as `hlines_y`.  If None will take from the color pallette in the
        current plot style."""
ldocstrings[
    "hlines_linestyles"
] = """hlines_linestyles:
        [optional, defaults to None]

        List of linestyles for the horizontal lines.  If a single linestyle
        then will be used as the linestyle for all horizontal lines.  If a list
        must be same length as `hlines_y`.  If None will take for the standard
        linestyles list."""
ldocstrings[
    "vlines_x"
] = """vlines_x:
        [optional, defaults to None]

        List of x values where to place a vertical line."""
ldocstrings[
    "vlines_ymin"
] = """vlines_ymin:
        [optional, defaults to None]

        List of minimum y values to start the vertical line.  If a list must be
        same length as `vlines_x`.  If a single number will be used as the
        minimum x values for all vertical lines.  A missing value or None will
        start at the minimum x value for the entire plot."""
ldocstrings[
    "vlines_ymax"
] = """vlines_ymax:
        [optional, defaults to None]

        List of maximum x values to end each vertical line.  If a list must be
        same length as `vlines_x`.  If a single number will be the maximum
        x value for all vertical lines.  A missing value or None will end at
        the maximum x value for the entire plot."""
ldocstrings[
    "vlines_colors"
] = """vlines_colors:
        [optional, defaults to None]

        List of colors for the vertical lines.  If a single color then will be
        used as the color for all vertical lines.  If a list must be same
        length as `vlines_x`.  If None will take from the color pallette in the
        current plot style."""
ldocstrings[
    "vlines_linestyles"
] = """vlines_linestyles:
        [optional, defaults to None]

        List of linestyles for the vertical lines.  If a single linestyle then
        will be used as the linestyle for all vertical lines.  If a list must
        be same length as `vlines_x`.  If None will take for the standard
        linestyles list."""
ldocstrings[
    "bar_hatchstyles"
] = r"""bar_hatchstyles
        [optional, default to "auto", only used if type equal to "bar", "barh",
        "bar_stacked", and "barh_stacked"]

        If 'auto' will iterate through the available matplotlib hatch types.
        Otherwise on the command line a comma separated list, or a list of
        strings if using the Python API.

        +-----------------+-------------------+
        | bar_hatchstyles | Description       |
        +=================+===================+
        | /               | diagonal hatching |
        +-----------------+-------------------+
        | ``\``           | back diagonal     |
        +-----------------+-------------------+
        | ``|``           | vertical          |
        +-----------------+-------------------+
        | ``-``           | horizontal        |
        +-----------------+-------------------+
        | ``+``           | crossed           |
        +-----------------+-------------------+
        | ``x``           | crossed diagonal  |
        +-----------------+-------------------+
        | o               | small circle      |
        +-----------------+-------------------+
        | O               | large circle      |
        +-----------------+-------------------+
        | .               | dots              |
        +-----------------+-------------------+
        | ``*``           | stars             |
        +-----------------+-------------------+"""
ldocstrings[
    "xaxis"
] = """xaxis : str
        [optional, default is 'arithmetic']

        Defines the type of the xaxis.  One of 'arithmetic', 'log'."""
ldocstrings[
    "yaxis"
] = """yaxis : str
        [optional, default is 'arithmetic']

        Defines the type of the yaxis.  One of 'arithmetic', 'log'."""
ldocstrings[
    "invert_xaxis"
] = """invert_xaxis
        [optional, default is False]

        Invert the x-axis."""
ldocstrings[
    "invert_yaxis"
] = """invert_yaxis
        [optional, default is False]

        Invert the y-axis."""
ldocstrings[
    "mark_right"
] = """mark_right
        [optional, default is True]

        When using a secondary_y axis, should the legend label the axis of the
        various time-series automatically."""
ldocstrings[
    "plotting_position"
] = """plotting_position : str
        [optional, default is 'weibull']

        {plotting_position_table}

        Only used for norm_xaxis, norm_yaxis, lognorm_xaxis,
        lognorm_yaxis, weibull_xaxis, and weibull_yaxis."""
ldocstrings[
    "prob_plot_sort_values"
] = """prob_plot_sort_values : str
        [optional, default is 'descending']

        How to sort the values for the probability plots.

        Only used for norm_xaxis, norm_yaxis, lognorm_xaxis,
        lognorm_yaxis, weibull_xaxis, and weibull_yaxis."""

MARKER_LIST = [
    ".",
    ",",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "D",
    "d",
    "|",
    "_",
]

LINE_LIST = ["-", "--", "-.", ":", "solid", "dashed", "dashdot", "dotted"]

HATCH_LIST = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


def know_your_limits(xylimits, axis="arithmetic"):
    """Establish axis limits.

    This defines the xlim and ylim as lists rather than strings.
    Might prove useful in the future in a more generic spot.  It
    normalizes the different representations.
    """
    nlim = tsutils.make_list(xylimits, n=2)

    if axis == "normal":
        if nlim is None:
            nlim = [None, None]
        if nlim[0] is None:
            nlim[0] = 0.01
        if nlim[1] is None:
            nlim[1] = 0.99
        if nlim[0] < 0 or nlim[0] > 1 or nlim[1] < 0 or nlim[1] > 1:
            raise ValueError(
                tsutils.error_wrapper(
                    """
Both limits must be between 0 and 1 for the 'normal', 'lognormal', or 'weibull'
axis.

Instead you have {}.
""".format(
                        nlim
                    )
                )
            )

    if nlim is None:
        return nlim

    if nlim[0] is not None and nlim[1] is not None:
        if nlim[0] >= nlim[1]:
            raise ValueError(
                tsutils.error_wrapper(
                    """
The second limit must be greater than the first.

You gave {}.
""".format(
                        nlim
                    )
                )
            )

    if axis == "log":
        if (nlim[0] is not None and nlim[0] <= 0) or (
            nlim[1] is not None and nlim[1] <= 0
        ):
            raise ValueError(
                tsutils.error_wrapper(
                    """
If log plot cannot have limits less than or equal to 0.

You have {}.
""".format(
                        nlim
                    )
                )
            )

    return nlim


@typic.al
def check_column_legend(plottype, tsd, legend_names):
    # print("in ck_col_lg :", plottype, type(tsd), len(tsd.columns), type(legend_names), legend_names)
    # Check number of columns.
    if plottype in ["bootstrap", "heatmap", "autocorrelation", "lag_plot"]:
        if len(tsd.columns) != 1:
            raise ValueError(
                tsutils.error_wrapper(
                    """
The '{1}' plot can only work with 1 time-series in the DataFrame.
The DataFrame that you supplied has {0} time-series.
""".format(
                        len(tsd.columns), plottype
                    )
                )
            )

    # Check legend_names.
    if not legend_names:
        legend_names = tsd.columns
        return tsd, legend_names

    if len(legend_names) != len(set(legend_names)):
        raise ValueError(
            tsutils.error_wrapper(
                """
Each name in legend_names must be unique.
"""
            )
        )
    if len(tsd.columns) == len(legend_names):
        renamedict = dict(list(zip(tsd.columns, legend_names)))
    elif plottype in ["xy", "double_mass"] and (
        len(tsd.columns) // 2 == len(legend_names) or len(tsd.columns) == 1
    ):
        renamedict = dict(list(zip(tsd.columns[2::2], legend_names[1:])))
        renamedict[tsd.columns[1]] = legend_names[0]
    else:
        raise ValueError(
            tsutils.error_wrapper(
                """
For 'legend_names' and most plot types you must have the same number of comma
separated names as columns in the input data.  The input data has {} where the
number of 'legend_names' is {}.

If `type` is 'xy' or 'double_mass' you need to have legend names as
l1,l2,l3,...  where l1 is the legend for x1,y1, l2 is the legend for x2,y2,
...etc.
""".format(
                    len(tsd.columns), len(legend_names)
                )
            )
        )
        tsd.rename(columns=renamedict, inplace=True)
    legend_names = tsd.columns

    return tsd, legend_names


def prepare_styles(ntrace, style, colors, linestyles, markerstyles):
    if colors == "auto":
        colors = None
    else:
        colors = tsutils.make_list(colors)

    if linestyles == "auto":
        linestyles = LINE_LIST
    else:
        linestyles = tsutils.make_list(linestyles)

    if markerstyles == "auto":
        markerstyles = MARKER_LIST
    else:
        markerstyles = tsutils.make_list(markerstyles)
        if markerstyles is None:
            markerstyles = " "
    if style != "auto":
        nstyle = tsutils.make_list(style)
        if len(nstyle) != ntrace:
            raise ValueError(
                tsutils.error_wrapper(
                    """
You have to have the same number of style strings as traces to plot.
You supplied '{}' for style which has {} style strings,
but you have {} traces.
""".format(
                        style, len(nstyle), ntrace
                    )
                )
            )
        colors = []
        markerstyles = []
        linestyles = []
        for st in nstyle:
            colors.append(st[0])
            if len(st) == 1:
                markerstyles.append(" ")
                linestyles.append("-")
                continue
            if st[1] in MARKER_LIST:
                markerstyles.append(st[1])
                try:
                    linestyles.append(st[2:])
                except IndexError:
                    linestyles.append(" ")
            else:
                markerstyles.append(" ")
                linestyles.append(st[1:])
    if linestyles is None:
        linestyles = [" "]
    else:
        linestyles = [" " if i in ["  ", None] else i for i in linestyles]
    markerstyles = [" " if i is None else i for i in markerstyles]

    if colors is not None:
        icolors = itertools.cycle(colors)
    else:
        icolors = None
    imarkerstyles = itertools.cycle(markerstyles)
    ilinestyles = itertools.cycle(linestyles)

    return style, colors, linestyles, markerstyles, icolors, ilinestyles, imarkerstyles
