# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import os
import warnings

import mando
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mando.rst_text_formatter import RSTHelpFormatter
from tstoolbox import tsutils

from .. import plotutils

matplotlib.use("Agg")

warnings.filterwarnings("ignore")


@mando.command("double_mass", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(plotutils.ldocstrings)
def double_mass_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    ofilename="plot.png",
    xtitle="",
    ytitle="",
    title="",
    figsize="10,6.0",
    legend=None,
    legend_names=None,
    colors="auto",
    linestyles="auto",
    markerstyles=" ",
    style="auto",
    xaxis="arithmetic",
    yaxis="arithmetic",
    xlim=None,
    ylim=None,
    grid=False,
    drawstyle="default",
    por=False,
    invert_xaxis=False,
    invert_yaxis=False,
    round_index=None,
    source_units=None,
    target_units=None,
    plot_styles="bright",
    hlines_y=None,
    hlines_xmin=None,
    hlines_xmax=None,
    hlines_colors=None,
    hlines_linestyles="-",
    vlines_x=None,
    vlines_ymin=None,
    vlines_ymax=None,
    vlines_colors=None,
    vlines_linestyles="-",
):
    r"""Double mass curve - cumulative sum of x against cumulative sum of y.

    "double_mass" creates a 'x,y' plot of the cumulative sum of x and y.

    ${xydata}

    Parameters
    ----------
    ${input_ts}
    ${columns}
    ${start_date}
    ${end_date}
    ${clean}
    ${skiprows}
    ${index_type}
    ${names}
    ${ofilename}
    ${xtitle}
    ${ytitle}
    ${title}
    ${figsize}
    ${legend}
    ${legend_names}
    ${colors}
    ${linestyles}
    ${markerstyles}
    ${style}
    ${xaxis}
    ${yaxis}
    ${xlim}
    ${ylim}
    ${grid}
    ${drawstyle}
    ${por}
    ${invert_xaxis}
    ${invert_yaxis}
    ${round_index}
    ${source_units}
    ${target_units}
    ${plot_styles}
    ${hlines_y}
    ${hlines_xmin}
    ${hlines_xmax}
    ${hlines_colors}
    ${hlines_linestyles}
    ${vlines_x}
    ${vlines_ymin}
    ${vlines_ymax}
    ${vlines_colors}
    ${vlines_linestyles}
    """
    plot = double_mass(
        input_ts=input_ts,
        columns=columns,
        start_date=start_date,
        end_date=end_date,
        clean=clean,
        skiprows=skiprows,
        index_type=index_type,
        names=names,
        ofilename=ofilename,
        xtitle=xtitle,
        ytitle=ytitle,
        title=title,
        figsize=figsize,
        legend=legend,
        legend_names=legend_names,
        colors=colors,
        linestyles=linestyles,
        markerstyles=markerstyles,
        style=style,
        xaxis=xaxis,
        yaxis=yaxis,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        drawstyle=drawstyle,
        por=por,
        invert_xaxis=invert_xaxis,
        invert_yaxis=invert_yaxis,
        round_index=round_index,
        source_units=source_units,
        target_units=target_units,
        plot_styles=plot_styles,
        hlines_y=hlines_y,
        hlines_xmin=hlines_xmin,
        hlines_xmax=hlines_xmax,
        hlines_colors=hlines_colors,
        hlines_linestyles=hlines_linestyles,
        vlines_x=vlines_x,
        vlines_ymin=vlines_ymin,
        vlines_ymax=vlines_ymax,
        vlines_colors=vlines_colors,
        vlines_linestyles=vlines_linestyles,
    )
    return plot


def double_mass(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    ofilename="plot.png",
    xtitle="",
    ytitle="",
    title="",
    figsize="10,6.0",
    legend=None,
    legend_names=None,
    colors="auto",
    linestyles="auto",
    markerstyles=" ",
    style="auto",
    xaxis="arithmetic",
    yaxis="arithmetic",
    xlim=None,
    ylim=None,
    grid=False,
    drawstyle="default",
    por=False,
    invert_xaxis=False,
    invert_yaxis=False,
    round_index=None,
    source_units=None,
    target_units=None,
    plot_styles="bright",
    hlines_y=None,
    hlines_xmin=None,
    hlines_xmax=None,
    hlines_colors=None,
    hlines_linestyles="-",
    vlines_x=None,
    vlines_ymin=None,
    vlines_ymax=None,
    vlines_colors=None,
    vlines_linestyles="-",
    **kwds,
):
    r"""Plot data."""

    # set up dataframe
    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna="all",
        source_units=source_units,
        target_units=target_units,
        clean=clean,
        por=por,
    )
    # check dataframe
    if tsd.shape[1] > 1:
        if tsd.shape[1] % 2 != 0:
            raise AttributeError(
                tsutils.error_wrapper(
                    """
The 'double_mass' type must have an even number of columns arranged
as x,y pairs or an x-index and one y data column.  You supplied {} columns.
""".format(
                        tsd.shape[1]
                    )
                )
            )
    colcnt = tsd.shape[1] // 2

    # Need to work around some old option defaults with the implementation of
    # mando
    legend = bool(legend == "" or legend == "True" or legend is None or legend is True)
    plottype = "double_mass"
    lnames = tsutils.make_list(legend_names)
    tsd, lnames = plotutils.check_column_legend(plottype, tsd, lnames)

    # check axis scales
    if xaxis == "log":
        logx = True
    else:
        logx = False
    if yaxis == "log":
        logy = True
    else:
        logy = False
    xlim = plotutils.know_your_limits(xlim, axis=xaxis)
    ylim = plotutils.know_your_limits(ylim, axis=yaxis)

    # process styles: colors, linestyles, markerstyles
    (
        style,
        colors,
        linestyles,
        markerstyles,
        icolors,
        ilinestyles,
        imarkerstyles,
    ) = plotutils.prepare_styles(colcnt, style, colors, linestyles, markerstyles)

    plot_styles = tsutils.make_list(plot_styles) + ["no-latex"]
    style_loc = os.path.join(
        os.path.dirname(__file__), os.pardir, "SciencePlots_styles"
    )
    plot_styles = [
        os.path.join(style_loc, i + ".mplstyle")
        if os.path.exists(os.path.join(style_loc, i + ".mplstyle"))
        else i
        for i in plot_styles
    ]
    plt.style.use(plot_styles)

    figsize = tsutils.make_list(figsize, n=2)
    _, ax = plt.subplots(figsize=figsize)

    plotdict = {
        (False, True): ax.semilogy,
        (True, False): ax.semilogx,
        (True, True): ax.loglog,
        (False, False): ax.plot,
    }

    # PANDAS was not doing the right thing with xy plots
    # if you wanted lines between markers.
    # Fell back to using raw matplotlib.
    # Boy I do not like matplotlib.

    for colindex in range(colcnt):
        if colcnt == 0:
            ndf = tsd.reset_index()
        else:
            ndf = tsd.iloc[:, colindex * 2 : colindex * 2 + 2]

        ndf = ndf.dropna()
        ndf.iloc[:, 0] = ndf.iloc[:, 0].cumsum()
        ndf.iloc[:, 1] = ndf.iloc[:, 1].cumsum()
        oxdata = np.array(ndf.iloc[:, 0])
        oydata = np.array(ndf.iloc[:, 1])

        if icolors is not None:
            c = next(icolors)
        else:
            c = None

        plotdict[(logx, logy)](
            oxdata,
            oydata,
            linestyle=next(ilinestyles),
            color=c,
            marker=next(imarkerstyles),
            label=lnames[colindex],
            drawstyle=drawstyle,
        )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if legend is True:
        ax.legend(loc="best")

    xtitle = xtitle or "Cumulative {}".format(tsd.columns[0])
    ytitle = ytitle or "Cumulative {}".format(tsd.columns[1])

    if hlines_y is not None:
        hlines_y = tsutils.make_list(hlines_y)
        hlines_xmin = tsutils.make_list(hlines_xmin)
        hlines_xmax = tsutils.make_list(hlines_xmax)
        hlines_colors = tsutils.make_list(hlines_colors)
        hlines_linestyles = tsutils.make_list(hlines_linestyles)
        nxlim = ax.get_xlim()
        if hlines_xmin is None:
            hlines_xmin = nxlim[0]
        if hlines_xmax is None:
            hlines_xmax = nxlim[1]
    if vlines_x is not None:
        vlines_x = tsutils.make_list(vlines_x)
        vlines_ymin = tsutils.make_list(vlines_ymin)
        vlines_ymax = tsutils.make_list(vlines_ymax)
        vlines_colors = tsutils.make_list(vlines_colors)
        vlines_linestyles = tsutils.make_list(vlines_linestyles)
        nylim = ax.get_ylim()
        if vlines_ymin is None:
            vlines_ymin = nylim[0]
        if vlines_ymax is None:
            vlines_ymax = nylim[1]

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    if invert_xaxis is True:
        plt.gca().invert_xaxis()
    if invert_yaxis is True:
        plt.gca().invert_yaxis()

    plt.grid(grid)

    plt.title(title)
    plt.tight_layout()
    if ofilename is not None:
        plt.savefig(ofilename)
    return plt


double_mass.__doc__ = double_mass_cli.__doc__
