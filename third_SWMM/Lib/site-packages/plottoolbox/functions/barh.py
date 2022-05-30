# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import itertools
import os
import warnings

import mando
import matplotlib
import matplotlib.pyplot as plt
from mando.rst_text_formatter import RSTHelpFormatter
from tstoolbox import tsutils

from .. import plotutils

matplotlib.use("Agg")


warnings.filterwarnings("ignore")

plotutils.HATCH_LIST = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]


@mando.command("barh", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(plotutils.ldocstrings)
def barh_cli(
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
    bar_hatchstyles="auto",
    style="auto",
    xaxis="arithmetic",
    yaxis="arithmetic",
    xlim=None,
    ylim=None,
    grid=False,
    label_rotation=None,
    label_skip=1,
    force_freq=None,
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
    r"""Bar plot, sometimes called a "column" plot.

    "barh" creates a horizontal bar plot.

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
    ${bar_hatchstyles}
    ${style}
    ${xaxis}
    ${yaxis}
    ${xlim}
    ${ylim}
    ${grid}
    ${label_rotation}
    ${label_skip}
    ${force_freq}
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
    plot = barh(
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
        bar_hatchstyles=bar_hatchstyles,
        style=style,
        xaxis=xaxis,
        yaxis=yaxis,
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        label_rotation=label_rotation,
        label_skip=label_skip,
        force_freq=force_freq,
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


def barh(
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
    bar_hatchstyles="auto",
    style="auto",
    xaxis="arithmetic",
    yaxis="arithmetic",
    xlim=None,
    ylim=None,
    grid=False,
    label_rotation=None,
    label_skip=1,
    force_freq=None,
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

    # Need to work around some old option defaults with the implementation of
    # mando
    legend = bool(legend == "" or legend == "True" or legend is None or legend is True)
    plottype = "barh"
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
    ) = plotutils.prepare_styles(
        len(tsd.columns), style, colors, linestyles, markerstyles
    )

    if bar_hatchstyles == "auto":
        bar_hatchstyles = plotutils.HATCH_LIST
    else:
        bar_hatchstyles = tsutils.make_list(bar_hatchstyles)
    ibar_hatchstyles = itertools.cycle(bar_hatchstyles)

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

    stacked = False
    kind = "barh"
    if icolors is not None:
        c = [next(icolors) for i in range(len(tsd.columns))]
    else:
        c = None
    tsd.plot(
        ax=ax,
        kind=kind,
        legend=legend,
        stacked=stacked,
        logx=logx,
        logy=logy,
        xlim=xlim,
        ylim=ylim,
        figsize=figsize,
        linestyle=None,
        color=c,
    )

    hatches = [next(ibar_hatchstyles) for i in range(len(tsd.columns))]
    hatches = "".join(h * len(tsd.index) for h in hatches)
    for patch, hatch in zip(ax.patches, hatches):
        patch.set_hatch(hatch)

    freq = tsutils.asbestfreq(tsd, force_freq=force_freq).index.freqstr
    if freq is not None:
        if "A" in freq:
            endchar = 4
        elif "M" in freq:
            endchar = 7
        elif "D" in freq:
            endchar = 10
        elif "H" in freq:
            endchar = 13
        else:
            endchar = None
        nticklabels = []
        taxis = ax.yaxis
        for index, i in enumerate(taxis.get_majorticklabels()):
            if index % label_skip:
                nticklabels.append(" ")
            else:
                nticklabels.append(i.get_text()[:endchar])
        taxis.set_ticklabels(nticklabels)
        plt.setp(taxis.get_majorticklabels(), rotation=label_rotation)

    if legend is True:
        plt.legend(loc="best")

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
    if hlines_y is not None:
        plt.hlines(
            hlines_y,
            hlines_xmin,
            hlines_xmax,
            colors=hlines_colors,
            linestyles=hlines_linestyles,
        )
    if vlines_x is not None:
        plt.vlines(
            vlines_x,
            vlines_ymin,
            vlines_ymax,
            colors=vlines_colors,
            linestyles=vlines_linestyles,
        )

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


barh.__doc__ = barh_cli.__doc__
