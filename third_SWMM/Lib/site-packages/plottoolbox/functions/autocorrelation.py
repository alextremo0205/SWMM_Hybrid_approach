# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
import matplotlib
import matplotlib.pyplot as plt
from mando.rst_text_formatter import RSTHelpFormatter
from pandas.plotting import autocorrelation_plot
from tstoolbox import tsutils

from .. import plotutils

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


@mando.command("autocorrelation", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(plotutils.ldocstrings)
def autocorrelation_cli(
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
    xlim=None,
    ylim=None,
    grid=False,
    xlabel_rotation=0,
    ylabel_rotation=0,
    por=False,
    round_index=None,
    source_units=None,
    target_units=None,
    plot_styles="bright",
):
    r"""Autocorrelation plot.

    The "autocorrelation" creates an autocorrelation plot.

    The horizontal lines in the plot correspond to 95% and 99% confidence
    bands.

    The dashed line is 99% confidence band.

    Only available for a single time-series.

    ${yone}

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
    ${xlim}
    ${ylim}
    ${grid}
    ${xlabel_rotation}
    ${ylabel_rotation}
    ${por}
    ${round_index}
    ${source_units}
    ${target_units}
    ${plot_styles}
    """
    plot = autocorrelation(
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
        xlim=xlim,
        ylim=ylim,
        grid=grid,
        xlabel_rotation=xlabel_rotation,
        ylabel_rotation=ylabel_rotation,
        por=por,
        round_index=round_index,
        source_units=source_units,
        target_units=target_units,
        plot_styles=plot_styles,
    )
    return plot


def autocorrelation(
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
    xlim=None,
    ylim=None,
    grid=False,
    xlabel_rotation=0,
    ylabel_rotation=0,
    por=False,
    round_index=None,
    source_units=None,
    target_units=None,
    plot_styles="bright",
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
    plottype = "autocorrelation"
    lnames = tsutils.make_list(legend_names)
    tsd, lnames = plotutils.check_column_legend(plottype, tsd, lnames)

    plt.style.use(plot_styles)

    figsize = tsutils.make_list(figsize, n=2)
    _, ax = plt.subplots(figsize=figsize)

    autocorrelation_plot(tsd, ax=ax)
    # This is to help pretty print the frequency
    try:
        try:
            pltfreq = str(tsd.index.freq, "utf-8").lower()
        except TypeError:
            pltfreq = str(tsd.index.freq).lower()
        if pltfreq.split(" ")[0][1:] == "1":
            beginstr = 3
        else:
            beginstr = 1
        if pltfreq == "none":
            short_freq = ""
        else:
            # short freq string (day) OR (2 day)
            short_freq = f"({pltfreq[beginstr:-1]})"
    except AttributeError:
        short_freq = ""
    xtitle = xtitle or f"Time Lag {short_freq}"

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)

    plt.xticks(rotation=xlabel_rotation)
    plt.yticks(rotation=ylabel_rotation)

    plt.grid(grid)

    plt.title(title)

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.tight_layout()
    if ofilename is not None:
        plt.savefig(ofilename)
    return plt


autocorrelation.__doc__ = autocorrelation_cli.__doc__
