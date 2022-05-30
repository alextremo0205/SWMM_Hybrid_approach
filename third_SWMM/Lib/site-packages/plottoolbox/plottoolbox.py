# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import os.path
import sys
import warnings

from mando import command, main

from .functions.autocorrelation import autocorrelation
from .functions.bar import bar
from .functions.bar_stacked import bar_stacked
from .functions.barh import barh
from .functions.barh_stacked import barh_stacked
from .functions.bootstrap import bootstrap
from .functions.boxplot import boxplot
from .functions.double_mass import double_mass
from .functions.heatmap import heatmap
from .functions.histogram import histogram
from .functions.kde import kde
from .functions.kde_time import kde_time
from .functions.lag_plot import lag_plot
from .functions.lognorm_xaxis import lognorm_xaxis
from .functions.lognorm_yaxis import lognorm_yaxis
from .functions.norm_xaxis import norm_xaxis
from .functions.norm_yaxis import norm_yaxis
from .functions.probability_density import probability_density
from .functions.scatter_matrix import scatter_matrix
from .functions.target import target
from .functions.taylor import taylor
from .functions.time import time
from .functions.weibull_xaxis import weibull_xaxis
from .functions.weibull_yaxis import weibull_yaxis
from .functions.xy import xy

warnings.filterwarnings("ignore")


@command()
def about():
    """Display version number and system information."""
    from tstoolbox import tsutils

    tsutils.about(__name__)


def lmain():
    """Set debug and run mando.main function."""
    if not os.path.exists("debug_plottoolbox"):
        sys.tracebacklimit = 0
    main()


if __name__ == "__main__":
    lmain()
