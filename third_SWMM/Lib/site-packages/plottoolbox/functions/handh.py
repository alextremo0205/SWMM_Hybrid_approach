# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

# this file needs definitions of input parameters and a doc string


def plotHH(t, sP, sQ):
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])

    # HYDROGRAM CHART
    ax = plt.subplot(gs[1])
    ax.plot(t, sQ)
    ax.set_ylabel("Q(m³/s)", color="b")
    ax.set_xlabel("Time (min.)")
    ax.tick_params(axis="y", colors="b")
    ax.xaxis.grid(b=True, which="major", color=".7", linestyle="-")
    ax.yaxis.grid(b=True, which="major", color=".7", linestyle="-")
    ax.set_xlim(min(t), max(t))
    ax.set_ylim(0, max(sQ) * 1.2)

    # PRECIPITATION/HYETOGRAPH CHART
    ax2 = plt.subplot(gs[0])
    ax2.bar(t, sP, 1, color="#b0c4de")
    ax2.xaxis.grid(b=True, which="major", color=".7", linestyle="-")
    ax2.yaxis.grid(b=True, which="major", color="0.7", linestyle="-")
    ax2.set_ylabel("P(mm)")
    ax2.set_xlim(min(t), max(t))
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.tight_layout()
    ax2.invert_yaxis()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()
    # plt.savefig(filename,format='pdf')
    plt.close(fig)
