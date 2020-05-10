import matplotlib as mpl
import matplotlib.pyplot as plt

from cycler import cycler


def init_mpl(
    plot_type="line", dpi=200, figsize=(5.31, 3.99), grid=False, **custom_params
):
    """Updates matplotlib styling"""
    mpl.rcParams.update(
        {
            **{"figure.dpi": dpi, "figure.figsize": figsize, "axes.grid": grid},
            **figure_params,
            **custom_params,
        }
    )


axes_linewidth = 1.0
major_tick = 0.5
minor_tick = 0.4
top_right_spines = False
figure_params = {
    "font.family": "Roboto Light",
    "font.size": 10.0,
    "figure.subplot.top": 0.95,  # Position of graph top
    "figure.subplot.bottom": 0.15,  # Position of graph bottom
    "figure.dpi": 100,
    "grid.alpha": 0.5,
    "grid.color": "#b0b0b0",
    "grid.linestyle": "-",
    "grid.linewidth": minor_tick,
    "axes.grid.axis": "both",
    "axes.grid.which": "major",
    "axes.labelcolor": "k",
    "axes.labelpad": 8.0,
    "axes.labelsize": 10.0,
    "axes.labelweight": 1.75,
    "axes.prop_cycle": cycler(
        color=["#111111", "#333333", "#555555", "#777777",],
        linestyle=["-", "-.", "--", ":"],
    ),
    "axes.linewidth": axes_linewidth,
    "axes.spines.right": top_right_spines,
    "axes.spines.top": top_right_spines,
    "lines.color": "#008000",
    "lines.dash_capstyle": "butt",
    "lines.dash_joinstyle": "round",
    "lines.dashdot_pattern": [6.4, 1.6, 1.0, 1.6],
    "lines.dashed_pattern": [3.7, 1.6],
    "lines.dotted_pattern": [1.0, 1.65],
    "lines.linestyle": "-",
    "lines.linewidth": 1.7,
    "lines.marker": "None",
    "lines.markeredgewidth": 1.0,
    "lines.markersize": 0.0,
    "lines.scale_dashes": True,
    "lines.solid_capstyle": "projecting",
    "lines.solid_joinstyle": "round",
    "xtick.alignment": "center",
    "xtick.bottom": True,
    "xtick.color": "k",
    "xtick.direction": "in",
    "xtick.labelbottom": True,
    "xtick.labelsize": 10.0,
    "xtick.labeltop": False,
    "xtick.major.bottom": True,
    "xtick.major.pad": 10.0,
    "xtick.major.size": 4.0,
    "xtick.major.top": top_right_spines,
    "xtick.major.width": major_tick,
    "xtick.minor.bottom": True,
    "xtick.minor.pad": 10.0,
    "xtick.minor.size": 2.0,
    "xtick.minor.top": top_right_spines,
    "xtick.minor.visible": False,
    "xtick.minor.width": minor_tick,
    "xtick.top": True,
    "ytick.alignment": "center_baseline",
    "ytick.color": "k",
    "ytick.direction": "in",
    "ytick.labelleft": True,
    "ytick.labelright": False,
    "ytick.labelsize": 10.0,
    "ytick.left": True,
    "ytick.major.left": True,
    "ytick.major.pad": 10.0,
    "ytick.major.right": top_right_spines,
    "ytick.major.size": 4.0,
    "ytick.major.width": major_tick,
    "ytick.minor.left": True,
    "ytick.minor.pad": 6.0,
    "ytick.minor.right": top_right_spines,
    "ytick.minor.size": 2.0,
    "ytick.minor.visible": False,
    "ytick.minor.width": minor_tick,
    "ytick.right": True,
}
