"""Random functions and scipts not relavent to other files"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from .params import figure_params, line_params, scatter_params

label_dict = {"V": "Voltage", "I": "Current", "Z": "Impedance ($\Omega$)"}


def init_mpl(
    plot_type="line", dpi=200, figsize=(5.31, 3.99), grid=False, **custom_params
):
    """Updates matplotlib styling"""
    graph_params = line_params if plot_type == "line" else scatter_params
    mpl.rcParams.update(
        {
            **{"figure.dpi": dpi, "figure.figsize": figsize, "axes.grid": grid},
            **figure_params,
            **graph_params,
            **custom_params,
        }
    )
