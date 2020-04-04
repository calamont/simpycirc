"""Random functions and scipts not relavent to other files"""
import numpy as np
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


def bode(V, node, freq, measure="V", Z_ground=None, ax=None, **mpl_kwargs):
    """Bode plot of simulated circuit response"""

    ax = plot(V, node, freq, measure, Z_ground, plot_type="bode", ax=ax, **mpl_kwargs)

    return ax


def nyquist(V, node, freq, measure="Z", Z_ground=None, ax=None, **mpl_kwargs):
    """Bode plot of simulated circuit response"""

    ax = plot(
        V, node, freq, measure, Z_ground, plot_type="nyquist", ax=ax, **mpl_kwargs
    )

    return ax


def plot(
    V, node, freq, measure="V", Z_ground=None, ax=None, plot_type="bode", **mpl_kwargs
):

    if measure == "Z":
        if Z_ground is None:
            raise AttributeError(
                "To calulate impedance a value must be set for `Z_ground`."
            )
        data = _v2z(V, node, Z_ground)
    elif measure == "I":
        data = (V[:, node] / Z_ground).flatten()
    elif measure == "V":
        data = V[:, node].flatten()

    if plot_type == "bode":
        data = np.abs(data), np.angle(data, deg=True)
        x_lim = [np.min(freq), np.max(freq)]
        y_lim = [
            [np.min(data[0]) * 0.5, np.max(data[0]) * 2],
            [np.min([0, *data[1]]) * 1.2, np.max([0, *data[1]]) * 1.2 + 5],
        ]
        figsize = (5.31, 5)

    elif plot_type == "nyquist":
        x_lim = [0, np.max(data.real) * 1.2]

        if np.sum(np.sign([np.min(data.imag), np.max(data.imag)])):
            idx_min = np.argmin(np.abs((data.imag)))
            idx_max = np.argmax(np.abs((data.imag)))
            y_lim = [[data.imag[idx_min] * 1.2, data.imag[idx_max] * 1.2]]
            figsize = (5.31, 3.25)
        else:
            # idx_min = np.argmin((data.imag))
            # idx_max = np.argmax((data.imag))
            y_max = np.max(np.abs((data.imag)))
            y_lim = [[-y_max * 1.2, y_max * 1.2]]
            figsize = (5.31, 4.5)

            # y_lim = [[data.imag[idx_min] * 1.2, data.imag[idx_max] * 1.2]]

    if ax is None:
        fig, ax = _create_graph(
            x_lim, y_lim, measure, plot_type=plot_type, figsize=figsize, **mpl_kwargs
        )
        # ax.spines["bottom"].set_position("zero")
    return _display_data(data, freq=freq, ax=ax, plot_type=plot_type)


def _create_graph(x_lim, y_lim, measure, plot_type="bode", sharex=True, **mpl_kwargs):
    if plot_type == "bode":

        fig, ax = plt.subplots(2, 1, sharex=True, **mpl_kwargs)
        ax[0].set_ylim(y_lim[0])

        ax[1].set_xlim(x_lim)
        ax[1].set_ylim(y_lim[1])

        # _set_labels("", f"Abs({measure})", ax=ax[0])
        # _set_labels("Frequency (Hz)", "Phase", ax=ax[1])

        ax[0].set_ylabel(f"Abs({measure})", fontname="Roboto")
        ax[1].set_xlabel("Frequency (Hz)", fontname="Roboto")
        ax[1].set_ylabel("Phase", fontname="Roboto")

    elif plot_type == "nyquist":

        # fig, ax = plt.subplots(**mpl_kwargs, figsize=figsize)
        fig = plt.figure(**mpl_kwargs)
        ax = fig.add_subplot(1, 1, 1, adjustable="box")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim[0])

        ax.set_xlabel(f"Real {measure}", fontname="Roboto")
        ax.set_ylabel(f"Imag {measure}", fontname="Roboto")

        # Change position of x-axis and its label if plot goes through
        # negative and positive values
        if (y_lim[0][0] < 0) & (y_lim[0][1] > 0):
            ax.spines["bottom"].set_position(("data", 0.0))
            ax.xaxis.set_label_coords(
                0.95, 0.55,
            )

    plt.tight_layout()

    return fig, ax


def _set_labels(x, y, ax=None, **kwargs):
    """To set the X & Y labels.

    A convenience function for setting the labels for the current figure.
    Any keyword arguments can also be forwarded onto the Matplotlib API.
    By supplying a Matplotlib Axes object to `axis`, it is possible to also
    change the labels of a figure which isn't the currently active plot.

    Args:
        x (str): The x-axis label
        y (str): The y-axis label
        axis (Axes): A Matplotlib Axes object.
    """

    if not ax:
        ax = plt.gca()
    ax.set_xlabel(x, fontname="Roboto", **kwargs)
    ax.set_ylabel(y, fontname="Roboto", **kwargs)


def _display_data(Z, ax, plot_type, freq=None):
    """Plot data on axes"""

    if plot_type == "bode":
        ax[0].loglog(freq, Z[0], linewidth=1.5, color="k")
        ax[1].semilogx(freq, Z[1], linewidth=1.5, color="k")
    elif plot_type == "nyquist":
        ax.plot(Z.real, Z.imag, linewidth=1.5, color="k")
    return ax


def _v2z(V, node, Z_ground):
    """Converts voltage to impedance"""

    I = V[:, node] / Z_ground
    Z_sim = (1 / I).flatten()

    return Z_sim
