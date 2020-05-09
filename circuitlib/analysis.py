import numpy as np
import matplotlib.pyplot as plt

y_label = {"V": "Voltage", "I": "Current", "Z": "Impedance"}


class FrequencyAnalysis:
    """Object to interface with circuit."""

    def __init__(self, circuit):
        self.circuit = circuit
        self.netlist = circuit.netlist.copy()
        self.freq = circuit.freq

    def multimeter(self, pos_node, neg_node=0, mode="V", **kwargs):
        V = self.circuit(**kwargs)
        if mode == "V":
            if not neg_node:
                return V[:, pos_node - 1]
            return V[:, pos_node - 1] - V[:, neg_node - 1]
        # If calculating current or impedances then we must find the impedances
        # between the circuit nodes, given by the A matrix
        tmp_circuit = self.circuit.copy()
        tmp_circuit.update(**kwargs)
        G = tmp_circuit.G_matrix
        if not neg_node:
            Z = 1 / np.sum(G[:, pos_node - 1, :], axis=-1)
            if mode == "Z":
                return Z
            elif mode == "I":
                return V[:, pos_node - 1] / Z
        Z = -1 / G[:, pos_node - 1, neg_node - 1]
        if mode == "Z":
            return Z
        elif mode == "I":
            return (V[:, pos_node - 1] - V[:, neg_node - 1]) / Z

    def bode(
        self,
        pos_node,
        neg_node=0,
        mode="V",
        linewidth=1.5,
        color="k",
        figsize=(5.31, 5),
        ax=None,
        **kwargs,
    ):
        clb_kwargs = {}
        mpl_kwargs = kwargs.copy()
        for k, v in kwargs.items():
            if k in self.circuit.components:
                clb_kwargs[k] = mpl_kwargs.pop(k)

        data = self.multimeter(pos_node, neg_node, mode, **clb_kwargs)
        Z, phase = np.abs(data), np.angle(data, deg=True)

        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize)

        ax[1].set_xlim([np.min(self.freq), np.max(self.freq)])
        y_lim = [
            [np.min(Z) * 0.5, np.max(Z) * 2],
            [np.min([0, *phase]) * 1.2, np.max([0, *phase]) * 1.2 + 5],
        ]
        ax[0].set_ylim(y_lim[0])
        ax[1].set_ylim(y_lim[1])

        ax[0].set_ylabel(y_label[mode], fontname="Roboto")
        ax[1].set_ylabel("Phase (°)", fontname="Roboto")
        ax[1].set_xlabel("Frequency (Hz)", fontname="Roboto")
        ax[0].loglog(self.freq, Z, linewidth=linewidth, color=color, **mpl_kwargs)
        ax[1].semilogx(self.freq, phase, linewidth=linewidth, color=color, **mpl_kwargs)

        return ax

    def nyquist(
        self,
        pos_node,
        neg_node=0,
        mode="V",
        linewidth=1.5,
        color="k",
        figsize=(5.31, 5),
        ax=None,
        **kwargs,
    ):
        clb_kwargs = {}
        mpl_kwargs = kwargs.copy()
        for k, v in kwargs.items():
            if k in self.circuit.components:
                clb_kwargs[k] = mpl_kwargs.pop(k)

        data = self.multimeter(pos_node, neg_node, mode, **clb_kwargs)

        x_lim = [0, np.max(data.real) * 1.2]
        if np.sum(np.sign([np.min(data.imag), np.max(data.imag)])):
            idx_min = np.argmin(np.abs((data.imag)))
            idx_max = np.argmax(np.abs((data.imag)))
            y_lim = [data.imag[idx_min] * 1.2, data.imag[idx_max] * 1.2]
            figsize = (5.31, 3.25)
        else:
            y_max = np.max(np.abs((data.imag)))
            y_lim = [-y_max * 1.2, y_max * 1.2]
            figsize = (5.31, 4.5)

        if ax is None:
            fig, ax = plt.subplots(sharex=True, figsize=figsize)

        # ax.set_xlim(x_lim)
        # ax.set_ylim(y_lim)
        ax.set_xlabel(f"Real {y_label[mode].lower()}", fontname="Roboto")
        ax.set_ylabel(f"Imag {y_label[mode].lower()}", fontname="Roboto")

        # Change position of x-axis and its label if plot goes through
        # negative and positive values
        if (y_lim[0] < 0) & (y_lim[1] > 0):
            ax.spines["bottom"].set_position(("data", 0.0))
            ax.xaxis.set_label_coords(
                0.95, 0.55,
            )

        ax.plot(data.real, data.imag, linewidth=linewidth, color=color, **mpl_kwargs)

        # plt.tight_layout()

        return ax

        # ax[1].set_xlim([np.min(self.freq), np.max(self.freq)])
        # y_lim = [
        #     [np.min(Z) * 0.5, np.max(Z) * 2],
        #     [np.min([0, *phase]) * 1.2, np.max([0, *phase]) * 1.2 + 5],
        # ]
        # ax[0].set_ylim(y_lim[0])
        # ax[1].set_ylim(y_lim[1])

        # ax[0].set_ylabel(y_label[mode], fontname="Roboto")
        # ax[1].set_ylabel("Phase (°)", fontname="Roboto")
        # ax[1].set_xlabel("Frequency (Hz)", fontname="Roboto")
        # ax[0].loglog(self.freq, Z, linewidth=linewidth, color=color, **mpl_kwargs)
        # ax[1].semilogx(self.freq, phase, linewidth=linewidth, color=color, **mpl_kwargs)

        # if self.V is None:
        #     raise AttributeError(
        #         "No value for `Circuit.V`. Circuit must be called"
        #         + "at least once before plotting results."
        #     )
        # return nyquist(
        #     V=self.V,
        #     node=node_label,
        #     freq=self.freq,
        #     measure=measure,
        #     Z_ground=Z_ground,
        #     ax=ax,
        #     **mpl_kwargs,
        # )
