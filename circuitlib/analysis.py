"""Classes for the analysis of node voltages/currents."""

import numpy as np
import matplotlib.pyplot as plt

y_label = {"V": "Voltage", "I": "Current", "Z": "Impedance"}


class FrequencyAnalysis:
    """Convenient analysis of circuit over its defined frequencies.

    Attributes:
        circuit (`circuitlib.NodalAnalysis`): A defined circuit that has been
            parsed into a `NodalAnalysis` object.
    """

    def __init__(self, circuit):
        """Constructor of `FrequencyAnalysis` class.

        Args:
            circuit (`circuitlib.NodalAnalysis`): A defined circuit that has been
            parsed into a `NodalAnalysis` object.
        """
        self.circuit = circuit

    def multimeter(self, pos_node, neg_node=0, mode="V", **kwargs):
        """Measures voltage, current or impedance between circuit nodes.

        Measurements are taken between two nodes, where the negative node
        (`neg_node`) acts as the reference. The values of circuit components
        can be updated by passing these in as kwargs.

        .. code:: python

            fra = FrequencyAnalysis(my_circuit)
            fra.multimeter(pos_node=2, R1=10e3, R2=20e3)


        Args:
            pos_node (int): The circuit node of interest, equivalent to the
                positive lead on a multimeter.
            neg_node (int, optional): The reference node from which the
                measurement is defined, equivalent to the negative lead on a
                multimeter. Defaults to 0 (i.e. ground).
            mode (str, optional): The type of measurement to perform.
                Options are "V" (voltage), "I" (current), and "Z" (impedance).
                Defaults to "V".

        Returns:
            numpy.ndarray: The measurement between `pos_node` and `neg_node`
                over the frequencies defined for the circuit.
        """
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
        figsize=(5.31, 5),
        dpi=100,
        ax=None,
        **kwargs,
    ):
        """Displays Bode plot of the circuit.

        Measurements are taken between two nodes, where the negative node
        (`neg_node`) acts as the reference. The values of circuit components
        can be updated by passing these in as kwargs. Similarly any kwargs
        for the `matplotlib.pyplot.plot` function may be passed as well.

        .. code:: python

            fra = FrequencyAnalysis(my_circuit)
            fra.bode(pos_node=2, R1=10e3, linestyle="--")


        Args:
            pos_node (int): The circuit node of interest, equivalent to the
                positive lead on a multimeter.
            neg_node (int, optional): The reference node from which the
                measurement is defined, equivalent to the negative lead on a
                multimeter. Defaults to 0 (i.e. ground).
            mode (str, optional): The type of measurement to perform.
                Options are "V" (voltage), "I" (current), and "Z" (impedance).
                Defaults to "V".
            figsize (tuple, optional): Width and height of the figure in inches.
                Defaults to (5.31, 5).
            dpi (int, optional): The figure resolution. Defaults to 100.
            ax (`matplotlib.pyplot.Axes`, optional): Object or array of
                `matplotlib.pyplot.Axes` objects to draw Bode plot on.
                Defaults to None.
            **kwargs:
                Circuit component values (i.e. R1=100) or additional arguments
                passed to `matplotlib.pyplot.plot` call.

        Returns:
            ax: Object or array of `matplotlib.pyplot.Axes` objects.
        """
        # Split kwargs into `circuitlib` components and
        # `matplotlib.pyplot.plot` kwargs
        clb_kwargs = {}
        mpl_kwargs = kwargs.copy()
        for k, v in kwargs.items():
            if k in self.circuit.components:
                clb_kwargs[k] = mpl_kwargs.pop(k)

        data = self.multimeter(pos_node, neg_node, mode, **clb_kwargs)
        Z, phase = np.abs(data), np.angle(data, deg=True)

        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize, dpi=dpi)

        # TODO: Choose best limits if multiple plots are drawn on the same fig.
        ax[1].set_xlim([np.min(self.circuit.freq), np.max(self.circuit.freq)])
        y_lim = [
            [np.min(Z) * 0.5, np.max(Z) * 2],
            [np.min([0, *phase]) * 1.2, np.max([0, *phase]) * 1.2 + 5],
        ]
        ax[0].set_ylim(y_lim[0])
        ax[1].set_ylim(y_lim[1])
        ax[0].set_ylabel(y_label[mode], fontname="Roboto")
        ax[1].set_ylabel("Phase (°)", fontname="Roboto")
        ax[1].set_xlabel("Frequency (Hz)", fontname="Roboto")
        ax[0].loglog(self.circuit.freq, Z, **mpl_kwargs)
        ax[1].semilogx(self.circuit.freq, phase, **mpl_kwargs)
        return ax

    def nyquist(
        self,
        pos_node,
        neg_node=0,
        mode="V",
        figsize=(5.31, 3.25),
        dpi=100,
        ax=None,
        **kwargs,
    ):
        """Displays Nyquist plot of the circuit.

        Measurements are taken between two nodes, where the negative node
        (`neg_node`) acts as the reference. The values of circuit components
        can be updated by passing these in as kwargs. Similarly any kwargs
        for the `matplotlib.pyplot.plot` function may be passed as well.

        .. code:: python

            fra = FrequencyAnalysis(my_circuit)
            fra.nyquist(pos_node=2, R1=10e3, linestyle="--")


        Args:
            pos_node (int): The circuit node of interest, equivalent to the
                positive lead on a multimeter.
            neg_node (int, optional): The reference node from which the
                measurement is defined, equivalent to the negative lead on a
                multimeter. Defaults to 0 (i.e. ground).
            mode (str, optional): The type of measurement to perform.
                Options are "V" (voltage), "I" (current), and "Z" (impedance).
                Defaults to "V".
            figsize (tuple, optional): Width and height of the figure in inches.
                Defaults to (5.31, 5).
            dpi (int, optional): The figure resolution. Defaults to 100.
            ax (`matplotlib.pyplot.Axes`, optional): Object or array of
                `matplotlib.pyplot.Axes` objects to draw Bode plot on.
                Defaults to None.
            **kwargs:
                Circuit component values (i.e. R1=100) or additional arguments
                passed to `matplotlib.pyplot.plot` call.

        Returns:
            ax: Object or array of `matplotlib.pyplot.Axes` objects.
        """
        # Split kwargs into `circuitlib` components and
        # `matplotlib.pyplot.plot` kwargs
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
        else:
            y_max = np.max(np.abs((data.imag)))
            y_lim = [-y_max * 1.2, y_max * 1.2]

        if ax is None:
            fig, ax = plt.subplots(sharex=True, figsize=figsize, dpi=dpi)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel(f"Real {y_label[mode].lower()}", fontname="Roboto")
        ax.set_ylabel(f"Imag {y_label[mode].lower()}", fontname="Roboto")

        # Change position of x-axis and its label if plot goes through
        # negative and positive values
        if (y_lim[0] < 0) & (y_lim[1] > 0):
            ax.spines["bottom"].set_position(("data", 0.0))
            ax.xaxis.set_label_coords(
                0.95, 0.55,
            )
        ax.plot(data.real, data.imag, **mpl_kwargs)
        return ax
