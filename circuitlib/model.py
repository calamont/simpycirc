import numpy as np
from .circuit import simulator
from .figures import bode, nyquist

from functools import partial
from scipy.optimize import minimize


class Model:
    """Dummy model to perform all tasks"""

    def __init__(self, schematic, freq):

        self.simulate = self._modify_circuit(schematic, freq=freq)
        self.freq = freq
        self.V = None
        self.default_vals = None

    def _modify_circuit(self, schematic, freq):
        """Create simulator circuit and update V attribute when called"""

        circuit = simulator(schematic, freq=freq)

        def mod_circuit(*args, **kwargs):
            self.V = circuit(*args, **kwargs)
            self.default_vals = dict(circuit.__closure__[4].cell_contents)
            return self.V

        mod_circuit.__doc__ = circuit.__doc__

        return mod_circuit

    def node(
        self, *circuit_args, node_label, measure="V", Z_ground=None, **circuit_kwargs
    ):

        V = self.simulate(*circuit_args, **circuit_kwargs)

        if measure == "Z":
            if Z_ground is None:
                raise AttributeError(
                    "To calulate impedance a value must be set for `Z_ground`."
                )
            data = _v2z(V, node_label, Z_ground)
        elif measure == "I":
            data = V[:, node_label] / Z_ground
        elif measure == "V":
            data = V[:, node_label]
        return data

    def bode(self, node_label, measure="V", Z_ground=None, ax=None, **mpl_kwargs):
        if self.V is None:
            raise AttributeError(
                "No value for `Model.V`. Circuit must be called"
                + "at least once before plotting results."
            )
        return bode(
            V=self.V,
            node=node_label,
            freq=self.freq,
            measure=measure,
            Z_ground=Z_ground,
            ax=ax,
            **mpl_kwargs,
        )

    def nyquist(self, node_label, measure="V", Z_ground=None, ax=None, **mpl_kwargs):
        if self.V is None:
            raise AttributeError(
                "No value for `Model.V`. Circuit must be called"
                + "at least once before plotting results."
            )
        return nyquist(
            V=self.V,
            node=node_label,
            freq=self.freq,
            measure=measure,
            Z_ground=Z_ground,
            ax=ax,
            **mpl_kwargs,
        )

    def fit(
        self, data, node_label, measure, Z_ground, return_results=False, **scipy_kwargs
    ):

        components = [
            key for key, val in self.default_vals.items() if val["value"] is None
        ]
        init_guess = []
        # Provide reasonable guess for component value
        for key in sorted(components):
            if key[0] == "R":
                init_guess.append(5)
            elif key[0] == "C":
                init_guess.append(-10)
            elif key[0] == "L":
                init_guess.append(-3)

        results = minimize(
            self._cost,
            init_guess,
            args=(data, node_label, measure, Z_ground),
            **scipy_kwargs,
        )

        self.fit_vals = {c: v for c, v in zip(sorted(components), 10 ** results.x)}

        if return_results:
            return results

    def _cost(self, params, data, node_label, measure, Z_ground):
        sim = np.abs(
            self.node(
                *10 ** params, node_label=node_label, measure=measure, Z_ground=Z_ground
            )
        )
        return np.square(np.subtract(np.log10(data), np.log10(sim))).mean()
        # sim = self.node(
        #     *10 ** params, node_label=node_label, measure=measure, Z_ground=Z_ground
        # )
        # return (
        #     np.square(np.subtract((data.real), (sim.real))).mean()
        #     + np.square(np.subtract((data.imag), (sim.imag))).mean()
        # )

        return np.log10(np.square(np.abs((np.subtract(data, sim))))).mean()

    def fit_components(data, **kwargs):

        fit = minimize(cost, [10], args=(circuit, data), **kwargs)

        return 10 ** fit.x

