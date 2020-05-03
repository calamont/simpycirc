import copy
import functools
import numpy as np
import matplotlib.pyplot as plt
from .figures import bode, nyquist
from .build import _solve_circuit, _component_impedance, _stamp
from .parse import _parse_func
from scipy.optimize import minimize


class NodalAnalysis:
    def __init__(self, netlist, freq):
        """Takes in netlist and builds node matrices for MNA."""
        if isinstance(freq, (int, float)):
            freq = [float(freq)]
        self.freq = np.array(freq)  # TODO: Use property to prevent changes to freq?
        self.netlist = netlist
        self.stamp_values = dict()
        self.update(**self.netlist.items())

    def update(self, *args, **kwargs):
        """Updates component values."""
        for key, val in kwargs.items():
            if key not in self.netlist.components:
                raise KeyError(f"{key} not defined for the original circuit.")
            self.netlist.components[key]["value"] = val
            if val is not None:
                self.stamp_values[key] = _component_impedance(key, val, self.freq)

    def copy(self):
        """Returns a deep copy of the netlist."""
        return copy.deepcopy(self)

    @property
    def A_matrix(self):
        if len(self.undefined) > 0:
            raise TypeError(f"{self} missing argument values for {self.undefined}")
        GB = np.concatenate([self.G_matrix, self.B_matrix], axis=2)
        CD = np.concatenate([self.C_matrix, self.D_matrix], axis=2)
        return np.concatenate([GB, CD], axis=1)

    @property
    def G_matrix(self):
        G = np.zeros(
            (len(self.freq), self.n_nodes - 1, self.n_nodes - 1), dtype=complex
        )
        for key, val in self.netlist.components.items():
            if not isinstance(val, dict):
                continue
            elif val.get("source", 0) != 0:
                continue
            if val["value"] is not None:
                Z = _component_impedance(key, val["value"], self.freq)
                _stamp(G, val["nodes"], Z)
        return G

    @property
    def B_matrix(self):
        B = np.zeros((len(self.freq), self.n_nodes - 1, self.netlist._n_sources))
        for key, val in self.netlist.components.items():
            if not isinstance(val, dict):
                continue
            elif val.get("source", 0) == 0:
                continue
            idx = reversed([i - 1 for i in val["nodes"] if i > 0])
            mat_val = 1
            for i in idx:
                B[:, i, val["source"] - 1] = mat_val
                mat_val *= -1
        return B

    @property
    def C_matrix(self):
        return np.moveaxis(self.B_matrix.T, -1, 0)

    @property
    def D_matrix(self):
        return np.zeros(
            (len(self.freq), self.netlist._n_sources, self.netlist._n_sources)
        )

    @property
    def z_matrix(self):
        z = np.zeros(self.n_nodes - 1 + self.netlist._n_sources)[None, :]
        for key, val in self.netlist.components.items():
            if val["source"]:
                z[:, -val["source"]] = val["value"]
        return z

    @property
    def nodes(self):
        nodes = []
        for key, val in self.netlist.components.items():
            if not isinstance(val, dict):
                continue
            nodes.extend(val.get("nodes", []))
        return set(nodes)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def undefined(self):
        return sorted(
            [
                key
                for key, val in self.netlist.components.items()
                if val.get("value", None) is None
            ]
        )

    @property
    def defined(self):
        return sorted(
            [
                key
                for key, val in self.netlist.components.items()
                if val.get("value", None) is not None
            ]
        )


class FrequencyAnalysis:
    """Object to interface with circuit."""

    def __init__(self, circuit):

        # self.simulate = self._modify_circuit(freq, circuit)
        self.circuit = circuit
        self.netlist = self.circuit.__closure__[1].cell_contents.copy()
        self.freq = self.netlist.freq

    def _modify_circuit(self, circuit, freq):
        """Create simulator circuit and update V attribute when called"""

        circuit = populate(circuit, freq=freq)

        def mod_circuit(*args, **kwargs):
            self.V = circuit(*args, **kwargs)
            self.default_vals = dict(circuit.__closure__[4].cell_contents)
            return self.V

        mod_circuit.__doc__ = circuit.__doc__

        return mod_circuit

    def multimeter(self, pos_node, neg_node=0, mode="V", **kwargs):
        V = self.circuit(**kwargs)
        if mode == "V":
            if not neg_node:
                return V[:, pos_node - 1]
            return V[:, pos_node - 1] - V[:, neg_node - 1]
        # If calculating current or impedances then we must find the impedances
        # between the circuit nodes, given by the A matrix
        tmp_netlist = self.netlist.copy()
        tmp_netlist.update(**kwargs)
        G = tmp_netlist.G_matrix
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
        ax=None,
        mpl_kwargs={},
        **kwargs,
    ):
        data = self.multimeter(pos_node, neg_node, mode, **kwargs)
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

        ax[0].set_ylabel(f"{mode}", fontname="Roboto")
        ax[1].set_ylabel("Phase (°)", fontname="Roboto")
        ax[1].set_xlabel("Frequency (Hz)", fontname="Roboto")
        ax[0].loglog(self.freq, Z, linewidth=1.5, color="k", **mpl_kwargs)
        ax[1].semilogx(self.freq, phase, linewidth=1.5, color="k", **mpl_kwargs)

        return ax

    def nyquist(self, node_label, measure="V", Z_ground=None, ax=None, **mpl_kwargs):
        if self.V is None:
            raise AttributeError(
                "No value for `Circuit.V`. Circuit must be called"
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


def populate2(freq=1000.0, circuit=None, **kwargs):

    freq = np.array(freq).reshape(-1)

    def decorator(circuit=None, **kwargs):

        if callable(circuit):
            netlist = _parse_func(circuit)
        else:
            netlist = circuit
        mna = NodalAnalysis(netlist, freq)

        @functools.wraps(circuit)
        def wrapper(**kwargs):

            if len(kwargs) > 0:
                mna_ = mna.copy()
                mna_.update(**kwargs)
            return np.linalg.solve(mna_.A_matrix, mna_.z_matrix)

        # wrapper.__doc__ = (
        #     "Automatically generated docstring. This function can now be used "
        #     + "to simulate the voltages across the nodes between the frequencies "
        #     + f"of {np.min(freq):.3f}-{np.max(freq):.3f}Hz.\n\nValue(s) for the "
        #     + f"remaning keyword arguments {netlist.undefined} need to be supplied to "
        #     + "complete the circuit.\n"
        # )

        return wrapper

    return decorator


def populate(circuit=None, freq=1000.0, **kwargs):
    """Generates function for simulating circuit.

    Modified nodal analysis defines a circuit by Kichhoff's circuit laws. The
    equation that must be solved is

    .. math::
    Ax = z

    Where A describes the various impedances and currents flowing in
    and out of each node, x is a vector of the voltages on each node of the
    circuit, and z is a vector of the voltage and current source values. As the
    impedance and source values are known, x can be solved for by taking the
    inverse of A

    .. math::
    x = A^{-1}\ z

    A is an (n+m) x (n+m) matrix, where n is the number of circuit nodes and m
    is the number of voltage sources. It is composed of four smaller matrices

    .. math::
    G B
    C D


    G is an n x n matrix that is composed of the inverse impedances connecting
    each node. B is an n x m matrix of the connctions of the voltage sources.
    C = B.T and D is an m x m matrix of zeros.

    Args:
        circuit (, optional): Function or spc.Netlist to create simulating
        function from. Defaults to None.
        freq (float, optional): Frequency/frequencies to simulate circuit over.
        Defaults to 1000.0.

    Raises:
        SyntaxError: Raised if voltage source not specified for circuit.

    Returns:
        function that takes circuit component values as position and keyword
        arguments and simulates voltage response of the circuit nodes.
    """
    freq = np.array(freq).reshape(-1)

    if circuit is not None:
        # If circuit is supplied as a function or string
        if callable(circuit):
            netlist = _parse_func(circuit)
        else:
            netlist = circuit

        mna = NodalAnalysis(netlist, freq)
        # z = np.zeros(mna.n_nodes - 1 + mna.sources)[None, :]
        # for k, v in mna.netlist.components.items():
        #     if v["source"]:
        #         z[:, -v["source"]] = v["value"]

        @functools.wraps(circuit)
        def wrapper(stamp_matrix=False, *args, **kwargs):

            if len(kwargs) > 0:
                mna_ = mna.copy()
                mna_.update(**kwargs)
            return np.linalg.solve(mna_.A_matrix, mna_.z_matrix)
            # return _solve_circuit(A.copy(), z, netlist, stamp_matrix, *args, **kwargs,)

        # wrapper.__doc__ = (
        #     "Automatically generated docstring. This function can now be used "
        #     + "to simulate the voltages across the nodes between the frequencies "
        #     + f"of {np.min(freq):.3f}-{np.max(freq):.3f}Hz.\n\nValue(s) for the "
        #     + f"remaning keyword arguments {netlist.undefined} need to be supplied to "
        #     + "complete the circuit.\n"
        # )

        return wrapper

    else:
        return functools.partial(populate, freq=freq, **kwargs)
