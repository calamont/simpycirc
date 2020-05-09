import copy
import types
import numpy as np
from .build import _component_impedance, _stamp
from .parse import _parse_func


class NodalAnalysis:
    def __init__(self, freq, netlist=None):
        """Takes in netlist and builds node matrices for MNA."""
        if isinstance(freq, (int, float)):
            freq = [float(freq)]
        self.freq = np.array(freq)  # TODO: Use property to prevent changes to freq?
        self.stamp_values = dict()
        self._initialised = False
        if netlist is not None:
            self.__call__(netlist)

    def __call__(self, *args, **kwargs):
        if not self._initialised:
            # If circuit is supplied as a function or string
            if callable(args[0]):
                self.netlist = _parse_func(args[0])
            else:
                self.netlist = args[0]

            def solve_matrix(**kwargs):
                if len(kwargs) > 0:
                    mna_ = self.copy()
                    mna_.update(**kwargs)
                else:
                    mna_ = self
                return np.linalg.solve(mna_.A_matrix, mna_.z_matrix)

            solve_matrix = self._add_func_signature(solve_matrix)
            self.__call__ = solve_matrix
            self._initialised = True
            return self

        return self.__call__(**kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def _add_func_signature(self, func):
        func_args = [
            0,
            len(self.netlist),
            func.__code__.co_nlocals,
            func.__code__.co_stacksize,
            func.__code__.co_flags,
            func.__code__.co_code,
            (),
            (),
            tuple(self.defined + self.undefined + ["kwargs"]),
            func.__code__.co_filename,
            func.__code__.co_name,
            func.__code__.co_firstlineno,
            func.__code__.co_lnotab,
        ]
        func_code = types.CodeType(*func_args)
        new_func = types.FunctionType(func_code, globals())
        func.__wrapped__ = new_func
        func.__doc__ = """Generates function for simulating circuit.

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
        return func

    def update(self, **kwargs):
        """Updates component values."""
        for key, val in kwargs.items():
            if key not in self.netlist.components:
                raise KeyError(f"{key} not defined for the original circuit.")
            self.netlist.components[key]["value"] = val
            if val is not None:
                self.stamp_values[key] = _component_impedance(key, val, self.freq)

    @property
    def A_matrix(self):
        if len(self.undefined) > 0:
            # Create readable string of components without defined values
            missing_vars = self.undefined[0]
            if len(self.undefined) == 2:
                missing_vars += " and " + self.undefined[-1]
            elif len(self.undefined) > 2:
                missing_vars += (
                    ", "
                    + ", ".join(self.undefined[1:-1])
                    + " and "
                    + self.undefined[-1]
                )
            raise TypeError(f"{self} missing argument values for {missing_vars}")
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

    @property
    def components(self):
        return self.defined + self.undefined
