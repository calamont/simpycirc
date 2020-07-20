"""Code for constructing and solving the circuit's nodal analysis matrices."""

import sys
import copy
import types
import inspect
import numpy as np
import matplotlib.pyplot as plt

from .stamps import Stamps
from .netlist import Netlist
from .parse import _parse_func

y_label = {"V": "Voltage", "I": "Current", "Z": "Impedance"}


class ModifiedNodalAnalysis:
    """Constructs MNA matrices to solve."""

    def __init__(self, circuit=None):
        """Takes in netlist and builds node matrices for MNA."""
        if circuit is not None:
            if callable(circuit):
                self.netlist = _parse_func(circuit)
            elif isinstance(circuit, Netlist):
                self.netlist = circuit
            else:
                raise TypeError(
                    "The passed circuit must be a function or a `circuitlib.netlist.Netlist` object."
                )

            self.A1, self.A2, self.s = self._stamp_matrices(self.netlist)

    def __call__(self, *args, **kwargs):
        pass

    def _stamp_matrices(self, netlist):
        """Samp values into two square matrices for group 1 and group 2 components."""
        mat_size = len(netlist.group1_components + netlist.group2_components)
        A1 = np.zeros((mat_size, mat_size))
        A2 = np.zeros((mat_size, mat_size))
        s = np.zeros(mat_size)

        stamp_funcs = Stamps()  # TODO: does this need to be a class or could be dict?
        for key, val in netlist.components.items():
            if val["value"] is not None:
                stamp = getattr(stamp_funcs, val["type"])
                A1, A2, s = stamp(A1, A2, s, **val)
        return A1, A2, s

    def update(self, **kwargs):
        """Updates default component values."""
        stamp_funcs = Stamps()
        # Check if kwargs match existing component values.
        for key, val in kwargs.items():
            if key not in self.netlist.components:
                raise KeyError(f"{key} not defined for the original circuit.")

            comp = self.netlist.components[key]
            comp["value"] = val
            stamp = getattr(stamp_funcs, comp["type"])
            self.A1, self.A2, self.s = stamp(self.A1, self.A2, self.s, **{**comp})

    def copy(self):
        """Deep copy of object. Needed for `__call__` if **kwargs supplied."""
        return copy.deepcopy(self)

    @property
    def nodes(self):
        """The nodes in the defined circuit."""
        nodes = []
        for key, val in self.netlist.components.items():
            if not isinstance(val, dict):
                continue
            nodes.extend(val.get("nodes", []))
        return set(nodes)

    @property
    def n_nodes(self):
        """The number of nodes in the defined circuit."""
        return len(self.nodes)

    @property
    def undefined(self):
        """The circuit components without a defined value."""
        return sorted(
            [
                key
                for key, val in self.netlist.components.items()
                if val.get("value", None) is None
            ]
        )

    @property
    def defined(self):
        """The circuit components with a defined value."""
        return sorted(
            [
                key
                for key, val in self.netlist.components.items()
                if val.get("value", None) is not None
            ]
        )

    @property
    def components(self):
        """The circuit components."""
        return self.defined + self.undefined


class AC(ModifiedNodalAnalysis):
    """AC analysis of circuit."""

    def __init__(self, circuit):
        super().__init__(circuit=circuit)
        if callable(circuit):
            argspec = inspect.getfullargspec(circuit)
            arg_dict = dict(zip(argspec.args, argspec.defaults))
            self.freq = arg_dict.get("freq", None)
        else:
            self.freq = self.netlist.freq
        if self.freq is not None:
            if isinstance(self.freq, (int, float)):
                self.freq = [
                    float(self.freq)
                ]  # TODO: do we need to put brackets around freq here?
            self.freq = np.array(self.freq)  # make freq array like for calculations

        # Create function for solving MNA on the fly
        def solve_matrix(freq, **kwargs):
            if len(kwargs) > 0:
                # Make copy so default values are preserved even if multiple
                # calls made to function with different component values.
                mna_ = self.copy()
                mna_.update(**kwargs)
            else:
                mna_ = self

            if len(mna_.undefined) > 0:
                # Raise error if components have undefined values. Create
                # readable string listing these components.
                missing_vars = mna_.undefined[0]
                if len(mna_.undefined) == 2:
                    missing_vars += " and " + mna_.undefined[-1]
                elif len(mna_.undefined) > 2:
                    missing_vars += (
                        ", "
                        + ", ".join(mna_.undefined[1:-1])
                        + " and "
                        + mna_.undefined[-1]
                    )
                raise TypeError(f"{self} missing argument values for {missing_vars}")

            return np.linalg.solve(
                mna_.A1[None, :, :] + 1j * mna_.A2[None, :, :] * freq[:, None, None],
                mna_.s[None, :],
            )

        self._solve_matrix = self._add_func_signature(solve_matrix)
        self.__call__ = self._solve_matrix

    def __call__(self, *args, **kwargs):
        return self._solve_matrix(*args, **kwargs)

    def _add_func_signature(self, func):
        """Adds informative signature to `solve_matrix` substituted to `__call__`"""

        func_args = [
            0,
            len(self.netlist) + 1,
            func.__code__.co_nlocals,
            func.__code__.co_stacksize,
            func.__code__.co_flags,
            func.__code__.co_code,
            (),
            (),
            tuple(["freq"] + self.components + ["kwargs", "kwargs"]),
            func.__code__.co_filename,
            func.__code__.co_name,
            func.__code__.co_firstlineno,
            func.__code__.co_lnotab,
        ]
        if sys.version_info >= (3, 8):
            func_args.insert(func.__code__.co_posonlyargcount, 1)
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
        freq = kwargs.pop("freq", self.freq)
        if freq is None:
            raise ValueError("freq has not been defined.")
        elif not isinstance(freq, np.ndarray):
            if isinstance(freq, (int, float)):
                freq = [
                    float(freq)
                ]  # TODO: do we need to put brackets around freq here?
            freq = np.array(freq)  # make freq array like for calculations
        V = self._solve_matrix(freq, **kwargs)
        if mode == "V":
            if not neg_node:
                return V[:, pos_node - 1]
            # If negative lead isn't connected to ground then calculate the
            # potential difference between the two nodes
            return V[:, pos_node - 1] - V[:, neg_node - 1]
        # If calculating current or impedances then we must find the impedances
        # between the circuit nodes. Currently this involves making a copy
        # of the object, updateing any component values, and calculating the
        # impedances using the A matrices and the measurement frequency.
        # TODO: Update the below calls as now we don't have a G matrix
        tmp_circuit = self.copy()
        tmp_circuit.update(**kwargs)
        G = (
            tmp_circuit.A1[None, :, :]
            + 1j * tmp_circuit.A2[None, :, :] * freq[:, None, None]
        )
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
        freq = kwargs.get("freq", self.freq)

        clb_kwargs = {}
        mpl_kwargs = kwargs.copy()
        for k, v in kwargs.items():
            if k in self.components + ["freq"]:
                clb_kwargs[k] = mpl_kwargs.pop(k)

        data = self.multimeter(pos_node, neg_node, mode, **clb_kwargs)
        Z, phase = np.abs(data), np.angle(data, deg=True)

        if ax is None:
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=figsize, dpi=dpi)

        # TODO: Choose best limits if multiple plots are drawn on the same fig.
        ax[1].set_xlim([np.min(freq), np.max(freq)])
        y_lim = [
            [np.min(Z) * 0.5, np.max(Z) * 2],
            [np.min([0, *phase]) * 1.2, np.max([0, *phase]) * 1.2 + 5],
        ]
        if all(phase < 0):  # reverse y-limits if all values negative
            y_lim[1] = [y_lim[1][1], y_lim[1][0]]
        ax[0].set_ylim(y_lim[0])
        ax[1].set_ylim(y_lim[1])
        ax[0].set_ylabel(y_label[mode], fontname="Roboto")
        ax[1].set_ylabel("Phase (°)", fontname="Roboto")
        ax[1].set_xlabel("Frequency (Hz)", fontname="Roboto")
        ax[0].loglog(freq, Z, **mpl_kwargs)
        ax[1].semilogx(freq, phase, **mpl_kwargs)
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
            if k in self.components:
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


class Transient(ModifiedNodalAnalysis):
    """Transient analysis of circuit."""

    def __init__(self, circuit, freq=None):
        super().__init__(circuit=circuit)
        if freq is not None:
            if isinstance(freq, (int, float)):
                freq = [float(freq)]
            self.freq = np.array(freq)

        def solve_matrix(freq, **kwargs):
            if len(kwargs) > 0:
                mna_ = self.copy()
                mna_.update(**kwargs)
            else:
                mna_ = self

            if len(mna_.undefined) > 0:
                # Create readable string of components without defined values
                missing_vars = mna_.undefined[0]
                if len(mna_.undefined) == 2:
                    missing_vars += " and " + mna_.undefined[-1]
                elif len(mna_.undefined) > 2:
                    missing_vars += (
                        ", "
                        + ", ".join(mna_.undefined[1:-1])
                        + " and "
                        + mna_.undefined[-1]
                    )
                raise TypeError(f"{self} missing argument values for {missing_vars}")

            return np.linalg.solve(
                mna_.A1[None, :, :] + mna_.A2[None, :, :] * freq[:, None, None],
                mna_.s[None, :],
            )

        self._solve_matrix = self._add_func_signature(solve_matrix)
        self.__call__ = self._solve_matrix

    def __call__(self, *args, **kwargs):
        return self._solve_matrix(*args, **kwargs)

    def _add_func_signature(self, func):
        """Adds informative signature to `solve_matrix` substituted to `__call__`"""

        func_args = [
            0,
            len(self.netlist) + 1,
            func.__code__.co_nlocals,
            func.__code__.co_stacksize,
            func.__code__.co_flags,
            func.__code__.co_code,
            (),
            (),
            tuple(["freq"] + self.components + ["kwargs", "kwargs"]),
            func.__code__.co_filename,
            func.__code__.co_name,
            func.__code__.co_firstlineno,
            func.__code__.co_lnotab,
        ]
        if sys.version_info >= (3, 8):
            func_args.insert(func.__code__.co_posonlyargcount, 1)
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


class NodalAnalysis:
    """Solves for the response of a defined circuit by modified nodal analysis.

    Modified nodal analysis defines a circuit by Kichhoff's circuit laws. The
    equation that must be solved is

    .. math::

        Ax = z


    Where :math:`A` describes the various impedances and currents flowing in
    and out of each node, :math:`x` is a vector of the voltages on each node of the
    circuit, and :math:`z` is a vector of the voltage and current source values.
    :math:`A` is an :math:`(n+m) \\times (n+m)` matrix, where :math:`n` is the
    number of circuit nodes and :math:`m` is the number of voltage sources.
    It is composed of four smaller matrices.

    .. math::

        A = \\begin{bmatrix}G & B\\\C & D\\end{bmatrix}


    :math:`G` is an :math:`n x n` matrix that is composed of the inverse impedances connecting
    each node. :math:`B` is an :math:`n x m` matrix of the connctions of the voltage sources.
    :math:`C = B.T` and :math:`D` is an :math:`m x m` matrix of zeros.

    As the impedance and source values are known, the node voltages given by :math:`x`
    can be solved for by taking the inverse of :math:`A`.

    .. math::

        x = A^{-1}\ z


    Attributes:
        circuit (`circuitlib.NodalAnalysis`): A defined circuit that has been
            parsed into a `NodalAnalysis` object.
    """

    def __init__(self, freq, circuit=None):
        """Takes in netlist and builds node matrices for MNA."""
        if isinstance(freq, (int, float)):
            freq = [float(freq)]
        self.freq = np.array(freq)  # TODO: Use property to prevent changes to freq?
        self.stamp_values = dict()
        self._initialised = False
        if circuit is not None:
            self.__call__(circuit)

    def __call__(self, *args, **kwargs):
        """Solves the constructed matrices for the modified nodal analysis.

        When first called the class will parse the circuit, which can be
        supplied as a decorated function, or by directly instantiating the
        class with a `circuitlib.netlist.Netlist` object. Subsequent calls
        then construct and solve the modified nodal analysis matrices."""

        if not self._initialised:
            if callable(args[0]):
                self.netlist = _parse_func(args[0])
            elif isinstance(args[0], Netlist):
                self.netlist = args[0]
            else:
                raise TypeError(
                    "The passed circuit must be a function or a `circuitlib.netlist.Netlist` object."
                )

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

    def _add_func_signature(self, func):
        """Adds informative signature to `solve_matrix` substituted to `__call__`"""

        func_args = [
            0,
            len(self.netlist),
            func.__code__.co_nlocals,
            func.__code__.co_stacksize,
            func.__code__.co_flags,
            func.__code__.co_code,
            (),
            (),
            tuple(self.components + ["kwargs", "kwargs"]),
            func.__code__.co_filename,
            func.__code__.co_name,
            func.__code__.co_firstlineno,
            func.__code__.co_lnotab,
        ]
        if sys.version_info >= (3, 8):
            func_args.insert(func.__code__.co_posonlyargcount, 1)
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

    def copy(self):
        """Deep copy of object. Needed for `__call__` if **kwargs supplied."""
        return copy.deepcopy(self)

    def update(self, **kwargs):
        """Updates component values."""
        for key, val in kwargs.items():
            if key not in self.netlist.components:
                raise KeyError(f"{key} not defined for the original circuit.")
            self.netlist.components[key]["value"] = val
            if val is not None:
                self.stamp_values[key] = self._component_impedance(key, val)

    def _stamp(self, G, idxs, val, subtract=False):
        """Stamps reciprocal of component's impedance on the G matrix for the
        modified nodal analysis."""
        if subtract:
            val = val * -1

        # Allow node indices to be compatible with the zero-indexed G array
        arr_idxs = [idx_ - 1 for idx_ in idxs[::-1] if idx_ > 0]
        G[:, arr_idxs[0], arr_idxs[0]] = G[:, arr_idxs[0], arr_idxs[0]] + val
        if len(arr_idxs) > 1:
            G[:, arr_idxs[1], arr_idxs[1]] = G[:, arr_idxs[1], arr_idxs[1]] + val
            G[:, arr_idxs[0], arr_idxs[1]] = G[:, arr_idxs[0], arr_idxs[1]] - val
            G[:, arr_idxs[1], arr_idxs[0]] = G[:, arr_idxs[1], arr_idxs[0]] - val

    def _component_impedance(self, component, value):
        """Calculates the reciprocal of a component's impedance."""
        if component[0].upper() == "R":
            return np.full_like(self.freq, 1 / value, dtype=complex)
        elif component[0].upper() == "C":
            return 1 / (1 / (2j * np.pi * self.freq * value))
        elif component[0].upper() == "L":
            return 1 / (2j * np.pi * self.freq * value)

    @property
    def A_matrix(self):
        """The A matrix for modified nodal analysis.
        Composed of the G, B, C, and D matrices."""
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
        """The G matrix for modified nodal analysis."""
        G = np.zeros(
            (len(self.freq), self.n_nodes - 1, self.n_nodes - 1), dtype=complex
        )
        for key, val in self.netlist.components.items():
            if not isinstance(val, dict):
                continue
            elif val.get("source", 0) != 0:
                continue
            if val["value"] is not None:
                Z = self._component_impedance(key, val["value"])
                self._stamp(G, val["nodes"], Z)
        return G

    @property
    def B_matrix(self):
        """The B matrix for modified nodal analysis."""
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
        """The C matrix for modified nodal analysis."""
        return np.moveaxis(self.B_matrix.T, -1, 0)

    @property
    def D_matrix(self):
        """The D matrix for modified nodal analysis."""
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
        """The nodes in the defined circuit."""
        nodes = []
        for key, val in self.netlist.components.items():
            if not isinstance(val, dict):
                continue
            nodes.extend(val.get("nodes", []))
        return set(nodes)

    @property
    def n_nodes(self):
        """The number of nodes in the defined circuit."""
        return len(self.nodes)

    @property
    def undefined(self):
        """The circuit components without a defined value."""
        return sorted(
            [
                key
                for key, val in self.netlist.components.items()
                if val.get("value", None) is None
            ]
        )

    @property
    def defined(self):
        """The circuit components with a defined value."""
        return sorted(
            [
                key
                for key, val in self.netlist.components.items()
                if val.get("value", None) is not None
            ]
        )

    @property
    def components(self):
        """The circuit components."""
        return self.defined + self.undefined
