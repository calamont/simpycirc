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

from .differential import DAE_solve

y_label = {"V": "Voltage", "I": "Current", "Z": "Impedance"}


class ModifiedNodalAnalysis:
    """Constructs MNA matrices to solve."""

    def __init__(self, circuit=None, transient=False):
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

            self.A1, self.A2, self.s = self._stamp_matrices(self.netlist, transient)

    def __call__(self, *args, **kwargs):
        pass

    def _stamp_matrices(self, netlist, transient):
        """Samp values into two square matrices for group 1 and group 2 components."""
        mat_size = len(netlist.group2_components) + self.n_nodes - 1
        # Arrays must be fortran contiguous for compatibility with LAPACK solvers
        A1 = np.zeros((mat_size, mat_size), order="F")
        A2 = np.zeros((mat_size, mat_size), order="F")
        s = np.zeros(mat_size)

        stamp_funcs = Stamps(
            self.transient
        )  # TODO: does this need to be a class or could be dict?
        for key, val in netlist.components.items():
            if val["value"] is not None:
                stamp = getattr(stamp_funcs, val["type"])
                A1, A2, s = stamp(A1, A2, s, **val)
        return A1, A2, s

    def update(self, **kwargs):
        """Updates default component values."""
        # stamp_funcs = Stamps(self.transient)
        # Check if kwargs match existing component values.
        for key, val in kwargs.items():
            if key not in self.netlist.components:
                raise KeyError(f"{key} not defined for the original circuit.")

            self.netlist.components[key]["value"] = val
        self.A1, self.A2, self.s = self._stamp_matrices(self.netlist, transient=True)
        # TODO: need to unstamp value first! Currently we will just rebuild
        # the entire A arrays as this should be relatively quick compared
        # to the actual simulation step.
        # comp = self.netlist.components[key]
        # comp["value"] = val
        # stamp = getattr(stamp_funcs, comp["type"])
        # self.A1, self.A2, self.s = stamp(self.A1, self.A2, self.s, **{**comp})

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
        self.transient = False
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

    def __init__(self, circuit):
        self.transient = True
        super().__init__(circuit=circuit)
        if callable(circuit):
            argspec = inspect.getfullargspec(circuit)
            arg_dict = dict(zip(argspec.args, argspec.defaults))
            self.time = arg_dict.get("time", None)
        else:
            self.time = self.netlist.time
        if self.time is not None:
            if isinstance(self.time, (int, float)):
                self.time = [float(self.time)]
            self.time = np.array(self.time)  # make time array like for calculations

        # TODO: Add DAE solver
        # Create function for solving differentiable algebraic equations
        # on the fly
        def solve_DAE(time, step=0.001, initialise="zeros", **kwargs):
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

            # Solve for intitial conditions under DC
            # Should solve for each voltage/current source at t=0
            # Need to solve how to hadle the signal generators for each source
            # TODO: could this use a DC class to solve for this?
            init = self._initial_conditions(mna_, time[0], initialise)
            # step = 0.001
            # def transient():
            #     print(mna_.A1)
            #     print(mna_.A2)
            #     print(init)
            #     print(time[-1])
            #     print(step)
            #     print(mna_.netlist.components)
            # transient = DAE_solve(
            #     mna_.A1, mna_.A2, init, time[0], time[-1], step, mna_.netlist.components
            # )
            if isinstance(time, np.ndarray):
                # time = np.ascontiguousarray(time)  # ensure array is C-contiguous
                pass
            elif len(time) == 2:
                time = np.linspace(time[0], time[1], 1000)
            transient = DAE_solve(time, mna_.A1, mna_.A2, init, mna_.netlist.components)
            return transient

        self._solve_DAE = self._add_func_signature(solve_DAE)
        self.__call__ = self._solve_DAE

    def _initial_conditions(self, mna, time, initialise):
        """Determines initial conditions before transient analysis."""
        if initialise == "auto":
            initialise = {}
        elif initialise == "zeros":
            initialise = dict.fromkeys(mna.netlist.components, 0)
        if not isinstance(initialise, dict):
            raise TypeError("initialise parameter not a dictionary")
        s = self._source_array(mna, time, initialise)
        # Solve AC circuit when freq=0 as this is equivalent to DC
        return np.linalg.solve(mna.A1, s)

    def _source_array(self, mna, time, initialise):
        # Iterates through each voltage source and solves for signal in self.s
        # and returns this column vector.
        s = np.zeros_like(mna.s)
        for key, val in mna.netlist.components.items():
            # TODO: Need to handle what happens with current source
            if "signal" in val:
                # Try to get initial value for signal, defaulting to the
                # signal at that time step if not defined.
                s[-1 * val["group2_idx"]] = initialise.get(key, val["signal"](time))
        return s

    def __call__(self, *args, **kwargs):
        return self._solve_DAE(*args, **kwargs)

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
            tuple(["time"] + self.components + ["kwargs", "kwargs"]),
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
                time (float, optional): Frequency/frequencies to simulate circuit over.
                Defaults to 1000.0.

            Raises:
                SyntaxError: Raised if voltage source not specified for circuit.

            Returns:
                function that takes circuit component values as position and keyword
                arguments and simulates voltage response of the circuit nodes.
        """
        return func

    def multimeter(self, nodes, mode="V", **kwargs):
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
        time = kwargs.pop("time", self.time)
        if time is None:
            raise ValueError("time has not been defined.")
        V = self._solve_DAE(time, **kwargs)
        if mode == "V":
            if isinstance(nodes, (int, float)):
                return V[:, nodes - 1]
            # If negative lead isn't connected to ground then calculate the
            # potential difference between the two nodes
            return V[:, nodes[0] - 1] - V[:, nodes[1] - 1]
        # TODO: NOT SURE IF ANY OF THE BELOW WORKS FOR TRANSIENT ANALYSIS...
        # If calculating current or impedances then we must find the impedances
        # between the circuit nodes. Currently this involves making a copy
        # of the object, updateing any component values, and calculating the
        # impedances using the A matrices and the measurement timeuency.
        # TODO: Update the below calls as now we don't have a G matrix
        # tmp_circuit = self.copy()
        # tmp_circuit.update(**kwargs)
        # G = (
        #     tmp_circuit.A1[None, :, :]
        #     + 1j * tmp_circuit.A2[None, :, :] * time[:, None, None]
        # )
        # if not neg_node:
        #     Z = 1 / np.sum(G[:, pos_node - 1, :], axis=-1)
        #     if mode == "Z":
        #         return Z
        #     elif mode == "I":
        #         return V[:, pos_node - 1] / Z
        # Z = -1 / G[:, pos_node - 1, neg_node - 1]
        # if mode == "Z":
        #     return Z
        # elif mode == "I":
        #     return (V[:, pos_node - 1] - V[:, neg_node - 1]) / Z

    def oscilloscope(
        self, nodes, mode="V", figsize=(6, 5), dpi=100, ax=None, **kwargs,
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
        time = kwargs.get("time", self.time)

        clb_kwargs = {}
        mpl_kwargs = kwargs.copy()
        for k, v in kwargs.items():
            if k in self.components + ["time", "initialise"]:
                clb_kwargs[k] = mpl_kwargs.pop(k)

        data = self.multimeter(nodes, mode, **clb_kwargs)

        if ax is None:
            fig, ax = plt.subplots(sharex=True, figsize=figsize, dpi=dpi)

        ax.set_xlim([np.min(time), np.max(time)])
        peak_to_peak = np.abs(np.max(data) - np.min(data))
        ax.set_ylim(
            [np.min(data) - peak_to_peak * 0.25, np.max(data) + peak_to_peak * 0.25]
        )
        ax.set_xlabel("Time (s)", fontname="Roboto")
        ax.set_ylabel(y_label[mode], fontname="Roboto")
        ax.plot(time, data, **mpl_kwargs)
        return ax
