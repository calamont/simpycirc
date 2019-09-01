import inspect
import functools
import numpy as np

from .utils import flatten, create_nodelist, netlist_converter, Netlist

default_freq = 1000.0


def simulator(circuit=None, freq=1000.0, phase=True, **kwargs):
    """Generates function for simulating circuit.

    Args:
        circuit (, optional): Function or spc.Netlist to create simulating
        function from. Defaults to None.
        freq (float, optional): Frequency/frequencies to simulate circuit over.
        Defaults to 1000.0.
        complex (bool, optional): To simulate complex response of circuit. Defaults to True.

    Raises:
        SyntaxError: Raised if voltage source not specified for circuit.

    Returns:
        function that takes circuit component values as position and keyword
        arguments and simulates voltage response of the circuit nodes.
    """
    freq = np.array(freq).reshape(-1)
    dtype = np.complex if phase else np.float64
    wj = 2j * np.pi if phase else 2 * np.pi
    print(phase, dtype)
    if circuit is not None:

        if callable(circuit):
            args = inspect.getfullargspec(circuit)
            try:
                arg_dict = dict(zip(args.args, args.defaults))
            except TypeError:
                arg_dict = {}

            undef_args = sorted(list(circuit.__code__.co_names))
            def_args = args.args

            nodes = create_nodelist(circuit)
            if "V" not in flatten(nodes.values()):
                raise SyntaxError("V not defined for circuit")
            else:
                total_nodes = nodes.keys()
            netlist = netlist_converter(nodes, arg_dict)
        else:
            if isinstance(circuit, Netlist):
                circuit = circuit.to_dict()
            netlist = {}
            total_nodes = []
            for key, val in circuit.items():
                new_nodes = []
                for node in val["nodes"]:
                    if node > 0:
                        new_nodes.append(node - 1)
                total_nodes.extend(new_nodes)
                netlist[key] = {"nodes": new_nodes, "value": circuit[key]["value"]}
            undef_args = sorted(
                [key for key, val in circuit.items() if not val["value"]]
            )
            if kwargs:
                for key, val in kwargs.items():
                    if isinstance(val, tuple):
                        netlist[key] = {
                            "nodes": [val[0][0] - 1, val[0][1] - 1],
                            "value": val[1],
                        }
                    else:
                        netlist[key]["value"] = val
                def_args = list(set(netlist.keys()).union(set(kwargs.keys())))
            else:
                def_args = []
            try:
                netlist.pop("V")
            except ValueError:
                raise SyntaxError("V not defined for circuit")

        length = len(set(total_nodes)) + 1
        A = np.zeros((freq.shape[0], length, length), dtype=dtype)
        comp_vals = {}
        for component, val in netlist.items():
            if val["value"]:
                Z = impedance(component, val["value"], freq, dtype)
                comp_vals[component] = Z
                stamp(A, val["nodes"], Z)
        total_args = undef_args + def_args

        @functools.wraps(circuit)
        def wrapper(*args, **kwargs):
            A_new = A.copy()
            undefined_args = undef_args.copy()
            defined_args = def_args
            passed_args = []

            for arg in args:
                component = undefined_args.pop(0)
                Z = impedance(component, arg, freq, dtype)
                stamp(A_new, netlist[component]["nodes"], Z)
                passed_args.append(component)
            for component, val in kwargs.items():
                if component in passed_args:
                    raise Exception(f"{component} defined twice")
                else:
                    passed_args.append(component)

                if component not in undefined_args:
                    old_val = comp_vals[component]
                    unstamp(A_new, netlist[component]["nodes"], old_val)
                else:
                    undefined_args.remove(component)

                Z = impedance(component, val, freq, dtype)
                stamp(A_new, netlist[component]["nodes"], Z)

            try:
                undefined_args.remove("V")
            except ValueError:
                pass
            if len(undefined_args) > 0:
                raise Exception(f"{undefined_args[0]} undefined")

            A_new[:, 0, -1] = 1
            A_new[:, -1, 0] = 1
            b = np.zeros(length)[None, :]
            b[:, -1] = 1
            # return np.dot(np.linalg.inv(A_new), b.T)
            return np.linalg.solve(A_new, b)

        wrapper.__doc__ = f"Automatically generated docstring. This function can now be used to simulate the voltages across the nodes between the frequencies of {np.min(freq):.3f}-{np.max(freq):.3f}Hz.\n\nValue(s) for the remaning keyword arguments {undef_args} need to be supplied to complete the circuit.\n"

        return wrapper

    else:
        return functools.partial(simulator, freq=freq, phase=phase, **kwargs)


def CPE(Rs, Cs, freq=None):
    """
    Simulates CPE behaviour through a series of R||C components.

    Args:
        freq (array-like):
            The frequencies over which the circuit is simulated.
    Returns:
        Complex impedances or the absolute impedances and phase for
        the CPE.
    """
    if freq is None:
        freq = default_freq

    r_vals = np.array(Rs, dtype=np.complex)
    c_vals = np.array(Cs)
    cpe = np.zeros_like(freq, dtype=np.complex)

    for r, c in zip(r_vals, c_vals):
        cpe = cpe + parallel_Z(r, C(c, freq))

    return cpe


def C(capacitance, freq):
    """Returns complex impedance of capacitor"""
    if freq is None:
        freq = default_freq
    return 1 / (2j * np.pi * freq * capacitance)


def parallel_Z(Z1, Z2):
    """Combined impedance of parallel impedances"""
    return Z1 * Z2 / (Z1 + Z2)


def impedance(component, value, freq, dtype):
    """Calculates impedance from transfer function of component

    Args:
        component (str): Component name in the simpycirc nomenclature
        value (float): Value of component to be simulated in the standard SI
        unit (i.e. farads for capacitor, ohms for resistor)

    Returns:
        np.ndarray: impedance of component over frequency range
    """
    if component[0].upper() == "R":
        return np.full_like(freq, 1 / value, dtype=dtype)
    elif component[:3].upper() == "CPE":
        return 1 / CPE(value[0], value[1], freq)
    elif component[0].upper() == "C":
        return 1 / (1 / (wj * freq * value))


def stamp(A, idxs, val, subtract=False):
    """Stamp used to update the MNA array of the circuit.

    Args:
        A (np.ndarray): The MNA representation of the circuit.
        idxs (list): The row/column indices representing the components being
        added to the MNA matrix
        val (float): The value of the component to be simulated in the standard SI
        unit (i.e. farads for capacitor, ohms for resistor)
        subtract (bool, optional): To be used to remove a component from the MNA matrix. Defaults to False.
    """

    if subtract:
        val *= -1

    A[:, idxs[0], idxs[0]] = A[:, idxs[0], idxs[0]] + val
    if len(idxs) > 1:
        A[:, idxs[1], idxs[1]] = A[:, idxs[1], idxs[1]] + val
        A[:, idxs[0], idxs[1]] = A[:, idxs[0], idxs[1]] - val
        A[:, idxs[1], idxs[0]] = A[:, idxs[1], idxs[0]] - val


# def unstamp(A, idxs, val):
#     A[:, idxs[0], idxs[0]] = A[:, idxs[0], idxs[0]] - val
#     if len(idxs) > 1:
#         A[:, idxs[1], idxs[1]] = A[:, idxs[1], idxs[1]] - val
#         A[:, idxs[0], idxs[1]] = A[:, idxs[0], idxs[1]] + val
#         A[:, idxs[1], idxs[0]] = A[:, idxs[1], idxs[0]] + val
