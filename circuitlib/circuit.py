import re
import ast
import inspect
import functools
import numpy as np

from .utils import flatten, stamp, component_impedance
from collections import defaultdict
from itertools import chain


default_freq = 1000.0
dtype = np.complex
wj = 2j * np.pi


def simulator(circuit=None, freq=1000.0, **kwargs):
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
        if callable(circuit) or (isinstance(circuit, str)):
            netlist, total_nodes, undef_args, def_args = _parse_circuit_func(circuit)
        else:
            netlist = circuit
            
        A = netlist.A_matrix()
        z = np.zeros(netlist.n_nodes -1 + netlist.sources)[None, :]
        for k, v in netlist.components.items():
            if v["source"]:
                z[:, -v["source"]] = v["value"]

        @functools.wraps(circuit)
        def wrapper(*args, **kwargs):

            return _solve_circuit(
                A.copy(),
                z,
                netlist,
                *args,
                **kwargs,
            )

        # wrapper.__doc__ = (
        #     "Automatically generated docstring. This function can now be used "
        #     + "to simulate the voltages across the nodes between the frequencies "
        #     + f"of {np.min(freq):.3f}-{np.max(freq):.3f}Hz.\n\nValue(s) for the "
        #     + f"remaning keyword arguments {undef_args} need to be supplied to "
        #     + "complete the circuit.\n"
        # )

        return wrapper

    else:
        return functools.partial(simulator, freq=freq, **kwargs)


def _parse_circuit_func(circuit):
    """Extract netlist from circuit function"""

    nodes = create_nodelist(circuit)

    try:
        args = inspect.getfullargspec(circuit)
        arg_dict = dict(zip(args.args, args.defaults))
        undef_args = sorted(list(circuit.__code__.co_names))
        def_args = args.args
    except TypeError:  # if circuit supplied as a string
        arg_dict = {}
        # Get unique components defined in `nodes`
        undef_args = np.unique(list(chain(*nodes.values()))).tolist()
        def_args = []

    if "V" not in flatten(nodes.values()):
        # TODO: Perhaps raise warning instead
        raise SyntaxError("V not defined for circuit")
    else:
        total_nodes = nodes.keys()
    netlist = _netlist_converter(nodes, arg_dict)

    return netlist, total_nodes, undef_args, def_args


def _parse_circuit_dict(circuit, **kwargs):
    """Extract netlist from circuit dictionary/netlist"""
    pass

def _netlist_converter(node_list, argspec):
    """Converts a node list into a netlist"""
    d = defaultdict(dict)
    for node, keys in node_list.items():
        for key in keys:
            if "V" in key.upper():
                continue
            d[key].setdefault("nodes", []).append(node)
            try:
                d[key]["value"] = argspec[key]
            except KeyError:
                d[key]["value"] = None
    return d


def _solve_circuit(A, z, netlist, *args, **kwargs):
    """Solves circuit using modified nodal analysis"""
    undef_args = netlist.undefined.copy()
    def_args = netlist.defined.copy()
    passed_args = []
    
    # Step through arguments and stamp values to the `A` matrix
    for arg in args:
        component = undef_args.pop(0)
        if component[0] == "V":
            idx = netlist.components[component]["source"]
            z[:, -idx] = arg
        else:
            Z = component_impedance(component, arg, netlist.freq)
            stamp(A, netlist.components[component]["nodes"], Z)
        passed_args.append(component)

    # Step through kwargs and stamp values to the `A` matrix. 
    for component, val in kwargs.items():
        if component in passed_args:
            raise Exception(f"{component} defined twice")
        passed_args.append(component)
        
        # Voltage sources are independent of each other so we don't need to
        # subtract previously stamped values, we can simply overwrite them.
        if component[0] == "V":
            idx = netlist.components[component]["source"]
            z[:, -idx] = val
            if component in undef_args:
                undef_args.remove()
            continue

        # If the component has been previously defined then subtract previously
        # stamped value.
        if component not in undef_args:
            old_val = netlist["stamp_values"][component]
            stamp(A, netlist.components[component]["nodes"], old_val, subtract=True)
        else:
            undef_args.remove(component)
        Z = component_impedance(component, val, netlist.freq)
        stamp(A, netlist.components[component]["nodes"], Z)

    if len(undef_args) > 0:
        raise Exception(f"{undef_args[0]} undefined")

    return np.linalg.solve(A, z)


def create_nodelist(circuit):
    """Identifies nodes in the circuit defined as a function"""
    if callable(circuit):
        func_string = inspect.getsource(circuit)
    elif isinstance(circuit, str):
        func_string = re.sub(r"\n\s*", "\n", circuit)
    tree = ast.parse(func_string)

    netlist = {}
    subcircuits = {}

    def recurse_tree(node):
        nodes = ast.iter_child_nodes(node)

        for i, node in enumerate(nodes):
            node_type = type(node).__name__
            if node_type == "Assign":
                subcircuits[node.targets[0].id] = node.value
            if node_type == "BinOp":
                if type(node.op).__name__ == "Add":

                    netlist[len(netlist) + 1] = [
                        find_component(node.left, "left", subcircuits),
                        find_component(node.right, "right", subcircuits),
                    ]
            recurse_tree(node)

    recurse_tree(tree)

    keys, vals = zip(*netlist.items())
    netlist = {key - 1: flatten(val) for key, val in zip(keys[::-1], vals)}

    return netlist


def find_component(node, side, subcircuit):
    try:
        node = subcircuit[node.id]
    except AttributeError:
        pass
    except KeyError:
        return node.id  # node is Name type and not a subcomponent

    node_type = type(node).__name__

    if node_type == "BinOp":
        if type(node.op).__name__ == "Add":
            if side == "left":
                return find_component(node.right, side, subcircuit)
            elif side == "right":
                return find_component(node.left, side, subcircuit)

        elif (type(node.op).__name__ == "BitOr") or (type(node.op).__name__ == "Mult"):
            return [
                find_component(node.left, side, subcircuit),
                find_component(node.right, side, subcircuit),
            ]


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
        cpe = cpe + _parallel_Z(r, C(c, freq))

    return cpe


def _parallel_Z(Z1, Z2):
    """Combined impedance of parallel impedances"""
    return Z1 * Z2 / (Z1 + Z2)


def C(cap, freq):
    return 1 / (2j * np.pi * freq * cap)


def impedance(component, value, freq):
    """Calculates the reciprocal of the impedance from transfer function of the component

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
    elif component[0].upper() == "L":
        return 1 / (wj * freq * value)


def _stamp(G, idxs, val, subtract=False, voltage_source=False):
    """Stamp used to update the MNA array of the circuit.

    Modified nodal analysis defines a circuit by Kichhoff's circuit laws. The
    equation that must be solved is
    
    Ax = z

    Where A describes the various impedances and currents flowing in 
    and out of each node, x is a vector of the voltages on each node of the 
    circuit, and z is a vector of the voltage and current source values. As the
    impedance and source values are known, x can be solved for by taking the 
    inverse of A

    x = A^{-1}\ z

    A is an (n+m) x (n+m) matrix, where n is the number of circuit nodes and m 
    is the number of voltage sources. It is composed of four smaller matrices

    G B
    C D

    G is an n x n matrix that is composed of the inverse impedances connecting
    each node. B is an n x m matrix of the connctions of the voltage sources. 
    C = B.T and D is an m x m matrix of zeros.


    Args:
        G (np.ndarray): The MNA representation of the circuit.
        idxs (list): The row/column indices representing the components being
        added to the MNA matrix
        val (float): The value of the component to be simulated in the standard SI
        unit (i.e. farads for capacitor, ohms for resistor)
        subtract (bool, optional): To be used to remove a component from the MNA matrix. Defaults to False.
    """

    if subtract:
        val = val * -1

    # Allow node indices to be compatible with the zero-indexed G array
    arr_idxs = [idx_ - 1 for idx_ in idxs if idx_ > 0]

    G[:, arr_idxs[0], arr_idxs[0]] = G[:, arr_idxs[0], arr_idxs[0]] + val
    if len(arr_idxs) > 1:
        G[:, arr_idxs[1], arr_idxs[1]] = G[:, arr_idxs[1], arr_idxs[1]] + val
        G[:, arr_idxs[0], arr_idxs[1]] = G[:, arr_idxs[0], arr_idxs[1]] - val
        G[:, arr_idxs[1], arr_idxs[0]] = G[:, arr_idxs[1], arr_idxs[0]] - val

