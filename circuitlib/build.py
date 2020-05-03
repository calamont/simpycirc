import numpy as np

def _solve_circuit(A, z, netlist, stamp_matrix, *args, **kwargs):
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
            Z = _component_impedance(component, arg, netlist.freq)
            _stamp(A, netlist.components[component]["nodes"], Z)
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
            old_val = netlist.stamp_values[component]
            _stamp(A, netlist.components[component]["nodes"], old_val, subtract=True)
        else:
            undef_args.remove(component)
        Z = _component_impedance(component, val, netlist.freq)
        _stamp(A, netlist.components[component]["nodes"], Z)

    if len(undef_args) > 0:
        raise Exception(f"{undef_args[0]} undefined")
    
    if stamp_matrix:
        return A
    return np.linalg.solve(A, z)

def _stamp(A, idxs, val, subtract=False):
    if subtract:
        val = val * -1

    # Allow node indices to be compatible with the zero-indexed A array
    arr_idxs = [idx_ - 1 for idx_ in idxs[::-1] if idx_ > 0]
    A[:, arr_idxs[0], arr_idxs[0]] = A[:, arr_idxs[0], arr_idxs[0]] + val
    if len(arr_idxs) > 1:
        A[:, arr_idxs[1], arr_idxs[1]] = A[:, arr_idxs[1], arr_idxs[1]] + val
        A[:, arr_idxs[0], arr_idxs[1]] = A[:, arr_idxs[0], arr_idxs[1]] - val
        A[:, arr_idxs[1], arr_idxs[0]] = A[:, arr_idxs[1], arr_idxs[0]] - val

def _component_impedance(component, value, freq, dtype=complex):
    """Calculates the reciprocal of the impedance from transfer function of the component

    Args:
        component (str): Component name in the simpycirc nomenclature
        value (float): Value of component to be simulated in the standard SI
            unit (i.e. farads for capacitor, ohms for resistor)
        freq (float or array like): Frequency to calculate impedance over

    Returns:
        np.ndarray: impedance of component over frequency range
    """
    if component[0].upper() == "R":
        return np.full_like(freq, 1 / value, dtype=dtype)
    elif component[:3].upper() == "CPE":
        return 1 / CPE(value[0], value[1], freq)
    elif component[0].upper() == "C":
        return 1 / (1 / (2j * np.pi * freq * value))
    elif component[0].upper() == "L":
        return 1 / (2j * np.pi * freq * value)

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


# def C(cap, freq):
#     return 1 / (2j * np.pi * freq * cap)

