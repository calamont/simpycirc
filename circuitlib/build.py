import numpy as np


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
    elif component[0].upper() == "C":
        return 1 / (1 / (2j * np.pi * freq * value))
    elif component[0].upper() == "L":
        return 1 / (2j * np.pi * freq * value)
