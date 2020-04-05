import ast
import inspect
import numpy as np
import pandas as pd

from functools import lru_cache
from collections import defaultdict, abc


def flatten(nested_list):
    """Flattens arbitrarily nested list of lists"""
    tmp_list = []
    for i in nested_list:
        if isinstance(i, (int, str)):
            tmp_list.append(i)
        elif isinstance(i, (list, tuple)):
            tmp_list.extend(flatten(i))
    return tmp_list

def stamp(A, idxs, val, subtract=False):

    if subtract:
        val = val * -1

    # Allow node indices to be compatible with the zero-indexed A array
    arr_idxs = [idx_ - 1 for idx_ in idxs[::-1] if idx_ > 0]
    A[:, arr_idxs[0], arr_idxs[0]] = A[:, arr_idxs[0], arr_idxs[0]] + val
    if len(arr_idxs) > 1:
        A[:, arr_idxs[1], arr_idxs[1]] = A[:, arr_idxs[1], arr_idxs[1]] + val
        A[:, arr_idxs[0], arr_idxs[1]] = A[:, arr_idxs[0], arr_idxs[1]] - val
        A[:, arr_idxs[1], arr_idxs[0]] = A[:, arr_idxs[1], arr_idxs[0]] - val

def component_impedance(component, value, freq, dtype=complex):
    if component[0].upper() == "R":
        return np.full_like(freq, 1 / value, dtype=dtype)
    elif component[:3].upper() == "CPE":
        return 1 / CPE(value[0], value[1], freq)
    elif component[0].upper() == "C":
        return 1 / (1 / (2j * np.pi * freq * value))
    elif component[0].upper() == "L":
        return 1 / (2j * np.pi * freq * value)
        



class Netlist(pd.DataFrame):
    """A tabular dataframe for defining a circuit's netlist"""

    def __init__(self, *args, **kwargs):
        super().__init__(data={}, index=["nodes", "value"])

    def __setattr__(self, key, value):
        if isinstance(value[0], list):
            self[key] = value
        else:
            raise ValueError("Expected [nodes], value")

    def B(self):
        return 'B'

