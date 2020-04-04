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

# TODO: Create class that inherits from MutableMapping instead of pd.DataFrame
# TODO: Be able to set component unit as letter (e.g. 200 pF). Probably regex.
class Netlist(abc.MutableMapping):

    def __init__(self):
        pass

    def _create_entry(self, nodes, value):
        """Returns dict with keys 'nodes' and 'value'"""
        self._check_input(nodes, value)
        # TODO: Include "source" as a key to the dict here
        return dict(*zip(["nodes", "value"], [nodes, value]))


    def __setattr__(self, key, value):
        self.__dict__.update()

    def _check_input(self, input):
        # TODO: Add type checking for inputs
        pass

class Netlist(pd.DataFrame):
    """A tabular dataframe for defining a circuit's netlist"""

    def __init__(self, *args, **kwargs):
        super().__init__(data={}, index=["nodes", "value"])

    def __setattr__(self, key, value):
        if isinstance(value[0], list):
            self[key] = value
        else:
            raise ValueError("Expected [nodes], value")

