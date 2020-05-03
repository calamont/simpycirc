import copy
import numpy as np
from .build import _stamp, _component_impedance


class Netlist:
    def __init__(self):
        self.__dict__["_n_sources"] = 0
        self.__dict__["components"] = dict()

    def __setattr__(self, key, value):
        nodes, value = value
        if key[0] == "V":
            self.__dict__["_n_sources"] += 1
            source = self._n_sources
        else:
            source = 0

        self.__dict__["components"][key] = {
            "nodes": nodes,
            "value": value,
            "source": source,
        }

    def items(self):
        return {key: val["value"] for key, val in self.components.items()}

    def copy(self):
        """Returns a deep copy of the netlist."""
        return copy.deepcopy(self)
