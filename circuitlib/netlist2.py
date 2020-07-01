import copy
import pprint


class Netlist:
    def __init__(self):
        self.__dict__["_n_sources"] = 0
        self.__dict__["components"] = dict()

    def __setattr__(self, key, value):
        nodes, value = value
        if key[0] in ["V", "I", "L"]:
            self.__dict__["_n_sources"] += 1
            source = self._n_sources
        else:
            source = 0

        self.__dict__["components"][key] = {
            "nodes": nodes,
            "value": value,
            "source": source,
        }

    def __repr__(self):
        return pprint.pformat(self.components, width=40)

    def __len__(self):
        return len(self.components)

    def items(self):
        return {key: val["value"] for key, val in self.components.items()}

    def copy(self):
        """Returns a deep copy of the netlist."""
        return copy.deepcopy(self)
