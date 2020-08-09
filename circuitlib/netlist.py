import copy
import pprint
from . import signal_generator

kwarg_defaults = {
    "period": 0,
    "x_offset": 0,
    "y_offset": 0,
    "mod": 0,
}


class Netlist:
    def __init__(self):
        self.components = dict()
        self.component_count = dict()

    def R(self, node1, node2, value, name=None):
        """Linear resistor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals
            of the component. value (float): Value of the component in SI units.
        """
        self.component_count["R"] = self.component_count.get("R", 0) + 1
        if name is None:
            name = f"R{self.component_count['R']}"
        self.components[name] = {
            "nodes": (node1, node2),
            "dependent_nodes": None,
            "value": value,
            "group2_idx": 0,
            "type": "R",
        }

    def C(self, node1, node2, value, name=None):
        """Linear resistor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals
            of the component. value (float): Value of the component in SI units.
        """
        self.component_count["C"] = self.component_count.get("C", 0) + 1
        if name is None:
            name = f"C{self.component_count['C']}"
        self.components[name] = {
            "nodes": (node1, node2),
            "dependent_nodes": None,
            "value": value,
            "group2_idx": 0,
            "type": "C",
        }

    def L(self, node1, node2, value, name=None):
        """Linear resistor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals of the component.
            value (float): Value of the component in SI units.
        """
        self.component_count["L"] = self.component_count.get("L", 0) + 1
        if name is None:
            name = f"L{self.component_count['L']}"
        self.components[name] = {
            "nodes": (node1, node2),
            "dependent_nodes": None,
            "value": value,
            "group2_idx": len(self.group2_components) + 1,
            "type": "L",
        }

    def V(self, node1, node2, value, name=None, signal=None, **signal_kwargs):
        """Independent voltage source.

        Args:
            nodes (list): Nodes connected the positive and negative terminals of the component.
            value (float): Value of the component in SI units.
        """
        self.component_count["V"] = self.component_count.get("V", 0) + 1
        if name is None:
            name = f"V{self.component_count['V']}"
        self.components[name] = {
            "nodes": (node1, node2),
            "dependent_nodes": None,
            "value": value,
            "group2_idx": len(self.group2_components) + 1,
            "type": "V",
            "signal": signal_generator.partial(
                getattr(signal_generator, signal), value=value, **signal_kwargs
            ),  # can we get the function name to then find the similarly named C function in the transient call?
            "set_kwargs": {**kwarg_defaults, **signal_kwargs},
        }

    def VCVS(self, nodes, value):
        """Linear resistor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals of the component.
            value (float): Value of the component in SI units.
        """
        pass

    def CCVS(self, nodes, value):
        """Linear resistor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals of the component.
            value (float): Value of the component in SI units.
        """
        pass

    def VCCS(self, nodes, value):
        """Linear resistor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals of the component.
            value (float): Value of the component in SI units.
        """
        pass

    def CCCS(self, nodes, value):
        """Linear resistor.

        Args:
            nodes (list): Nodes connected the positive and negative terminals of the component.
            value (float): Value of the component in SI units.
        """
        pass

    def __repr__(self):
        return pprint.pformat(self.components, width=40)

    def __len__(self):
        return len(self.components)

    def items(self):
        return {key: val["value"] for key, val in self.components.items()}

    def copy(self):
        """Returns a deep copy of the netlist."""
        return copy.deepcopy(self)

    @property
    def group1_components(self):
        return [key for key, val in self.components.items() if val["group2_idx"] == 0]

    @property
    def group2_components(self):
        return [key for key, val in self.components.items() if val["group2_idx"] > 0]
