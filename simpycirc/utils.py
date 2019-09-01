import ast
import inspect
import numpy as np
import pandas as pd

from functools import lru_cache
from collections import defaultdict


def create_nodelist(circuit):

    func_string = inspect.getsource(circuit)
    tree = ast.parse(func_string)

    netlist = {}
    subcircuits = {}

    #     @lru_cache()
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


def flatten(nested_list):
    tmp_list = []
    for i in nested_list:
        if isinstance(i, (int, str)):
            tmp_list.append(i)
        elif isinstance(i, (list, tuple)):
            tmp_list.extend(flatten(i))
    return tmp_list


def netlist_converter(node_list, argspec):
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


class Netlist(pd.DataFrame):
    """
    A tabular dataframe for defining a circuit's netlist
    """

    def __init__(self, *args, **kwargs):
        super().__init__(data={"V": [[0, 1], 1]}, index=["nodes", "value"])

    def __setattr__(self, key, value):
        if isinstance(value[0], list):
            self[key] = value
        else:
            raise ValueError("Expected [nodes], value")

