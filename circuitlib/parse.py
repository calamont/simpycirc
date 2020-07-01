"""Methods to parse and convert code object of a function into a netlist."""

import ast
import inspect
from collections import defaultdict
from .netlist import Netlist


def _parse_func(circuit):
    """Extract netlist from circuit function"""

    nodes = _create_nodelist(circuit)
    args = inspect.getfullargspec(circuit)
    try:
        arg_dict = dict(zip(args.args, args.defaults))
    except TypeError:
        arg_dict = {}
    if "V" not in _flatten(nodes.values()):
        # TODO: Perhaps raise warning instead
        pass
        # raise SyntaxError("V not defined for circuit")
    return _netlist_converter(nodes, arg_dict)


def _flatten(nested_list):
    """Flattens arbitrarily nested list of lists"""
    tmp_list = []
    for i in nested_list:
        if isinstance(i, (int, str)):
            tmp_list.append(i)
        elif isinstance(i, (list, tuple)):
            tmp_list.extend(_flatten(i))
    return tmp_list


def _netlist_converter(node_list, argspec):
    """Converts a node list into a netlist"""
    d = defaultdict(dict)
    print(node_list)
    for node, keys in node_list.items():
        for key in keys:
            # if "V" in key.upper():
            #     continue)
            d[key].setdefault("nodes", []).append(node)
            try:
                d[key]["value"] = argspec[key]
            except KeyError:
                d[key]["value"] = None

    netlist = Netlist()
    print(d)
    for k, v in d.items():
        if len(v["nodes"]) == 1:
            print(k)
            v["nodes"].append(0)  # Connect lose connections to ground
        getattr(netlist, k[0])(v["nodes"][0], v["nodes"][1], v["value"], k)
    return netlist


def _create_nodelist(circuit):
    """Identifies nodes in the circuit defined as a function"""
    if callable(circuit):
        func_string = inspect.getsource(circuit)
    tree = ast.parse(func_string)

    nodelist = {}
    nodelist = defaultdict(list)
    subcircuits = {}

    def recurse_tree(node):
        nodes = ast.iter_child_nodes(node)

        for i, node in enumerate(nodes):
            node_type = type(node).__name__
            if node_type == "Assign":
                subcircuits[node.targets[0].id] = node.value
            if node_type == "BinOp":
                if type(node.op).__name__ == "Add":

                    nodelist[len(nodelist) + 1].extend(
                        [
                            _find_component(node.left, "left", subcircuits),
                            _find_component(node.right, "right", subcircuits),
                        ]
                    )
            recurse_tree(node)

    recurse_tree(tree)

    keys, vals = zip(*nodelist.items())
    nodelist = {key: _flatten(val) for key, val in zip(keys[::-1], vals)}
    return nodelist


def _find_component(node, side, subcircuit):
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
                return _find_component(node.right, side, subcircuit)
            elif side == "right":
                return _find_component(node.left, side, subcircuit)

        elif (type(node.op).__name__ == "BitOr") or (type(node.op).__name__ == "Mult"):
            return [
                _find_component(node.left, side, subcircuit),
                _find_component(node.right, side, subcircuit),
            ]
