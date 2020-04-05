import numpy as np
from .utils import stamp, component_impedance

class Netlist:

    def __init__(self, freq=1000.0):
        if isinstance(freq, (int, float)):
            freq = [float(freq)]
        # Add attributes with __dict__ to prevent __setattr__ call
        self.__dict__["freq"] = np.array(freq)
        self.__dict__["sources"] = 0
        self.__dict__["components"] = dict()
        self.__dict__["stamp_values"] = dict()
    
    def __setattr__(self, key, entry):
        d = {key: self._create_entry(key, *entry)}
        self.__dict__["components"].update(d)

    def _create_entry(self, key, nodes, value):
        """Returns dict with keys 'nodes' and 'value'"""
        source = self._parse_input(key, nodes, value)
        return {"nodes": nodes,  "value": value, "source": source}

    def _parse_input(self, key, nodes, value):
        # TODO: Add type checking for inputs
        # TODO: Handle input for current sources e.g. I_input
        if key[0] == "V":
            self.__dict__["sources"] += 1
            # self.__dict__["stamp_values"][key] = value 
            return self.sources
        if value is not None:
            self.__dict__["stamp_values"][key] = component_impedance(key,
                                                                 value,
                                                                 self.freq)
        return 0
    
    @property
    def A_matrix(self):
        GB = np.concatenate([self.G_matrix, self.B_matrix], axis=2)
        CD = np.concatenate([self.C_matrix, self.D_matrix], axis=2)
        return np.concatenate([GB, CD], axis=1)

    @property
    def G_matrix(self):
        G = np.zeros((len(self.__dict__["freq"]), self.n_nodes-1, self.n_nodes-1), dtype=complex)
        for key, val in self.__dict__["components"].items():
            if not isinstance(val, dict):
                continue
            elif val.get("source", 0) != 0:
                continue
            if val["value"] is not None:
                Z = component_impedance(key, val["value"], self.freq)
                stamp(G, val["nodes"], Z)
        return G

    @property
    def B_matrix(self):
        B = np.zeros((len(self.__dict__["freq"]), self.n_nodes-1, self.sources))
        for key, val in self.__dict__["components"].items():
            if not isinstance(val, dict):
                continue
            elif val.get("source", 0) == 0:
                continue
            idx = reversed([i-1 for i in val["nodes"] if i > 0])
            mat_val = 1
            for i in idx:
                B[:, i, val["source"]-1] = mat_val
                mat_val *= -1
        return B

    @property
    def C_matrix(self):
        return np.moveaxis(self.B_matrix().T, -1, 0)

    @property
    def D_matrix(self):
        return np.zeros((len(self.__dict__["freq"]), self.sources, self.sources))
        
    @property
    def nodes(self):
        nodes = []
        for key, val in self.__dict__["components"].items():
            if not isinstance(val, dict):
                continue
            nodes.extend(val.get("nodes", []))
        return set(nodes)

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def undefined(self):
        return sorted([key for key, val in self.__dict__["components"].items() 
                       if val.get("value", None) is None])
    
    @property
    def defined(self):
        return sorted([key for key, val in self.__dict__["components"].items() 
                       if val.get("value", None) is not None])
