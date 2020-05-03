import numpy as np

def multimeter(circuit, node1, node2=0, mode="V", **kwargs):
    V = circuit(**kwargs)
    if mode == "V":
        if not node2:
            return V[:,node1-1]
        return V[:,node1-1] - V[:,node2-1]
    # If calculating current or impedances then we must find the impedances
    # between the circuit nodes, given by the A matrix
    # TODO: could also just create a copy of the netlist then update its compenents
    # then spit out the G_matrix?
    netlist = circuit.__closure__[1].cell_contents.copy()
    netlist.update(**kwargs)
    G = netlist.G_matrix
    if not node2:
        Z =  1/np.sum(G[:, node1-1, :], axis=-1)
        if mode == "Z":
            return Z
        elif mode == "I":
            return V[:,node1-1] / Z
    Z = -1/G[:, node1-1, node2-1]
    if mode == "Z":
        return Z
    elif mode == "I":
        return (V[:,node1-1] - V[:,node2-1]) / Z

