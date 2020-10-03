class Stamps:
    """Defines the component stamps for the MNA matrices."""

    def __init__(self, transient):
        if transient:
            self.transient_modifier = 2
        else:
            self.transient_modifier = 1

    def R(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        if group2_idx == 0:
            nodes = [n - 1 for n in nodes if n > 0]
            A1[nodes[0], nodes[0]] += 1 / value
            if len(nodes) < 2:
                return A1, A2, s
            A1[nodes[1], nodes[1]] += 1 / value
            A1[nodes[0], nodes[1]] += -1 / value
            A1[nodes[1], nodes[0]] += -1 / value
        else:
            for n, sign in zip(nodes, [1, -1]):
                if n - 1 < 0:
                    continue
                A1[-group2_idx, n - 1] += sign
                A1[n - 1, -group2_idx] += sign
            A1[-group2_idx, -group2_idx] -= value

        return A1, A2, s

    def C(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        value *= self.transient_modifier  # stamp the value * 2 for transient analysis
        nodes = [n - 1 for n in nodes if n > 0]
        A2[nodes[0], nodes[0]] += value
        if len(nodes) < 2:
            return A1, A2, s
        A2[nodes[1], nodes[1]] += value
        A2[nodes[0], nodes[1]] += -value
        A2[nodes[1], nodes[0]] += -value
        return A1, A2, s

    def L(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        value *= self.transient_modifier  # stamp the value * 2 for transient analysis
        for n, sign in zip(nodes, [1, -1]):
            if n - 1 < 0:
                continue
            A1[-group2_idx, n - 1] += sign
            A1[n - 1, -group2_idx] += sign
        A2[-group2_idx, -group2_idx] -= value
        return A1, A2, s

    def _stamp_r2(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        for n, sign in zip(nodes, [1, -1]):
            if n - 1 < 0:
                continue
            A1[-group2_idx, n - 1] += sign
            A1[n - 1, -group2_idx] += sign
        A1[-group2_idx, -group2_idx] -= value
        return A1, A2, s

    def _stamp_c2(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        for n, sign in zip(nodes, [1, -1]):
            if n - 1 < 0:
                continue
            A1[n - 1, -group2_idx] += sign
            A2[-group2_idx, n - 1] -= sign * value
        A1[-group2_idx, -group2_idx] += 1
        return A1, A2, s

    def V(
        self,
        A1,
        A2,
        s,
        nodes,
        dependent_nodes,
        value,
        group2_idx,
        type,
        signal,
        set_kwargs,
    ):
        s[-1 * group2_idx] = signal(
            0
        )  # TODO: this only initialise voltage to value at t=0. Needs to handle any start time...
        for n, sign in zip(nodes, [-1, 1]):
            if n - 1 < 0:
                continue
            A1[-group2_idx, n - 1] += sign
            A1[n - 1, -group2_idx] += sign
        return A1, A2, s

    def _stamp_vcvs(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        s[-1] = value
        for n, sign in zip(nodes, [-1, 1]):
            if n - 1 < 0:
                continue
            A1[-group2_idx, n - 1] += sign
            A1[n - 1, -group2_idx] += sign

        for n, sign in zip(
            dependent_nodes, [-1, 1]
        ):  # might need to swap around those signs
            if n - 1 < 0:
                continue
            # TODO: Check if this is meant to be A1 or A2
            A1[-group2_idx, n - 1] += value * sign
        return A1, A2, s

    def _stamp_i(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        for n, sign in zip(nodes, [-1, 1]):
            if n - 1 < 0:
                continue
            s[n - 1] += value * sign
        return A1, A2, s

    def _stamp_l(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        for n, sign in zip(nodes, [1, -1]):
            if n - 1 < 0:
                continue
            A1[-group2_idx, n - 1] += sign
            A1[n - 1, -group2_idx] += sign
        A2[-group2_idx, -group2_idx] -= value
        return A1, A2, s
