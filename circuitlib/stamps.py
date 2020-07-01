class Stamps:
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
                A1[-n_source, n - 1] += sign
                A1[n - 1, -group2_idx] += sign
            A1[-group2_idx, -group2_idx] -= value

        return A1, A2, s

    def C(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        nodes = [n - 1 for n in nodes if n > 0]
        A2[nodes[0], nodes[0]] += value
        if len(nodes) < 2:
            return A1, A2, s
        A2[nodes[1], nodes[1]] += value
        A2[nodes[0], nodes[1]] += -value
        A2[nodes[1], nodes[0]] += -value
        return A1, A2, s

    def _stamp_r2(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        for n, sign in zip(nodes, [1, -1]):
            if n - 1 < 0:
                continue
            A1[-n_source, n - 1] += sign
            A1[n - 1, -n_source] += sign
        A1[-n_source, -n_source] -= val
        return A1

    def _stamp_c2(self, A1, A2, nodes, val, n_source):
        for n, sign in zip(nodes, [1, -1]):
            if n - 1 < 0:
                continue
            A1[n - 1, -n_source] += sign
            A2[-n_source, n - 1] -= sign * val
        A1[-n_source, -n_source] += 1
        return A1, A2

    def V(self, A1, A2, s, nodes, dependent_nodes, value, group2_idx, type):
        s[-1] = value
        for n, sign in zip(nodes, [1, -1]):
            if n - 1 < 0:
                continue
            A1[-group2_idx, n - 1] += sign
            A1[n - 1, -group2_idx] += sign
        return A1, A2, s

    def _stamp_vcvs(self, A, nodes, ctrl_nodes, val, n_source):
        s[-1] = val
        for n, sign in zip(nodes, [-1, 1]):
            if n - 1 < 0:
                continue
            A[-n_source, n - 1] += sign
            A[n - 1, -n_source] += sign

        for n, sign in zip(
            ctrl_nodes, [-1, 1]
        ):  # might need to swap around those signs
            if n - 1 < 0:
                continue
            A[-n_source, n - 1] += val * sign
        return A

    def _stamp_i(self, s, nodes, val, n_source):
        for n, sign in zip(nodes, [-1, 1]):
            if n - 1 < 0:
                continue
            s[n - 1] += val * sign
        return s

    def _stamp_l(self, A1, A2, nodes, val, n_source):
        for n, sign in zip(nodes, [1, -1]):
            if n - 1 < 0:
                continue
            A1[-n_source, n - 1] += sign
            A1[n - 1, -n_source] += sign
        A2[-n_source, -n_source] -= val
        return A1, A2
