from BooleanNetwork import BooleanNetwork

def build_test_network():
    transitions = [

        # --------------------------------------------------
        # f0 = ¬x0 ∨ (¬x3 ∧ ¬x5)
        # parents = [x0, x3, x5]
        #
        # little-endian order:
        # index = x0*1 + x3*2 + x5*4
        #
        # index | x0 x3 x5 | f0
        #   0   | 0  0  0  | 1
        #   1   | 1  0  0  | 1
        #   2   | 0  1  0  | 1
        #   3   | 1  1  0  | 1
        #   4   | 0  0  1  | 1
        #   5   | 1  0  1  | 0
        #   6   | 0  1  1  | 0
        #   7   | 1  1  1  | 0
        #
        ([0, 3, 5], [1, 1, 1, 1, 1, 0, 0, 0]),


        # --------------------------------------------------
        # f1 = ¬x2
        # parents = [x2]
        #
        # index | x2 | f1
        #   0   | 0  | 1
        #   1   | 1  | 0
        #
        ([2], [1, 0]),


        # --------------------------------------------------
        # f2 = ¬x1 ∨ ¬x5
        # parents = [x1, x5]
        #
        # index = x1*1 + x5*2
        #
        # index | x1 x5 | f2
        #   0   | 0  0  | 1
        #   1   | 1  0  | 1
        #   2   | 0  1  | 1
        #   3   | 1  1  | 0
        #
        ([1, 5], [1, 1, 1, 0]),


        # --------------------------------------------------
        # f3 = x0 ∨ ¬x3
        # parents = [x0, x3]
        #
        # index = x0*1 + x3*2
        #
        # index | x0 x3 | f3
        #   0   | 0  0  | 1
        #   1   | 1  0  | 1
        #   2   | 0  1  | 0
        #   3   | 1  1  | 1
        #
        ([0, 3], [1, 1, 0, 1]),


        # --------------------------------------------------
        # f4 =
        # (¬x0 ∧ x1 ∧ x3)
        # ∨ (¬x0 ∧ x3 ∧ ¬x5)
        # ∨ (x1 ∧ ¬x3 ∧ ¬x5)
        # ∨ (¬x1 ∧ ¬x3 ∧ x5)
        #
        # parents = [x0, x1, x3, x5]
        #
        # index = x0*1 + x1*2 + x3*4 + x5*8
        #
        # index | x0 x1 x3 x5 | f4
        #   0   | 0  0  0  0  | 0
        #   1   | 1  0  0  0  | 0
        #   2   | 0  1  0  0  | 1
        #   3   | 1  1  0  0  | 0
        #   4   | 0  0  1  0  | 1
        #   5   | 1  0  1  0  | 0
        #   6   | 0  1  1  0  | 1
        #   7   | 1  1  1  0  | 0
        #   8   | 0  0  0  1  | 1
        #   9   | 1  0  0  1  | 0
        #  10   | 0  1  0  1  | 0
        #  11   | 1  1  0  1  | 0
        #  12   | 0  0  1  1  | 0
        #  13   | 1  0  1  1  | 0
        #  14   | 0  1  1  1  | 0
        #  15   | 1  1  1  1  | 0
        #
        ([0, 1, 3, 5],
         [0, 0, 1, 0,
          1, 0, 1, 0,
          1, 0, 0, 0,
          0, 0, 0, 0]),


        # --------------------------------------------------
        # f5 = x5
        # parents = [x5]
        #
        # index | x5 | f5
        #   0   | 0  | 0
        #   1   | 1  | 1
        #
        ([5], [0, 1])
    ]

    return BooleanNetwork(number_of_nodes=6, transitions=transitions)


class AsyncAnalyzerTarjan:
    """
    Asynchronous Boolean Network analyzer using Tarjan's SCC algorithm.

    Detects asynchronous attractors defined as SCCs without outgoing edges.
    """

    def __init__(self, network):
        self.network = network
        self.n = network.number_of_nodes
        self.num_states = 2 ** self.n

        self.state_info = {}     # state -> {is_attractor, attractor_id}
        self.attractors = []     # list of sets (each set is one attractor)

        # Tarjan algorithm variables
        self.index = 0
        self.stack = []
        self.indices = {}
        self.lowlink = {}
        self.on_stack = {}

    def analyze(self):
        """
        Run Tarjan's algorithm on the asynchronous state transition graph.
        """
        for state in range(self.num_states):
            if state not in self.indices:
                self._strongconnect(state)

        # Assign attractor IDs
        attractor_id = 0
        for scc in self.attractors:
            for state in scc:
                self.state_info[state] = {
                    "is_attractor": True,
                    "attractor_id": attractor_id
                }
            attractor_id += 1

        # Mark remaining states as transient
        for state in range(self.num_states):
            if state not in self.state_info:
                self.state_info[state] = {
                    "is_attractor": False,
                    "attractor_id": None
                }

        return self.state_info

    def _async_successors(self, state_int):
        """
        Return all asynchronous successors of a state.
        """
        cur = self.network.int_to_binary(state_int)
        succs = []
        for i in range(self.n):
            new_state = cur.copy()
            new_state[i] = self.network.compute_next_node_state(i, cur)
            succ = self.network.binary_to_int(new_state)
            if succ != state_int:
                succs.append(succ)
        return succs

    def _strongconnect(self, state):
        """
        Tarjan DFS step.
        """
        self.indices[state] = self.index
        self.lowlink[state] = self.index
        self.index += 1

        self.stack.append(state)
        self.on_stack[state] = True

        for succ in self._async_successors(state):
            if succ not in self.indices:
                self._strongconnect(succ)
                self.lowlink[state] = min(self.lowlink[state], self.lowlink[succ])
            elif self.on_stack.get(succ, False):
                self.lowlink[state] = min(self.lowlink[state], self.indices[succ])

        # Root of SCC found
        if self.lowlink[state] == self.indices[state]:
            scc = set()
            while True:
                w = self.stack.pop()
                self.on_stack[w] = False
                scc.add(w)
                if w == state:
                    break

            # Check if SCC is an attractor (no outgoing edges)
            if all(succ in scc for node in scc for succ in self._async_successors(node)):
                self.attractors.append(scc)

    # -------------------- Printing utilities --------------------

    def print_summary(self):
        print("Total states:", self.num_states)
        print("Number of asynchronous attractors:", len(self.attractors))
        for i, attr in enumerate(self.attractors):
            print(f"  Attractor {i}: size {len(attr)}")

    def print_attractors_binary(self):
        print("\nAttractors (binary states):")
        for i, attr in enumerate(self.attractors):
            print(f"Attractor {i}:")
            for s in sorted(attr):
                print("  ", self.network.int_to_binary(s))

    def print_state_info(self):
        print("\nState information:")
        for s in range(self.num_states):
            info = self.state_info[s]
            print(
                f"State {s} {self.network.int_to_binary(s)} | "
                f"is_attractor={info['is_attractor']} | "
                f"attractor_id={info['attractor_id']}"
            )

def test_async_tarjan():
    bn = build_test_network()
    print("===== TEST NETWORK (TARJAN) =====")
    bn.print_network()

    analyzer = AsyncAnalyzerTarjan(bn)
    state_info = analyzer.analyze()

    print("\n===== ATTRACTORS (TARJAN) =====")
    for i, scc in enumerate(analyzer.attractors):
        print(f"Attractor {i}:")
        for s in sorted(scc):
            print(" ", s, bn.int_to_binary(s))

    print("\n===== STATE CLASSIFICATION (TARJAN) =====")
    for s in range(bn.number_of_nodes ** 2):
        info = state_info.get(s)
        if info:
            print(
                f"State {s:2d} {bn.int_to_binary(s)} | "
                f"is_attractor={info['is_attractor']} | "
                f"attractor_id={info['attractor_id']}"
            )

if __name__ == "__main__":
    test_async_tarjan()
