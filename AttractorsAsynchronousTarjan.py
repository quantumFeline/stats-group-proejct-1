from __future__ import print_function
from BooleanNetwork import BooleanNetwork

def build_test_network():
    transitions = [

        # --------------------------------------------------
        # f0 = not x0 or (not x3 and not x5)
        # parents = [x0, x3, x5]
        #
        # little-endian order:
        # index = x0*1 + x3*2 + x5*4
        #
        # index | x0 x3 x5 | f0
        #   0   | 0  0  0  | 1
        #   1   | 1  0  0  | 1
        #   2   | 0  1  0  | 1
        #   3   | 1  1  0  | 0
        #   4   | 0  0  1  | 1
        #   5   | 1  0  1  | 0
        #   6   | 0  1  1  | 1
        #   7   | 1  1  1  | 0
        #
        ([0, 3, 5], [1, 1, 1, 0, 1, 0, 1, 0]),


        # --------------------------------------------------
        # f1 = not x2
        # parents = [x2]
        #
        # index | x2 | f1
        #   0   | 0  | 1
        #   1   | 1  | 0
        #
        ([2], [1, 0]),


        # --------------------------------------------------
        # f2 = not x1 or not x5
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
        # f3 = x0 or not x3
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
        # (not x0 and x1 and x3)
        # or (not x0 and x3 and not x5)
        # or (x1 and not x3 and not x5)
        # or (not x1 and not x3 and x5)
        #
        # parents = [x0, x1, x3, x5]
        #
        # index = x0*1 + x1*2 + x3*4 + x5*8
        #
        # index | x0 x1 x3 x5 | f4
        #   0   | 0  0  0  0  | 0
        #   1   | 1  0  0  0  | 0
        #   2   | 0  1  0  0  | 1 # x1 and not x3 and not x5
        #   3   | 1  1  0  0  | 1 # x1 and not x3 and not x5
        #   4   | 0  0  1  0  | 1 # not x0 and x3 and not x5
        #   5   | 1  0  1  0  | 0
        #   6   | 0  1  1  0  | 1 # not x0 and x1 and x3
        #   7   | 1  1  1  0  | 0
        #   8   | 0  0  0  1  | 1 # not x1 and not x3 and x5
        #   9   | 1  0  0  1  | 1 # not x1 and not x3 and x5
        #  10   | 0  1  0  1  | 0
        #  11   | 1  1  0  1  | 0
        #  12   | 0  0  1  1  | 0
        #  13   | 1  0  1  1  | 0
        #  14   | 0  1  1  1  | 1 # not x0 and x1 and x3
        #  15   | 1  1  1  1  | 0
        #
        ([0, 1, 3, 5],
         [0, 0, 1, 1,
          1, 0, 1, 0,
          1, 1, 0, 0,
          0, 0, 1, 0]),


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
            new_state = cur[:]
            new_state[i] = self.network.compute_next_node_state(i, cur)
            succ = self.network.binary_to_int(new_state)
            if succ != state_int:
                succs.append(succ)
            #succs.append(succ)
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
        """
        Print a brief summary of the state space.
        """
        print("Total number of states:", self.num_states)
        print("Number of attractors:", len(self.attractors))
        for i, attr in enumerate(self.attractors):
            print("  Attractor {}: {} states".format(i, len(attr)))
    
    def print_attractors_binary(self, show_decimal=True):
        """
        Print all attractors in binary representation.
        Optionally also show decimal value of each state.
    
        :param show_decimal: if True, print decimal state before binary
        """
        print("\nAttractors (binary states):")
        for i, attr in enumerate(self.attractors):
            print("  Attractor {}:".format(i))
            for state in sorted(attr):
                bin_state = self.network.int_to_binary(state)
                if show_decimal:
                    print("    {:2d} {}".format(state, bin_state))
                else:
                    print("    {}".format(bin_state))

    def print_state_info(self):
        """
        Print information for all states: binary state, is_attractor, attractor_id.
        """
        print("\nState information:")
        for state in range(self.num_states):
            info = self.state_info[state]
            bin_state = self.network.int_to_binary(state)
            print("State {} (binary {}): is_attractor={}, attractor_id={}".format(
                state, bin_state, info['is_attractor'], info['attractor_id']
            ))

def test_async_tarjan():
    bn = build_test_network()
    print("===== TEST NETWORK (Tarjan) =====")
    bn.print_network()

    analyzer = AsyncAnalyzerTarjan(bn)
    state_info = analyzer.analyze()
    
    print("\n===== SUMMARY =====")
    analyzer.print_summary()

    print("\n===== ATTRACTORS (Tarjan) =====")
    analyzer.print_attractors_binary(show_decimal=True)

    print("\n===== STATE CLASSIFICATION (Tarjan) =====")
    analyzer.print_state_info()

if __name__ == "__main__":
    test_async_tarjan()
