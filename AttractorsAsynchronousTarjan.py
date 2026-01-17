from AttractorAsynchronous import AsyncAnalyzer

class AsyncAnalyzerTarjan(AsyncAnalyzer):
    """
    Asynchronous Boolean Network analyzer using Tarjan's SCC algorithm.

    Detects asynchronous attractors defined as SCCs without outgoing edges.
    """

    def __init__(self, network):
        AsyncAnalyzer.__init__(self, network)
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
                self._strong_connect(state)

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

    def _strong_connect(self, state):
        """
        Tarjan DFS step.
        """
        self.indices[state] = self.index
        self.lowlink[state] = self.index
        self.index += 1

        self.stack.append(state)
        self.on_stack[state] = True

        for successor in self._async_successors(state):
            if successor not in self.indices:
                self._strong_connect(successor)
                self.lowlink[state] = min(self.lowlink[state], self.lowlink[successor])
            elif self.on_stack.get(successor, False):
                self.lowlink[state] = min(self.lowlink[state], self.indices[successor])

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
    bn = AsyncAnalyzerTarjan.build_test_network()
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
