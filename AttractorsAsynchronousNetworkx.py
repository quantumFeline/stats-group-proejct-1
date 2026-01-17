import networkx as nx
from AttractorAsynchronous import AsyncAnalyzer

class AsyncAnalyzerNX(AsyncAnalyzer):
    """
    Asynchronous Boolean Network Analyzer using networkx.

    Detects attractors (SCCs without outgoing edges) and stores
    information for each state: is_attractor and attractor_id.
    """

    def analyze(self):
        """
        Analyze all states asynchronously and find attractors.
        Returns state_info dictionary.
        """
        G = nx.DiGraph()
        # Build the asynchronous state transition graph
        for state in range(self.num_states):
            for succ in self._async_successors(state):
                G.add_edge(state, succ)

        attractor_id = 0
        # Find strongly connected components (SCCs)
        for scc in nx.strongly_connected_components(G):
            # SCC without outgoing edges is an attractor
            if all(succ in scc for state in scc for succ in G.successors(state)):
                self.attractors.append(scc)
                for state in scc:
                    self.state_info[state] = {"is_attractor": True, "attractor_id": attractor_id}
                attractor_id += 1

        # Mark all other states as non-attractor
        for state in range(self.num_states):
            if state not in self.state_info:
                self.state_info[state] = {"is_attractor": False, "attractor_id": None}

        return self.state_info

    # ------------------- Visualization / Printing -------------------

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


def test_async_nx():
    bn = AsyncAnalyzerNX.build_test_network()
    print("===== TEST NETWORK (NX) =====")
    bn.print_network()

    analyzer = AsyncAnalyzerNX(bn)
    state_info = analyzer.analyze()
    
    print("\n===== SUMMARY =====")
    analyzer.print_summary()

    print("\n===== ATTRACTORS (NX) =====")
    analyzer.print_attractors_binary(show_decimal=True)

    print("\n===== STATE CLASSIFICATION (NX) =====")
    analyzer.print_state_info()

if __name__ == "__main__":
    test_async_nx()
