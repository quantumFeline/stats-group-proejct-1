from __future__ import print_function
from BooleanNetwork import BooleanNetwork
import networkx as nx

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


class AsyncAnalyzerNX:
    """
    Asynchronous Boolean Network Analyzer using networkx.

    Detects attractors (SCCs without outgoing edges) and stores
    information for each state: is_attractor and attractor_id.
    """

    def __init__(self, network):
        self.network = network
        self.n = network.number_of_nodes
        self.num_states = 2 ** self.n
        self.state_info = {}   # state -> {"is_attractor": bool, "attractor_id": int}
        self.attractors = []   # list of sets of states forming attractors

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

    #def _async_successors(self, state_int):
    #    """
    #    Return all possible next states from state_int in asynchronous update mode.
    #    Each successor differs from the current state by updating exactly one node.
    #    """
    #    cur = self.network.int_to_binary(state_int)
    #    succs = []
    #    for i in range(self.n):
    #        new_state = cur[:]
    #        new_state[i] = self.network.compute_next_node_state(i, cur)
    #        succ = self.network.binary_to_int(new_state)
    #        if succ != state_int:
    #            succs.append(succ)
    #    return succs
        
    def _async_successors(self, state_int):
        """
        Return all possible next states from state_int in asynchronous update mode.
        Each successor differs from the current state by updating exactly one node.
        """
        cur = self.network.int_to_binary(state_int)
        succs = []
        for i in range(self.n):
            new_state = cur[:]
            new_state[i] = self.network.compute_next_node_state(i, cur)
            succ = self.network.binary_to_int(new_state)
            if succ != state_int:
                succs.append(succ)
        return succs

    # ------------------- Visualization / Printing -------------------

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


def test_async_nx():
    bn = build_test_network()
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
