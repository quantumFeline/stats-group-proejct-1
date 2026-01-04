from __future__ import print_function
from BooleanNetwork import BooleanNetwork

class StateSpaceAnalyzer:
    """
    Analyze the synchronous state transition graph of a Boolean network.

    For each state:
    - determine whether it belongs to an attractor (cycle),
    - assign an attractor ID,
    - compute distance to the attractor (0 for attractor states).
    """

    def __init__(self, network):
        """
        :param network: BooleanNetwork instance
        """
        self.network = network
        self.n = network.number_of_nodes
        self.num_states = 2 ** self.n

        # state -> info dict
        self.state_info = {}

        # internal counter for attractor IDs
        self._next_attractor_id = 0

    def analyze(self):
        """
        Perform full state space analysis.
        """
        for state in range(self.num_states):
            if state not in self.state_info:
                self._analyze_from_state(state)

        return self.state_info

    def _analyze_from_state(self, start_state):
        """
        Analyze trajectory starting from a single state.
        """
        visited_local = {}   # state -> index in path
        path = []

        current = start_state

        while True:
            # Case 1: we found a cycle (new attractor)
            if current in visited_local:
                cycle_start = visited_local[current]
                self._register_new_attractor(path, cycle_start)
                return

            # Case 2: we reached an already analyzed state
            if current in self.state_info:
                self._propagate_existing_info(path, current)
                return

            # Otherwise: continue simulation
            visited_local[current] = len(path)
            path.append(current)
            current = self.network.compute_next_network_state(
                current, synchronous=True
            )

    def _register_new_attractor(self, path, cycle_start):
        """
        Register a newly discovered attractor and its basin.
        """
        attractor_id = self._next_attractor_id
        self._next_attractor_id += 1

        # Attractor states (cycle)
        for state in path[cycle_start:]:
            self.state_info[state] = {
                "is_attractor": True,
                "attractor_id": attractor_id,
                "distance": 0
            }

        # Transient states leading to the attractor
        for i in range(cycle_start - 1, -1, -1):
            state = path[i]
            self.state_info[state] = {
                "is_attractor": False,
                "attractor_id": attractor_id,
                "distance": cycle_start - i
            }

    def _propagate_existing_info(self, path, known_state):
        """
        Propagate attractor information from an already known state.
        """
        known_info = self.state_info[known_state]

        for i in range(len(path) - 1, -1, -1):
            state = path[i]
            self.state_info[state] = {
                "is_attractor": False,
                "attractor_id": known_info["attractor_id"],
                "distance": known_info["distance"] + (len(path) - i)
            }

    # -------------------- Convenience methods --------------------

    def get_attractors(self):
        """
        Return a dictionary attractor_id -> list of states.
        """
        attractors = {}
        for state, info in self.state_info.items():
            if info["is_attractor"]:
                aid = info["attractor_id"]
                attractors.setdefault(aid, []).append(state)
        return attractors

    def get_states_by_distance(self):
        """
        Return a dictionary distance -> list of states.
        """
        dist_map = {}
        for state, info in self.state_info.items():
            d = info["distance"]
            dist_map.setdefault(d, []).append(state)
        return dist_map

    def print_summary(self):
        """
        Print a short summary of the state space structure.
        """
        attractors = self.get_attractors()
        print("Number of states:", self.num_states)
        print("Number of attractors:", len(attractors))
        for aid, states in attractors.items():
            print("  Attractor", aid, "size:", len(states))

def main():
    # -------------------------
    # Define the network manually
    # -------------------------
    # Note: little-endian order for binary sequences
    
    ###### Source: Example1 and Example3 from lab about boolean networks on moodle

    
    transitions = [
        # f0 = not x1 or (x0 and x1)
        # parents: x0, x1 => output for 00,10,01,11
        ([0, 1], [1, 1, 0, 1]),

        # f1 = x0 and x1
        # parents: x0, x1 => output for 00,10,01,11
        ([0, 1], [0, 0, 0, 1]),

        # f2 = x2 and not(x0 and x1)
        # parents: x0, x1, x2 => output for 000,100,010,110,001,101,011,111
        ([0, 1, 2], [0,0,0,0,1,1,1,0])
    ]

    bn = BooleanNetwork(number_of_nodes=3, transitions=transitions)
    
    print("===== Boolean Network =====")
    bn.print_network()

    # -------------------------
    # Analyze state space
    # -------------------------
    analyzer = StateSpaceAnalyzer(bn)
    state_info = analyzer.analyze()

    # -------------------------
    # Print summary
    # -------------------------
    print("\n===== State Space Summary =====")
    analyzer.print_summary()

    # -------------------------
    # Print all states with info (binary)
    # -------------------------
    print("\nState info (binary representation):")
    for state_int, info in state_info.items():
        state_bin = bn.int_to_binary(state_int)
        print("State {} (binary {}): is_attractor={}, attractor_id={}, distance={}".format(
            state_int, state_bin, info['is_attractor'], info['attractor_id'], info['distance']
        ))

if __name__ == "__main__":
    main()

