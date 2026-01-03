from __future__ import print_function
import random

class BooleanNetwork:
    def __init__(self, number_of_nodes=5, transitions = None):
        """
        Initialize a random BooleanNetwork.
        :param number_of_nodes: total number of nodes in the network.
        :param transitions: transitions between nodes in this network. If not provided, generated on the spot randomly.
        """
        self.number_of_nodes = number_of_nodes
        if transitions is None:
            self.transitions = self.create_random_boolean_network(number_of_nodes)
        else:
            self.transitions = transitions

    def create_random_boolean_network(self, number_of_nodes):
        """
        Generates a random BooleanNetwork.
        :param number_of_nodes: total number of nodes in the network.
        :return: a list of length number_of_nodes of tuples of the format
        (list of parents, binary sequence of length 2^number_of_parents representing the transition)
        """
        # Create random functions governing individual nodes.
        transitions = [] # format:
        for n in range(number_of_nodes):
            number_of_parents = random.randint(0,3)
            parents = random.sample(range(number_of_nodes), number_of_parents) # if number_of_parents = 0, then this returns an empty list
            transition = self.create_random_transition(number_of_parents)
            # NOTE: If number_of_parents == 0, transition is a list with a single value [0] or [1].
            # Such nodes have no parents, so their value should remain constant in the dataset.
            # To ensure this, compute_next_node_state must check if len(parents) == 0 and return the current state.
            # Without this check, the node's state would be overwritten by transition[0] at every step.
            
            transitions.append((parents, transition))
        # Now the whole network is constructed in a reproducible way
        return transitions

    @staticmethod
    def create_random_transition(number_of_variables):
        """
        Creates a random transition function from n variables.
        :param number_of_variables: number of input variables.
        :return: a list of length 2**n describing the random transition function.
        Each element corresponds to the output value of the function, where the input if the list index.
        """
        binary_sequence = []
        for i in range(2**number_of_variables):
            bite = random.randint(0,1)
            binary_sequence.append(bite)
        return binary_sequence

    # Printing the network in a comprehensible way:
    def print_network(self):
        print("=======================================")
        print("The network has ", self.number_of_nodes, " nodes:", sep="")
        for i in range(self.number_of_nodes):
            parents, transition = self.transitions[i]
            print("")
            print("For node ", i, ":", sep="")
            if len(parents)>0:
                print(" the parents are of indexes", end=" ")
                for p in parents:
                    print(p, end=", ")
                print("")
            else:
                print(" the node has no parents,")
            print(" and the transition is ", transition, ".", sep="")
        print("=======================================")

    def save_network(self, filename):
        """
        Save Boolean network to a file in format:
        ```
        n_nodes
        [parent]; [transition]
        ```
        :param filename: name of the output file
        :return: None
        """
        with open(filename, 'w') as f:
            f.write(str(self.number_of_nodes))
            f.write("\n")
            for parents, transition in self.transitions:
                f.write(str(parents))
                f.write("; ")
                f.write(str(transition))
                f.write("\n")

    @staticmethod
    def load_network(filename):
        """
        Load Boolean network ground truth from a file in format:
        ```
        n_nodes
        [parent]; [transition]
        ```
        :param filename: name of the input file
        :return: a Boolean network
        """
        with open(filename, 'r') as f:
            n_nodes = int(f.readline())
            transitions = []
            for line in f:
                # Split the line into parents and transition parts
                # Example line: "[0, 1]; [0, 1, 1, 1]\n"
                parents_str, transition_str = line.split('; ')
                # parents_str = "[0, 1]"
                # transition_str = "[0, 1, 1, 1]\n" (includes the newline character at the end)

                # Parse parents
                parents_inner = parents_str[1:-1].strip() # remove brackets
                if parents_inner:
                    parents = [int(p) for p in parents_inner.split(', ')]
                else:
                    parents = []

                # Parse transition
                transition_inner = transition_str.strip() # remove newline and spaces
                transition_inner = transition_inner.lstrip('[').rstrip(']') # remove brackets explicitly from both sides
                if transition_inner:
                    transition = [int(x) for x in transition_inner.split(', ')]
                else:
                    transition = []

                transitions.append((parents, transition))
        return BooleanNetwork(n_nodes, transitions)


    # Helper functions for creating the datasets:

    @staticmethod
    def binary_to_int(binary_list):
        """
         Converts a list representing a binary in little-endian format to an integer.
        :param binary_list: a list representing a binary number.
        :return: a single integer.
        """
        number = 0
        for i in range(len(binary_list)):
            number += binary_list[i] * (2 ** i)
        return number

    def int_to_binary(self, number):
        """
        Converts an integer to a list representing a binary number in little-endian format.
        :param number: a single integer.
        :return: a list.
        """
        binary_list = []
        for i in range(self.number_of_nodes):
            binary_list.append(number%2)
            number = number // 2
        return binary_list
    
    def compute_next_node_state(self, node_index, current_state_binary): #which_node: int, current_state_binary: list[int]
        """
        Compute the next node's state according to a single transition.
        
        - Nodes with parents: follow the Boolean function (transition).
        - Nodes without parents: keep their current value (do not use transition).
    
        :param node_index: which node to compute the next state for.
        :param current_state_binary: current state of the network, represented as a little-endian binary list.
        :return: next state of the node, either 1 or 0.
        """
        parents, transition = self.transitions[node_index]
        
        if len(parents) == 0:
            # Node without parents keeps its current value
            return current_state_binary[node_index]
        
        index_binary = []
        for p in parents:
            index_binary.append(current_state_binary[p])
        index = self.binary_to_int(index_binary)
        return transition[index]
    
    def compute_next_network_state(self, current_state, synchronous): #current_state: int, synchronous: bool
        """
        Compute the next state of the network.
        :param current_state: a single integer representing the current state of the network.
        :param synchronous: If done synchronously, do all transitions at once for all nodes;
        otherwise, do it for a random single node.
        :return: a single integer representing the new network state.
        """
        current_state_binary = self.int_to_binary(current_state)
        new_state_binary = [0] * self.number_of_nodes
        if synchronous:
            # Do all transitions at once
            for i in range(self.number_of_nodes):
                new_state_binary[i] = self.compute_next_node_state(i, current_state_binary)
        else:
            # Pick a random node to make one transition on, the rest stays the same
            for i in range(self.number_of_nodes):
                new_state_binary[i] = current_state_binary[i]
            i = random.randint(0, self.number_of_nodes-1)
            new_state_binary[i] = self.compute_next_node_state(i, current_state_binary)
        return self.binary_to_int(new_state_binary)

    def create_dataset(self, number_of_datapoints = 1, trajectory_length = 10, synchronous = False, sampling_frequency = 1,
                       starting_states=None):
        #number_of_datapoints: int = 1, trajectory_length: int = 10, synchronous: bool = False, sampling_frequency: int = 1, starting_states: list[int|None] = []
        """
        Creates a dataset of consecutive states of a network.

        :param number_of_datapoints: Number of trajectories in our dataset, aka the length of output.
        Default 1

        :param trajectory_length: the length of our trajectories; even if sampling_frequency!=1, we still get the same length.
        Default 10

        :param synchronous: If True, all transitions will be synchronous, otherwise, they will be asynchronous.
        Default False

        :param sampling_frequency: Which states in a trajectory we take for our datapoint.
        For example, if =3, then we take the starting state, skip 2, take the next one, skip next 2, and so on.
        Default 1

        :param starting_states: a list of ints and Nones:
        if the value is an integer, this will be our starting point (in binary);
        if None, the starting point will be randomized.
        If the list is left empty, it will initialize as all Nones.
        Default []

        :return: is a two-element list [output, sampling_frequency],
        where output is a matrix (list of lists) of size number_of_datapoints x trajectory_length,
        where each element (i, j) represent the consecutive j-th state in i-th trajectory
        (sampling_frequency is required by BNFinder2 when saving the dataset)
        """
        # Prepare the list of starting points
        if starting_states is None:
            starting_states = []
        if len(starting_states) == 0:
            starting_states = [None] * number_of_datapoints
        for i in range(len(starting_states)):
            if starting_states[i] is None:
                starting_states[i] = random.randint(0, (2**self.number_of_nodes)-1)

        # Simulate the trajectories
        output = []
        for i in range(number_of_datapoints):
            # Setting the starting point
            current_trajectory = [starting_states[i]] # a list of ints that represent the states
            current_state = current_trajectory[-1]
            states_skipped = 0
            while len(current_trajectory) < trajectory_length:
                # Do a transition form current_state to the next one and save it as a new current_state
                current_state = self.compute_next_network_state(current_state, synchronous)
                if states_skipped == sampling_frequency-1:
                    current_trajectory.append(current_state)
                    states_skipped = 0
                else:
                    states_skipped += 1
            output.append(current_trajectory)
        return [output, sampling_frequency]
    
    # Printing and saving the datasets created:

    def print_dataset(self, dataset_tuple, as_binary=True): # dataset: list[list[int]], as_binary: bool = False
        """
        Print the dataset in human-readable format.
        :param dataset_tuple: tuple of the format (dataset, frequency). For details, see #create_dataset
        :param as_binary: whether to output the transitions as integers or as lists (default: lists)
        :return: None
        """
        dataset, sampling_frequency = dataset_tuple
        print("=======================================")
        print("The sampling frequency for this dataset was ", sampling_frequency, ".", sep="")
        print("The dataset has ", len(dataset), " datapoint(s):", sep="")
        for i in range(len(dataset)):
            print("")
            print("Datapoint ", i, " is: ", sep="", end="")
            for j in range(len(dataset[i])):
                if as_binary:
                    print(self.int_to_binary(dataset[i][j]), end="") # Correct this if the format turns out to be different
                else:
                    print(dataset[i][j], end="")
                if j<len(dataset[i])-1:
                    print(" --> ", end="")
        print("")
        print("=======================================")

    def save_dataset(self, filename, dataset_tuple): #filename: string, dataset: list[list[int]]
        """
        Save the dataset in a table of the following format:
        `conditions serie:time ... serie:time
        x0
        ...
        xn
        :param filename: output filename for the dataset in the format required by BNF
        :param dataset_tuple: tuple of the format (dataset, frequency). For details, see #create_dataset
        :return: None
        """
        dataset, sampling_frequency = dataset_tuple
        with open(filename, 'w') as f:
            f.write("conditions") # Header
            for index_of_trajectory in range(len(dataset)):
                for index_of_state in range(len(dataset[index_of_trajectory])):
                    f.write(" ")
                    f.write(str(index_of_trajectory))
                    f.write(":")
                    f.write(str(index_of_state))

            f.write("\n")

            # Unwrap the binaries in the dataset
            dataset_unwrapped = []
            for index_of_trajectory in range(len(dataset)):
                trajectory_unwrapped = []
                for index_of_state in range(len(dataset[index_of_trajectory])):
                    state_binary = self.int_to_binary(dataset[index_of_trajectory][index_of_state])
                    trajectory_unwrapped.append(state_binary)
                dataset_unwrapped.append(trajectory_unwrapped)

            # Write
            for index_of_variable in range(len(dataset_unwrapped[0][0])):
                f.write("x")
                f.write(str(index_of_variable))

                for index_of_trajectory in range(len(dataset_unwrapped)):
                    for index_of_state in range(len(dataset_unwrapped[index_of_trajectory])):
                        f.write("   ")
                        f.write(str(dataset_unwrapped[index_of_trajectory][index_of_state][index_of_variable]))

                f.write("\n")
            print("File saved as ", filename, sep="")
