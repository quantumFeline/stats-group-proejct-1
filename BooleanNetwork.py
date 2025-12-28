from __future__ import print_function
import random

class BooleanNetwork:
    def __init__(self, number_of_nodes=5):
        self.number_of_nodes = number_of_nodes
        #creating random functions governing individual nodes:
        self.transitions = [] # format:
        # a list of length number_of_nodes of tuples:
        # first element is a list of parents,
        # second is a binary sequence (a list) of length 2^number_of_parents representing the transition
        for n in range(number_of_nodes):
            #pick 0-3 random nodes as parents
            number_of_parents = random.randint(0,3)
            parents = random.sample(range(number_of_nodes), number_of_parents) # if number_of_parents = 0, then this returns an empty list
            #pick random behaviour of our transition
            transition = self.create_random_transition(number_of_parents)
            #append to self.transitions
            self.transitions.append((parents, transition))
        # Now the whole network is constructed in a reproducible way

    # Helper function for __init__:
    @staticmethod
    def create_random_transition(number_of_variables):
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

    # Saving the network to a file 
    @staticmethod
    def save_network(filename):
        print("Under construction")

    # Loading the network from a file
    @staticmethod
    def load_network(filename):
        print("Under construction")


    # Helper functions for creating the datasets:

    @staticmethod
    def binary_to_int(binary_list): # Converts a binary list to a number in a "reverse order" (0th index gives 1, 1st index gives 2, 2nd gives 4 etc)
        number = 0
        for i in range(len(binary_list)):
            number += binary_list[i] * (2 ** i)
        return number

    def int_to_binary(self, number): # Converts a number to a binary list in a "reverse order" (0th index gives 1, 1st index gives 2, 2nd gives 4 etc)
        binary_list = []
        for i in range(self.number_of_nodes):
            binary_list.append(number%2)
            number = number // 2
        return binary_list
    
    def transition_on_a_single_node(self, which_node, current_state_binary): #which_node: int, current_state_binary: list[int]
        parents, transition = self.transitions[which_node]
        index_binary = []
        for p in parents:
            index_binary.append(current_state_binary[p])
        index = self.binary_to_int(index_binary)
        return transition[index]
    
    def do_a_transition(self, current_state, synchronous): #current_state: int, synchronous: bool
        current_state_binary = self.int_to_binary(current_state)
        new_state_binary = [0]*self.number_of_nodes
        if synchronous:
            # Do all transitions at once
            for i in range(self.number_of_nodes):
                new_state_binary[i] = self.transition_on_a_single_node(i, current_state_binary)
        else:
            # Pick a random node to make one transition on, the rest stays the same
            for i in range(self.number_of_nodes):
                new_state_binary[i] = current_state_binary[i]
            i = random.randint(0, self.number_of_nodes-1)
            new_state_binary[i] = self.transition_on_a_single_node(i, current_state_binary)
        return self.binary_to_int(new_state_binary)
    
    # Main dataset-creating function:
    def create_dataset(self, number_of_datapoints = 1, length = 10, synchronous = False, sampling_frequency = 1,
                       starting_states=None):
        #number_of_datapoints: int = 1, length: int = 10, synchronous: bool = False, sampling_frequency: int = 1, starting_states: list[int|None] = []
        """
        :param number_of_datapoints: Number of trajectories in our dataset, aka the length of output.
        Default 1

        :param length: the length of our trajectories; even if sampling_frequency!=1, we still get the same length.
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
        where output is a list of length `number_of_datapoints`
        of lists of length `length`
        of ints that represent the consecutive states in trajectories
        (sampling_frequency is required by BNFinder2 when saving the dataset)
        """
        # Preparing the list of starting points:
        if starting_states is None:
            starting_states = []
        if len(starting_states)==0:
            starting_states = [None]*number_of_datapoints
        for i in range(len(starting_states)):
            if starting_states[i] is None:
                starting_states[i] = random.randint(0, (2**self.number_of_nodes)-1)

        # Simulating the trajectories:
        output = []
        for i in range(number_of_datapoints):
            # Setting the starting point
            current_trajectory = [starting_states[i]] # a list of ints that represent the states
            current_state = current_trajectory[-1]
            states_skipped = 0
            while len(current_trajectory)<length:
                # Do a transition form current_state to the next one and save it as a new current_state
                current_state = self.do_a_transition(current_state, synchronous)
                if states_skipped == sampling_frequency-1:
                    current_trajectory.append(current_state)
                    states_skipped=0
                else:
                    states_skipped+=1
            output.append(current_trajectory)
        return [output, sampling_frequency]
    
    # Printing and saving the datasets created:

    def print_dataset(self, dataset, as_binary): #dataset: list[list[int]], as_binary: bool = False
        output, sampling_frequency = dataset
        print("=======================================")
        print("The sampling frequency for this dataset was ", sampling_frequency, ".", sep="")
        print("The dataset has ", len(output), " datapoint(s):", sep="")
        for i in range(len(output)):
            print("")
            print("Datapoint ", i, " is: ", sep="", end="")
            for j in range(len(output[i])):
                if as_binary:
                    print(self.int_to_binary(output[i][j]), end="") # Correct this if the format turns out to be different
                else:
                    print(output[i][j], end="")
                if j<len(output[i])-1:
                    print(" --> ", end="")
        print("")
        print("=======================================")

    def save_dataset(self, filename, dataset): #filename: string, dataset: list[list[int]]
        output, sampling_frequency = dataset
        with open(filename, 'w') as f:
            f.write("#trajectory    time") # Headers
            for index_of_node in range(self.number_of_nodes):
                f.write("   x")
                f.write(str(index_of_node))
            f.write("\n")
            for index_of_trajectory in range(len(output)):
                time_of_observation = 0
                for index_of_state in range(len(output[index_of_trajectory])): # Each state governs a single line
                    f.write(str(index_of_trajectory+1)) # Index of a trajectory starting from 1
                    f.write("   ")
                    f.write(str(time_of_observation)) # Time of a given observation starting from 0
                    time_of_observation += sampling_frequency
                    state_binary = self.int_to_binary(output[index_of_trajectory][index_of_state])
                    for bite in state_binary:
                        f.write("   ")
                        f.write(str(bite))
                    f.write("\n")
                f.write("#")
                if index_of_trajectory < len(output)-1:
                    f.write("\n")
            print("File saved as ", filename, sep="")
