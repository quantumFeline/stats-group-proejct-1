from __future__ import print_function
from BooleanNetwork import BooleanNetwork
import random
import sys

RANDOM_SEED = 42
DEFAULT_N_NODES = 5
AS_BINARY = True # False is more concise, but is harder to understand where the resulting trajectory came from

if __name__ == '__main__':
    # Creating a network in a reproducible way.
    # Usage: python CreateDatasets.py [output_filename] [n_nodes]

    random.seed(RANDOM_SEED)

    if len(sys.argv) > 2:
        network = BooleanNetwork(sys.argv[2])
    else:
        network = BooleanNetwork(DEFAULT_N_NODES)

    network.print_network()

    # Under construction!
    # For now, they do nothing:
    # filename = 'Network.txt'
    # network.save_network(filename)
    # network.load_network(filename) # overwrites the current network

    #======================================================================================================================================================

    # Create a dataset in a reproducible way

    number_of_datapoints = 5
    length_of_one_trajectory = 5
    synchronous = False
    sampling_frequency = 1
    starting_states = [] # if left empty, the starting states will be randomized for each datapoint; uncomment the next line for always starting at 0
    # starting_states = [0] * number_of_datapoints
    # If you want to specify some of the starting states, you have to write a list of length `number_of_datapoints` of ints and Nones,
    # where an integer value is the index of the state (in reverse binary), and None is "take a random starting state"
    dataset = network.create_dataset(number_of_datapoints, length_of_one_trajectory, synchronous, sampling_frequency, starting_states)

    #======================================================================================================================================================

    # Print and save the dataset
    network.print_dataset(dataset, AS_BINARY)

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'output.txt' # default

    network.save_dataset(filename, dataset) #saved in the format required for BNFinder2