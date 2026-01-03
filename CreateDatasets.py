from __future__ import print_function
from BooleanNetwork import BooleanNetwork
import random
import sys
import argparse

RANDOM_SEED = 42
DEFAULT_N_NODES = 5
AS_BINARY = True # False is more concise, but is harder to understand where the resulting trajectory came from
DEFAULT_DATASET_LENGTH = 5
DEFAULT_TRAJECTORY_LENGTH = 5
DEFAULT_OUTPUT_FILENAME = "output.txt"
DEFAULT_GROUND_TRUTH_FILENAME = "group_truth_network.txt"
DEFAULT_SYNC = False

if __name__ == '__main__':
    # Creating a network in a reproducible way.
    # Usage:  python2.7 CreateDatasets.py -o output.txt -g network.txt -n 10 -d 20 -l 15 -s
    parser = argparse.ArgumentParser(description='Create Boolean network datasets')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_FILENAME,
                        help='Output dataset filename (default: output.txt)')
    parser.add_argument('-g', '--ground-truth', default=DEFAULT_GROUND_TRUTH_FILENAME,
                        help='Ground truth network filename (default: ground_truth_network.txt)')
    parser.add_argument('-n', '--nodes', type=int, default=DEFAULT_N_NODES,
                        help='Number of nodes (default: 5)')
    parser.add_argument('-d', '--datapoints', type=int, default=DEFAULT_DATASET_LENGTH,
                        help='Number of trajectories (default: 5)')
    parser.add_argument('-l', '--length', type=int, default=DEFAULT_TRAJECTORY_LENGTH,
                        help='Length of each trajectory (default: 5)')
    parser.add_argument('-s', '--synchronous', action='store_true',
                        help='Use synchronous updates (default: asynchronous)')

    args = parser.parse_args()

    random.seed(RANDOM_SEED)

    network = BooleanNetwork(args.nodes)
    network.print_network()

    network.save_network(args.ground_truth)
    # network.load_network(args.ground_truth) # overwrites the current network
    
    loaded_network = BooleanNetwork.load_network(args.ground_truth)
    print("\n===== ORIGINAL NETWORK =====")
    network.print_network()

    print("\n===== LOADED NETWORK =====")
    loaded_network.print_network()
    #======================================================================================================================================================

    # Create a dataset in a reproducible way
    sampling_frequency = 1
    starting_states = [] # if left empty, the starting states will be randomized for each datapoint; uncomment the next line for always starting at 0
    # starting_states = [0] * number_of_datapoints
    # If you want to specify some of the starting states, you have to write a list of length `number_of_datapoints` of ints and Nones,
    # where an integer value is the index of the state (in reverse binary), and None is "take a random starting state"
    dataset = network.create_dataset(args.datapoints, args.length, args.synchronous, sampling_frequency, starting_states)

    #======================================================================================================================================================

    # Print and save the dataset
    network.print_dataset(dataset, AS_BINARY)
    network.save_dataset(args.output, dataset) #saved in the format required for BNFinder2
    
    # to run BNF, run:
    # bnf -e output.txt -s MDL -n network_mdl.txt -g
    # bnf -e output.txt -s BDE -n network_bde.txt [-i 10 -o 0.01]
