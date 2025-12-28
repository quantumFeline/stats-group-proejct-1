from __future__ import print_function
from Boolean_Network import Boolean_Network
import random

#======================================================================================================================================================

# Creating a network in a reproducable way:
random_seed = 42
random.seed(random_seed)

number_of_nodes = 5
network = Boolean_Network(number_of_nodes)
network.print_network()

# Under construction!
# For now they do nothing:
# filename = 'Network.txt'
# network.save_network(filename)
# network.load_network(filename) # overwrites the current network

#======================================================================================================================================================

# Creating a detaset in a reproducible way:
random_seed = 42
random.seed(random_seed)

number_of_datapoints = 5
length_of_one_trajectory = 5
synchronous = False
sampling_frequency = 1
starting_states = [] # if left empty, the starting states will be randomized for each datapoint; uncomment the next line for always starting at 0
# starting_states = [0]*number_of_datapoints
# If you want to specify some of the starting states, you have to write a list of length `number_of_datapoints` of ints and Nones,
# where an integer value is the index of the state (in reverse binary), and None is "take a random starting state"
dataset = network.create_dataset(number_of_datapoints, length_of_one_trajectory, synchronous, sampling_frequency, starting_states)

#======================================================================================================================================================

# Printing and saving the dataset
as_binary = True #False is more concise, but is harder to understand where the resulting trajectory came from
network.print_dataset(dataset, as_binary)

filename = 'output.txt'
network.save_dataset(filename, dataset) #saved in the format required for BNFinder2