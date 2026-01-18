# -*- coding: utf-8 -*-
from __future__ import print_function
import random
from BooleanNetwork import BooleanNetwork
from StateSpaceAnalysisSynchronous import StateSpaceAnalyzer
from AttractorsAsynchronousNetworkx import AsyncAnalyzerNX
from itertools import product
import os

random.seed(1)

def collect_attractor_states(state_info, mode):
    attractor_states = set()
    for state, info in state_info.items():
        if mode == 'synchronous':
            if info.get("distance", 0) == 0:
                attractor_states.add(state)
        elif mode == 'asynchronous':
            if info.get("is_attractor", False):
                attractor_states.add(state)
        else:
            raise ValueError("Unknown mode: %s" % mode)
    return attractor_states

def generate_long_trajectories(network, state_info, max_length=1000, mode='synchronous'):
    """
    Generate a large set of long trajectories for a Boolean network.

    :param network: BooleanNetwork instance
    :param state_info: dict {state_int: {...}}
                       For synchronous: includes 'distance' and 'is_attractor'
                       For asynchronous: includes 'is_attractor'
    :param max_length: maximum length of trajectories to generate
    :param mode: 'synchronous' or 'asynchronous'
    :return: list of dicts, each with:
        {'trajectory': [state_int, ...],
         'transient_count': int,
         'attractor_count': int}
    """

    # -----------------------------
    # Prepare attractor info
    # -----------------------------
    attractor_states = collect_attractor_states(state_info, mode)

    all_states = list(state_info.keys())
    trajectories = []

    # -----------------------------
    # Generate trajectories
    # -----------------------------
    for start_state in all_states:
        traj = []
        current = start_state
        transient_count = 0
        attractor_count = 0

        for _ in range(max_length):
            traj.append(current)
            if current in attractor_states:
                attractor_count += 1
            else:
                transient_count += 1

            # use built-in function of BooleanNetwork
            current = network.compute_next_network_state(
                current, synchronous=(mode == 'synchronous')
            )

        trajectories.append({
            'trajectory': traj,
            'transient_count': transient_count,
            'attractor_count': attractor_count
        })

    print("Generated %d long trajectories" % len(trajectories))
    return trajectories

def sample_dataset_from_long_trajectories_unique(
    long_trajectories,
    state_info,
    mode,
    num_trajectories,
    trajectory_length,
    transient_fraction,
    sampling_frequency=1
):
    """
    Sample unique dataset fragments from pre-generated long trajectories.

    Ensures all returned fragments are unique. Returns None if not enough candidates
    or not enough unique fragments are available.

    :param long_trajectories: list of dicts, each with 'trajectory', 'transient_count', 'attractor_count'
    :param state_info: dict {state_int: {...}} from StateSpaceAnalyzer or AsyncAnalyzerNX
    :param mode: 'synchronous' or 'asynchronous'
    :param num_trajectories: number of trajectories to sample
    :param trajectory_length: length of each trajectory
    :param transient_fraction: desired fraction of transient states (0..1)
    :param sampling_frequency: sampling step between states in the fragment
    :return: list of sampled trajectories (lists of state_int) or None if not enough candidates
    """
    attractor_states = collect_attractor_states(state_info, mode)

    required_transient = int(round(transient_fraction * trajectory_length))
    required_attractor = trajectory_length - required_transient

    # -----------------------------
    # Build candidate fragments
    # -----------------------------
    candidates = []
    for traj_dict in long_trajectories:
        traj = traj_dict['trajectory']
        for offset in range(sampling_frequency):
            fragment = traj[offset::sampling_frequency]

            transient_count = sum(1 for s in fragment if s not in attractor_states)
            attractor_count = sum(1 for s in fragment if s in attractor_states)

            if transient_count >= required_transient and attractor_count >= required_attractor:
                candidates.append((traj, offset))

    if len(candidates) < num_trajectories:
        print("Not enough candidate fragments (requested {}, found {})".format(
            num_trajectories, len(candidates)
        ))
        return None

    # -----------------------------
    # Sample unique fragments
    # -----------------------------
    sampled_fragments_set = set()
    sampled_trajectories = []

    random.shuffle(candidates)

    for traj, offset in candidates:
        if len(sampled_trajectories) >= num_trajectories:
            break

        fragment = traj[offset::sampling_frequency]

        transient_positions = [i for i, s in enumerate(fragment) if s not in attractor_states]
        attractor_positions = [i for i, s in enumerate(fragment) if s in attractor_states]

        # last required_transient transient states, first required_attractor attractor states
        selected_transient_indices = transient_positions[-required_transient:]
        selected_attractor_indices = attractor_positions[:required_attractor]

        final_fragment = tuple(fragment[i] for i in selected_transient_indices + selected_attractor_indices)

        if final_fragment not in sampled_fragments_set:
            sampled_fragments_set.add(final_fragment)
            sampled_trajectories.append(list(final_fragment))

    # -----------------------------
    # Check if enough unique fragments were sampled
    # -----------------------------
    if len(sampled_trajectories) < num_trajectories:
        print("Not enough unique fragments (requested {}, found {})".format(
            num_trajectories, len(sampled_trajectories)
        ))
        return None

    return sampled_trajectories


def ensure_dirs():
    if not os.path.exists("ground_truth_networks"):
        os.makedirs("ground_truth_networks")
    if not os.path.exists("datasets"):
        os.makedirs("datasets")

def generate_and_sample_network(
    num_nodes,
    do_synchronous=True,
    do_asynchronous=True,
    max_length=100
):
    ensure_dirs()

    # -----------------------------
    # 1. Create ONE Boolean network
    # -----------------------------
    network = BooleanNetwork(num_nodes)
    print("Boolean network with %d nodes created." % num_nodes)

    network_filename = "ground_truth_networks/network_{}nodes.txt".format(num_nodes)
    network.save_network(network_filename)
    print("Network saved as {}".format(network_filename))

    # -----------------------------
    # 2. Run selected modes
    # -----------------------------
    for mode in ['synchronous', 'asynchronous']:

        if mode == 'synchronous' and not do_synchronous:
            continue
        if mode == 'asynchronous' and not do_asynchronous:
            continue

        print("\n--- Mode:", mode, "---")

        # Analyze state space
        if mode == 'synchronous':
            analyzer = StateSpaceAnalyzer(network)
        else:
            analyzer = AsyncAnalyzerNX(network)

        state_info = analyzer.analyze()
        print("{} state space analyzed.".format(mode.capitalize()))

        # Generate long trajectories
        long_trajectories = generate_long_trajectories(
            network=network,
            state_info=state_info,
            max_length=max_length,
            mode=mode
        )

        # Dataset parameters
        sampling_frequencies = [1, 2, 3]
        num_trajectories_list = [ i for i in range(3,40,3)]
        trajectory_lengths = [ i for i in range(3,40,3)]
        transient_fractions = [1.0/5, 2.0/5, 3.0/5, 4.0/5]

        for sample_frequency, n_traj, traj_len, transient_fraction in product(
            sampling_frequencies,
            num_trajectories_list,
            trajectory_lengths,
            transient_fractions
        ):
            sampled_dataset = sample_dataset_from_long_trajectories(
                long_trajectories,
                state_info,
                mode,
                n_traj,
                traj_len,
                transient_fraction,
                sample_frequency
            )

            if not sampled_dataset:
                print("Skipping dataset (insufficient fragments)")
                continue

            dataset_filename = (
                "datasets/dataset_{}nodes_{}_sf{}_n{}_len{}_tf{}.txt"
                .format(num_nodes, mode, sample_frequency, n_traj, traj_len, int(transient_fraction*100))
            )

            network.save_dataset(dataset_filename, [sampled_dataset, sample_frequency])
            print("Dataset saved as {}".format(dataset_filename))



if __name__ == "__main__":
    for num_nodes_in_network in range(5, 17):
        print("\n===============================================")
        print("Boolean network with {} nodes".format(num_nodes_in_network))
        print("===============================================")
        generate_and_sample_network(
            num_nodes=num_nodes_in_network,
            max_length=300
        )
