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

def sample_dataset_from_long_trajectories(
    long_trajectories,
    state_info,
    mode,
    num_trajectories,
    trajectory_length,
    transient_fraction,
    sampling_frequency=1
):
    """
    Sample dataset fragments from pre-generated long trajectories.

    Logic:
    - For each trajectory, try all offsets 0...sampling_frequency-1
    - Take the trajectory going in steps `sampling_frequency` from the offset to the end
    - Check if it contains enough transient and attractor states
    - If yes, candidate is accepted
    - Later, deterministically cut last required_transient transient states and first required_attractor attractor states

    :param long_trajectories: list of dicts, each with 'trajectory', 'transient_count', 'attractor_count'
    :param state_info: dict {state_int: {...}} from StateSpaceAnalyzer or AsyncAnalyzerNX
    :param mode: 'synchronous' or 'asynchronous'
    :param num_trajectories: number of trajectories to sample
    :param trajectory_length: length of each trajectory
    :param transient_fraction: desired fraction of transient states (0..1)
    :param sampling_frequency: sampling step between states in the fragment
    :return: list of sampled trajectories (lists of state_int)
    """

    # -----------------------------
    # Identify attractor states
    # -----------------------------
    attractor_states = collect_attractor_states(state_info, mode)

    required_transient = int(round(transient_fraction * trajectory_length))
    required_attractor = trajectory_length - required_transient

    # -----------------------------
    # Build candidate pairs (traj, offset)
    # -----------------------------
    candidates = []
    for traj_dict in long_trajectories:
        traj = traj_dict['trajectory']
        for offset in range(sampling_frequency):
            # Take fragment from offset to end with sampling frequency
            fragment = traj[offset::sampling_frequency]

            transient_count = sum(1 for s in fragment if s not in attractor_states)
            attractor_count = sum(1 for s in fragment if s in attractor_states)

            if transient_count >= required_transient and attractor_count >= required_attractor:
                candidates.append((traj, offset))

    if len(candidates) < num_trajectories:
        print("WARNING: Not enough candidate fragments available (requested {}, found {})".format(
            num_trajectories, len(candidates)
        ))
        return None

    # -----------------------------
    # Sample from candidates
    # -----------------------------
    sampled_pairs = random.sample(candidates, min(num_trajectories, len(candidates)))
    sampled_trajectories = []

    for traj, offset in sampled_pairs:
        fragment = traj[offset::sampling_frequency]

        # Positions of transient and attractor states in the fragment
        transient_positions = [i for i, s in enumerate(fragment) if s not in attractor_states]
        attractor_positions = [i for i, s in enumerate(fragment) if s in attractor_states]

        # Deterministically take last required_transient transient states
        selected_transient_indices = transient_positions[-required_transient:]
        # Deterministically take first required_attractor attractor states
        selected_attractor_indices = attractor_positions[:required_attractor]

        # Merge and sort by original order
        final_indices = sorted(selected_transient_indices + selected_attractor_indices)
        final_fragment = [fragment[i] for i in final_indices]

        sampled_trajectories.append(final_fragment)

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
        num_trajectories_list = [1, 2, 3]
        trajectory_lengths = [5, 10, 15, 20]
        transient_fractions = [2.0/5, 3.0/5, 4.0/5]

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
    for synchronicity_mode in ['synchronous', 'asynchronous']:
        print("\n===============================================")
        print("Mode:", synchronicity_mode)
        print("===============================================")
        for num_nodes_in_network in range(5, 17):
            print("\n--- Boolean network with {} nodes ---".format(num_nodes_in_network))
            generate_and_sample_network(num_nodes=num_nodes_in_network, max_length=100)
