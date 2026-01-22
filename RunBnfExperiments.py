# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import subprocess

DATASETS_DIR = "datasets"
RESULTS_DIR = "results"

SAMPLING_FREQUENCIES = [1, 2, 3]
N_TRAJECTORIES_LIST = [3, 15, 27, 39]
TRAJECTORY_LENGTHS = [3, 15, 27, 39]
TRANSIENT_FRACTIONS = [1.0 / 5, 3.0 / 5]

modes = ["synchronous", "asynchronous"]
#modes = ["asynchronous"]
scores = ["MDL", "BDE"]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_bnf(dataset_path, result_network_path, score):
    cmd = [
        "bnf",
        "-e", dataset_path,   # INPUT dataset
        "-s", score,          # score
        "-n", result_network_path,
        "-g"
    ]

    print("Command:", " ".join(cmd))

    try:
        subprocess.check_call(cmd)
        print("Finished:", os.path.basename(result_network_path))
    except subprocess.CalledProcessError:
        print("ERROR running BNF on:", dataset_path)

def main():
    ensure_dir(RESULTS_DIR)

    for mode in modes:
        print("\n=======================================")
        print("MODE:", mode)
        print("=======================================")

        for num_nodes in range(14, 17):
            print("\n--- {} nodes ---".format(num_nodes))

            for sf in SAMPLING_FREQUENCIES:
                for ntraj in N_TRAJECTORIES_LIST:
                    for tlen in TRAJECTORY_LENGTHS:
                        for tf in TRANSIENT_FRACTIONS:

                            tf_int = int(tf * 100)

                            dataset_name = (
                                "dataset_{}nodes_{}_sf{}_n{}_len{}_tf{}.txt"
                                .format(num_nodes, mode, sf, ntraj, tlen, tf_int)
                            )
                            dataset_path = os.path.join(DATASETS_DIR, dataset_name)

                            if not os.path.exists(dataset_path):
                                print("Skipping (missing):", dataset_name)
                                continue

                            print("\nDataset:", dataset_name)

                            for score in scores:
                                result_name = (
                                    "result_{}nodes_{}_sf{}_n{}_len{}_tf{}_{}.txt"
                                    .format(num_nodes, mode, sf, ntraj, tlen, tf_int, score)
                                )
                                result_path = os.path.join(RESULTS_DIR, result_name)

                                run_bnf(dataset_path, result_path, score)

    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()

