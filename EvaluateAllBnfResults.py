# -*- coding: utf-8 -*-
from __future__ import print_function
import os

from EvaluationMetrics import (
    compute_metrics,
    adjacency_matrix_from_file,
    adjacency_matrix_from_sif
)

# ================= CONFIG =================

GROUND_TRUTH_DIR = "ground_truth_networks"
RESULTS_DIR = "results"
TABLES_DIR = "tables"

SAMPLING_FREQUENCIES = [1, 2, 3]
N_TRAJECTORIES_LIST = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]
TRAJECTORY_LENGTHS = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]
TRANSIENT_FRACTIONS = [1.0 / 5, 2.0 / 5, 3.0 / 5, 4.0 / 5]

modes = ["synchronous", "asynchronous"]
scores = ["MDL", "BDE"]

# =========================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    ensure_dir(TABLES_DIR)

    for mode in modes:
        print("\n=======================================")
        print("MODE:", mode)
        print("=======================================")

        table_path = os.path.join(
            TABLES_DIR, "evaluation_{}.tsv".format(mode)
        )

        with open(table_path, "w") as out:
            out.write(
                "nodes\tmode\tscore\tsf\tntraj\tlen\ttf\t"
                "hd_frac\tshd_frac\tprecision\trecall\tf1\n"
            )

            for num_nodes in range(5, 17):

                gt_file = os.path.join(
                    GROUND_TRUTH_DIR,
                    "network_{}nodes.txt".format(num_nodes)
                )

                if not os.path.exists(gt_file):
                    print("Missing ground truth:", gt_file)
                    continue

                gt_adj = adjacency_matrix_from_file(gt_file)

                for sf in sampling_frequencies:
                    for ntraj in num_trajectories_list:
                        for tlen in trajectory_lengths:
                            for tf in transient_fractions:

                                tf_int = int(tf * 100)

                                for score in scores:
                                    result_file = os.path.join(
                                        RESULTS_DIR,
                                        "result_{}nodes_{}_sf{}_n{}_len{}_tf{}_{}.txt"
                                        .format(num_nodes, mode, sf, ntraj, tlen, tf_int, score)
                                    )

                                    if not os.path.exists(result_file):
                                        out.write(
                                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t"
                                            "NA\tNA\tNA\tNA\tNA\n"
                                            .format(num_nodes, mode, score, sf, ntraj, tlen, tf)
                                        )
                                        continue

                                    try:
                                        pred_adj = adjacency_matrix_from_sif(
                                            result_file, gt_file
                                        )

                                        hd_f, shd_f, p, r, f1 = compute_metrics(
                                            gt_adj, pred_adj
                                        )

                                        out.write(
                                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t"
                                            "{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n"
                                            .format(
                                                num_nodes, mode, score,
                                                sf, ntraj, tlen, tf,
                                                hd_f, shd_f, p, r, f1
                                            )
                                        )

                                    except Exception as e:
                                        print("Error processing:", result_file)
                                        print(e)
                                        out.write(
                                            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t"
                                            "NA\tNA\tNA\tNA\tNA\n"
                                            .format(num_nodes, mode, score, sf, ntraj, tlen, tf)
                                        )

        print("Saved:", table_path)

    print("\nAll evaluations completed.")

if __name__ == "__main__":
    main()

