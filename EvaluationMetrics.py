import numpy as np
import sys
from BooleanNetwork import BooleanNetwork

"""
Evaluation metrics for Boolean network reconstruction.

All functions assume:
- Input matrices are square adjacency matrices of directed graphs.
- mat[i][j] = 1 means there is a directed edge from node i to node j.
- Ground truth matrix is the reference, predicted matrix is what we want to evaluate.
"""

def _check_matrices(A, B):
    """
    Check that A and B have the same shape and are square matrices.
    """
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same dimensions")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrices must be square")
    return A, B

def hamming_distance(A, B):
    """
    Compute the Hamming distance between two adjacency matrices.
    Counts the number of differing edges.

    :param A: ground truth adjacency matrix
    :param B: predicted adjacency matrix
    :return: number of differing edges
    """
    A, B = _check_matrices(A, B)
    return np.sum(A != B)

def precision(A, B):
    """
    Compute precision = TP / (TP + FP)
    :param A: ground truth adjacency matrix
    :param B: predicted adjacency matrix
    :return: precision score
    """
    A, B = _check_matrices(A, B)
    TP = np.sum((A == 1) & (B == 1))
    FP = np.sum((A == 0) & (B == 1))
    return float(TP) / (TP + FP) if (TP + FP) > 0 else 0.0

def recall(A, B):
    """
    Compute recall = TP / (TP + FN)
    :param A: ground truth adjacency matrix
    :param B: predicted adjacency matrix
    :return: recall score
    """
    A, B = _check_matrices(A, B)
    TP = np.sum((A == 1) & (B == 1))
    FN = np.sum((A == 1) & (B == 0))
    return float(TP) / (TP + FN) if (TP + FN) > 0 else 0.0

def f1_score(A, B):
    """
    Compute F1 score = 2 * (precision * recall) / (precision + recall)
    :param A: ground truth adjacency matrix
    :param B: predicted adjacency matrix
    :return: F1 score
    """
    p = precision(A, B)
    r = recall(A, B)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def structural_hamming_distance(A, B):
    """
    Compute the Structural Hamming Distance (SHD) between two directed graphs.

    Graphs are represented by adjacency matrices:
        mat[i][j] = 1  means there is a directed edge from node i to node j
        mat[i][j] = 0  means no such edge exists

    SHD counts the minimum number of structural operations needed to transform
    graph A (ground truth) into graph B (inferred graph).

    Allowed operations (each counts as ONE):
        - edge addition
        - edge removal
        - edge reversal

    For each unordered pair of distinct nodes {i, j}, i != j, we compare
    the joint edge configuration:
        (A[i, j], A[j, i])  vs  (B[i, j], B[j, i])

    Possible configurations for i != j:
        (0, 0) : no edge
        (1, 0) : i -> j
        (0, 1) : j -> i
        (1, 1) : bidirectional edges

    Contribution to SHD for i != j
    (configuration in A -> configuration in B):

    1) A = (0, 0)
        - B = (0, 0) -> 0
        - B = (1, 0) -> 1  (edge addition)
        - B = (0, 1) -> 1  (edge addition)
        - B = (1, 1) -> 1  (edge addition)

    2) A = (1, 0)
        - B = (1, 0) -> 0
        - B = (0, 1) -> 1  (edge reversal)
        - B = (0, 0) -> 1  (edge removal)
        - B = (1, 1) -> 1  (edge addition)

    3) A = (0, 1)
        - B = (0, 1) -> 0
        - B = (1, 0) -> 1  (edge reversal)
        - B = (0, 0) -> 1  (edge removal)
        - B = (1, 1) -> 1  (edge addition)

    4) A = (1, 1)
        - B = (1, 1) -> 0
        - B = (1, 0) -> 1  (edge removal)
        - B = (0, 1) -> 1  (edge removal)
        - B = (0, 0) -> 1  (edge removal)

    For self-loops (i == j):
        - A[i, i] != B[i, i] -> 1
        - A[i, i] == B[i, i] -> 0

    :param A: ground truth adjacency matrix (square numpy array)
    :param B: inferred adjacency matrix (square numpy array)
    :return: structural Hamming distance (int)
    """
    A, B = _check_matrices(A, B)

    n = A.shape[0]
    shd = 0

    # Self-loops
    for i in range(n):
        if A[i, i] != B[i, i]:
            shd += 1

    # Unordered pairs of distinct nodes
    for i in range(n):
        for j in range(i + 1, n):
            a_ij, a_ji = A[i, j], A[j, i]
            b_ij, b_ji = B[i, j], B[j, i]

            if (a_ij, a_ji) != (b_ij, b_ji):
                shd += 1

    return shd

########################## Helper functions ###################################

def adjacency_matrix_from_boolean_network(network):
    """
    Convert a BooleanNetwork object into an adjacency matrix.

    The adjacency matrix represents a directed graph:
        mat[i][j] = 1  means there is a directed edge from node i to node j
        mat[i][j] = 0  otherwise

    Interpretation for BooleanNetwork:
    - If node j is listed as a parent of node i,
      then there is a directed edge j -> i in the graph.

    Self-loops (i -> i) are allowed and included if present.

    :param network: BooleanNetwork instance
    :return: adjacency matrix as a numpy array of shape (n, n)
    """
    n = network.number_of_nodes
    adj = np.zeros((n, n), dtype=int)

    for child_index, (parents, _) in enumerate(network.transitions):
        for parent_index in parents:
            adj[parent_index, child_index] = 1

    return adj

def adjacency_matrix_from_file(filename):
    """
    Create an adjacency matrix from a BooleanNetwork ground truth file.

    The file should be in the format produced by BooleanNetwork.save_network:
        n_nodes
        [parent]; [transition]

    The resulting adjacency matrix represents the network as a directed graph:
        mat[i][j] = 1  if there is a directed edge from node i to node j
        mat[i][j] = 0  otherwise

    Self-loops (edges i -> i) are included if present in the network file.

    :param filename: path to the BooleanNetwork ground truth file
    :return: adjacency matrix as a numpy array of shape (n_nodes, n_nodes)
    """
    network = BooleanNetwork.load_network(filename)
    adj_matrix = adjacency_matrix_from_boolean_network(network)

    return adj_matrix

def adjacency_matrix_from_sif(sif_file, ground_truth_file):
    """
    Create adjacency matrix from a SIF file using the number of nodes from the ground truth file.

    Notes:
    - Nodes in SIF are named xi (i=0,1,...).
    - Interactions must be '+' or '-'.
    - Self-loops are allowed if present in SIF.
    - This function by default includes all interactions in the SIF file.
      It is NOT intended for files containing posterior probabilities or weighted edges.

    :param sif_file: path to the SIF file
    :param ground_truth_file: path to ground truth BNF file (first line contains number of nodes)
    :return: adjacency matrix as a numpy array
    """
    # Determine number of nodes from ground truth
    with open(ground_truth_file, 'r') as f:
        n_nodes = int(f.readline().strip())

    adj = np.zeros((n_nodes, n_nodes), dtype=int)

    with open(sif_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # skip malformed lines
            source, interaction, target = parts
            # assert interaction type
            assert interaction in ('+', '-'), (
                "SIF file contains interaction other than '+' or '-'. "
                "This function assumes all interactions are present and does not handle probabilities."
            )
            # Convert node names xi to indices
            i = int(source[1:])
            j = int(target[1:])
            adj[i, j] = 1

    return adj

########################## Evaluation #################################

def main(ground_truth_file, test_file):
    """
    Main function to compute adjacency matrices and evaluation metrics.

    :param ground_truth_file: path to ground truth BooleanNetwork file
    :param test_file: path to test/inferred network file
    """
    
    gt_adj = adjacency_matrix_from_file(ground_truth_file)
    print("Ground truth adjacency matrix:")
    print(gt_adj)

    test_adj = adjacency_matrix_from_sif(test_file, ground_truth_file)
    print("Test adjacency matrix:")
    print(test_adj)

    hd = hamming_distance(gt_adj, test_adj)
    shd = structural_hamming_distance(gt_adj, test_adj)
    prec = precision(gt_adj, test_adj)
    rec = recall(gt_adj, test_adj)
    f1 = f1_score(gt_adj, test_adj)

    
    n_elements = gt_adj.size  # total number of entries in the adjacency matrix
    n = gt_adj.shape[0]
    
    # Total unordered pairs excluding self-loops
    num_pairs = n * (n - 1) // 2  
    # Self-loops separately
    num_self_loops = n
    total_shd_entries = num_pairs + num_self_loops
    
    # Hamming distance
    hd_fraction = float(hd) / n_elements
    print("Hamming distance: " + str(hd) + " / " + str(n_elements) + " (fraction: {:.6f})".format(hd_fraction))

    # Structural Hamming distance
    shd_fraction = float(shd) / total_shd_entries
    print("Structural Hamming distance: " + str(shd) + " / " + str(total_shd_entries) + " (fraction: {:.6f})".format(shd_fraction))

    # Precision, recall, F1
    print("Precision: {:.6f}".format(prec))
    print("Recall: {:.6f}".format(rec))
    print("F1 score: {:.6f}".format(f1))

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python EvaluationMetrics.py <ground_truth_file> <test_sif_file>")
        sys.exit(1)

    ground_truth_file = sys.argv[1]
    test_sif_file = sys.argv[2]

    main(ground_truth_file, test_sif_file)

