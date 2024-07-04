import numpy as np
import os
import math
import pickle
from scipy import sparse
import sys

import torch
from torch_geometric.utils import negative_sampling


EPS = 1e-7

def save_dict(path: str, filename: str, data: dict):
    """Save dictionary"""
    with open(f'{os.path.join(path, filename)}', 'wb') as f:
        pickle.dump(data, f)

def load_dict(path: str, filename: str) -> dict:
    """Load dictionary."""
    with open(f'{os.path.join(path, filename)}', 'rb') as f:
        data = pickle.load(f)
    return data

def check_exists(path: str, filename: str, force_run: bool = False):
    """Terminate program if file exists."""
    if not force_run and os.path.exists(os.path.join(path, filename)):
        sys.exit(f'File "{filename}" already exists.')

def adjacency_torch2sparse(torch_data):
    """Convert Adjacency matrix from torch format (Data) into Sparse format."""
    # nodes and edges
    rows = np.asarray(torch_data.edge_index[0])
    cols = np.asarray(torch_data.edge_index[1])
    data = np.ones(len(rows))
    n = torch_data.x.shape[0]
    adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    assert(adjacency.nnz == len(rows))
    assert(adjacency.shape[0] == n)
    return adjacency

def matrix_cosine(x: np.ndarray, y: np.ndarray):
    """Cosine similarity for each pairs of corresponding rows in matrices x and y.
        Source: https://stackoverflow.com/questions/49218285/cosine-similarity-between-matching-rows-in-numpy-ndarrays
    """
    return np.einsum('ij,ij->i', x, y) / (EPS + np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))

def sparse_matrix_cosine(x, y):
    """Cosine similarity between rows of two sparse matrices x and y."""
    num = x.multiply(y).dot(np.ones(x.shape[1]))
    denom = EPS + sparse.linalg.norm(x, axis=1) * sparse.linalg.norm(y, axis=1)
    return num / denom

def sigmoid(x):
    """Sigmoid"""
    return 1 / (1 + math.exp(-x))

def negative_sample(data, force_undirected: bool = False):
    """Negative sampling using torch function: randomly samples negative edges in the graph.
    Returns indexes and labels for all new edges, i.e. both positive and negative edges (positive edges
    are contained in the first half of arrays).
    """
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=round(data.edge_label_index.size(1)),
        method='sparse',
        force_undirected=force_undirected)
    
    # Creates a tensor of src and dst indexes for postive (first half) and negative (second half) edges in the graph.
    # Shape: [2, n_edges]
    edge_label_index = torch.cat(
        [data.edge_label_index, neg_edge_index],
        dim=-1,
    )

    # Creates a tensor of 1s for postive (first half) and 0s for negative edges in the graph.
    # Shape: [n_edges]
    edge_label = torch.cat([
        data.edge_label,
        data.edge_label.new_zeros(neg_edge_index.size(1))
    ], dim=0)

    return edge_label, edge_label_index

def find_index(arr: np.ndarray, n: int, K: float) -> int:
    """Find insert position of K in arr, using binary search (https://www.geeksforgeeks.org/search-insert-position-of-k-in-a-sorted-array/).
    
    Parameters
    ----------
        arr: Array to traverse. Values must be sorted.
        n: size of arr.
        K: value of element for which index is searched.

    Output
    ------
        Insert position (integer) of K in arr, so that arr remains sorted.
    """
    # Lower and upper bounds
    start = 0
    end = n - 1
 
    # Traverse the search space
    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == K:
            return mid
        elif arr[mid] >= K:
            start = mid + 1
        else:
            end = mid - 1
    # Return the insert position
    return end + 1