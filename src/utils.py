import itertools
from multiprocessing import Pool

import networkx as nx
import scipy.sparse as sp
import numpy as np


def JSD(prob_dists): # (node, trial, class_dist)
    num_classes = prob_dists.shape[-1]
    mean_dist = np.mean(prob_dists, axis=1) # (node, class_dist)
    left_term = -np.sum(mean_dist * np.log(mean_dist, out=np.zeros_like(mean_dist, dtype=mean_dist.dtype), where=(mean_dist != 0)), axis=-1) # (node,)
    right_term = np.mean(-np.sum(
        prob_dists * np.log(prob_dists, out=np.zeros_like(prob_dists, dtype=prob_dists.dtype), where=(prob_dists != 0)), axis=-1
    ), axis=-1) # (node,)
    return (left_term - right_term) / np.log(num_classes)

# calculate the percentage of elements larger than the k-th element
def percd(x): # [1,2,3] -> [1, 0.66, 0.33] Reversed!
    return 1 - np.argsort(np.argsort(x, kind='stable'), kind='stable') / len(x)

# calculate the percentage of elements smaller than the k-th element
def perc(x): # [1,2,3] -> [0, 0.33, 0.66] In order!
    return 1 - np.argsort(np.argsort(-x, kind='stable'), kind='stable') / len(x)


def chunks(l, n):
    """Divide a list of nodes `l` in `n` chunks"""
    l_c = iter(l)
    while 1:
        x = tuple(itertools.islice(l_c, n))
        if not x:
            return
        yield x

def betweenness_centrality_parallel(G, node_pool, processes=None):
    """Parallel betweenness centrality  function"""
    p = Pool(processes=processes)
    node_divisor = len(p._pool) * 4
    node_chunks = list(chunks(node_pool, len(node_pool) // node_divisor))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        zip(
            [G] * num_chunks,
            node_chunks,
            [list(G)] * num_chunks,
            [True] * num_chunks,
            [None] * num_chunks,
        ),
    )

    # Reduce the partial solutions
    bt_c = bt_sc[0]
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]
    return bt_c

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col))
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape
