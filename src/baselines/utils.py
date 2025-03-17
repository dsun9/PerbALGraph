import numpy as np
import networkx as nx

# calculate the percentage of elements larger than the k-th element
def percd(input):
    return 1 - np.argsort(np.argsort(input, kind="stable"), kind="stable") / len(input)

# calculate the percentage of elements smaller than the k-th element
def perc(input):
    return 1 - np.argsort(np.argsort(-input, kind="stable"), kind="stable") / len(input)


def centralissimo(G, method="pagerank"):
    if method == "pagerank":
        centrality = nx.pagerank(G)
    elif method == "degree":
        centrality = nx.degree_centrality(G)
    elif method == "betweenness":
        centrality = nx.betweenness_centrality(G)
    elif method == "harmonic":
        centrality = nx.harmonic_centrality(G)
    else:
        raise ValueError("Invalid method")
    L = len(centrality)
    cenarray = np.zeros(L)
    cenarray[list(centrality.keys())] = list(centrality.values())
    normcen = (cenarray.astype(float) - np.min(cenarray)) / (np.max(cenarray) - np.min(cenarray))
    return normcen


def multiclassentropy_numpy(tens, dim=1):
    reverse = 1 - tens
    ent_1 = -np.log(np.clip(tens, a_min=1e-7, a_max=None)) * tens
    ent_2 = -np.log(np.clip(reverse, a_min=1e-7, a_max=None)) * reverse
    ent = ent_1 + ent_2
    entropy = np.mean(ent, axis=1)
    return entropy
