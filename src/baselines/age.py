## this is the code borrowed from AGE's public code

import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import scipy as sc

from baselines.utils import centralissimo, perc, percd, multiclassentropy_numpy

class AGEQuery(object):
    def __init__(self, G, multilabel, num_classes, method, basef=0.95, **_):
        self.G = G
        self.normcen = centralissimo(self.G, method)
        self.cenperc = perc(self.normcen)
        self.basef = basef
        self.multilabel = multilabel
        self.NCL = num_classes

    def __call__(self, outputs, pool, epoch):
        ret = []
        for id, row in enumerate(pool):
            selected = self.selectOneNode(outputs[id], row, epoch)
            ret.append(selected)
        ret = np.array(ret)  # .reshape(-1,1)
        return ret
    
    def getScores(self, output, alpha=0.25, beta=0.25, gamma=0.5):
        if self.multilabel:
            probs = 1.0 / (1.0 + np.exp(-output))
            entropy = multiclassentropy_numpy(probs)
        else:
            entropy = sc.stats.entropy(output.transpose())
        assert type(entropy) == np.ndarray, "entropy type {}".format(type(entropy))

        entrperc = perc(entropy)
        kmeans = KMeans(n_clusters=self.NCL, random_state=0, n_init="auto").fit(output)
        ed = euclidean_distances(output, kmeans.cluster_centers_)
        ed_score = np.min(ed, axis=1)  # the larger ed_score is, the far that node is away from cluster
        # centers, the less representativeness the node is

        edprec = percd(ed_score)
        finalweight = alpha * entrperc + beta * edprec + gamma * self.cenperc
        return finalweight

    def selectOneNode(self, output, pool, epoch, **_):
        gamma = np.random.beta(1, 1.005 - self.basef**epoch)
        alpha = beta = (1 - gamma) / 2
        finalweight = self.getScores(output, alpha, beta, gamma)
        finalweight = finalweight[pool]
        select = pool[np.argmax(finalweight)]
        return select


class EntropyQuery(object):
    def __init__(self, G, multilabel, num_classes, **_):
        self.G = G
        self.multilabel = multilabel
        self.NCL = num_classes

    def __call__(self, outputs, pool, epoch):
        ret = []
        for id, row in enumerate(pool):
            selected = self.selectOneNode(outputs[id], row, epoch)
            ret.append(selected)
        ret = np.array(ret)  # .reshape(-1,1)
        return ret

    def selectOneNode(self, output, pool, **_):
        if self.multilabel:
            probs = 1.0 / (1.0 + np.exp(-output))
            entropy = multiclassentropy_numpy(probs)
        else:
            entropy = sc.stats.entropy(output.transpose())
        assert type(entropy) == np.ndarray, "entropy type {}".format(type(entropy))

        finalweight = perc(entropy)
        finalweight = finalweight[pool]
        select = pool[np.argmax(finalweight)]
        return select


class CentralityQuery(object):
    def __init__(self, G, multilabel, num_classes, method, basef=0.95, **_):
        self.G = G
        self.normcen = centralissimo(self.G, method)
        self.cenperc = perc(self.normcen)
        self.basef = basef
        self.multilabel = multilabel
        self.NCL = num_classes

    def __call__(self, outputs, pool, epoch):
        ret = []
        for id, row in enumerate(pool):
            selected = self.selectOneNode(outputs[id], row, epoch)
            ret.append(selected)
        ret = np.array(ret)  # .reshape(-1,1)
        return ret

    def selectOneNode(self, pool, **_):
        finalweight = self.cenperc
        finalweight = finalweight[pool]
        select = pool[np.argmax(finalweight)]
        return select


class EdgeQuery(object):
    def __init__(self, G, method, **_):
        self.G = G
        self.adj = self.G.adj.cpu().to_dense().numpy()
        self.adjidx = np.where(self.adj > 0)
        self.adj += np.eye(self.adj.shape[0])
        self.normcen = centralissimo(self.G, method)
        self.cenperc = perc(self.normcen)
        self.basef = 0.95
        self.multilabel = self.G.stat["multilabel"]
        self.NCL = self.G.stat["nclass"]

    def __call__(self, outputs, pool, epoch):
        ret = []
        for id, row in enumerate(pool):
            selected = self.selectOneNode(outputs[id], row, epoch)
            ret.append(selected)
        ret = np.array(ret)  # .reshape(-1,1)
        return ret

    def selectOneNode(self, output, pool, **_):

        if self.multilabel:
            probs = 1.0 / (1.0 + np.exp(-output))
            entropy = multiclassentropy_numpy(probs)
        else:
            entropy = sc.stats.entropy(output.transpose())
        assert type(entropy) == np.ndarray, "entropy type {}".format(type(entropy))

        entropy /= np.log(float(self.G.stat["nclass"]))
        row, col = self.adjidx

        N = entropy.shape[0]

        b = entropy[row]
        c = entropy[col]
        d = np.vstack([b, c])

        weight = np.array([0.8, 0.2]).transpose()

        e = np.matmul(weight, d)
        eta = 1.5
        e = eta * (e - 0.5) + 0.5

        f = sp.csr_matrix((e, (row, col)), shape=(N, N))

        g = np.asarray(np.sum(f, axis=1))
        g = np.squeeze(g, axis=1)
        # entrperc = perc(entropy)
        # kmeans = KMeans(n_clusters=self.NCL, random_state=0, n_init='auto').fit(output)
        # ed = euclidean_distances(output, kmeans.cluster_centers_)
        # ed_score = np.min(ed,axis=1)  # the larger ed_score is, the far that node is away from cluster
        #                               # centers, the less representativeness the node is

        finalweight = perc(g)
        # finalweight = alpha * entrperc + beta * edprec + gamma * self.cenperc

        finalweight = finalweight[pool]
        select = pool[np.argmax(finalweight)]
        return select
