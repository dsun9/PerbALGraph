# from src.baselines.sampling_methods.bandit_discrete import BanditDiscreteSampler
import numpy as np
import scipy as sc
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from baselines.utils import centralissimo, perc, percd

class ProbSampler(object):
    def __init__(self):
        pass

    def select_batch(self,wraped_feat):
        entropy = wraped_feat[0]
        selected = np.argmax(entropy)
        return selected

class DegSampler(object):
    def __init__(self):
        pass
    def select_batch(self,wraped_feat):
        deg = wraped_feat[1]
        selected = np.argmax(deg)
        # print("in DegSampler {}".format(np.max(deg)))
        return selected

class ClusterSampler(object):
    def __init__(self):
        pass
    def select_batch(self,wraped_feat):
        edscore = wraped_feat[2]
        selected = np.argmax(-edscore)
        # print("in DegSampler {}".format(np.max(deg)))
        return selected





    

class BanditDiscreteSampler(object):
    """Wraps EXP3 around mixtures of indicated methods.

    Uses EXP3 mult-armed bandit algorithm to select sampler methods.
    """
    def __init__(self,budget,num_nodes,n_arms,seed=123,
                reward_function = lambda AL_acc: AL_acc[-1],
                gamma=0.5):
                
        # self.name = 'bandit_discrete'
        # np.random.seed(seed)
        # self.seed = seed
        # self.initialize_samplers(samplers)
        # self.gamma = gamma

        # print(budget,num_nodes,n_arms)
        self.n_arms = n_arms
        self.reward_function = reward_function
        self.num_arm = float(self.n_arms)
        self.pmin=np.sqrt(np.log(self.num_arm) / (self.num_arm * budget))

        self.pull_history = []
        self.acc_history = []
        self.w = np.ones(self.n_arms)
        # self.x = np.zeros(self.n_arms)
        self.p = None
        # self.probs = []
        self.selectionhistory = []
        self.num_nodes = float(num_nodes)
        self.rewardlist = []
        self.phi = []
        self.budget = budget
        self.Q = []
        self.lastselected = []
        
    def select_batch(self, r, wraped_feat):
        
        # print(r)
        # self.acc_history.append(eval_acc)
        
        wraped_feat = np.exp(wraped_feat*20.) #make the prob sharper
        wraped_feat = wraped_feat/np.sum(wraped_feat,axis=-1,keepdims=True)
        if self.p is not  None :
            self.rewardlist.append(1.0 / (self.phi[self.lastselected] * float(wraped_feat.shape[-1])) * r)
            # print("rhat {} {} {}".format(self.rewardlist[-1],self.phi[self.lastselected] , float(wraped_feat.shape[-1])))
            reward = 1/float(self.budget)*sum(self.rewardlist)
            self.rhat = reward*self.Q[:, self.lastselected] / self.phi[self.lastselected]
            self.w = self.w*np.exp(1*0.5*self.pmin*(self.rhat+1/self.p*np.sqrt(np.log(self.num_nodes/0.1)/(self.num_arm*self.budget))))
        
        self.p = (1 - self.num_arm * self.pmin) * self.w / np.sum(self.w) + self.pmin
        # print(self.p,np.sum(self.p),self.pmin)
        self.Q = wraped_feat
        self.phi = np.matmul(self.p.reshape((1, -1)), wraped_feat).squeeze()

        t= 10
        # self.phi = np.exp(self.phi * 100.)
        # self.phi = self.phi/np.sum(self.phi)
        # print(self.phi.shape)

        
        selected = np.random.choice(range(wraped_feat.shape[-1]), p=self.phi)
        self.lastselected = selected
        
        # print(wraped_feat[:,selected])
        return selected
        
        
class AnrmabQuery:
    def __init__(self, G, num_classes,budget,num_nodes,method):
        self.G = G
        self.NCL =  num_classes
        self.normcen = centralissimo(self.G, method)
        self.cenperc = perc(self.normcen)
        self.q = BanditDiscreteSampler(budget=budget,num_nodes=num_nodes,n_arms=3)

    def selectOneNode(self, acc, output,pool):
        
        # if self.multilabel:
        #     probs = 1. / (1. + np.exp(-output))
        #     entropy = multiclassentropy_numpy(probs)
        # else:
        entropy = sc.stats.entropy(output.transpose())
        # print("entropy shape{}".format(entropy.shape))

        # validentropy = entropy[pool]
        # validdeg = self.normcen[pool]

        kmeans = KMeans(n_clusters=self.NCL, random_state=0).fit(output)
        ed = euclidean_distances(output, kmeans.cluster_centers_)
        ed_score = np.min(ed, axis=1)
        # valided_score = ed_score[pool]


        entrperc = perc(entropy)
        edprec = percd(ed_score)


        wraped_feat = np.stack([entrperc, edprec, self.cenperc])

        wraped_feat_valid = wraped_feat[:,pool]
        # print("warpedfaet {}".format(wraped_feat_valid.shape))
        selected = self.q.select_batch( acc, wraped_feat_valid)
        
        realselected = pool[selected]
        # print(realselected)
        # print(self.normcen[realselected])

        return realselected


def unitTest():
    q = CoreSetQuery(3)

    # pool = np.array([[2,3],[2,3],[4,6]])

    for i in range(3):
        features = np.random.randn(3, 10, 5)
        selected = q(features)
        print(selected)


if __name__ == "__main__":
    unitTest()


