from baselines.sampling_methods.kcenter_greedy import kCenterGreedy
import numpy as np

class CoreSetQuery:
    def __init__(self, *_, **__):
        self.q = kCenterGreedy()

    def selectOneNode(self, output, selected, **_):
        ret = self.q.select_batch(output, selected, 1)
        return ret[0]
