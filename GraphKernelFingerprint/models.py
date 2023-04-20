# Implementation of TLS SM-based classification model and other models for comparison.
# The original implementation for the two following models is in https://github.com/WSPTTH/MaMPF/blob/master/Markov/models.py

import numpy as np
import GraphKernel.Markov as MK

class SMarkovModel(object):

    def __init__(self, order=1):
        self.order = order
        self.markovs = None

    def fit(self, status):
        self.markovs = [MK.Markov(data, order=self.order) for data in status]

    def predict(self, status):
        res = []
        for dataset in status:
            res_dataset = []
            for seq in dataset:
                lab = np.argmax(np.array([e.predict(seq) for e in self.markovs]))
                res_dataset.append(lab)
            res.append(res_dataset)
        return res

    def predict_prob(self, status):
        res = []
        for dataset in status:
            res_dataset = []
            for seq in dataset:
                res_dataset.append(np.array([e.predict(seq) for e in self.markovs]))
            res.append(np.array(res_dataset))
        return res


class LMarkovModel(SMarkovModel):

    def __init__(self, order=1):
        SMarkovModel.__init__(self, order)

    def fit(self, _, length):
        self.markovs = [MK.LengthMarkov(data, order=self.order) for data in length]

    def predict(self, length):
        res = []
        for dataset in length:
            res_dataset = []
            for seq in dataset:
                lab = np.argmax(np.array([e.predict(seq) for e in self.markovs]))
                res_dataset.append(lab)
            res.append(res_dataset)
        return res


class SLMarkovModel(SMarkovModel):

    def __init__(self, order=1):
        SMarkovModel.__init__(self, order)
        self.smarkovs = None
        self.lmarkovs = None

    def fit(self, status, length):
        self.smarkovs = [MK.Markov(data, order=self.order) for data in status]
        self.lmarkovs = [MK.LengthMarkov(data, order=self.order) for data in length]

    def predict(self, status, length):
        res = []
        for sd, ld in zip(status, length):
            res_dataset = []
            for sseq, lseq in zip(sd, ld):
                sx = np.array([e.predict(sseq) for e in self.smarkovs])
                lx = np.array([e.predict(lseq) for e in self.lmarkovs])
                lab = np.argmax(sx * lx)
                res_dataset.append(lab)
            res.append(res_dataset)
        return res
