# coding: utf-8
import numpy as np
# Modified implementation for Markov Chain based classifier and MaMPF.
# The original codebase is in https://github.com/WSPTTH/MaMPF/blob/master/Markov/markov.py

class Markov(object):
    def __init__(self, data, order=1):
        """
        compute the markov matrix
        :param data: input sequence,a nested list as follows
        [ [1,2,3], [2,3,1,1,1], [1,2] ]
        
        :param order: the order of the markov matrix, only support 1 and 2, and 1 is the default
        """
        self.startLabel = 'ss'
        self.endLabel = 'ee'
        if order not in [1, 2]:
            raise ValueError

        self.order = order
        if data is None:
            return

        self.keys = None
        self.fit(data)

    def _getKeys(self, data):
        self.keys = {}
        count = 0
        for seq in data:
            for state in seq:
                if state not in self.keys:
                    self.keys[state] = count
                    count += 1
        self.keys[self.startLabel] = count
        self.keys[self.endLabel] = count + 1
        return None

    def _fitOneOrderMarkov(self, data):
        self.P1 = np.zeros((len(self.keys), len(self.keys)))
        for seq in data:
            self.P1[self.keys[self.startLabel]][self.keys[seq[0]]] += 1
            self.P1[self.keys[seq[-1]]][self.keys[self.endLabel]] += 1
            for ix in range(0, len(seq) - 1):
                self.P1[self.keys[seq[ix]]][self.keys[seq[ix + 1]]] += 1

        for ix in range(len(self.P1)):
            sumx = np.sum(self.P1[ix])
            if sumx != 0:
                self.P1[ix] /= sumx

        return None

    def _predictOneOrder(self, seq):
        p = self.P1[self.keys[self.startLabel]][self.keys[seq[0]]]
        p *= self.P1[self.keys[seq[-1]]][self.keys[self.endLabel]]
        for ix in range(0, len(seq) - 1):
            p *= self.P1[self.keys[seq[ix]]][self.keys[seq[ix + 1]]]
        return p

    def fit(self, data):
        self._getKeys(data)
        if self.order == 1:
            self.P1 = None
            self._fitOneOrderMarkov(data)

    def predict(self, seq):
        if len(seq) == 0:  # empty sequence
            return 0
        for ix in seq:
            if ix not in self.keys:  # unknown status
                return 0

        if self.order == 1:
            return self._predictOneOrder(seq)


class LengthMarkov(object):

    def __init__(self, data, order=1, prob=0.9):
        self.order = order
        self.prob = prob

        self.keysLength = None
        self.keysLengthMap = None
        self.data = None
        self.markov = None
        if data is not None:
            self.fit(data)

    def _getKeyLength(self, data):
        if self.keysLength is not None:
            return None
        lenFeq = {}
        for seq in data:
            for lx in seq:
                if lx not in lenFeq:
                    lenFeq[lx] = 1
                else:
                    lenFeq[lx] += 1
        lenFeq = sorted(list(lenFeq.items()), key=lambda x: -x[1])
        thre = np.sum([xx[1] for xx in lenFeq]) * self.prob
        cum = 0
        keys = []
        for num in lenFeq:
            cum += num[1]
            if cum < thre:
                keys.append(num[0])
        if len(keys) == 0:
            keys.append(lenFeq[0][0])
        self.keysLength = np.array(sorted(keys))

    def _setKeyLengthMap(self):
        n = self.keysLength[-1]
        if len(self.keysLength) == 1:
            self.keysLengthMap = np.ones(n + 1)

        keyMap = np.ones(n + 1) * n
        now = 0
        for nx in range(n + 1):
            if now + 1 < len(self.keysLength) and abs(nx - self.keysLength[now]) > abs(nx - self.keysLength[now + 1]):
                now += 1
            keyMap[nx] = self.keysLength[now]
        self.keysLengthMap = keyMap

    def _searchKey(self, num):
        if self.keysLengthMap is None:
            self._setKeyLengthMap()
        if num > self.keysLength[-1]:
            return self.keysLength[-1]
        return self.keysLengthMap[num]

    def seqExchange(self, seq):
        return [self._searchKey(lx) for lx in seq]

    def dataExchange(self, data):
        self._getKeyLength(data)
        return [self.seqExchange(seq) for seq in data]

    def fit(self, data):
        self.data = self.dataExchange(data)
        self.markov = Markov(self.data, self.order)

    def predict(self, seq):
        if len(seq) == 0:
            return 0
        return self.markov.predict(self.seqExchange(seq))


if __name__ == '__main__':
    # a = [[1, 2], [1, 1], [2, 2], [2, 1]]
    # mvfora = Markov(a, order=2)
    # b = [1, 2]
    # print(mvfora.predict(b))
    a = LengthMarkov([[1]])
