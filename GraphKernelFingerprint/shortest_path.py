from .StateMachine import StateMachine as SM
from .kernel import Kernel
from .algorithm import dirac_kernel, brownian_kernel
from itertools import product
import numpy as np

class ShortestPath(Kernel):
    def __init__(self, node_metric='dirac', edge_metric='dirac', normalizer=3.0):
        '''
        Implemention of shortest-path kernel, use specified similarity metrics.
        Params:
            node_metric: Metric measures similarity between node labels.
            edge_metric: Metric measures similarity between edge lengths.
        '''
        self.node_metric = dirac_kernel if node_metric == 'dirac' else brownian_kernel
        self.edge_metric = dirac_kernel if edge_metric == 'dirac' else brownian_kernel
        self.normalizer = normalizer

    def similarity(self, SM_1: SM, SM_2: SM, c=3.0):
        '''
        Performs similarity computation between to StateMachines.
        Args:
            SM_1, SM_2: SM,
            The StateMachine object on which we compute similarity.
            c: float,
            The normalizer of similarity, which is forced to lie between [0,1]
        Returns:
            k: float,
            The similarity between SM_1, SM_2
        '''
        sim = 0
        non_inf_1 = np.where(np.isfinite(SM_1.adj))
        non_inf_2 = np.where(np.isfinite(SM_2.adj))
        for i_1, j_1 in zip(list(non_inf_1[0]), list(non_inf_1[1])):
            u_1, v_1 = SM_1.vertices[i_1], SM_1.vertices[j_1]
            e_1 = SM_1.adj[i_1, j_1]
            label_u_1 = u_1.label
            label_v_1 = v_1.label
            for i_2, j_2 in zip(list(non_inf_2[0]), list(non_inf_2[1])):
                u_2, v_2 = SM_2.vertices[i_2], SM_2.vertices[j_2]
                e_2 = SM_2.adj[i_2,j_2]
                label_u_2 = u_2.label
                label_v_2 = v_2.label
                node_sim = (label_u_1 == label_u_2) and (label_v_1 == label_v_2) or \
                           (label_v_1 == label_u_2) and (label_u_1 == label_v_2)

                edge_sim = self.edge_metric(e_1, e_2, c)
                sim += float(node_sim) * edge_sim / c
        return sim
    
    def compute(self, SMs: list):
        '''
        Compute the Gram Matrix of a list of SMs.
        Args:
            SMs: list,
            A list of StateMachines.
        Returns:
            K: np.array,
            The Gram Matrix.
        '''
        K = np.zeros(shape=(len(SMs), len(SMs)))
        for i in range(len(SMs)):
            SM = SMs[i]
            for j in range(i+1, len(SMs)):
                K[i,j] = self.similarity(SM, SMs[j])
                K[j,i] = K[i,j]
        return K

    def transform(self, SM_test, SM_train):
        row = len(SM_test)
        col = len(SM_train)
        K = np.zeros(shape=(row, col))
        for i in range(len(SM_test)):
            SM_test = SM_test[i]
            for j in range(i + 1, len(SM_train)):
                K[i,j] = self.similarity(SM_test, SM_train[j])
                if j < row:
                    K[j,i] = K[i,j]
        return K
