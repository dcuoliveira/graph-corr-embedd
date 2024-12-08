import numpy as np
import torch.nn as nn
from scipy.stats import spearmanr

from src.stats.Stats import Stats

class Frobenius(Stats):
    def __init__(self):
        super(Frobenius, self).__init__()

    def forward(self, adj1, adj2):
        adj_dis = adj1 - adj2
        fro_norm = np.linalg.norm(adj_dis, ord="fro")

        return fro_norm

if __name__ == "__main__":
    pass