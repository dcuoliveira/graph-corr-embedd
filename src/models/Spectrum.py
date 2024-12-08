import numpy as np
import torch.nn as nn
from scipy.stats import spearmanr

from src.stats.Stats import Stats

class Spectrum(Stats):
    def __init__(self):
        super(Spectrum, self).__init__()

    def forward(self, adj, k=-1):
        if adj.shape[0] == adj.shape[1]:
            eigenvalues = np.linalg.eigvalsh(adj)
        else:
            eigenvalues = np.linalg.eigvalsh(adj @ adj.T)

        if k == -1:
            max_eigenvalue = np.max(eigenvalues)
        else:
            max_eigenvalue = np.sort(eigenvalues)[-k:]

        return max_eigenvalue

if __name__ == "__main__":
    pass