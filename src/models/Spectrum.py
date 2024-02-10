import numpy as np
import torch.nn as nn
from scipy.stats import spearmanr

from stats.Stats import Stats

class Spectrum(Stats):
    def __init__(self):
        super(Spectrum, self).__init__()

    def forward(self, adj):
        if adj.shape[0] == adj.shape[1]:
            eigenvalues = np.linalg.eigvalsh(adj)
        else:
            eigenvalues = np.linalg.eigvalsh(adj @ adj.T)

        max_eigenvalue = np.max(eigenvalues)

        return max_eigenvalue

if __name__ == "__main__":
    pass