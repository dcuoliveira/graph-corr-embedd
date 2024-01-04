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
        return eigenvalues

    def largest_eigenvalue(self, adj):
        eigenvalues = self.forward(adj)
        return np.max(eigenvalues)

    def compute_largest_eigenvalues(self, graphs):
        return [self.largest_eigenvalue(g) for g in graphs]

    def calculate_spearman_correlation(self, g1, g2):
        largest_eigenvalues_g1 = self.compute_largest_eigenvalues(g1)
        largest_eigenvalues_g2 = self.compute_largest_eigenvalues(g2)
        correlation, _ = spearmanr(largest_eigenvalues_g1, largest_eigenvalues_g2)
        return correlation


if __name__ == "__main__":
    pass