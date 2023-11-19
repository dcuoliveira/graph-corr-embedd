import torch
import torch.nn as nn
import numpy as np

class Spectrum(nn.Module):
    def __init__(self):
        super(Spectrum, self).__init__()

    def forward(self, adj):

        if adj.shape[0] == adj.shape[1]:
            eigenvalues, eigenvectors = np.linalg.eig(adj)
        else:
            eigenvalues, eigenvectors = np.linalg.eig(adj @ adj.T)

        return eigenvalues, eigenvectors

