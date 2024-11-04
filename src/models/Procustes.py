import numpy as np
import torch.nn as nn
from scipy.spatial import procrustes
import torch

from stats.Stats import Stats

class Procustes(Stats):
    def __init__(self):
        super(Procustes, self).__init__()

    def compute_laplacian(self, adj_matrix):
        """
        Computes the symmetric normalized Laplacian of a graph from its adjacency matrix.

        Parameters:
        adj_matrix (numpy.ndarray): The adjacency matrix of the graph.

        Returns:
        numpy.ndarray: The Laplacian matrix.
        """

        # Ensure adj_matrix is on the correct device
        adj_matrix = adj_matrix.to(torch.float32)

        # Degree matrix as diagonal
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))

        # D^(-1/2)
        d_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(adj_matrix, dim=1)))

        # Symmetric normalized Laplacian L = I - D^(-1/2) * A * D^(-1/2)
        identity = torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        laplacian = identity - d_inv_sqrt @ adj_matrix @ d_inv_sqrt
        return laplacian

    def top_k_eigenvectors(self, matrix, k):
        """
        Computes the top k principal eigenvectors of a matrix.

        Parameters:
        matrix (numpy.ndarray): The matrix to decompose.

        Returns:
        numpy.ndarray: The matrix of top k eigenvectors.
        """
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        
        # Sort eigenvectors based on eigenvalues in descending order
        idx = torch.argsort(eigenvalues, descending=True)[:k]
        return eigenvectors[:, idx]

    def forward(self, adj1, adj2, k):
        """
        Computes the Procrustes distance between the top k eigenvectors of two graph Laplacians.

        Parameters:
        adj1 (numpy.ndarray): Adjacency matrix of the first graph.
        adj2 (numpy.ndarray): Adjacency matrix of the second graph.

        Returns:
        float: The Procrustes distance.
        """
        # Compute Laplacians
        laplacian1 = self.compute_laplacian(adj1)
        laplacian2 = self.compute_laplacian(adj2)

        # Get top k eigenvectors
        eigvecs1 = self.top_k_eigenvectors(laplacian1, k)
        eigvecs2 = self.top_k_eigenvectors(laplacian2, k)

        # Perform Procrustes alignment
        _, _, fro_norm = procrustes(eigvecs1, eigvecs2)
        
        return fro_norm


if __name__ == "__main__":
    pass