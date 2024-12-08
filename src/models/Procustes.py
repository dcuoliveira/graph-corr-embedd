import numpy as np
import torch.nn as nn
from scipy.spatial import procrustes
import torch

from src.stats.Stats import Stats

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

        # Convert to float32 and ensure it's on the correct device
        adj_matrix = adj_matrix.to(torch.float32)
        
        # Compute degree
        degrees = torch.sum(adj_matrix, dim=1)
        
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        degrees = degrees + epsilon
        
        # Compute D^(-1/2)
        d_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees))
        
        # Compute normalized Laplacian
        identity = torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        laplacian = identity - d_inv_sqrt @ adj_matrix @ d_inv_sqrt
        
        # Explicitly enforce symmetry by averaging with transpose
        laplacian = 0.5 * (laplacian + laplacian.t())
        
        # Ensure numerical stability
        laplacian = laplacian.to(torch.float32)
        
        # Verify symmetry
        is_symmetric = torch.allclose(laplacian, laplacian.t(), rtol=1e-5, atol=1e-8)
        if not is_symmetric:
            print("Warning: Laplacian is not perfectly symmetric after corrections")
    
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