import torch
import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh

class LaplacianEncoding:
    def __init__(self):
        pass

    def forward(self, adj_matrix: torch.Tensor, l: int=3):
        """
        Compute Laplacian (positional) encoding for nodes in a graph given its adjacency matrix.

        Parameters:
        adj_matrix (torch.Tensor): Adjacency matrix of the graph (n x n)
        l (int): Number of principal eigenvectors to use in the encoding

        Returns:
        torch.Tensor: Laplacian encoding of the nodes (n x l)
        """
        # Ensure the adjacency matrix is square
        assert adj_matrix.size(0) == adj_matrix.size(1), "Adjacency matrix must be square"

        # Convert adjacency matrix to numpy array for compatibility with scipy
        adj_matrix_np = adj_matrix.numpy()

        # Compute the degree matrix
        degree_matrix = np.diag(adj_matrix_np.sum(axis=1))

        # Compute the normalized Laplacian matrix
        d_root_inv = np.diag(1.0 / np.sqrt(adj_matrix_np.sum(axis=1)))
        laplacian = np.eye(adj_matrix_np.shape[0]) - d_root_inv @ adj_matrix_np @ d_root_inv

        # Compute the eigenvalues and eigenvectors of the Laplacian matrix
        eigvals, eigvecs = eigh(laplacian)
        
        # Select the first l eigenvectors (corresponding to the smallest eigenvalues)
        encoding = eigvecs[:, :l]
        
        return torch.tensor(encoding, dtype=torch.float)