import torch

class DegreeEncoding:
    def __init__(self):
        pass

    def compute_degree_encoding(self, adj_matrix: torch.Tensor):
        """
        Compute degree encoding for nodes in a graph given its adjacency matrix.

        Parameters:
        adj_matrix (torch.Tensor): Adjacency matrix of the graph (n x n)

        Returns:
        torch.Tensor: Degree encoding of the nodes (n x 1)
        """
        # Ensure the adjacency matrix is square
        assert adj_matrix.size(0) == adj_matrix.size(1), "Adjacency matrix must be square"

        # Compute the degree of each node (sum of each row in the adjacency matrix)
        degree = adj_matrix.sum(dim=1, keepdim=True)

        return degree
