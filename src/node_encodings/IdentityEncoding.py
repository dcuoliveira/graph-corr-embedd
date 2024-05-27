import torch

class IdentityEncoding:
    def __init__(self):
        pass

    def compute_identity_encoding(self, adj_matrix: torch.Tensor):
        """
        Compute identity encoding (one-hot encoding) for nodes in a graph.

        Parameters:
        adj_matrix (torch.Tensor): Adjacency matrix of the graph (n x n)

        Returns:
        torch.Tensor: Identity encoding (one-hot encoding) of the nodes (n x n)
        """
        # Ensure the adjacency matrix is square
        assert adj_matrix.size(0) == adj_matrix.size(1), "Adjacency matrix must be square"

        num_nodes = adj_matrix.size(0)

        # Create an identity matrix of size (num_nodes x num_nodes)
        identity_matrix = torch.eye(num_nodes)

        return identity_matrix