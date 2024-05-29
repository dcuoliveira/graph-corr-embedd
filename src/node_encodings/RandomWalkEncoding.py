import torch

class RandomWalkEncoding:
    def __init__(self):
        pass

    def forward(self, adj_matrix: torch.Tensor, l: int=3):
        """
        Compute random-walk encoding for nodes in a graph given its adjacency matrix.

        Parameters:
        adj_matrix (torch.Tensor): Adjacency matrix of the graph (n x n)
        l (int): Number of dimensions in the encoding

        Returns:
        torch.Tensor: Random-walk encoding of the nodes (n x l)
        """
        # Ensure the adjacency matrix is square
        assert adj_matrix.size(0) == adj_matrix.size(1), "Adjacency matrix must be square"

        # Compute the degree matrix
        degree_matrix = torch.diag(adj_matrix.sum(dim=1))
        
        # Compute the inverse of the degree matrix
        degree_matrix_inv = torch.inverse(degree_matrix)
        
        # Compute the random-walk operator R
        random_walk_operator = torch.matmul(adj_matrix, degree_matrix_inv)
        
        # Initialize the encoding matrix
        n = adj_matrix.size(0)
        encoding = torch.zeros((n, l))
        
        # Compute the powers of R and fill the encoding matrix
        R_power = torch.eye(n)  # R^0 is the identity matrix
        for i in range(l):
            R_power = torch.matmul(R_power, random_walk_operator)  # R^i
            encoding[:, i] = R_power.diag()
        
        return encoding