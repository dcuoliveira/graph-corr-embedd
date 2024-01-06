import torch

class Stats:
    def __init__(self):
        pass

    def compute_spearman_correlation(self, x, y):
        """
        Computer the Spearman correlation between two tensors.

        Parameters
        ----------
        x: torch.Tensor
            First tensor
        y: torch.Tensor
            Second tensor

        Returns
        -------
        correlation: float
            Spearman correlation between x and y

        """

        correlation, _ = torch.spearmanr(x, y)
        return correlation

    def compute_eigenvalues(self, adj):
        """
        Computer the eigenvalues of a matrix.

        Parameters
        ----------
        adj: torch.Tensor
            Matrix

        Returns
        -------
        eigenvalues: torch.Tensor
            Eigenvalues of the matrix

        """

        if adj.shape[0] == adj.shape[1]:
            eigenvalues = torch.linalg.eigvalsh(adj)
        else:
            eigenvalues = torch.linalg.eigvalsh(adj @ adj.T)
            
        return eigenvalues