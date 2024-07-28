import torch
from scipy.stats import spearmanr

class Stats:
    def __init__(self):
        pass

    def compute_spearman_rank_correlation_tensor(self, x, y):
        """
        Compute the Spearman rank correlation between two tensors.

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

        # Rank the data
        def rankdata(data):
            sorted_data, indices = torch.sort(data)
            ranks = torch.zeros_like(data)
            ranks[indices] = torch.arange(1, len(data) + 1, device=data.device, dtype=data.dtype)
            return ranks

        x_rank = rankdata(x)
        y_rank = rankdata(y)

        def pearsonr(x, y):
            mean_x = torch.mean(x)
            mean_y = torch.mean(y)
            xm = x - mean_x
            ym = y - mean_y
            r_num = torch.sum(xm * ym)
            r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
            r = r_num / r_den
            return r

        correlation = pearsonr(x_rank, y_rank)
        return correlation.item()

    def compute_spearman_rank_correlation(self, x, y):
        """
        Computer the Spearman rank correlation between two tensors.

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
        # Move tensors to CPU before converting to NumPy arrays
        x_cpu = x.cpu().numpy()
        y_cpu = y.cpu().numpy()

        correlation = spearmanr(x_cpu, y_cpu)

        return correlation.correlation

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