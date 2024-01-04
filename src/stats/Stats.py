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