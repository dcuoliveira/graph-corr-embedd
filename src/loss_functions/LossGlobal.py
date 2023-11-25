import torch.nn as nn
import torch

class LossGlobal(nn.Module):
    def __init__(self):
        pass

    def forward(self, adj: torch.tensor, x: torch.tensor, b_mat: torch.tensor):
        """"
        Compute global loss, or adjusted reconstructions loss. 
        This loss function is used to preserve the the second-order proximity structure of the network.
        Furthermore, it also takes into account the sparsity of the network by penalizing the 
        reconstruction of non-existing edges.
        
        :param adj: Adjacency matrix
        :param x: Output of encoder
        :param b_mat: Weight matrix
        
        :return: Loss


        Daixin Wang, Peng Cui, and Wenwu Zhu (2016). Structural deep network embedding. SIGKDD.
        """

        loss = torch.sum(((adj - x) * b_mat) * ((adj - x) * b_mat))

        return loss
        