import torch.nn as nn
import torch

class LossLocal(nn.Module):
    def __init__(self):
        pass

    def forward(self, adj, z):
        """"
        Compute local loss,.
        This loss function is used to preserve the the first-order proximity structure of the network.
        It does so by comparing the local similarity between nodes in the adjacency matrix and the corresponding
        similarity between the nodes in the embedding space. It penalizes the difference between the two.
        
        :param adj: Adjacency matrix
        :param z: Output of encoder
        
        :return: Loss


        Daixin Wang, Peng Cui, and Wenwu Zhu (2016). Structural deep network embedding. SIGKDD.
        """

        z_norm = torch.sum(z * z, dim=1, keepdim=True) 

        loss = torch.sum(adj * (z_norm - 2 * torch.mm(z, torch.transpose(z, dim0=0, dim1=1)) + torch.transpose(z_norm, dim0=0, dim1=1)))

        return loss
        