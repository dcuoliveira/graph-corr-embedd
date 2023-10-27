import torch.nn as nn
import torch

class LossLocal(nn.Module):
    def __init__(self):
        pass

    def forward(self, z, adj_mat):

        z_norm = torch.sum(embedding * embedding, dim=1, keepdim=True) 

        loss = torch.sum(adj_mat * (z_norm - 2 * torch.mm(z, torch.transpose(z, dim0=0, dim1=1)) + torch.transpose(z_norm, dim0=0, dim1=1)))

        return loss
        