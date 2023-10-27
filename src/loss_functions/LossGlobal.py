import torch.nn as nn
import torch

class LossGlobal(nn.Module):
    def __init__(self):
        pass

    def forward(self, adj, x, b_mat):

        loss = torch.sum(((adj - x) * b_mat) * ((adj - x) * b_mat))

        return loss
        