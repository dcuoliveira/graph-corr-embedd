import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class SDNE(nn.Module):
    def __init__(self, node_size, n_hidden, droput):
        super(SDNE, self).__init__()
        self.encode = nn.Linear(node_size, n_hidden)
        self.decode = nn.Linear(n_hidden, node_size)
        self.droput = droput

    def loss_regularization(self):
        pass

    def forward(self, adj_batch):

        # encoder
        z = F.leaky_relu(self.encode(adj_batch))
                
        # decode
        x = F.leaky_relu(self.decode(z))

        # normalize embeddings
        z_norm = torch.sum(z * z, dim=1, keepdim=True)

        return x, z, z_norm


if __name__ == "__main__":
    pass