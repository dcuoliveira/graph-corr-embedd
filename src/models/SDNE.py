import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class SDNE(nn.Module):
    def __init__(self,
                 node_size: int,
                 n_hidden: int,
                 n_layers_enc: int,
                 n_layers_dec: int,
                 bias_enc: bool,
                 bias_dec: bool,
                 droput: float):
        super(SDNE, self).__init__()

        # encoder
        self.encoder = nn.ModuleList([nn.Linear(node_size, n_hidden, bias=bias_enc) if i == 0
                                       else nn.Linear(n_hidden, n_hidden)
                                         for i in range(n_layers_enc)])

        # decoder
        self.decoder = nn.ModuleList([nn.Linear(n_hidden, n_hidden, bias=bias_dec) if i == 0
                                       else nn.Linear(n_hidden, n_hidden)
                                         for i in range(n_layers_dec)])

        # sparsity
        self.droput = droput

    def loss_regularization(self):
        pass

    def forward(self, adj_batch):

        # encoder
        z = F.leaky_relu(self.encoder(adj_batch))
                
        # decode
        x = F.leaky_relu(self.decoder(z))

        # normalize embeddings
        z_norm = torch.sum(z * z, dim=1, keepdim=True)

        return x, z, z_norm


if __name__ == "__main__":
    pass