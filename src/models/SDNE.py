import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stats.Stats import Stats

class SDNE(nn.Module, Stats):
    def __init__(self,
                 node_size: int,
                 n_hidden: int,
                 n_layers_enc: int,
                 n_layers_dec: int,
                 bias_enc: bool,
                 bias_dec: bool,
                 droput: float):
        super(SDNE, self).__init__()

        # parameters
        self.node_size = node_size
        self.n_hidden = n_hidden
        self.n_layers_dec = n_layers_dec
        self.n_layers_enc = n_layers_enc

        # encoder
        encoder_layers = []
        for i in range(n_layers_enc + 1):
            if i == 0:
                encoder_layers.append(nn.Linear(node_size, n_hidden // 2, bias=bias_enc))
            else:
                encoder_layers.append(nn.Linear(n_hidden // 2, n_hidden // 2, bias=bias_enc))
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        decoder_layers = []
        for i in range(n_layers_dec + 1):
            if (i == 0) and (n_layers_dec == 1):
                decoder_layers.append(nn.Linear(n_hidden // 2, node_size, bias=bias_dec))
            elif (i == 0) and (n_layers_dec > 1):
                decoder_layers.append(nn.Linear(n_hidden // 2, n_hidden, bias=bias_dec))
            elif (i > 0) and  (i < n_layers_dec) and (n_layers_dec > 1):
                decoder_layers.append(nn.Linear(n_hidden, n_hidden, bias=bias_dec))
            else:
                decoder_layers.append(nn.Linear(n_hidden, node_size))
        self.decoder = nn.Sequential(*decoder_layers)

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