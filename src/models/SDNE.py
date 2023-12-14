import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class SDNE(nn.Module):
    """
    Implementation of the SDNE (Structural Deep Network Embedding) model.

    Args:
        node_size (int): Number of nodes in the graph.
        n_hidden (int): Number of hidden units in the encoder and decoder layers.
        n_layers_enc (int): Number of layers in the encoder.
        n_layers_dec (int): Number of layers in the decoder.
        bias_enc (bool): Whether to include bias in the encoder layers.
        bias_dec (bool): Whether to include bias in the decoder layers.
        droput (float): Dropout rate for regularization.

    Attributes:
        encoder (nn.Sequential): Encoder layers of the SDNE model.
        decoder (nn.Sequential): Decoder layers of the SDNE model.
        droput (float): Dropout rate for regularization.

    Methods:
        loss_regularization: Placeholder method for loss regularization.
        forward: Forward pass of the SDNE model.

    """

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
        encoder_layers = []
        for i in range(n_layers_enc):
            if i == 0:
                encoder_layers.append(nn.Linear(node_size, n_hidden, bias=bias_enc))
            else:
                encoder_layers.append(nn.Linear(n_hidden, n_hidden, bias=bias_enc))
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        decoder_layers = []
        for i in range(n_layers_dec):
            if i != (n_layers_dec-1):
                decoder_layers.append(nn.Linear(n_hidden, n_hidden, bias=bias_dec))
            else:
                decoder_layers.append(nn.Linear(n_hidden, node_size))
        self.decoder = nn.Sequential(*decoder_layers)

        # sparsity
        self.droput = droput

    def loss_regularization(self):
        """
        Placeholder method for loss regularization.
        """
        pass

    def forward(self, adj_batch):
        """
        Forward pass of the SDNE model.

        Args:
            adj_batch (torch.Tensor): Batch of adjacency matrices.

        Returns:
            x (torch.Tensor): Reconstructed adjacency matrices.
            z (torch.Tensor): Encoded node embeddings.
            z_norm (torch.Tensor): Normalized node embeddings.

        """
        # encoder
        z = F.leaky_relu(self.encoder(adj_batch))
                
        # decode
        x = F.leaky_relu(self.decoder(z))

        # normalize embeddings
        z_norm = torch.sum(z * z, dim=1, keepdim=True)

        return x, z, z_norm


if __name__ == "__main__":
    pass