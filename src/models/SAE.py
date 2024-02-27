import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stats.Stats import Stats
class StackedSparseAutoencoder(nn.Module, Stats):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list,
                 bias: bool = True,
                 dropout: float = 0.0,
                 sparsity_penalty: float = 1e-4):
        super(StackedSparseAutoencoder, self).__init__()

        # parameters
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.sparsity_penalty = sparsity_penalty

        # encoder
        encoder_layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_size, hidden_size, bias=bias))
            encoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder
        decoder_layers = []
        hidden_sizes.reverse()  # reverse the hidden sizes for symmetric decoder
        for hidden_size in hidden_sizes[:-1]:
            decoder_layers.append(nn.Linear(prev_size, hidden_size, bias=bias))
            decoder_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        # add the last layer without ReLU to reconstruct the input
        decoder_layers.append(nn.Linear(prev_size, input_size, bias=bias))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x, z

    def loss_function(self, recon_x, x):
        # Mean squared error for reconstruction loss
        recon_loss = F.mse_loss(recon_x, x)

        # L1 loss for sparsity penalty
        sparsity_loss = 0
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                sparsity_loss += torch.sum(torch.abs(layer.weight))

        total_loss = recon_loss + self.sparsity_penalty * sparsity_loss
        return total_loss