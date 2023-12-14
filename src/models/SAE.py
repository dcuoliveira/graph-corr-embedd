import torch
import torch.nn as nn

class SAE(nn.Module):
    def __init__(self, layers_sizes):
        super(SAE, self).__init__()
        self.autoencoders = nn.ModuleList()
        for i in range(len(layers_sizes) - 1):
            encoder = nn.Sequential(
                nn.Linear(layers_sizes[i], layers_sizes[i+1]),
                nn.Sigmoid()
            )
            decoder = nn.Sequential(
                nn.Linear(layers_sizes[i+1], layers_sizes[i]),
                nn.Sigmoid()
            )
            self.autoencoders.append(nn.ModuleDict({'encoder': encoder, 'decoder': decoder}))

    def forward(self, X):
        hidden_activations = []
        for autoencoder in self.autoencoders:
            X = autoencoder['encoder'](X)
            hidden_activations.append(X)
            X = autoencoder['decoder'](X)
        return X, hidden_activations

    def get_hidden_representation(self, X, layer_idx):
        for i, autoencoder in enumerate(self.autoencoders):
            X = autoencoder['encoder'](X)
            if i == layer_idx:
                return X

    def get_best_representation(self, X, criterion_func):
        best_representation = None
        best_score = None

        for i in range(len(self.autoencoders)):
            current_representation = self.get_hidden_representation(X, i)
            current_score = criterion_func(current_representation)

            if best_score is None or current_score < best_score:
                best_score = current_score
                best_representation = current_representation

        return best_representation
