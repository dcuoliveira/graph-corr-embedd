
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from sklearn.cluster import KMeans

class SAE(nn.Module):
    """
    Sparse Autoencoder (SAE) neural network model.

    Args:
        layers (list): List of integers representing the number of units in each layer.

    Attributes:
        layers (nn.Sequential): Sequential container for the layers of the SAE.
        outputs (dict): Dictionary to store the output activations of specific layers.

    Methods:
        get_activation: Hook function to retrieve the activation output of a specific layer.
        forward: Forward pass of the SAE.
        layer_activations: Computes the average activation of a specific layer.
        sparse_result: Computes the sparse regularization term for a specific layer.
        kl_div: Computes the Kullback-Leibler divergence for the sparse regularization.
        loss: Computes the total loss function of the SAE.
        get_embedding: Retrieves the embedding representation from the SAE.
    """

    def __init__(self, layers):
        super(SAE, self).__init__()

        self.layers = nn.Sequential(OrderedDict({
            'lin1': nn.Linear(layers[0], layers[1]),
            'sig1': nn.Sigmoid(),
            'lin2': nn.Linear(layers[1], layers[2]),
            'sig2': nn.Sigmoid(),
            'lin3': nn.Linear(layers[2], layers[3]),
            'sig3': nn.Sigmoid(),
            'lin4': nn.Linear(layers[3], layers[4]),
            'sig4': nn.Sigmoid(),
            }))

        self.outputs = {}

        self.layers[0].register_forward_hook(self.get_activation('lin1'))
        self.layers[2].register_forward_hook(self.get_activation('lin2'))
        self.layers[4].register_forward_hook(self.get_activation('lin3'))
    
    def get_activation(self, name):
        def hook(module, input, output):
            self.outputs[name] = output
        return hook

    def forward(self, x):
        """
        Forward pass of the SAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        output = self.layers(x)
        return output

    def layer_activations(self, layername):
        """
        Computes the average activation of a specific layer.

        Args:
            layername (str): Name of the layer.

        Returns:
            torch.Tensor: Average activation tensor.
        """
        return torch.mean(torch.sigmoid(self.outputs[layername]), dim=0)

    def sparse_result(self, rho, layername):
        """
        Computes the sparse regularization term for a specific layer.

        Args:
            rho (float): Sparsity parameter.
            layername (str): Name of the layer.

        Returns:
            torch.Tensor: Sparse regularization term.
        """
        rho_hat = self.layer_activations(layername)
        return rho * np.log(rho) - rho * torch.log(rho_hat) + (1 - rho) * np.log(1 - rho) \
                - (1 - rho) * torch.log(1 - rho_hat)

    def kl_div(self, rho):
        """
        Computes the Kullback-Leibler divergence for the sparse regularization.

        Args:
            rho (float): Sparsity parameter.

        Returns:
            torch.Tensor: KL divergence.
        """
        first = torch.mean(self.sparse_result(rho, 'lin1'))
        second = torch.mean(self.sparse_result(rho, 'lin2'))
        return first + second

    def loss(self, x_hat, x, beta, rho):
        """
        Computes the total loss function of the SAE.

        Args:
            x_hat (torch.Tensor): Reconstructed input tensor.
            x (torch.Tensor): Original input tensor.
            beta (float): Regularization parameter.
            rho (float): Sparsity parameter.

        Returns:
            torch.Tensor: Total loss.
        """
        loss = F.mse_loss(x_hat, x) + beta * self.kl_div(rho)
        return loss

    def get_embedding(self, linname):
        """
        Retrieves the embedding representation from the SAE.

        Returns:
            numpy.ndarray: Embedding representation.
        """
        return self.outputs[linname].detach().cpu().numpy()


if __name__ == "__main__":
    pass