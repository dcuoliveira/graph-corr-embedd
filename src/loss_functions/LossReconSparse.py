import torch.nn as nn
import torch
import torch.nn.functional as F

class LossReconSparse(nn.Module):
    def __init__(self):
        pass

    def forward(self, recon_x, x, model):
        # Mean squared error for reconstruction loss
        recon_loss = F.mse_loss(recon_x, x)

        # L1 loss for sparsity penalty
        sparsity_loss = 0
        for layer in model.encoder:
            if isinstance(layer, nn.Linear):
                sparsity_loss += torch.sum(torch.abs(layer.weight))

        total_loss = recon_loss + model.sparsity_penalty * sparsity_loss
        return total_loss