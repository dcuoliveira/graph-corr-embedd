import torch.nn as nn
import torch

class LossDistance(nn.Module):
    def __init__(self):
        pass

    def forward(self, pred, true):
        
        loss_value = pred - true

        return loss_value