import torch.nn as nn
import torch

class LossAbsDistance(nn.Module):
    def __init__(self):
        pass

    def forward(self, pred, true):
        
        loss_value = torch.abs(pred - true)

        return loss_value