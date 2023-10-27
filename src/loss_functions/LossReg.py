import torch.nn as nn
import torch

class LossReg(nn.Module):
    def __init__(self):
        pass

    def forward(self, nu1, nu2, model):

        loss = 0
        for param in model.parameters():
            L_reg += nu1 * torch.sum(torch.abs(param)) + nu2 * torch.sum(param * param)

        return loss