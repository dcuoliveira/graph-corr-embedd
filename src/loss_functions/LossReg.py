import torch.nn as nn
import torch

class LossReg(nn.Module):
    def __init__(self):
        pass

    def forward(self, model):

        loss = 0
        for param in model.parameters():

            if len(param.shape) != 1:
                # loss += nu1 * torch.sum(torch.abs(param)) + nu2 * torch.sum(param * param)
                loss += 1/2 * (torch.linalg.norm(param, 'fro') + torch.linalg.norm(param * param, 'fro'))

        return loss