import torch.nn as nn
import torch

class LossReg(nn.Module):
    def __init__(self):
        pass

    def forward(self, model):
        """"
        Regularization loss function.
        
        :param model: model to be regularized
        
        :return: 1/2 * \sum_i ||w_i||^2
        """

        loss = 0
        for param in model.parameters():

            if len(param.shape) != 1:
                loss += torch.linalg.norm(param, 'fro')

        return (1/2 * loss)