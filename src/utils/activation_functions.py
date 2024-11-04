import numpy as np
import torch

def scaled_arctan(x):
    return ((2 / torch.pi) * torch.atan(torch.tensor(x)) - 1).item()

def arctan(x):
   return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def sigmoid(x):
 return 1/(1 + np.exp(-x))

def min_max_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def standardization(x):
    return (x - np.mean(x)) / np.std(x)

def sum_normalization(x):
    return x / np.sum(x)