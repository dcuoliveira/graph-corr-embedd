import numpy as np

def sigmoid(x):
 return 1/(1 + np.exp(-x))

def min_max_normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def standardization(x):
    return (x - np.mean(x)) / np.std(x)

def sum_normalization(x):
    return x / np.sum(x)