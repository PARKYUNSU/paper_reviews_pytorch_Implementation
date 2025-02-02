import torch
import torch.nn as nn
import numpy as np

def np2th(weights, conv=False):
    """
    Numpy : HWIO (Height, Width, Input_channels, Outout_channels)
    Pytorch : OIHW (Output_channels, Input_channels, Height, Width)
    if conv=True : Numpy -> Pytorch
    """
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

# swish function
def swish(x):
    return x * torch.sigmoid(x)

# GeLU, ReLU, Swish function dictionary
ACT2FN = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
}