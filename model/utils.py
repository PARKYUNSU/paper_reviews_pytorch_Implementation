import torch
import numpy as np
import torch.nn.functional as F

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
    'relu': F.relu,
    'gelu': F.gelu,
    'tanh': F.tanh,
    'sigmoid': F.sigmoid,
}