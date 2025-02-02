import torch
import torch.nn as nn
import numpy as np

def np2th(weights, conv=False):
    """
    변환:
    - numpy 배열이면 torch.Tensor로 변환합니다.
    - 이미 torch.Tensor인 경우 그대로 반환합니다.
    
    만약 conv=True이면, 커널의 차원 순서를 변환합니다.
    """
    if isinstance(weights, torch.Tensor):
        return weights
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