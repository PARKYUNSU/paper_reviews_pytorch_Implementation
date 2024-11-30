import torch
import torch.nn as nn

class GhostNet(nn.Module):
    def __init__(self, in_channels, out_channels, kerner_size, stride, use_se):
        super(GhostNet, self).__init__()
        self.conv1 = nn.Sequential()