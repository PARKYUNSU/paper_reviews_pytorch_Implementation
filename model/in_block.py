import torch
import torch.nn as nn
from se_block import SE_block

class InvertedResBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, kernel_size, stride, SE, AF):
        super().__init__()

        self.identity = (self.stride == 1 and in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels, momentum = 0.99),
            nn.Hardswish(inplace = True) if AF else nn.ReLU(inplace = True)
            )

        # Depth Wise
        self.depthwise = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, stride=stride, padding=(kernel_size - 1) // 2, groups = inner_channels, bias = False),
            nn.BatchNorm2d(inner_channels),
            nn.Hardswish(inplace = True) if AF else nn.ReLU(inplace = True)
            )
        
        # SE-Block
        se_block = SE_block(inner_channels) if SE else None
        
        # Point Wise
        self.pointwise = nn.Sequential(
            nn.Conv2d(inner_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
            )

        layers = []
        if in_channels < inner_channels:
            layers.append(self.expand)
        layers.append(self.depthwise)
        if se_block is not None:
            layers.append(se_block)
        layers.append(self.pointwise)

        # Residual
        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        if self.identity:
            out = self.residual(x) + x
        else:
            out = self.residual(x)
        return out