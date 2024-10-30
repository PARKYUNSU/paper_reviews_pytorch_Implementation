import torch
import torch.nn as nn
from in_block import InvertedResBlock

class MobilenetV3(nn.Module):
    def __init__(self, bottleneck, last_channels, num_classes):
        super(MobilenetV3).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace = True)
        )

        in_channels = 16
        bottleneck_layers = []
        for kernel_size, inner_channels, out_channels, use_se, use_hswish, stride in bottleneck:
            bottleneck_layers.append(InvertedResBlock(in_channels, inner_channels, out_channels, kernel_size, stride, use_se, use_hswish))

            in_channels = out_channels
        self.bottlenecks = nn.Sequential(*bottleneck_layers)

        # last_inner_channels = bottleneck[-1][1]
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 1, bias = False),
            nn.BatchNorm2d(inner_channels),
            nn.Hardswish(inplace = True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(inner_channels, last_channels),
            nn.Hardswish(inplace = True)
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channels, num_classes)
        )
    
    def forward(self, x):
        x = self.first_conv(x)
        x = self.bottlenecks(x)
        x = self.last_conv(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x