import torch
from torch import nn
from torchinfo import summary

from .in_block import InvertedResBlock

class MobilenetV3(nn.Module):
    def __init__(self, bottleneck, last_channels, num_classes):
        super(MobilenetV3, self).__init__()

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
    
def mobilenet_v3_large(num_classes=1000):
    cfgs = [
        [3,  16,  16, False, False, 1],
        [3,  64,  24, False, False, 2],
        [3,  72,  24, False, False, 1],
        [5,  72,  40,  True, False, 2],
        [5, 120,  40,  True, False, 1],
        [5, 120,  40,  True, False, 1],
        [3, 240,  80, False,  True, 2],
        [3, 200,  80, False,  True, 1],
        [3, 184,  80, False,  True, 1],
        [3, 184,  80, False,  True, 1],
        [3, 480, 112,  True,  True, 1],
        [3, 672, 112,  True,  True, 1],
        [5, 672, 160,  True,  True, 2],
        [5, 960, 160,  True,  True, 1],
        [5, 960, 160,  True,  True, 1],
    ]

    return MobilenetV3(cfgs, last_channels=1280, num_classes=num_classes)

def mobilenet_v3_small(num_classes=1000):
    cfgs = [
        [3,  16, 16,  True, False, 2],
        [3,  72, 24, False, False, 2],
        [3,  88, 24, False, False, 1],
        [5,  96, 40,  True,  True, 2],
        [5, 240, 40,  True,  True, 1],
        [5, 240, 40,  True,  True, 1],
        [5, 120, 48,  True,  True, 1],
        [5, 144, 48,  True,  True, 1],
        [5, 288, 96,  True,  True, 2],
        [5, 576, 96,  True,  True, 1],
        [5, 576, 96,  True,  True, 1],
        ]
   
    return MobilenetV3(cfgs, last_channels = 1024, num_classes=num_classes)

model = mobilenet_v3_large()
#model = mobilenet_v3_small()

# summary(model, input_size = (2, 3, 224, 224), device = "cpu")