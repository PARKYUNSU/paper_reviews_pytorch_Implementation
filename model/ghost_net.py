import torch
import torch.nn as nn
import math
from torchinfo import summary


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                      padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class DwSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = BasicConv(in_channels, in_channels, kernel_size=3, stride=stride, 
                                   padding=1, groups=in_channels)
        self.pointwise = BasicConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_kernel_size=3, stride=1, relu=True):
        super().__init__()
        self.oup = out_channels
        init_channels = math.ceil(out_channels / ratio)
        cheap_channels = init_channels * (ratio - 1)

        self.primary_conv = BasicConv(in_channels, init_channels, kernel_size, stride=stride, 
                                      padding=kernel_size // 2, relu=relu)
        self.cheap_operation = BasicConv(init_channels, cheap_channels, dw_kernel_size, stride=1, 
                                         padding=dw_kernel_size // 2, groups=init_channels, relu=relu)

    def forward(self, x):
        primary_output = self.primary_conv(x)
        cheap_output = self.cheap_operation(primary_output)
        return torch.cat([primary_output, cheap_output], dim=1)[:, :self.oup, :, :]

class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, stride, use_se):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels

        self.conv = nn.Sequential(
            GhostModule(in_channels, hidden_channels, kernel_size=1, relu=True),
            DwSepConv(hidden_channels, hidden_channels, stride=stride) if stride > 1 else nn.Identity(),
            SELayer(hidden_channels) if use_se else nn.Identity(),
            GhostModule(hidden_channels, out_channels, kernel_size=1, relu=False),
        )

        if not self.use_shortcut:
            self.shortcut = nn.Sequential(
                DwSepConv(in_channels, in_channels, stride=stride),
                BasicConv(in_channels, out_channels, kernel_size=1, stride=1, relu=False),
            )

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)

class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=80, width_mult=1.0):
        super().__init__()
        self.cfgs = cfgs

        # Initial layer
        init_channels = self._make_divisible(16 * width_mult, 4)
        self.initial_layer = BasicConv(3, init_channels, kernel_size=3, stride=2, padding=1)
        input_channels = init_channels

        # Ghost Bottleneck blocks
        layers = []
        for k, exp_size, c, use_se, s in cfgs:
            output_channels = self._make_divisible(c * width_mult, 4)
            hidden_channels = self._make_divisible(exp_size * width_mult, 4)
            layers.append(GhostBottleneck(input_channels, hidden_channels, output_channels, k, s, use_se))
            input_channels = output_channels
        self.blocks = nn.Sequential(*layers)

        # Final layers
        final_channels = self._make_divisible(exp_size * width_mult, 4)
        self.final_layer = nn.Sequential(
            BasicConv(input_channels, final_channels, kernel_size=1, stride=1, relu=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(final_channels, 1280, bias=False),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.blocks(x)
        x = self.final_layer(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def ghostnet(num_classes=80):
    cfgs = [
        [3, 16, 16, 0, 1],
        [3, 48, 24, 0, 2],
        [3, 72, 24, 0, 1],
        [5, 72, 40, 1, 2],
        [5, 120, 40, 1, 1],
        [3, 240, 80, 0, 2],
        [3, 200, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 184, 80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
    ]
    return GhostNet(cfgs, num_classes=num_classes)

model = ghostnet()

summary(model, input_size = (2, 3, 224, 224), device = "cpu")