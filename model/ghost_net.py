import torch
import torch.nn as nn
import math


def _make_divisible(v, divisor, min_value=None):
    """Ensures the channel number is divisible by the divisor."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, ratio=2, dw_size=3, relu=True):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = out_channels - init_channels

        # Primary Convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        # Cheap Operation
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # Pointwise Convolution
        self.ghost1 = GhostModule(in_channels, exp_channels, relu=True)

        # Depthwise Convolution
        self.dwconv = nn.Conv2d(
            exp_channels, exp_channels, kernel_size, stride, kernel_size // 2, groups=exp_channels, bias=False
        ) if stride > 1 else nn.Identity()

        self.bn_dw = nn.BatchNorm2d(exp_channels)

        # Squeeze-and-Excitation
        self.se = SELayer(exp_channels) if use_se else nn.Identity()

        # Pointwise Linear
        self.ghost2 = GhostModule(exp_channels, out_channels, relu=False)

        # Shortcut Connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if stride > 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)

        # GhostBottleneck operations
        x = self.ghost1(x)
        x = self.dwconv(x)
        x = self.bn_dw(x)
        x = self.se(x)
        x = self.ghost2(x)

        return x + shortcut


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.0):
        super(GhostNet, self).__init__()
        self.cfgs = cfgs

        # First Layer
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, output_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        input_channel = output_channel

        # Ghost Bottlenecks
        self.blocks = self._make_layers(input_channel, width_mult)

        # Head Layer
        output_channel = _make_divisible(960 * width_mult, 4)
        self.conv_head = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

        # Classification Layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(output_channel, 1280, bias=False),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

        self._initialize_weights()

    def _make_layers(self, input_channel, width_mult):
        layers = []
        for k, exp_size, c, se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(GhostBottleneck(input_channel, hidden_channel, output_channel, k, s, se))
            input_channel = output_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghostnet(num_classes=1000, width_mult=1.0):
    cfgs = [
        # kernel_size, expansion_size, output_channels, use_se, stride
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
    return GhostNet(cfgs, num_classes=num_classes, width_mult=width_mult)


# 테스트
if __name__ == "__main__":
    model = ghostnet(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")