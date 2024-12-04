import torch
import torch.nn as nn
from .ghost_module import Ghost_module


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, exp_channels, out_channels, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.stride = stride
        self.use_se = use_se

        # Pointwise conv (expansion)
        self.ghost1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(exp_channels),
            nn.ReLU(inplace=True)
        )

        # Depthwise conv (optional downsampling)
        self.dwconv = nn.Conv2d(
            exp_channels, exp_channels, kernel_size=kernel_size, stride=stride,
            padding=kernel_size // 2, groups=exp_channels, bias=False
        ) if stride > 1 else nn.Identity()
        self.bn_dw = nn.BatchNorm2d(exp_channels)

        # SE layer (optional)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(exp_channels, exp_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(exp_channels // 4, exp_channels, kernel_size=1),
            nn.Sigmoid()
        ) if use_se else nn.Identity()

        # Pointwise linear (reduction)
        self.ghost2 = nn.Sequential(
            nn.Conv2d(exp_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Residual connection adjustment (if needed)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride > 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        # Ghost module operations
        x = self.ghost1(x)
        x = self.dwconv(x)
        x = self.bn_dw(x)

        if self.use_se:
            x = x * self.se(x)

        x = self.ghost2(x)

        # Add residual connection
        return x + residual

    
class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=80, width_mult=1.):
        super(GhostNet, self).__init__()
        self.cfgs = cfgs

        # First Layer
        output_channel = _make_divisible(16 * width_mult, 4)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, output_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )
        input_channel = output_channel

        # GhostBottleneck Blocks
        self.blocks = self._make_layers(input_channel, width_mult)
        input_channel = self.blocks[-1][-1].ghost2[0].out_channels  # 마지막 블록 출력 채널 추적

        # Head Layer
        output_channel = _make_divisible(960 * width_mult, 4)
        self.conv_head = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )

        # Classification Layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(output_channel, 1280, bias=False),
            nn.BatchNorm1d(1280),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

        self._initialize_weights()

    def _make_layers(self, input_channel, width_mult):
        layers = []
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(GhostBottleneck(input_channel, hidden_channel, output_channel, k, s, use_se))
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


def ghostnet(num_classes=1000, width_mult=1.):
    cfgs = [
        # k, t, c, SE, s
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
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, num_classes=num_classes, width_mult=width_mult)


# 테스트
if __name__ == "__main__":
    model = ghostnet(num_classes=10)
    x = torch.randn(3, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")