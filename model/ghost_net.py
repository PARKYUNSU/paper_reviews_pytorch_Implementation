import torch
import torch.nn as nn
from ghost_module import Ghost_module

class GhosBottleNeck(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kerner_size, stride, use_se=False):
        super(GhosBottleNeck, self).__init__()
        self.stride = stride
        self.use_se = use_se

        # Ghost module with ReLU
        self.ghostR = Ghost_module(in_channels, hidden_channels, relu=True)

        # Depthwise Conv
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kerner_size, stride, kerner_size//2 , groups=hidden_channels, biase=False),
            nn.BatchNorm2d(hidden_channels)
        ) if stride > 1 else nn.Identity()

        # SE module
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(hidden_channels, hidden_channels//4, kerner_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels//4, hidden_channels, kerner_size=1, bias=True),
            nn.Hardsigmoid(inplace=True)
        ) if use_se else nn.Identity()

        # Ghost module without ReLU
        self.ghost = Ghost_module(hidden_channels, out_channels, relu=False)
        self.shortcut = nn.Sequential(
            nn.Convd2d(in_channels, out_channels, kerner_size=1, stride=stride, biase=False),
            nn.BatchNorm2d(out_channels)
        ) if stride > 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shorchut = self.shortcut(x)
        x = self.ghostR(x)
        x = self.depthwise(x)
        x = self.se(x) if self.use_se else x
        x = self.ghost
        return x + shorchut
    
class GhostNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GhostNet, self).__init__()
        self.configs = [
            # [kernel_size, exp_size, out_channels, use_se, stride]
            [3, 16, 16, False, 1],
            [3, 48, 24, False, 2],
            [3, 72, 24, False, 1],
            [5, 72, 40, True, 2],
            [5, 120, 40, True, 1],
            [5, 240, 80, False, 2],
            [3, 200, 80, False, 1],
            [3, 480, 112, True, 1],
            [3, 672, 160, True, 2],
            [3, 960, 160, True, 1],
        ]

        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.blocks = self._make_layers()

        self.conv_head = nn.Sequential(
            nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def _make_layers(self):
        layers = []
        in_channels = 16
        for k, exp, c, se, s in self.cfgs:
            layers.append(GhosBottleNeck(in_channels, exp, c, k, s, use_se=se))
            in_channels = c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 테스트
if __name__ == "__main__":
    model = GhostNet(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")