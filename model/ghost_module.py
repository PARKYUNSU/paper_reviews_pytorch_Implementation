import torch
import torch.nn as nn

class Ghost_module(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, ratio=2, dw_size=3, relu=True):
        super(Ghost_module, self).__init__()
        self.conv_channels = out_channels // ratio
        self.cheap_channels = out_channels - self.conv_channels

        # Intrinsic feature maps
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, self.conv_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        # Ghost feature maps
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.conv_channels, self.cheap_channels, dw_size, stride=1, padding=dw_size // 2, groups=self.conv_channels, bias=False),
            nn.BatchNorm2d(self.cheap_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        conv_features = self.conv(x)
        ghost_features = self.cheap_operation(conv_features)
        return torch.cat([conv_features, ghost_features], dim=1)


if __name__ == "__main__":
    # 테스트 입력
    x = torch.randn(1, 16, 32, 32)  # 배치 크기 1, 입력 채널 16, 32x32 이미지
    ghost_module = Ghost_module(16, 32, kernel_size=1, stride=1, ratio=2)  # Ghost Module 초기화
    y = ghost_module(x)
    print(f"input: {x.shape}, output: {y.shape}")  # 입력 및 출력 크기 출력
