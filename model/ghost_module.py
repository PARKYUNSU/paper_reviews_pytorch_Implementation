import torch
import torch.nn as nn

class Ghost_module(nn.Module):
    def __init__(self, in_channels, out_channels, kerner_size=1, stride=1, ratio=2, dw_size=3):
        super(Ghost_module, self).__init__()
        self.conv_channels = out_channels // ratio
        self.cheap_channels = out_channels - self.conv_channels

        # Intrinsic feature maps
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, self.conv_channels, kerner_size, stride, kerner_size//2, biae=False),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True)
        )

        # Ghost feature maps
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(self.conv_channels, self.cheap_channels, dw_size, stride=1, padding=dw_size//2, groups=self.conv_channels, bias=False),
            nn.BatchNorm2d(self.conv_channels),
            nn.ReLU(inplace=True)
        )

        def forward(self, x):
            conv_features = self.conv(x)
            ghost_features = self.cheap_operation(conv_features)
            return torch.cat([conv_features, ghost_features], dim=1)
        
if __name__ == "__main__":
    x = torch.randn(1, 16, 32, 32)
    ghost_module = Ghost_module(16, 32, kernel_size=1, stride=1, ratio=2)
    y = ghost_module(x)
    print(f"input: {x.shape}, opuput: {y.shape}")