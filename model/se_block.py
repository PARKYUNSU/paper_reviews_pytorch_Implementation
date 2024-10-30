import torch
from torch import nn

class SE_block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4): # MobilenetV3 r = 4
        super(SE_block, self).__init__()

        # Squeeze: Global Information Embedding
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        
        # Excitation: Adaptive Recalibration
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels //reduction_ratio, 1), # Fclayer -> 1x1 conv2d로 변경 : Gpu 병령 연산 더 친화적
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Hardsigmoid(inplace=True)
        )
    
    def forward(self, x):
        se_weight = self.squeeze(x)  # (N, C, H, W) -> (N, C, 1, 1)
        se_weight = self.excitation(se_weight)  # 채널별 중요도 계산
        return x * se_weight  # 입력에 중요도를 반영하여 조정