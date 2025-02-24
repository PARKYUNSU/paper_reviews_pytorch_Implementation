import torch
from torch import nn

class SE_block(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SE_block, self).__init__()

        # Squeeze: Global Information Embedding
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        
        # Excitation: Adaptive Recalibration
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        se_weight = self.squeeze(x)  # (N, C, H, W) -> (N, C, 1, 1)
        se_weight = se_weight.view(se_weight.size(0), -1)  # (N, C, 1, 1) -> (N, C)로 reshape
        se_weight = self.excitation(se_weight)  # 채널별 중요도 계산
        se_weight = se_weight.view(se_weight.size(0), se_weight.size(1), 1, 1)  # (N, C) -> (N, C, 1, 1)로 다시 reshape
        return x * se_weight  # 입력에 중요도를 반영하여 조정