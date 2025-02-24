import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair # tuple 처리 함수 # 이미지가 튜블로 되어있는 함수인데 튜플을 사용했어

class Patch_Embedding(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super(Patch_Embedding, self).__init__()
        img_size = _pair(img_size) # (img_size, img_size)
        patch_size = _pair(config.patches["size"])
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x : (B, C, H, W)
        x = self.proj(x)    # (B, hidden_size, H/patch, W/patch)
        x = x.flatten(2)    # (B, hidden_size, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, hidden_size)
        return x