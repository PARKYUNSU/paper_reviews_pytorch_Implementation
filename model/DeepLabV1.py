import torch
import torch.nn as nn
import torch.nn.functional as F
from model.VGG16_LargeFV import VGG16_LargeFV

class DeepLabV1(nn.Module):
    def __init__(self, num_classes=32, init_weights=True):
        super(DeepLabV1, self).__init__()
        self.backbone = VGG16_LargeFV(num_classes=num_classes, init_weights=init_weights)

    def forward(self, x):
        x = self.backbone(x)
        probmap = F.softmax(x, dim=1)
        
        return probmap