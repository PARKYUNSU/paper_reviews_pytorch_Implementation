import torch
import torch.nn as nn
from model.VGG16_LargeFV import VGG16_LargeFV
from utils.crf import DenseCRF

class DeepLabV1(nn.Module):
    def __init__(self, num_classes=21, input_size=321):
        super(DeepLabV1, self).__init__()
        self.backbone = VGG16_LargeFV(num_classes=num_classes, input_size=input_size)
        self.crf = DenseCRF(iter_max=10, 
                            pos_w=3, 
                            pos_xy_std=3, 
                            bi_w=5, 
                            bi_xy_std=140, 
                            bi_rgb_std=5)
        
    def forward(self, x):
        x = self.backbone(x)
        
        return x