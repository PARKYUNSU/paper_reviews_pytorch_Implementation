import torch
import torch.nn as nn
import torch.nn.functional as F
from aspp import ASPP
from xception import Xception

class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(low_level_channels + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, low_level_feat):
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        return x

    
class DeepLabV3P(nn.Module):
    def __init__(self, num_classes, output_stride):
        super(DeepLabV3P, self).__init__()

        # Xception
        self.backbone = Xception(output_stride=output_stride)
        
        ASPP
        self.aspp = ASPP(in_channels=2048, out_channels=128, output_stride=output_stride)

        # Decoder
        self.decoder = Decoder(num_classes=num_classes, low_level_channels=128)


    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        print(f"Backbone output shape: {x.shape}")
        print(f"Low-level features shape: {low_level_features.shape}")

        x = self.aspp(x)
        print(f"ASPP output shape: {x.shape}")

        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        print(f"Decoder output shape after first 4x upsample: {x.shape}")
        return x

if __name__ == "__main__":
    model = DeepLabV3P(num_classes=21, output_stride=16)
    input_tensor = torch.randn(3, 3, 512, 512)
    output = model(input_tensor)
    # 출력 형태 확인
    print("Output shape:", output.shape)
