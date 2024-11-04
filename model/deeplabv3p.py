import torch
import torch.nn as nn
import torch.nn.functional as F
from aspp import ASPP
from xception import Xception

class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        self.output = nn.Sequential(
            nn.Conv2d(48 + 128, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1, stride=1),
        )

    def forward(self, x, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.relu(self.bn1(low_level_features))
        H, W = low_level_features.size(2), low_level_features.size(3)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output(torch.cat((low_level_features, x), dim=1))
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
