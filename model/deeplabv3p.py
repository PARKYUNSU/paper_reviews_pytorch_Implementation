import torch
import torch.nn as nn
import torch.nn.functional as F
from aspp import ASPP
from xception import Xception

class DeepLabV3P(nn.Module):
    def __init__(self, num_classes, output_stride):
        super(DeepLabV3P, self).__init__()

        # Xception
        self.backbone = Xception(output_stride=output_stride)
        
        # ASPP
        self.aspp = ASPP(in_channels=2048, out_channels=256, output_stride=output_stride)

        # Decoder
        self.decoder = Decoder(num_classes=num_classes, low_level_channels=256)

    def forward(self, x):
        x, low_level_features = self.backbone(x)
        print(f"Backbone output shape: {x.shape}")
        print(f"Low-level features shape: {low_level_features.shape}")

        x = self.aspp(x)
        print(f"ASPP output shape: {x.shape}")

        x = self.decoder(x, low_level_features)
        print(f"Decoder output shape after first 4x upsample: {x.shape}")

        return x


# class Decoder(nn.Module):
#     def __init__(self, num_classes, low_level_channels):
#         super(Decoder, self).__init__()
#         self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(48)
#         self.relu = nn.ReLU(inplace=True)

#         # Table 2, best performance with two 3x3 convs
#         self.output = nn.Sequential(
#             nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.1),
#             nn.Conv2d(256, num_classes, 1, stride=1),
#         )

#     def forward(self, x, low_level_features):
#         low_level_features = self.conv1(low_level_features)
#         low_level_features = self.relu(self.bn1(low_level_features))
#         size = low_level_features.shape[-2:]

#         x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
#         x = self.output(torch.cat((low_level_features, x), dim=1))
#         return x
class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        # Best performance with two 3x3 convs
        self.output = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1, bias=False),
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
        # size = low_level_features.shape[2:]  # Get spatial dimensions of low-level features

        # 첫 번째 4배 업샘플링
        x = torch.cat((low_level_features, x), dim=1)
        x = self.output(x)
        
        # Decoder 단계에서 4배 업샘플링 적용
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        print(f"Decoder output shape after first 4x upsample: {x.shape}")
        
        return x


if __name__ == "__main__":
    model = DeepLabV3P(num_classes=21, output_stride=8)
    input_tensor = torch.randn(3, 3, 512, 512)
    output = model(input_tensor)
    # 출력 형태 확인
    print("Output shape:", output.shape)
