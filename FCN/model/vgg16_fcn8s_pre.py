import torch
import torch.nn as nn
from torchvision import models

class FCN8(nn.Module):
  def __init__(self, num_classes=21):
      super(FCN8, self).__init__()
      vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
      self.features = vgg.features
      self.pool3_conv = nn.Conv2d(256, num_classes, kernel_size=1)  # pool3 출력 맞춤
      self.pool4_conv = nn.Conv2d(512, num_classes, kernel_size=1)
      self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)
      self.upsample32x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      self.upsample16x = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
      self.upsample8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

  def forward(self, x):
      x = self.features[:17](x)  # pool3 전까지 진행
      pool3_out = self.pool3_conv(x)
      x = self.features[17:24](x)  # pool4 전까지 진행
      pool4_out = self.pool4_conv(x)
      x = self.features[24:](x)  # pool5 이후 컨볼루션 진행
      x = self.classifier(x)
      x = self.upsample32x(x)  # 32배 업샘플링
      x = x + pool4_out  # pool4와 합성
      x = self.upsample16x(x)  # 16배 업샘플링
      x = x + pool3_out  # pool3과 합성
      x = self.upsample8x(x)  # 8배 업샘플링으로 원본 크기 복원
      return x

# 모델 초기화 및 테스트
if __name__ == "__main__":
    model = FCN8(num_classes=21)
    input = torch.ones([1, 3, 224, 224])
    output = model(input)
    print(f"Final shapes - output: {output.shape}")
