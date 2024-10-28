import torch
import torch.nn as nn
from torchvision.models import vgg16

class FCN8s(nn.Module):
    def __init__(self, num_classes=32):  # CamVid 클래스 수에 맞춤
        super(FCN8s, self).__init__()
        self.vgg = vgg16(weights='IMAGENET1K_V1').features  # pretrained=True 대신 weights 사용
        self.f3, self.f4, self.f5 = 16, 23, 30  # Layer indices for VGG

        # Convolutional layers replacing the dense layers
        self.f5_conv1 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.f5_conv2 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.f5_conv3 = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Transposed convolutions for upsampling
        self.upsample_f5 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_f4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upsample_final = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=4, padding=2, bias=False)

        # f4와 f3를 num_classes 채널로 변환
        self.f4_conv1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.f3_conv1 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        f3 = self.vgg[:self.f3](x)
        f4 = self.vgg[self.f3:self.f4](f3)
        f5 = self.vgg[self.f4:self.f5](f4)

        f5 = self.f5_conv1(f5)
        f5 = nn.ReLU(inplace=True)(f5)
        f5 = nn.Dropout(p=0.5)(f5)  # 드롭아웃 확률 0.5
        f5 = self.f5_conv2(f5)
        f5 = nn.ReLU(inplace=True)(f5)
        f5 = nn.Dropout(p=0.5)(f5)
        f5 = self.f5_conv3(f5)

        upsampled_f5 = self.upsample_f5(f5)

        # f4 채널을 num_classes로 변환
        f4 = self.f4_conv1(f4)
        f4 = self.upsample_f4(f4 + upsampled_f5)

        # f3 채널을 num_classes로 변환
        f3 = self.f3_conv1(f3)
        f3 = self.upsample_final(f4 + f3)
        
        return f3

# 모델 초기화 및 테스트
if __name__ == "__main__":
    model = FCN8s(num_classes=32)  # CamVid 데이터셋에 맞춤
    input = torch.ones([1, 3, 224, 224])
    output = model(input)
    print(f"Final shapes - output: {output.shape}")
