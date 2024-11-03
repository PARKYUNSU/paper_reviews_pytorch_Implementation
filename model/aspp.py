import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ASPPConv, self).__init__()
        padding = 0 if kernel_size == 1 else dilation  # kernel_size가 1일 때 padding을 0으로 설정
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
            size = x.shape[2:]  # 입력의 공간 크기 저장
            x = self.pool(x)
            x = self.conv(x)
            print(f"ASPPPooling output shape: {x.shape} (after pooling and conv)")
            return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, output_stride):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise ValueError("output_stride == 8 or 16!!")

        self.conv1 = ASPPConv(in_channels, out_channels, kernel_size=1, dilation=dilations[0])
        self.conv2 = ASPPConv(in_channels, out_channels, kernel_size=3, dilation=dilations[1])
        self.conv3 = ASPPConv(in_channels, out_channels, kernel_size=3, dilation=dilations[2])
        self.conv4 = ASPPConv(in_channels, out_channels, kernel_size=3, dilation=dilations[3])
        self.pool = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        print(f"ASPP conv1 output shape: {x1.shape}")

        x2 = self.conv2(x)
        print(f"ASPP conv2 output shape: {x2.shape}")

        x3 = self.conv3(x)
        print(f"ASPP conv3 output shape: {x3.shape}")

        x4 = self.conv4(x)
        print(f"ASPP conv4 output shape: {x4.shape}")

        x5 = self.pool(x)
        print(f"ASPP pool output shape: {x5.shape}")

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # 채널 방향으로 연결
        print(f"Concatenated ASPP output shape: {x.shape}")

        return self.project(x)

# 테스트 코드
if __name__ == "__main__":
    # 입력 텐서 정의 (예: 3채널 RGB 이미지, 크기 64x64)
    input_tensor = torch.randn(3, 2048, 32, 32)  # (배치 크기, 채널, 높이, 너비)
    aspp = ASPP(in_channels=2048, out_channels=256, output_stride=8)
    
    # 평가 모드로 설정
    aspp.eval()

    # 모델 적용
    output = aspp(input_tensor)
    
    # 출력 형태 출력
    print("Final ASPP output shape:", output.shape)
