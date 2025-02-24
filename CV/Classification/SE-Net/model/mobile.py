import torch
from torch import nn
from torchinfo import summary

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    def forward(self, x):
        return self.conv1(x)
    
class DwSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = BasicConv(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = BasicConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

class MobileNetV1(nn.Module):
    def __init__(self, alpha, num_classes=10, init_weight=True):
        super(MobileNetV1).__init__()

        self.alpha = alpha
        # 32*32 CIFAR IMG
        self.stem = BasicConv(3, int(32*self.alpha), kernel_size=3, stride=1, padding=1)
        # # 224*224 IMG
        # self.stem = BasicConv(3, int(32*self.alpha), kernel_size=3, stride=2, padding=1)

        self.model = nn.Sequential(
            DwSepConv(int(32*self.alpha), int(64*self.alpha)),
            DwSepConv(int(64*self.alpha), int(128*self.alpha), stride=2),
            DwSepConv(int(128*self.alpha), int(128*self.alpha)),
            DwSepConv(int(128*self.alpha), int(256*self.alpha), stride=2),
            DwSepConv(int(256*self.alpha), int(256*self.alpha)),
            DwSepConv(int(256*self.alpha), int(512*self.alpha), stride=2),
            # 5층에서->3개층으로 줄임 CIFAR-10
            *[DwSepConv(int(512 * self.alpha), int(512 * self.alpha)) for _ in range(3)],
            DwSepConv(int(512*self.alpha), int(1024*self.alpha), stride=2),
            DwSepConv(int(1024*self.alpha), int(1024*self.alpha))
        )
        self.classfier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(int(1024*self.alpha), num_classes)
        )
        if init_weight: #초기화 구동함수 호출
            self._initialize_weight()

    def forward(self, x):
        x = self.stem(x)
        x = self.model(x)
        x = self.classfier(x)
        return x
    
    #모델의 초기 Random을 커스터마이징 하기 위한 함수
    def _initialize_weight(self):
        for m in self.modules(): #설계한 모델의 모든 레이어를 순회
            if isinstance(m, nn.Conv2d): #conv의 파라미터(weight, bias)의 초가깂설정
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm2d): #BN의 파라미터(weight, bias)의 초가깂설정
                nn.init.constant_(m.weight, 1) # 1로 다 채움
                nn.init.constant_(m.bias, 0) # 0으로 다 채움

            elif isinstance(m, nn.Linear): #FCL의 파라미터(weight, bias)의 초기값 설정
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# model = MobileNetV1(alpha = 1)
# summary(model, input_size = (1, 3, 224, 224), device = "cpu")