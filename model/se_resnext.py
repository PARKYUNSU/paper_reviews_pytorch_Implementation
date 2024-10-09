import torch
from torch import nn
from model.se_block import SE_block
from torchinfo import summary

class SE_BottleNeck(nn.Module):
    expansion_factor = 2
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32):
        super(SE_BottleNeck, self).__init__()

        D = int(out_channels / cardinality)  # 그룹당 채널 수
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, groups=cardinality),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels * self.expansion_factor, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channels * self.expansion_factor))

        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion_factor))

        self.se_block = SE_block(out_channels * self.expansion_factor)
        self.relu1 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.se_block(x)

        x += self.residual(identity)  # shortcut connection
        x = self.relu1(x)
        return x


class SE_Resnext(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, cardinality=32):
        super(SE_Resnext, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.conv2 = self._make_layer(block, 128, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 256, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 512, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 1024, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, num_classes)

        # 가중치 초기화
        self._init_layer()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks -1) # [stride, 1, 1, ..., 1] 1은 num_block -1 개
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, self.cardinality))  # cardinality 전달
            self.in_channels = out_channels * block.expansion_factor
        return nn.Sequential(*layers)

    def _init_layer(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def se_resnext50():
    return SE_Resnext(SE_BottleNeck, [3, 4, 6, 3], cardinality=32)

def se_resnext101():
    return SE_Resnext(SE_BottleNeck, [3, 4, 23, 3], cardinality=32)

# model = se_resnext50()
# summary(model, input_size=(2,3,224,224), device='cpu')
