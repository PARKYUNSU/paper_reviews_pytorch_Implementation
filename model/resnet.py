import torch
import torch.nn as nn
from torchinfo import summary

class BasicBlock(nn.Module):
    expansion_factor = 1
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())

        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*self.expansion_factor))

        self.relu1 = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)

        x += self.residual(identity)
        x = self.relu1(x)
        return x

class BottleNeck(nn.Module):
    expansion_factor=4
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels * self.expansion_factor, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channels * self.expansion_factor))

        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion_factor:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion_factor))

        self.relu1 = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += self.residual(identity)
        x = self.relu1(x)
        return x

class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=21, output_stride=16):
        super(Resnet, self).__init__()
        self.in_channels = 64
        self.output_stride = output_stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        s3, s4, d3, d4 = self._determine_stride_dilation()

        self.conv2 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=s3, dilation=d3)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=s4, dilation=d4)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2, dilation=1)

        self.classifier = nn.Conv2d(512 * block.expansion_factor, num_classes, kernel_size=1)

        self._init_layer()

    def _determine_stride_dilation(self):
        if self.output_stride == 16:
            return (2, 1, 1, 2)
        elif self.output_stride == 8:
            return (1, 1, 2, 4)
        return (2, 2, 1, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride, dilation=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
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
        low_level_features = self.conv2(x)
        x = self.conv3(low_level_features)
        x = self.conv4(x)
        x = self.conv5(x)

        # Apply 1x1 convolution for classification
        x = self.classifier(x)
        
        # Upsample to input image size
        x = nn.functional.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

        return x, low_level_features
    
class Model:
    def resnet18(self, output_stride=16, num_classes=21):
        return Resnet(BasicBlock, [2, 2, 2, 2], output_stride=output_stride, num_classes=num_classes)

    def resnet34(self, output_stride=16, num_classes=21):
        return Resnet(BasicBlock, [3, 4, 6, 3], output_stride=output_stride, num_classes=num_classes)

    def resnet50(self, output_stride=16, num_classes=21):
        return Resnet(BottleNeck, [3, 4, 6, 3], output_stride=output_stride, num_classes=num_classes)

    def resnet101(self, output_stride=16, num_classes=21):
        return Resnet(BottleNeck, [3, 4, 23, 3], output_stride=output_stride, num_classes=num_classes)

    def resnet152(self, output_stride=16, num_classes=21):
        return Resnet(BottleNeck, [3, 8, 36, 3], output_stride=output_stride, num_classes=num_classes)

model = Model().resnet101(output_stride=8, num_classes=21)
summary(model, input_size=(2, 3, 224, 224), device="cpu")
