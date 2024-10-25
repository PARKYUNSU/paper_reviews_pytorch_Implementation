import torch
import torch.nn as nn
from torchvision import models

class VGG16_FCN_pre(nn.Module):
    def __init__(self, num_classes=21):
        super(VGG16_FCN_pre, self).__init__()
        
        # Pretrained VGG16 로드
        vgg16 = models.vgg16(pretrained=True)
        vgg16_features = list(vgg16.features.children()) # VGG16 Layer 가져오기
        
        self.features1 = nn.Sequential(*vgg16_features[0:5])   # [1, 64, 112, 112]
        self.features2 = nn.Sequential(*vgg16_features[5:10])  # [1, 128, 56, 56]
        self.features3 = nn.Sequential(*vgg16_features[10:17]) # [1, 256, 28, 28]
        self.features4 = nn.Sequential(*vgg16_features[17:24]) # [1, 512, 14, 14]
        self.features5 = nn.Sequential(*vgg16_features[24:])   # [1, 512, 7, 7]
        
        # Fully connected layers
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.score = nn.Conv2d(4096, num_classes, kernel_size=1)
        
        # Up-sampling layers
        self.upscore32 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)
        self.deconv16 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=32, stride=16, padding=8, bias=False)
        self.deconv8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, x):
        x1 = self.features1(x)  # [1, 64, 112, 112]
        x2 = self.features2(x1)  # [1, 128, 56, 56]
        x3 = self.features3(x2)  # [1, 256, 28, 28]
        x4 = self.features4(x3)  # [1, 512, 14, 14]
        x5 = self.features5(x4)  # [1, 512, 7, 7]

        x6 = self.conv6(x5)  # [1, 4096, 7, 7]
        x7 = self.conv7(x6)  # [1, 4096, 7, 7]
        score = self.score(x7)  # [1, 21, 7, 7]

        fcn32 = self.upscore32(score)  # [1, 21, 224, 224]

        return fcn32

if __name__ == "__main__":
    model = VGG16_FCN_pre(num_classes=21)
    input = torch.ones([1, 3, 224, 224])
    fcn8 = model(input)