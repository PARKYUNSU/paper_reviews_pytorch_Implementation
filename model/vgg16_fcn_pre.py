import torch
import torch.nn as nn
import torchvision.models as models

class FCN32s(nn.Module):
    def __init__(self, num_classes):
        super(FCN32s, self).__init__()
        
        # Pretrained VGG16 모델을 불러오고 Fully Connected layer들을 제거
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        self.features = nn.Sequential(*features)
        self.score = nn.Conv2d(512, num_classes, kernel_size=1)

        # 최종 출력을 32배 업샘플링 (FCN-32s)
        self.upsample32x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16, bias=False)
    
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.score(x)
        # 32배 업샘플링하여 원래 이미지 크기로 복원
        x = self.upsample32x(x)
        
        return x
    
    def _initialize_weights(self):
        nn.init.xavier_normal_(self.score.weight)
        nn.init.xavier_normal_(self.upsample32x.weight)

model = FCN32s(num_classes=21)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)

print(output.shape)