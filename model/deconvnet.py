import torch
import torch.nn as nn

class DeconvNet(nn.Module):
    def __init__(self, num_classes=21):
        super(DeconvNet, self).__init__()

        self.encode1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/2 크기
        )
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/4 크기
        )
        
        self.encode3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/8 크기
        )

        self.encode4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/16 크기
        )

        self.encode5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/32 크기
        )

        # Fully connected layers
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        # deconv
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, kernel_size=7, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.deconv5 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        self.deconv4 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.deconv3 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )                
        self.deconv2 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.deconv1 = nn.Sequential(
            nn.MaxUnpool2d(kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Features extraction through VGG16
        x1 = self.encode1(x)
        x2 = self.encode2(x1)
        x3 = self.encode3(x2)
        x4 = self.encode4(x3)
        x5 = self.encode5(x4)

        # Fully connected layers
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        score = self.score(x7)

        # FCN-32s
        fcn32 = self.upscore32(score)
        
        # FCN-16s
        score2 = self.score_upsample(score)
        fcn16 = self.deconv16(score2)
        score4 = self.score4(x4) 

        score4_1 = score4 + score2
        
        # FCN-8s
        score3 = self.score3(x3)
        score4_2 = self.score_upsample2(score4_1)
        score3_1 = score3 + score4_2
        fcn8 = self.deconv8(score3_1)

        return fcn32, fcn16, fcn8

if __name__ == "__main__":
    model = DeconvNet(num_classes=21)
    input = torch.ones([1, 3, 224, 224])
    fcn32, fcn16, fcn8 = model(input)
    print(f"Final shapes - FCN32: {fcn32.shape}, FCN16: {fcn16.shape}, FCN8: {fcn8.shape}")