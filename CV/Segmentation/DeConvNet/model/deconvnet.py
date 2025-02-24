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
            nn.ReLU(inplace=True)
        )            
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)  # return_indices로 maxpooling 위치 값 기억 -> Unpooling에서 그 위치 사용
        
        self.encode2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)            
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.encode3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)                       
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.encode4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)            
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        
        self.encode5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)                        
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Fully connected layers
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, stride=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5)
        )

        # Deconv layers
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(4096, 512, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv5_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)            
        )
        
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv4_conv = nn.Sequential(
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
        
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv3_conv = nn.Sequential(
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

        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.score = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, dilation=1)

    def forward(self, x):
        # Encoding path
        x = self.encode1(x)
        x, indices1 = self.pool1(x)
        x = self.encode2(x)
        x, indices2 = self.pool2(x)
        x = self.encode3(x)
        x, indices3 = self.pool3(x)
        x = self.encode4(x)
        x, indices4 = self.pool4(x)
        x = self.encode5(x)
        x, indices5 = self.pool5(x)

        # Fully connected layers
        x = self.conv6(x)
        x = self.conv7(x)

        # Decoding path
        x = self.deconv6(x)
        x = self.unpool5(x, indices5)
        x = self.deconv5_conv(x)
        x = self.unpool4(x, indices4)
        x = self.deconv4_conv(x)
        x = self.unpool3(x, indices3)
        x = self.deconv3_conv(x)
        x = self.unpool2(x, indices2)
        x = self.deconv2_conv(x)
        x = self.unpool1(x, indices1)
        x = self.deconv1_conv(x)
        x = self.score(x)
        return x


if __name__ == "__main__":
    model = DeconvNet(num_classes=21)
    input = torch.ones([2, 3, 224, 224])
    output = model(input)
    print(f"Final shapes - output: {output.shape}")

###  배치 정규화(Batch Normalization) 레이어가 최소 두 개 이상의 값이 있는 채널을 필요로 하기 때문에 실험을 ([2, 3, 224, 224]) 베치 2로