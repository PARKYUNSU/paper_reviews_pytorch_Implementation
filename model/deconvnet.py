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
        print(f"x.shape after encode1: {x.shape}")
        
        x, indices1 = self.pool1(x)
        print(f"x.shape after pool1: {x.shape}")
        
        x = self.encode2(x)
        print(f"x.shape after encode2: {x.shape}")
        
        x, indices2 = self.pool2(x)
        print(f"x.shape after pool2: {x.shape}")
        
        x = self.encode3(x)
        print(f"x.shape after encode3: {x.shape}")
        
        x, indices3 = self.pool3(x)
        print(f"x.shape after pool3: {x.shape}")
        
        x = self.encode4(x)
        print(f"x.shape after encode4: {x.shape}")
        
        x, indices4 = self.pool4(x)
        print(f"x.shape after pool4: {x.shape}")
        
        x = self.encode5(x)
        print(f"x.shape after encode5: {x.shape}")
        
        x, indices5 = self.pool5(x)
        print(f"x.shape after pool5: {x.shape}")

        # Fully connected layers
        x = self.conv6(x)
        print(f"x.shape after conv6: {x.shape}")

        x = self.conv7(x)
        print(f"x.shape after conv7: {x.shape}")

        # Decoding path
        x = self.deconv6(x)
        print(f"x.shape after deconv6: {x.shape}")
        
        x = self.unpool5(x, indices5)
        print(f"x.shape after unpool5: {x.shape}")
        
        x = self.deconv5_conv(x)
        print(f"x.shape after deconv5_conv: {x.shape}")
        
        x = self.unpool4(x, indices4)
        print(f"x.shape after unpool4: {x.shape}")
        
        x = self.deconv4_conv(x)
        print(f"x.shape after deconv4_conv: {x.shape}")
        
        x = self.unpool3(x, indices3)
        print(f"x.shape after unpool3: {x.shape}")
        
        x = self.deconv3_conv(x)
        print(f"x.shape after deconv3_conv: {x.shape}")
        
        x = self.unpool2(x, indices2)
        print(f"x.shape after unpool2: {x.shape}")
        
        x = self.deconv2_conv(x)
        print(f"x.shape after deconv2_conv: {x.shape}")
        
        x = self.unpool1(x, indices1)
        print(f"x.shape after unpool1: {x.shape}")
        
        x = self.deconv1_conv(x)
        print(f"x.shape after deconv1_conv: {x.shape}")
        
        x = self.score(x)
        print(f"x.shape after score: {x.shape}")

        return x


if __name__ == "__main__":
    model = DeconvNet(num_classes=21)
    input = torch.ones([2, 3, 224, 224])
    output = model(input)
    print(f"Final shapes - output: {output.shape}")

###  배치 정규화(Batch Normalization) 레이어가 최소 두 개 이상의 값이 있는 채널을 필요로 하기 때문에 실험을 ([2, 3, 224, 224]) 베치 2로