import torch
import torch.nn as nn
from torchvision import models

class DeconvNet2(nn.Module):
    def __init__(self, num_classes=21):
        super(DeconvNet2, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):

            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels,
                          kernel_size=kernel_size, 
                          stride=stride, 
                          padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())

        def DCB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):

            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, 
                                   out_channels=out_channels,
                                   kernel_size=kernel_size, 
                                   stride=stride,
                                   padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())        


        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/2

        # conv2 
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/4

        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/8

        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/16

        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)

        # fc6
        self.fc6 = CBR(512, 4096, 7, 1, 0)
        self.drop6 = nn.Dropout2d(0.5)

        # fc7
        self.fc7 = CBR(4096, 4096, 1, 1, 0)
        self.drop7 = nn.Dropout2d(0.5)

        # fc6-deconv
        self.fc6_deconv = DCB(4096, 512, 7, 1, 0)

        # unpool5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.deconv5_1 = DCB(512, 512, 3, 1, 1)
        self.deconv5_2 = DCB(512, 512, 3, 1, 1)
        self.deconv5_3 = DCB(512, 512, 3, 1, 1)

        # unpool4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv4_1 = DCB(512, 512, 3, 1, 1)
        self.deconv4_2 = DCB(512, 512, 3, 1, 1)
        self.deconv4_3 = DCB(512, 256, 3, 1, 1)        

        # unpool3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3_1 = DCB(256, 256, 3, 1, 1)
        self.deconv3_2 = DCB(256, 256, 3, 1, 1)
        self.deconv3_3 = DCB(256, 128, 3, 1, 1)                          

        # unpool2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2_1 = DCB(128, 128, 3, 1, 1)
        self.deconv2_2 = DCB(128, 64, 3, 1, 1)

        # unpool1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1_1 = DCB(64, 64, 3, 1, 1)
        self.deconv1_2 = DCB(64, 64, 3, 1, 1)

        # Score
        self.score_fr = nn.Conv2d(64, num_classes, 1, 1, 0, 1)

    def forward(self, x):

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x, pool1_indices = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x, pool2_indices = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)        
        x, pool3_indices = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)        
        x, pool4_indices = self.pool4(x) 

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)        
        x, pool5_indices = self.pool5(x)

        x = self.fc6(x)
        x = self.drop6(x)

        x = self.fc7(x)
        x = self.drop7(x)

        x = self.fc6_deconv(x)     

        x = self.unpool5(x, pool5_indices)
        x = self.deconv5_1(x)        
        x = self.deconv5_2(x)                
        x = self.deconv5_3(x)                

        x = self.unpool4(x, pool4_indices)
        x = self.deconv4_1(x)        
        x = self.deconv4_2(x)                
        x = self.deconv4_3(x)                       

        x = self.unpool3(x, pool3_indices)
        x = self.deconv3_1(x)        
        x = self.deconv3_2(x)                
        x = self.deconv3_3(x)                            

        x = self.unpool2(x, pool2_indices)
        x = self.deconv2_1(x)        
        x = self.deconv2_2(x)                                         

        x = self.unpool1(x, pool1_indices)
        x = self.deconv1_1(x)        
        x = self.deconv1_2(x)                                    


        x = self.score_fr(x)           
        return x
    
if __name__ == "__main__":
    model = DeconvNet2(num_classes=21)
    input = torch.ones([2, 3, 224, 224])
    output = model(input)
    print(f"Final shapes - output: {output.shape}")

###  배치 정규화(Batch Normalization) 레이어가 최소 두 개 이상의 값이 있는 채널을 필요로 하기 때문에 실험을 ([2, 3, 224, 224]) 베치 2로