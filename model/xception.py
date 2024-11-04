import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1):
        super(BasicConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=groups, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv1(x)
    
class DwSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super(DwSepConv, self).__init__()
        padding = dilation if dilation > kernel_size // 2 else kernel_size // 2
        self.depthwise = BasicConv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, dilation=dilation)
        self.pointwise = BasicConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
# EnrtyFlow
class EntryFlow(nn.Module):
    def __init__(self, enf_s):
        super(EntryFlow, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2_residual = nn.Sequential(
            DwSepConv(64, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DwSepConv(128, 128, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DwSepConv(128, 128, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv2_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2, padding=0),
            nn.BatchNorm2d(128)
        )

        self.conv3_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            DwSepConv(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DwSepConv(256, 256),
            nn.BatchNorm2d(256),
            DwSepConv(256, 256, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)            
        )

        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2, padding=0),
            nn.BatchNorm2d(256)
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            DwSepConv(256, 728),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            DwSepConv(728, 728),
            nn.BatchNorm2d(728),
            DwSepConv(728, 728, stride=enf_s),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True)            
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1, stride=enf_s, padding=0),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        print(f"Conv1 output shape: {x.shape}")
        
        x = self.conv2_residual(x) + self.conv2_shortcut(x)
        print(f"Conv2 output shape: {x.shape}")

        low_level_features = x
        print(f"Low-level features shape (after Conv2): {low_level_features.shape}")

        x = self.conv3_residual(x) + self.conv3_shortcut(x)
        print(f"Conv3 output shape: {x.shape}")
        
        

        x = self.conv4_residual(x) + self.conv4_shortcut(x)
        print(f"Conv4 output shape: {x.shape}")

        return x, low_level_features
    
# MiddleFlow
class MiddleFlow(nn.Module):
    def __init__(self, dilation):
        super(MiddleFlow, self).__init__()
        self.conv_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            DwSepConv(728, 728, dilation=dilation),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            DwSepConv(728, 728, dilation=dilation),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            DwSepConv(728, 728, dilation=dilation),
            nn.BatchNorm2d(728)
        )

        self.conv_shortcut = nn.Sequential()

    def forward(self, x):
        x = self.conv_shortcut(x) + self.conv_residual(x)
        print(f"MiddleFlow output shape: {x.shape}")
        return x
    
# ExitFlow
class ExitFlow(nn.Module):
    def __init__(self, ef_dilation):
        super(ExitFlow, self).__init__()
        self.conv1_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            DwSepConv(728, 1024, dilation=ef_dilation[0]),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            DwSepConv(1024, 1024),
            nn.BatchNorm2d(1024),
            DwSepConv(1024, 1024)
        )

        self.conv1_shortcut = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=1, padding=0),
            nn.BatchNorm2d(1024)
        )

        self.conv2 = nn.Sequential(
            DwSepConv(1024, 1536, kernel_size=3, stride=1, dilation=ef_dilation[1]),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            DwSepConv(1536, 1536, kernel_size=3, stride=1, dilation=ef_dilation[1]),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),            
            DwSepConv(1536, 2048, kernel_size=3, stride=1, dilation=ef_dilation[1]),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1_residual(x) + self.conv1_shortcut(x)
        print(f"ExitFlow Conv1 output shape: {x.shape}")

        x = self.conv2(x)
        print(f"ExitFlow Conv2 output shape: {x.shape}")

        return x
    
# Xception
class Xception(nn.Module):
    def __init__(self, output_stride):
        super(Xception, self).__init__()
        if output_stride == 16:
            enf_s, mdf_d, exf_d = 2, 1, (1, 2)
        elif output_stride == 8:
            enf_s, mdf_d, exf_d = 1, 2, (2, 4)
        else:
            raise ValueError("output_stride == 8 or 16!!")
        
        self.entry = EntryFlow(enf_s)
        self.middle = self._make_middle_flow(mdf_d)
        self.exit = ExitFlow(exf_d)
        
        self._initialize_weights()

    def forward(self, x):
        x, low_level_features = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        return x, low_level_features

    def _make_middle_flow(self, dilation):
        middle = nn.Sequential()
        for i in range(16):
            middle.add_module('middle_block_{}'.format(i), MiddleFlow(dilation))
        return middle

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# check model
if __name__ == "__main__":
    model = Xception(output_stride=16)
    input_tensor = torch.randn([3, 3, 512, 512])
    output, low_level_features = model(input_tensor)
    print(f"Final output shape: {output.shape}")
    print(f"Low-level features shape: {low_level_features.shape}")
