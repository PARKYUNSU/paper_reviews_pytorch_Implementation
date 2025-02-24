import torch
import torch.nn as nn

class SimSiamWithoutPrediction(nn.Module):
    # dim : feature dimension 2048
    def __init__(self, base_encoder, dim=2048):
        super(SimSiamWithoutPrediction, self).__init__()
        # Base encoder 설정 (e.g., ResNet50)
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        prev_dim = self.encoder.fc.weight.shape[1]
        # Projection Head 정의
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                         nn.BatchNorm1d(prev_dim),
                                         nn.ReLU(inplace=True),  # first layer
                                         nn.Linear(prev_dim, prev_dim, bias=False),
                                         nn.BatchNorm1d(prev_dim),
                                         nn.ReLU(inplace=True),  # second layer
                                         self.encoder.fc,
                                         nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[6].bias.requires_grad = False

    def forward(self, x1, x2):
        """
        Forward Pass:
        - x1, x2: 두 입력 이미지 (증강된 쌍)
        """
        # Encoder와 Projection Head를 통해 z1, z2 추출
        z1 = self.encoder(x1)  # Feature from x1
        z2 = self.encoder(x2)  # Feature from x2

        # z1, z2만 반환
        return z1, z2
