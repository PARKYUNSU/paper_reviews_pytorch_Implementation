import torch
import torch.nn as nn
from torchvision.models import resnet50

class SimSiamWithoutProjection(nn.Module):
    def __init__(self, base_encoder=resnet50, dim=1000, pred_dim=512):
        """
        SimSiam 모델에서 Projection Head를 제거하고 ResNet50의 기본 출력 사용.
        
        Args:
            base_encoder: Backbone 네트워크 (default: resnet50)
            dim: ResNet50의 기본 출력 차원 (default: 1000)
            pred_dim: Predictor의 숨겨진 차원 (default: 512)
        """
        super(SimSiamWithoutProjection, self).__init__()
        # ResNet50 Encoder 설정 (num_classes는 변경 없음)
        self.encoder = base_encoder(pretrained=True)  # Pretrained ResNet50
        
        # Predictor 네트워크: ResNet50의 1000차원 출력 사용
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x1, x2):
        """
        SimSiam Forward Pass
        Args:
            x1, x2: 두 개의 입력 이미지 (증강된 쌍 또는 원본 이미지)

        Returns:
            p1, p2: Predictor의 출력
            z1, z2: Encoder(ResNet50)의 출력 (그대로 사용)
        """
        # ResNet50 출력 (Projection Head 없이)
        z1 = self.encoder(x1)  # ResNet50의 1000차원 출력
        z2 = self.encoder(x2)

        # Predictor를 통해 변환
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # z1, z2는 detach() 없이 그대로 반환
        return p1, p2, z1, z2


# 모델 초기화 예제
model = SimSiamWithoutProjection(resnet50)
model = model.cuda()

# 입력 데이터 예제
x1 = torch.randn(16, 3, 224, 224).cuda()  # 16개의 배치, 3채널, 224x224 크기
x2 = torch.randn(16, 3, 224, 224).cuda()

# Forward Pass 실행
p1, p2, z1, z2 = model(x1, x2)
print(f"p1 shape: {p1.shape}, p2 shape: {p2.shape}")
print(f"z1 shape: {z1.shape}, z2 shape: {z2.shape}")