import torch
import torchvision.transforms as transforms
from PIL import Image
from simsiam.builder2 import SimSiamWithoutPrediction  # Prediction 제거된 SimSiam
from torchvision.models import resnet50

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pre-trained SimSiamWithoutPrediction 모델 로드
pretrained_path = "path/to/pretrain_ckpt.pth.tar"
encoder = SimSiamWithoutPrediction(resnet50, dim=1000)  # dim=1000으로 설정
state_dict = torch.load(pretrained_path, map_location="cpu")["state_dict"]
encoder.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
encoder = encoder.to(device)
encoder.eval()

# 전처리 함수 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 이미지 로드 및 전처리
img1 = Image.open("path/to/image1.jpg").convert("RGB")
img2 = Image.open("path/to/image2.jpg").convert("RGB")
img1_tensor = transform(img1).unsqueeze(0).to(device)
img2_tensor = transform(img2).unsqueeze(0).to(device)

# 특징 벡터 추출 및 유사도 계산
with torch.no_grad():
    # z1, z2 특징 벡터 추출
    z1, z2 = encoder(img1_tensor, img2_tensor)

# 코사인 유사도 계산
cosine_similarity = torch.nn.functional.cosine_similarity(z1, z2, dim=1).mean().item()

# 결과 출력
print(f"유사도: {cosine_similarity:.4f}")