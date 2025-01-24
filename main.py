import torch
from model.vit import Vision_Transformer
from model.config import get_b16_config

if __name__ == "__main__":
    config = get_b16_config()
    model = Vision_Transformer(config, img_size=224, num_classes=1000, in_channels=3)
    
    # 임의의 입력 이미지 배치
    x = torch.randn(8, 3, 224, 224)
    logits = model(x)
    print("Logits shape:", logits.shape)  # Expected: (8, 1000)