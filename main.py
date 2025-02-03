import torch
import argparse
import train
from model.config import get_b16_config
from data import cifar_10
from model.vit import Vision_Transformer
from utils import save_model

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(
        description='Main script to train, evaluate, and visualize the Vision Transformer model.'
    )
    parser.add_argument('--mode', type=str, choices=['train', 'visualize'], required=True, 
                        help="Mode of operation: 'train' for training, 'visualize' for attention map visualization")
    parser.add_argument('--pretrained_path', type=str, required=True, 
                        help='Path to the pretrained model weights')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training or evaluation')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                        help='Learning rate for training')
    parser.add_argument('--save_fig', action='store_true', 
                        help='Save the loss and accuracy plot as a PNG file')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to the image for visualization (required in visualize mode)')
    
    return parser.parse_args()


def visualize_attention(image_path, model, device):
    model.eval()

    # 이미지 전처리: 224x224 사이즈, normalize 처리 (ViT 기준)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    # Patch embedding을 사용해 토큰 생성
    tokens = model.patch_embed(image_tensor)  # (B, num_patches, hidden_size)
    B = tokens.shape[0]
    # CLS 토큰 추가 및 위치 임베딩
    cls_tokens = model.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
    tokens = torch.cat((cls_tokens, tokens), dim=1)   # (B, num_tokens, hidden_size)
    tokens = tokens + model.pos_embed

    # 첫 번째 encoder block의 attention 모듈에서 시각화를 위해 vis 옵션 활성화
    attn_module = model.encoder.layers[0].attn
    attn_module.vis = True

    # Attention 모듈 호출 (출력은 무시하고, attention weight를 사용)
    _, attn_probs = attn_module(tokens)  # attn_probs shape: (B, num_heads, num_tokens, num_tokens)

    # 첫 번째 이미지(B=0), 첫 번째 head(=0)의 attention map 선택
    attn = attn_probs[0, 0].detach().cpu().numpy()  # shape: (num_tokens, num_tokens)

    # 시각화: heatmap 생성
    plt.figure(figsize=(8, 8))
    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.title("Attention Map - First Head of First Encoder Block")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    # plt.show() 대신 현재 figure 객체 반환
    return plt.gcf()


if __name__ == "__main__":
    args = parse_args()

    # device 설정 (cuda 사용 가능하면 cuda, 아니면 cpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == 'train':
        # CIFAR-10 데이터로더 준비
        train_loader, test_loader = cifar_10(batch_size=args.batch_size)

        # Vision Transformer 모델 준비 (ViT-B/16 config)
        config = get_b16_config()
        model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3, pretrained=False)
        model = model.to(device)
        # pretrained weight 로드 (여기서 args.pretrained_path 사용)
        pretrained_weights = torch.load(args.pretrained_path, map_location=device)
        model.load_from(pretrained_weights)

        print("Starting training...")
        train.train(model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    device=device,
                    save_fig=args.save_fig)
        save_model(model, "fine_tuned_model.pth")

    elif args.mode == 'visualize':
        # 시각화를 위해 이미지 경로가 필요합니다.
        if args.image_path is None:
            raise ValueError("Visualization mode requires --image_path argument.")
        
        print("Starting visualization...")

        # Vision Transformer 모델 준비 (평가/시각화 모드에서는 pretrained weight를 로드)
        config = get_b16_config()
        model = Vision_Transformer(config, img_size=224, num_classes=10, in_channels=3,
                                   pretrained=True, pretrained_path=args.pretrained_path)
        model = model.to(device)
        # 시각화 모드에서 첫 번째 encoder block의 attention 모듈의 vis 옵션 활성화
        model.encoder.layers[0].attn.vis = True
        
        # 시각화 함수 호출 후 figure 객체 반환
        fig = visualize_attention(args.image_path, model, device)
        # 그림 파일로 저장
        fig.savefig("attention_map.png")
        print("Figure saved as attention_map.png")