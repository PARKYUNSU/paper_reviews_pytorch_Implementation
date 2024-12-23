import torch
from train import train
from eval import evaluate
from data.generate_data import get_dataloaders

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    train_loader, valid_loader, test_loader, input_dim = get_dataloaders(batch_size=64)

    # 학습
    model = train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        input_dim=28,  # seq_dim의 입력 크기
        hidden_dim=128,
        num_layers=2,
        num_classes=10,  # MNIST 클래스 개수
        device=device
    )

    # 평가
    evaluate(model, test_loader, device)

if __name__ == "__main__":
    main()