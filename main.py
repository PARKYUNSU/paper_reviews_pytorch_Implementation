import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data import get_dataloader
from train import train_one_epoch, validate_one_epoch
from model.transformer import Transformer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--max_seq_len', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--total_samples', type=int, default=1000)
    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_fig', action='store_true', help='If set, saves the loss plot as a PNG file')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1) Train, Validation Dataloader 구분 (예시)
    # 실제로는 get_dataloader(mode='train') / get_dataloader(mode='val') 로 구분하는 방식을 권장.
    # 여기서는 예시로 total_samples=1000이 train, total_samples=200이 val이라 가정.
    train_dataloader = get_dataloader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        total_samples=args.total_samples
    )
    val_dataloader = get_dataloader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        total_samples=200  # validation set 크기 예시
    )

    # 2) 모델, Loss, Optimizer 생성
    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3) 학습 & 검증 루프
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss = validate_one_epoch(model, val_dataloader, criterion, device)

        print(f"Epoch [{epoch}/{args.epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # 4) 결과 시각화 (Train/Val Loss)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, marker='o', label='Train Loss')
    plt.plot(range(1, args.epochs + 1), val_losses, marker='x', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # --save_fig 옵션 처리
    if args.save_fig:
        plt.savefig('loss_plot.png')
        print("Loss plot saved as 'loss_plot.png'")
    else:
        plt.show()

if __name__ == '__main__':
    main()