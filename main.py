# main.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_dataloader
from train import train_one_epoch
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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1) Dataloader 준비
    dataloader = get_dataloader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        total_samples=args.total_samples
    )

    # 2) 모델 준비
    model = Transformer(
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)

    # 3) Loss, Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # pad_idx를 무시
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4) 학습 루프
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f}")

if __name__ == '__main__':
    main()