import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
    parser.add_argument('--save_fig', action='store_true', help='If set, saves the loss plot as a PNG file')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataloader = get_dataloader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        total_samples=args.total_samples
    )

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

    losses = []

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f}")
        losses.append(avg_loss)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), losses, marker='o', label='Training Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # --save_fig 옵션이 주어지면 PNG로 저장, 그렇지 않으면 단순히 show
    if args.save_fig:
        plt.savefig('loss_plot.png')
        print("Loss plot saved as 'loss_plot.png'")
    else:
        plt.show()

if __name__ == '__main__':
    main()