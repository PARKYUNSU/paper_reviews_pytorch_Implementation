import torch
from model import GhostNet
from utils import get_data_loaders
from train import train
from eval import evaluate
from torch import optim


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GhostNet(num_classes=10).to(device)
    train_loader, test_loader = get_data_loaders(batch_size=128)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    num_epochs = 200
    best_acc = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "ghostnet_best.pth")
            print("Best model saved!")

    print(f"Best Test Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()