import torch
from torch import optim
from model.mobilev3 import MobilenetV3
from utils import get_cifar10_data_loaders
from train import train
from eval import evaluate
import matplotlib.pyplot as plt

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MobilenetV3(num_classes=10).to(device)
    train_loader, val_loader = get_cifar10_data_loaders(batch_size=128)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    num_epochs = 30
    best_acc = 0

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    print(f"Dataset: CIFAR-10")
    print(f"Batch Size: {128}, Learning Rate: {0.001}, Weight Decay: {1e-4}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        # Log losses and accuracies
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Update scheduler
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"ghostnet_best_epoch_{epoch + 1}.pth")
            print("Best model saved!")

    # Plot accuracy and loss
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.show()

if __name__ == "__main__":
    main()