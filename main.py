import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from model.ghost_net import GhostNet
from utils import get_data_loaders
from train import train
from eval import evaluate
import matplotlib.pyplot as plt

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GhostNet(num_classes=10).to(device)
    train_loader, test_loader = get_data_loaders(batch_size=128)

    # FLOPs 계산
    print("\nCalculating FLOPs and Parameters...")
    example_input = torch.randn(1, 3, 32, 32).to(device)  # CIFAR-10은 32x32 크기
    flops = FlopCountAnalysis(model, example_input)
    print(f"FLOPs: {flops.total() / 1e6:.2f} MFLOPs")
    print(parameter_count_table(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    num_epochs = 200
    best_acc = 0

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "ghostnet_best.pth")
            print("Best model saved!")

    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.show()

    # Accuracy plot
    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig("accuracy_curve.png")
    plt.show()

if __name__ == "__main__":
    main()