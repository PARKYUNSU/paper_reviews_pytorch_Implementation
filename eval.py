import torch
import matplotlib.pyplot as plt

def evaluate_model(test_loader, model, save_path=None):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.unsqueeze(-1)  # [batch_size, seq_length, 1]
            labels = labels.unsqueeze(-1)  # [batch_size, 1]
            outputs = model(sequences)
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(labels.squeeze().tolist())

    # 예측 결과 vs 실제 값 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label="Actual", linestyle="--", alpha=0.7)
    plt.plot(predictions, label="Predicted", alpha=0.9)
    plt.legend()
    plt.title("Actual vs Predicted (Test Data)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")

    if save_path:  # 파일 저장
        plt.savefig(save_path)
        print(f"Prediction graph saved to {save_path}")
    else:
        plt.show()

def plot_training_curves(train_losses, save_path=None):
    # Loss 변화 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if save_path:  # 파일 저장
        plt.savefig(save_path)
        print(f"Training loss curve saved to {save_path}")
    else:
        plt.show()