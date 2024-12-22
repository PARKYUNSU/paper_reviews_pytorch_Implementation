import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model.lstm import LSTM
from data.generate_data import get_dataloaders
from utils import load_checkpoint


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    test_loader, vocab_size = get_dataloaders(data_dir="data", batch_size=32)

    # Load model
    embedding_dim = 100
    hidden_dim = 128
    layer_dim = 2
    output_dim = 2  # Positive or Negative classification
    model = LSTM(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim).to(device)
    
    # Load model checkpoint
    _, _ = load_checkpoint(model, None)  # Load only model weights
    
    # Evaluation
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Accuracy calculation
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_samples * 100
    print(f"Evaluation Loss: {avg_loss:.4f}")
    print(f"Evaluation Accuracy: {accuracy:.2f}%")

    # Save evaluation results
    with open("evaluation_results.txt", "w") as f:
        f.write(f"Evaluation Loss: {avg_loss:.4f}\n")
        f.write(f"Evaluation Accuracy: {accuracy:.2f}%\n")

    return avg_loss, accuracy


if __name__ == "__main__":
    avg_loss, accuracy = evaluate()