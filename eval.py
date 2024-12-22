import torch
import torch.nn as nn
from model.lstm import LSTM
from data.generate_data import get_dataloaders
from utils import load_checkpoint

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    _, test_loader, vocab_size = get_dataloaders(data_dir="data", batch_size=32)

    # Load model
    input_dim = vocab_size
    hidden_dim = 128
    layer_dim = 2
    output_dim = vocab_size
    model = LSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    
    # Load model checkpoint
    _, _ = load_checkpoint(model, None)  # Load only model weights
    
    # Evaluation
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad> token during loss calculation

    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(2))  # Reshape for CrossEntropyLoss
            targets = targets.view(-1)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Accuracy calculation
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.ne(0).sum().item()  # Exclude padding tokens

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