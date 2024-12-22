import torch
from model.lstm import LSTM
from data.ptb_data import get_dataloaders
from utils import load_checkpoint
from torch.nn.functional import cross_entropy
import numpy as np

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DataLoader
    _, vocab_size = get_dataloaders(batch_size=1)
    eval_loader, _ = get_dataloaders(split="valid", batch_size=1)
    
    # Load model
    input_dim = 50  # 임베딩 차원
    hidden_dim = 128
    layer_dim = 2
    output_dim = vocab_size
    model = LSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    
    _, _ = load_checkpoint(model, None)  # Load only model weights
    
    # Evaluation
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = cross_entropy(outputs.view(-1, output_dim), targets.view(-1))
            total_loss += loss.item()
    
    perplexity = np.exp(total_loss / len(eval_loader))
    print(f"Validation Perplexity: {perplexity:.4f}")

if __name__ == "__main__":
    evaluate()