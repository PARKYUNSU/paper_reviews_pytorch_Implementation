import torch
from model.lstm import LSTM
from data.generate_data import generate_sine_data
from utils import load_checkpoint

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate data
    X, y = generate_sine_data()
    X, y = X.to(device), y.to(device)
    
    # Load model
    input_dim = 1
    hidden_dim = 32
    layer_dim = 1
    output_dim = 1
    model = LSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    
    _, _ = load_checkpoint(model, None)  # Load only model weights
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions = []
        for i in range(len(X)):
            input_seq = X[i].unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions
            output = model(input_seq)
            predictions.append(output.item())
        
    print("Evaluation Complete")
    return predictions

if __name__ == "__main__":
    preds = evaluate()
    print("Sample Predictions:", preds[:10])