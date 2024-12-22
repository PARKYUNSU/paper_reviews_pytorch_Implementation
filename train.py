import torch
import torch.nn as nn
import torch.optim as optim
from model.lstm import LSTM
from data.generate_data import generate_sine_data
from utils import save_checkpoint

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate data
    X, y = generate_sine_data()
    X, y = X.to(device), y.to(device)
    
    # Hyperparameters
    input_dim = 1
    hidden_dim = 32
    layer_dim = 1
    output_dim = 1
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 32
    
    # Model, Loss, Optimizer
    model = LSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size].unsqueeze(-1)  # Add feature dimension
            y_batch = y[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, epoch_loss)

if __name__ == "__main__":
    train()