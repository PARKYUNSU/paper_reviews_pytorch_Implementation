import torch
import torch.nn as nn
import torch.optim as optim
from model.lstm import LSTM
from data.generate_data import get_dataloaders
from utils import save_checkpoint

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    train_loader, vocab_size = get_dataloaders(data_dir="data", batch_size=32, seq_len=50)
    
    # Hyperparameters
    input_dim = vocab_size
    hidden_dim = 128
    layer_dim = 2
    output_dim = vocab_size
    learning_rate = 0.001
    num_epochs = 10

    # Model, Loss, Optimizer
    model = LSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=1)  # Ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Reshape outputs for CrossEntropyLoss
            outputs = outputs.view(-1, outputs.size(2))
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, epoch_loss)

if __name__ == "__main__":
    train()