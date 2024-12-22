import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model.lstm import LSTM
from data.generate_data import get_dataloaders
from utils import save_checkpoint


def plot_loss(train_losses, val_losses, filename="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    train_loader, val_loader, vocab_size = get_dataloaders(data_dir="data", batch_size=32)
    
    # Hyperparameters
    embedding_dim = 100
    hidden_dim = 128
    layer_dim = 2
    output_dim = 2  # Positive or Negative classification
    learning_rate = 0.001
    num_epochs = 10

    # Model, Loss, Optimizer
    model = LSTM(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()  # Ignore padding index if needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Track losses
    train_losses = []
    val_losses = []
    
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_losses.append(val_loss / len(val_loader))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
        save_checkpoint(model, optimizer, epoch, train_losses[-1])
    
    # Plot Loss
    plot_loss(train_losses, val_losses)


if __name__ == "__main__":
    train()