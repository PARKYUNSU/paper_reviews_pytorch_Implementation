import torch
import torch.nn as nn
import torch.optim as optim
from model.lstm import LSTM
from data.ptb_data import get_dataloaders
from utils import save_checkpoint
import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    input_dim = 50  # 임베딩 차원
    hidden_dim = 128
    layer_dim = 2
    output_dim = 10000  # Vocabulary size (PTB)
    learning_rate = 0.001
    num_epochs = 10
    
    # DataLoader
    train_loader, vocab_size = get_dataloaders(batch_size=32)
    
    # Model, Loss, Optimizer
    model = LSTM(input_dim, hidden_dim, layer_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, output_dim), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        save_checkpoint(model, optimizer, epoch, epoch_loss)
    
    # Loss 그래프 저장
    plt.plot(range(1, num_epochs + 1), loss_history, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.savefig("loss_curve.png")
    print("Loss curve saved as loss_curve.png")

if __name__ == "__main__":
    train()