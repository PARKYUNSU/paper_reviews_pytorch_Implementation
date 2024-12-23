import torch
import torch.nn as nn
from model.lstm import LSTM
from utils import plot_loss

def train(train_loader, valid_loader, input_dim, hidden_dim, num_layers, num_classes, device, num_epochs=20, learning_rate=0.001):
    model = LSTM(num_classes, input_dim, hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            # LSTM 입력 크기에 맞게 변환
            inputs = images.view(-1, 28, 28).to(device)  # seq_dim=28, input_dim=28
            targets = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}")

    plot_loss(train_losses, val_losses)
    return model