import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTM
from dataset import SineWaveDataset, generate_data

def train_model(train_loader, model, criterion, optimizer, num_epochs):
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for sequences, labels in train_loader:
            sequences = sequences.unsqueeze(-1)
            labels = labels.unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return train_losses