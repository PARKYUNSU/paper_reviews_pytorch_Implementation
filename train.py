import torch
import torch.nn as nn
import torch.optim as optim
from model.lstm import LSTM

def train_model(train_loader, model, criterion, optimizer, num_epochs=1):
    model.train()
    epoch_loss = 0.0  # 단일 스칼라 값으로 초기화

    for sequences, labels in train_loader:
        sequences = sequences.unsqueeze(-1)
        labels = labels.unsqueeze(-1)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 배치 손실을 누적

    epoch_loss /= len(train_loader)  # 평균 손실 계산
    return epoch_loss  # 단일 스칼라 값 반환