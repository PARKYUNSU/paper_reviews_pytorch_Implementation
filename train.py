import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import plot_loss
from model.lstm import LSTM

def train(train_loader, valid_loader, input_dim, hidden_dim, num_layers, num_classes, device, num_epochs=20, learning_rate=0.001):
    model = LSTM(num_classes, input_dim, hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses, loss_list = [], [], []
    iter_count = 0
    seq_dim = 28  # MNIST 이미지에서 각 row가 시퀀스처럼 처리됨

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # LSTM 입력 크기에 맞게 변환
            inputs = Variable(images.view(-1, seq_dim, input_dim).to(device))
            targets = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loss_list.append(loss.item())
            iter_count += 1

            # 중간 정확도 및 손실 출력
            if iter_count % 500 == 0:
                correct, total = 0, 0
                model.eval()
                with torch.no_grad():
                    for val_images, val_labels in valid_loader:
                        val_inputs = Variable(val_images.view(-1, seq_dim, input_dim).to(device))
                        val_targets = Variable(val_labels.to(device))
                        val_outputs = model(val_inputs)
                        _, predicted = torch.max(val_outputs, 1)
                        total += val_targets.size(0)
                        correct += (predicted == val_targets).sum().item()
                accuracy = 100 * correct / total
                print(f"Iteration: {iter_count}, Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}%")

        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.4f}")

    plot_loss(train_losses, val_losses)
    return model