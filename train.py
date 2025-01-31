import torch
import torch.optim as optim
from model.vit import Vision_Transformer
from data import cifar_10
from model.config import get_b16_config

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, train_loader, test_loader, epochs, learning_rate, device, save_fig=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    
    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)            

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}')
        evaluate(model, test_loader, device)
        
    print('Training finished.')

    plot_metrics(train_losses, test_accuracies, save_fig)

# 평가 함수
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    # tqdm을 사용하여 평가 진행 상황을 시각화
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터도 CUDA로 이동
            outputs = model(inputs)  # 어텐션 맵은 받지 않음
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy: {100 * correct / total:.2f}%')

# loss와 accuracy 그래프를 그리는 함수
def plot_metrics(train_losses, test_accuracies, save_fig=False):
    # Plot Loss
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label="Test Accuracy", color='g')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()

    # 그래프를 PNG 파일로 저장
    if save_fig:
        os.makedirs('./plots', exist_ok=True)
        plt.savefig('./plots/loss_accuracy_plot.png')
        print("Graph saved as 'loss_accuracy_plot.png'")
    else:
        plt.show()