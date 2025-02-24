import torch
import torch.optim as optim
from model.vit import Vision_Transformer
from data import cifar_10
from model.config import get_b16_config

import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model, train_loader, test_loader, epochs, learning_rate, optimizer, criterion, device, save_fig=False):

    model = model.to(device)
    
    train_losses = []
    train_accuracies = []
    eval_losses = []
    eval_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
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
        train_accuracies.append(epoch_accuracy)
        
        eval_acc, eval_loss = evaluate(model, test_loader, device)
        eval_accuracies.append(eval_acc)
        eval_losses.append(eval_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}% | "
              f"Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%")
        
    print('Training finished.')
    plot_metrics(train_losses, train_accuracies, eval_losses, eval_accuracies, save_fig)


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return accuracy, avg_loss

def plot_metrics(train_losses, train_accuracies, eval_losses, eval_accuracies, save_fig=False):
    plt.figure(figsize=(12,5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(eval_losses, label="Eval Loss", linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(eval_accuracies, label="Eval Accuracy", linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy')
    plt.legend()

    if save_fig:
        os.makedirs('./plots', exist_ok=True)
        plt.savefig('./plots/loss_accuracy_plot.png')
        print("Graph saved as 'loss_accuracy_plot.png'")
    else:
        plt.show()