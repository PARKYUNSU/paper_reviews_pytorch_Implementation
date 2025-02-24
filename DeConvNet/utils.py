import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

def compute_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total
    return accuracy

def compute_iou(preds, labels, num_classes):
    iou_list = []
    for i in range(num_classes):
        pred_i = (preds == i)
        label_i = (labels == i)
        intersection = (pred_i & label_i).sum().item()
        union = (pred_i | label_i).sum().item()
        iou = intersection / union if union != 0 else 0
        iou_list.append(iou)
    return np.mean(iou_list)

def compute_precision_recall_f1(preds, labels):
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    precision = precision_score(labels.flatten(), preds.flatten(), average='weighted', zero_division=0)
    recall = recall_score(labels.flatten(), preds.flatten(), average='weighted', zero_division=0)
    f1 = f1_score(labels.flatten(), preds.flatten(), average='weighted', zero_division=0)

    return precision, recall, f1


def visualize_segmentation(model, dataloader, device, epoch, save_path="/kaggle/working/"):
    model.eval()
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            break  # Only visualize the first batch

        image = images[0].permute(1, 2, 0).cpu().numpy()
        true_mask = targets[0].squeeze(0).cpu().numpy()
        pred_mask = torch.argmax(outputs[0], dim=0).cpu().numpy()

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[1].imshow(true_mask, cmap="jet")
        axs[1].set_title("True Mask")
        axs[2].imshow(pred_mask, cmap="jet")
        axs[2].set_title("Predicted Mask")

        for ax in axs:
            ax.axis("off")

        plt.savefig(f"{save_path}/epoch_{epoch}.png")
        plt.close(fig)

import matplotlib.pyplot as plt

def plot_metrics(history, output_filename='/kaggle/working/training_metrics.png'):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Training metrics plot saved at {output_filename}")
