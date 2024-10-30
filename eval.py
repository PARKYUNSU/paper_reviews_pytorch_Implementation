import torch
import numpy as np
from metrics import compute_accuracy, compute_iou, compute_precision_recall_f1

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    pixel_accuracies = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            correct = (preds == labels).sum().item()
            total = labels.numel()
            pixel_acc = correct / total
            pixel_accuracies.append(pixel_acc)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    accuracy = compute_accuracy(all_preds, all_labels)
    iou = compute_iou(all_preds, all_labels, num_classes)
    precision, recall, f1 = compute_precision_recall_f1(all_preds, all_labels)
    avg_pixel_acc = np.mean(pixel_accuracies)

    print(f'Pixel Accuracy: {avg_pixel_acc * 100:.2f}%')
    print(f'Overall Accuracy: {accuracy * 100:.2f}%')
    print(f'Mean IoU: {iou:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    return avg_pixel_acc, accuracy, iou, precision, recall, f1
