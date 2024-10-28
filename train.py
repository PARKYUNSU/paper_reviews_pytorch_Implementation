import torch
import time
import matplotlib.pyplot as plt
import random
from metrics import compute_accuracy, compute_iou, compute_precision_recall_f1
from dataset import label_to_rgb_tensor
from early_stopping import EarlyStopping
from PIL import Image

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience, delta):
    model = model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    early_stopping = EarlyStopping(patience=patience, delta=delta)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()

        train_loss = running_loss / len(train_loader)
        train_acc = correct_pixels / total_pixels

        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 매 10 에포크마다 시각화
        if (epoch + 1) % 10 == 0:
            visualize_segmentation(model, val_loader, device, epoch + 1)

        # 조기 종료 확인
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
        
    return model, history

def validate_model(model, val_loader, criterion, device='cuda'):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == labels).sum().item()
            total_pixels += labels.numel()

    val_loss = running_loss / len(val_loader)
    val_acc = correct_pixels / total_pixels

    return val_loss, val_acc

def visualize_segmentation(model, val_loader, device, epoch):
    model.eval()
    batch_idx = random.randint(0, len(val_loader) - 1)
    images, labels = list(val_loader)[batch_idx]
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    img_idx = random.randint(0, len(images) - 1)
    img = images[img_idx].cpu().numpy().transpose(1, 2, 0)
    label = labels[img_idx].cpu().numpy()
    pred = preds[img_idx].cpu().numpy()
    pred_rgb = label_to_rgb_tensor(torch.tensor(pred), label_to_rgb_dict={}).cpu().numpy().transpose(1, 2, 0)
    label_rgb = label_to_rgb_tensor(torch.tensor(label), label_to_rgb_dict={}).cpu().numpy().transpose(1, 2, 0)

    # 시각화 및 이미지 저장
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[1].imshow(label_rgb)
    axes[1].set_title('Ground Truth')
    axes[2].imshow(pred_rgb)
    axes[2].set_title('Predicted Mask')

    # 이미지 파일 저장
    output_path = f'/kaggle/working/segmentation_epoch_{epoch}.png'
    plt.savefig(output_path)
    plt.close()

    print(f"Segmentation visualization saved at {output_path}")

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

    # 그래프를 .png 파일로 저장
    plt.savefig(output_filename)
    plt.close()

    print(f"Training metrics plot saved at {output_filename}")