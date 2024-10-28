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

            # 모델 예측
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # 예측 결과와 레이블을 리스트에 추가
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            # 픽셀 정확도 계산 (배치 크기 고려)
            correct = (preds == labels).sum().item()
            total = labels.numel()
            pixel_acc = correct / total
            pixel_accuracies.append(pixel_acc)

    # 모든 배치의 예측 결과를 하나로 결합
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 성능 지표 계산
    accuracy = compute_accuracy(all_preds, all_labels)
    iou = compute_iou(all_preds, all_labels, num_classes)
    precision, recall, f1 = compute_precision_recall_f1(all_preds, all_labels)
    avg_pixel_acc = np.mean(pixel_accuracies)

    # 결과 출력
    print(f'Pixel Accuracy: {avg_pixel_acc * 100:.2f}%')
    print(f'Overall Accuracy: {accuracy * 100:.2f}%')
    print(f'Mean IoU: {iou:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    # 6개의 성능 지표 반환
    return avg_pixel_acc, accuracy, iou, precision, recall, f1
