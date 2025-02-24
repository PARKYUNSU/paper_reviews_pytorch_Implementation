import torch
import numpy as np
from metrics import compute_accuracy, compute_iou, compute_precision_recall_f1
from model import DenseCRF  # CRF 클래스를 import

def evaluate_model(model, dataloader, device, num_classes, use_crf=False, crf_params=None):
    """
    모델 평가 함수. CRF 후처리 적용 여부와 CRF 파라미터를 인자로 받습니다.

    Args:
        model: 평가할 모델
        dataloader: 데이터로더 (테스트 세트)
        device: GPU 또는 CPU
        num_classes: 클래스 수
        use_crf: CRF 후처리 사용 여부
        crf_params: CRF 파라미터 (iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std)

    Returns:
        다양한 평가 지표 (평균 픽셀 정확도, IoU, Precision, Recall, F1)
    """
    model.eval()
    all_preds, all_labels, pixel_accuracies = [], [], []

    # CRF 파라미터 설정
    if use_crf:
        iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std = crf_params
        crf = DenseCRF(iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # CRF 후처리 적용 여부 확인
            if use_crf:
                crf_preds = []
                for i in range(images.size(0)):
                    crf_pred = crf(images[i].cpu(), outputs[i].cpu())
                    crf_pred = torch.from_numpy(crf_pred).float().cuda(device)
                    crf_pred = torch.argmax(crf_pred, dim=0)
                    crf_preds.append(crf_pred)
                preds = torch.stack(crf_preds).to(device)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            correct = (preds == labels).sum().item()
            total = labels.numel()
            pixel_acc = correct / total
            pixel_accuracies.append(pixel_acc)

    # 예측 및 레이블 연결
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # 평가 지표 계산
    accuracy = compute_accuracy(all_preds, all_labels)
    iou = compute_iou(all_preds, all_labels, num_classes)
    precision, recall, f1 = compute_precision_recall_f1(all_preds, all_labels)
    avg_pixel_acc = np.mean(pixel_accuracies)

    # 평가 결과 출력
    print(f'Pixel Accuracy: {avg_pixel_acc * 100:.2f}%')
    print(f'Overall Accuracy: {accuracy * 100:.2f}%')
    print(f'Mean IoU: {iou:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')

    return avg_pixel_acc, accuracy, iou, precision, recall, f1
