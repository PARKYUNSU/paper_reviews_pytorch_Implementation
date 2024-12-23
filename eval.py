import torch
import torch.nn.functional as F

def evaluate(model, val_iter, seq_dim, input_dim, device):
    corrects, total, total_loss = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for images, labels in val_iter:
            # 입력 크기 변환 및 장치 이동
            images = images.view(-1, seq_dim, input_dim).to(device)
            labels = labels.to(device)

            # 모델 출력
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction='sum')

            # 예측값 계산
            _, predicted = torch.max(logits, 1)

            # 정확도 및 손실 합산
            total += labels.size(0)
            corrects += (predicted == labels).sum().item()
            total_loss += loss.item()

    # 평균 손실 및 정확도 계산
    avg_loss = total_loss / len(val_iter.dataset)
    avg_accuracy = 100 * corrects / total

    return avg_loss, avg_accuracy