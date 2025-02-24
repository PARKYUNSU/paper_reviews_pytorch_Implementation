import torch

def model_eval(model, data_loader, criterion, device):
    model.eval()
    running_loss, correct_top1, correct_top5 = 0.0, 0.0, 0.0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Top-1 & Top-5 Accuracy 계산
            _, pred_top1 = outputs.topk(1, 1, True, True)  # Top-1 예측
            _, pred_top5 = outputs.topk(5, 1, True, True)  # Top-5 예측

            correct_top1 += pred_top1.eq(labels.view(-1, 1)).sum().item()  # Top-1 예측과 실제 값 비교
            correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()  # Top-5 예측과 비교

            running_loss += loss.item() * images.size(0)
            total += labels.size(0)

    avg_loss = running_loss / total
    top1_error_rate = 100 * (1 - correct_top1 / total)  # Top-1 Error Rate 계산
    top5_error_rate = 100 * (1 - correct_top5 / total)  # Top-5 Error Rate 계산

    return avg_loss, top1_error_rate, top5_error_rate
