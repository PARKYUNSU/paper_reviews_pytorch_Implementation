import torch
from tqdm import tqdm

def model_train(model, data_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss, correct_top1, correct_top5 = 0.0, 0.0, 0.0
    total = len(data_loader.dataset)
    
    pbar = tqdm(data_loader)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        
        # Top-1 & Top-5 Accuracy 계산
        _, pred_top1 = outputs.topk(1, 1, True, True)  # Top-1 예측
        _, pred_top5 = outputs.topk(5, 1, True, True)  # Top-5 예측
        
        correct_top1 += pred_top1.eq(labels.view(-1, 1)).sum().item()  # Top-1 예측과 실제 값 비교
        correct_top5 += pred_top5.eq(labels.view(-1, 1).expand_as(pred_top5)).sum().item()  # Top-5 예측과 비교
    
    avg_loss = running_loss / total
    top1_accuracy = 100 * correct_top1 / total  # 정확도 계산 (에러율 아님)
    top1_error_rate = 100 * (1 - correct_top1 / total)
    top5_error_rate = 100 * (1 - correct_top5 / total)

    return avg_loss, top1_accuracy, top1_error_rate, top5_error_rate