import torch

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    corrects = 0
    total_pixels = 0  # 픽셀 단위 정확도 계산을 위한 변수

    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets.squeeze(1).long())
            total_loss += loss.item()
            
            # 픽셀 단위 정확도 계산
            _, preds = torch.max(outputs, dim=1)
            corrects += torch.sum(preds == targets.squeeze(1))
            total_pixels += targets.numel()  # 총 픽셀 수

    average_loss = total_loss / len(dataloader)
    accuracy = (corrects / total_pixels).item() * 100  # 픽셀 기반 정확도 계산
    print(f"Evaluation complete. Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    return average_loss, accuracy
