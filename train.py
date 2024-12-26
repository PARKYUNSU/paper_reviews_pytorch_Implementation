import torch

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    한 epoch 동안 모델 학습을 수행하는 함수.

    Args:
        model (nn.Module): 학습할 모델.
        dataloader (DataLoader): 학습 데이터 로더.
        criterion (nn.Module): 손실 함수.
        optimizer (Optimizer): 옵티마이저.
        device (torch.device): 연산 장치 (CPU 또는 CUDA).

    Returns:
        tuple: 평균 학습 손실, 학습 정확도
    """
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        hidden = model.init_hidden(inputs.size(0), device)

        # Gradients 초기화
        optimizer.zero_grad()

        # Forward
        outputs = model(inputs, hidden).squeeze()

        # Loss 계산
        loss = criterion(outputs, labels)

        # Backward
        loss.backward()

        # Parameter 업데이트
        optimizer.step()

        # Loss 및 Accuracy 집계
        train_loss += loss.item()
        preds = (outputs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # 평균 Loss 및 Accuracy 계산
    avg_loss = train_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy