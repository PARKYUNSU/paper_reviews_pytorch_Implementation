import torch

def evaluate(model, val_loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            # 이미지 리스트를 Tensor로 변환
            inputs = torch.stack(inputs).to(device)
            # targets는 리스트 그대로 유지 (COCO annotations)

            # Forward pass
            outputs = model(inputs)
            # COCO 데이터셋의 주석 구조에 맞게 라벨을 가져옵니다
            labels = torch.tensor([target[0]['category_id'] for target in targets]).to(device)

            # 손실 계산
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            # Metrics 업데이트
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy
