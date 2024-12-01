import torch
import torch_xla.core.xla_model as xm

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, targets in train_loader:
        # 이미지 리스트를 Tensor로 변환
        inputs = torch.stack(inputs).to(device)
        # targets는 리스트 그대로 유지 (COCO annotations)

        # Forward pass
        outputs = model(inputs)
        # COCO 데이터셋의 주석 구조에 맞게 라벨을 가져옵니다
        labels = torch.tensor([target[0]['category_id'] for target in targets]).to(device)

        # 손실 계산
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward pass와 Optimizer step
        optimizer.zero_grad()
        loss.backward()
        xm.optimizer_step(optimizer)  # TPU를 사용할 경우 필요

        # Metrics 업데이트
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy
