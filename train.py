import torch
from tqdm import tqdm

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # tqdm으로 학습 진행 표시
    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        # COCO 주석이 없는 이미지를 건너뜀
        if any(len(target) == 0 for target in targets):
            continue

        # 이미지 리스트를 Tensor로 변환
        inputs = torch.stack(inputs).to(device)
        
        # 주석에서 category_id 추출
        labels = torch.tensor([target[0]['category_id'] for target in targets if len(target) > 0]).to(device)

        # Forward pass
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward pass와 Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics 업데이트
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = 100.0 * correct / total if total > 0 else 0
    return avg_loss, accuracy