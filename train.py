import torch
from tqdm import tqdm
import numpy as np
from utils import intersection_and_union

def model_train(model, data_loader, criterion, optimizer, epoch, device, num_classes):
    model.train()
    running_loss = 0.0
    intersection_meter, union_meter = 0.0, 0.0  # IoU 계산용

    total = len(data_loader.dataset)

    pbar = tqdm(data_loader)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)  # 모델 예측
        loss = criterion(outputs, labels)  # CrossEntropy 손실 함수
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # mIoU 계산
        outputs_np = outputs.detach().cpu().numpy()  # 예측 결과를 numpy로 변환
        labels_np = labels.cpu().numpy()

        intersection, union = intersection_and_union(outputs_np, labels_np, num_classes)
        intersection_meter += intersection
        union_meter += union

    avg_loss = running_loss / total
    iou = intersection_meter / (union_meter + 1e-10)  # IoU 계산
    mIoU = np.mean(iou)  # Mean IoU 계산

    return avg_loss, mIoU