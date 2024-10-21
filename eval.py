import torch
from utils import intersection_and_union

def model_val(model, data_loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    intersection_meter, union_meter = 0.0, 0.0

    total = len(data_loader.dataset)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # 모델 예측
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            outputs_np = outputs.detach().cpu().numpy()
            labels_np = labels.cpu().numpy()

            # mIoU 계산
            intersection, union = intersection_and_union(outputs_np, labels_np, num_classes)
            intersection_meter += intersection
            union_meter += union

    avg_loss = running_loss / total
    iou = intersection_meter / (union_meter + 1e-10)
    mIoU = np.mean(iou)  # Mean IoU 계산

    return avg_loss, mIoU