import torch
from tqdm import tqdm
from utils import build_metrics

def model_val(model, data_loader, device, num_classes):
    model.eval()
    running_loss = 0.0
    running_mIoU = 0.0
    total = len(data_loader.dataset)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            # build_metrics 사용하여 손실, 정확도, mIoU 계산
            loss_seg, accuracy, mIoU = build_metrics(model, batch, device, num_classes)

            running_loss += loss_seg.item()
            running_mIoU += mIoU

    avg_loss = running_loss / total
    avg_mIoU = running_mIoU / len(data_loader)
    return avg_loss, avg_mIoU