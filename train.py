import torch
from tqdm import tqdm
from utils import build_metrics

def model_train(model, data_loader, optimizer, epoch, device, num_classes):
    model.train()
    running_loss = 0.0
    running_mIoU = 0.0
    total = len(data_loader.dataset)
    
    pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}")

    for batch in pbar:
        optimizer.zero_grad()

        # build_metrics 사용하여 손실, 정확도, mIoU 계산
        loss_seg, accuracy, mIoU = build_metrics(model, batch, device, num_classes)

        loss_seg.backward()
        optimizer.step()

        running_loss += loss_seg.item()
        running_mIoU += mIoU

        # 현재 배치의 정확도, mIoU, 손실을 tqdm bar에 표시
        pbar.set_postfix({'loss': loss_seg.item(), 'accuracy': accuracy * 100, 'mIoU': mIoU})

    avg_loss = running_loss / total
    avg_mIoU = running_mIoU / len(data_loader)
    return avg_loss, avg_mIoU