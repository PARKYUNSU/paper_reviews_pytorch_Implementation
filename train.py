import torch
from tqdm import tqdm
from utils import build_metrics

def model_train(model, train_loader, epoch, optimizer, device, num_classes=21):
    model.train()
    running_loss = 0.0
    running_mIoU = 0.0
    total = len(train_loader.dataset)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in pbar:
        optimizer.zero_grad()

        # batch에서 image_ids가 없을 경우
        images, labels = batch  # 2개의 값만 받도록 수정
        loss_seg, accuracy, mIoU = build_metrics(model, (images, labels), device, num_classes)

        loss_seg.backward()
        optimizer.step()

        running_loss += loss_seg.item()
        running_mIoU += mIoU

        # 현재 배치의 정확도, mIoU, 손실을 tqdm bar에 표시
        pbar.set_postfix({'loss': loss_seg.item(), 'accuracy': accuracy * 100, 'mIoU': mIoU})

    avg_loss = running_loss / total
    avg_mIoU = running_mIoU / len(train_loader)
    return avg_loss, avg_mIoU
