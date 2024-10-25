import torch
from tqdm import tqdm
from utils import AverageMeter, intersectionAndUnionGPU
from utils import calculate_mIoU
VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
               "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tvmonitor"]

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    # tqdm을 이용해 프로그레스 바 표시
    for images, targets in tqdm(train_loader, desc="Training", leave=True):
        images = images.to(device)
        targets = targets.long().to(device)  # targets를 long 타입으로 변환
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # 손실 계산
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    
    # tqdm을 이용한 출력
    tqdm.write(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


from utils import calculate_mIoU  # utils.py에서 mIoU 함수 가져오기

def validate_per_class_iou(model, val_loader, criterion, num_classes, device):
    model.eval()
    class_names = VOC_CLASSES  # 클래스 이름 정의
    iou_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validating"):
            images, targets = images.to(device), targets.long().to(device)
            
            # 모델 출력
            outputs = model(images)
            
            # argmax로 클래스 예측
            outputs = outputs.argmax(dim=1)

            # 클래스별 IoU 계산
            ious = calculate_mIoU(outputs, targets, num_classes)
            
            # 각 클래스별 IoU 업데이트
            for idx, iou in enumerate(ious):
                if not torch.isnan(iou):
                    iou_meter.update(iou.item())

    # 클래스별 IoU 출력
    for idx, iou in enumerate(ious):
        print(f"Class '{class_names[idx]}': IoU = {iou:.4f}")

    # 최종 mIoU 계산
    miou = iou_meter.avg
    print(f"Validation - mIoU: {miou:.4f}")
    return miou
