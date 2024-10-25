import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss

from dataset import VOCDataset
from model import VGG16_FCN
from train import train, validate_per_class_iou

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 트레이닝용 이미지 변환 (이미지 크기 통일)
train_transform = A.Compose([
    A.Resize(224, 224),  # 이미지 크기를 224x224로 고정
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# 검증용 이미지 변환 (이미지 크기 통일)
val_transform = A.Compose([
    A.Resize(224, 224),  # 이미지 크기를 224x224로 고정
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def main():
    # CUDA 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(device)}")

    # 파라미터
    epochs = 100
    batch_size = 8
    learning_rate = 1e-6
    num_classes = 21
    voc_root_2007 = '/kaggle/input/voc0712/VOC_dataset/VOCdevkit/VOC2007'
    voc_root_2012 = '/kaggle/input/voc0712/VOC_dataset/VOCdevkit/VOC2012'
    save_path = "/kaggle/working/best_model.pt"
    
    # 2007과 2012 데이터를 각각 로드
    # VOCDataset 로드 시 변환 적용
    train_dataset_2007 = VOCDataset(root=voc_root_2007, is_train=True, transform=train_transform)
    train_dataset_2012 = VOCDataset(root=voc_root_2012, is_train=True, transform=train_transform)
    val_dataset_2007 = VOCDataset(root=voc_root_2007, is_train=False, transform=val_transform)
    val_dataset_2012 = VOCDataset(root=voc_root_2012, is_train=False, transform=val_transform)

    
    # 두 데이터셋을 결합
    train_dataset = ConcatDataset([train_dataset_2007, train_dataset_2012])
    val_dataset = ConcatDataset([val_dataset_2007, val_dataset_2012])

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16_FCN(num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    
    best_miou = float('-inf')  # mIoU를 초기화

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        miou = validate_per_class_iou(model, val_loader, criterion, num_classes, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, mIoU: {miou:.4f}")

        # mIoU를 기준으로 모델 저장
        if miou > best_miou:
            best_miou = miou
            print(f"New best model found at epoch {epoch+1}. Saving model...")
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()