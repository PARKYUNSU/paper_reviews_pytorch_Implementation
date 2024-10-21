import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
import config # 하이퍼파라미터 및 transform 설정
from datasets import PascalVOCSegmentationDataset
from model import VGG16_LargeFV
from train import model_train
from eval import model_val

def main():
    # Device 설정
    device = config.device

    # 모델 초기화
    model = VGG16_LargeFV(num_classes=21, input_size=321).to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.initial_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    # Mixed Precision Training 설정
    scaler = torch.cuda.amp.GradScaler()

    # 데이터셋 및 데이터로더 설정
    train_image_transform, train_mask_transform = config.get_transforms(input_size=321)
    val_image_transform, val_mask_transform = config.get_transforms(input_size=321)

    train_dataset = PascalVOCSegmentationDataset(
        data_folder='path_to_dataset', split='train',
        transform=lambda img, mask: (train_image_transform(img), train_mask_transform(mask))
    )
    val_dataset = PascalVOCSegmentationDataset(
        data_folder='path_to_dataset', split='val',
        transform=lambda img, mask: (val_image_transform(img), val_mask_transform(mask))
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

    # 학습 시작
    for epoch in range(config.num_epochs):
        train_loss, train_miou = model_train(model, train_loader, criterion, optimizer, device, scaler, num_classes=21)
        val_loss, val_miou = model_val(model, val_loader, criterion, device, num_classes=21)

        print(f'Epoch [{epoch+1}/{config.num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}')