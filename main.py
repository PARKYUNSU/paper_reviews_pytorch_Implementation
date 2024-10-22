import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import config
from datasets import PascalVOCSegmentationDataset
from model import VGG16_LargeFV
from train import model_train
from eval import model_val

def main():
    # Device
    device = config.device

    # Model
    model = VGG16_LargeFV(num_classes=21, input_size=321).to(device)
    
    # criterion, optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.initial_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    # Mixed Precision Training(Fp16)
    scaler = torch.cuda.amp.GradScaler()

    # Dataset Load
    train_image_transform, train_mask_transform = config.get_transforms(input_size=321)
    val_image_transform, val_mask_transform = config.get_transforms(input_size=321)

    train_dataset = PascalVOCSegmentationDataset(
        data_folder='/kaggle/working',  # 데이터 파일들이 저장된 경로
        split='train',  # 훈련용 데이터를 사용할 때 'train'
        transform=lambda img, mask: (train_image_transform(img), train_mask_transform(mask))
    )

    test_dataset = PascalVOCSegmentationDataset(
        data_folder='/kaggle/working',  # 데이터 파일들이 저장된 경로
        split='test',  # 검증 또는 테스트용 데이터를 사용할 때 'test'
        transform=lambda img, mask: (val_image_transform(img), val_mask_transform(mask))
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)

    # Train

    for epoch in range(config.num_epochs):
        # epoch 인자를 추가하여 호출
        train_loss, train_miou = model_train(model, train_loader, epoch, optimizer, device)
        val_loss, val_miou = model_val(model, val_loader, criterion, device)

        print(f'Epoch [{epoch+1}/{config.num_epochs}], '
            f'Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, '
            f'Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}')

if __name__ == "__main__":
    main()