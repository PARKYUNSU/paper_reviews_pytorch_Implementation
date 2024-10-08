import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from config import *
from train import model_train
from eval import model_eval
from utils import save_checkpoint
import argparse

# Argument parsing for model selection
parser = argparse.ArgumentParser(description='Train and Evaluate SE Models')
parser.add_argument('--model', type=str, required=True, choices=['se_mobile', 'se_resnet50', 'se_resnext50'],
                    help='Choose the model: se_mobile, se_resnet50, se_resnext50')
args = parser.parse_args()

# 데이터셋 로딩 (모델에 맞게 데이터 증강 선택)
train_transforms, val_transforms = get_transforms(args.model)
train_dataset = datasets.CIFAR10(root='/kaggle/working/data', train=True, transform=train_transforms, download=True)
val_dataset = datasets.CIFAR10(root='/kaggle/working/data', train=False, transform=val_transforms, download=True)

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 모델 선택 로직 (model 폴더 안에서 불러오기)
if args.model == 'se_mobile':
    from model.se_mobile import SE_MobileNetV1 as Model
elif args.model == 'se_resnet50':
    from model.se_resnet import se_resnet50 as Model
elif args.model == 'se_resnext50':
    from model.se_resnext import se_resnext50 as Model
# elif args.model == 'se_resnext101':
    # from model.se_resnext import se_resnext101 as Model  # se_resnext101 호출

# 모델, 옵티마이저 및 스케줄러 설정
model = Model().to(device)
optimizer = optim.RMSprop(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = nn.CrossEntropyLoss()

# 학습 및 평가 변수 초기화
losses, accuracies, train_error_rates, val_error_rates = [], [], [], []
best_val_loss = np.inf
early_stopping_counter = 0

# 체크포인트 디렉토리 설정
checkpoint_dir = '/kaggle/working/data/checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# 학습 및 평가 루프
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # 모델 학습
    train_loss, train_accuracy, train_error_rate = model_train(model, train_loader, criterion, optimizer, epoch, device)
    
    # 모델 평가
    val_loss, val_accuracy, val_error_rate = model_eval(model, val_loader, criterion, device)
    
    # 결과 저장
    losses.append([train_loss, val_loss])
    accuracies.append([train_accuracy, val_accuracy])
    train_error_rates.append(train_error_rate)
    val_error_rates.append(val_error_rate)
    
    # 스케줄러 업데이트
    scheduler.step()
    
    # 체크포인트 저장
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
    save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path)
    
    # 결과 출력
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Error Rate: {train_error_rate:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Error Rate: {val_error_rate:.2f}%")
    
    # Early Stopping 체크
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    
    if early_stopping_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

# 최종 모델 저장
model_save_path = f'/kaggle/working/{args.model}_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


# 결과 시각화
epochs = range(1, len(train_error_rates) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_error_rates, label='Train Error Rate')
plt.plot(epochs, val_error_rates, label='Val Error Rate')
plt.xlabel('Epoch')
plt.ylabel('Error Rate (%)')
plt.title('Train vs Val Error Rate')
plt.legend()
plt.show()
