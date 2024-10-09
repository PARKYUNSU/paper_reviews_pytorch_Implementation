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

# 모델 찾기
parser = argparse.ArgumentParser(description='Train and Evaluate SE Models')
parser.add_argument('--model', type=str, required=True, choices=['se_mobile', 'se_resnet50', 'se_resnext50', 'se_vgg16'],
                    help='Choose the model: se_mobile, se_resnet50, se_resnext50, se_vgg16')
args = parser.parse_args()

# 데이터셋 로딩
train_transforms, val_transforms = get_transforms()
train_dataset = datasets.CIFAR10(root='/kaggle/working/data', train=True, transform=train_transforms, download=True)
val_dataset = datasets.CIFAR10(root='/kaggle/working/data', train=False, transform=val_transforms, download=True)

print(len(train_dataset))
print(len(val_dataset))

# DataLoader 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 모델 선택
if args.model == 'se_mobile':
    from model.se_mobile import SE_MobileNetV1 as Model
elif args.model == 'se_resnet50':
    from model.se_resnet import se_resnet50 as Model
elif args.model == 'se_resnext50':
    from model.se_resnext import se_resnext50 as Model
elif args.model == 'se_vgg16':
    from model.se_vgg16 import SE_VGG16 as Model

# 모델, 옵티마이저 
model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
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
    train_loss, train_accuracy, train_error_rate, train_top5_error_rate = model_train(model, train_loader, criterion, optimizer, epoch, device)
    
    # 모델 평가 (Top-1 & Top-5 Error Rate 추가)
    val_loss, top1_error_rate, top5_error_rate = model_eval(model, val_loader, criterion, device)
    
    # 결과 저장
    losses.append([train_loss, val_loss])
    accuracies.append(train_accuracy)  # Train Accuracy 저장
    train_error_rates.append(train_error_rate)
    val_error_rates.append([top1_error_rate, top5_error_rate])

    # 스케줄러 업데이트
    scheduler.step()
    
    # 결과 출력
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train Error Rate: {train_error_rate:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Top-1 Error Rate: {top1_error_rate:.2f}%, Top-5 Error Rate: {top5_error_rate:.2f}%")
    # # Early Stopping 체크
    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     early_stopping_counter = 0
    # else:
    #     early_stopping_counter += 1
    
    # if early_stopping_counter >= patience:
    #     print(f"Early stopping triggered at epoch {epoch + 1}")
    #     break


# 학습 종료 후 최종 체크포인트 저장
checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path)
print(f"Final checkpoint saved after epoch {num_epochs}")

# 최종 모델 저장
model_save_path = f'/kaggle/working/{args.model}_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# 결과 시각화
epochs = range(1, len(train_error_rates) + 1)
plt.figure(figsize=(10, 5))

# Train Accuracy Plot
plt.plot(epochs, accuracies, label='Train Accuracy')
# Train Error Rate와 Val Error Rate 비교
plt.plot(epochs, train_error_rates, label='Train Error Rate')
plt.plot(epochs, [e[0] for e in val_error_rates], label='Top-1 Error Rate')
plt.plot(epochs, [e[1] for e in val_error_rates], label='Top-5 Error Rate')

plt.xlabel('Epoch')
plt.ylabel('Accuracy/Error Rate (%)')
plt.title('Train Accuracy and Error Rate vs Val Error Rate')
plt.legend()
plt.show()

