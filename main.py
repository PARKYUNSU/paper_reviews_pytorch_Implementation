import torch
from torch import nn
from torch.optim import Adam
from model import DeconvNet
from data_loader import get_voc_dataloader
from train import train
from eval import evaluate
from utils import *

model = DeconvNet(num_classes=21)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = get_voc_dataloader(batch_size=4)
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = Adam(model.parameters(), lr=0.001)
num_epochs = 100

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    
    val_loss, val_acc = evaluate(model, train_loader, criterion, device)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # 10 에포크마다 세그멘테이션 결과 시각화
    visualize_segmentation(model, train_loader, device, epoch, save_path="/kaggle/working/")


# 학습 및 검증 손실과 정확도를 저장된 경로에 시각화
plot_metrics(history, output_filename='/kaggle/working/training_metrics_final.png')
