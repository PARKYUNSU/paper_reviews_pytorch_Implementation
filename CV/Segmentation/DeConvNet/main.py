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
    train_loss = train(model, train_loader, criterion, optimizer, device, epoch, num_epochs)
    history['train_loss'].append(train_loss)
    
    val_loss = evaluate(model, train_loader, criterion, device)
    history['val_loss'].append(val_loss)

    history['train_acc'].append(0.0)
    history['val_acc'].append(0.0)


visualize_segmentation(model, train_loader, device, num_epochs, save_path=f"/kaggle/working/segmentation_final_epoch.png")


plot_metrics(history, output_filename='/kaggle/working/training_metrics_final.png')