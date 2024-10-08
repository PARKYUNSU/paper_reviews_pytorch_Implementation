import torch
from torchvision import transforms

# Hyperparameters
initial_lr = 0.6 
batch_size = 1024
momentum = 0.9
weight_decay = 0.0001
num_epochs = 100
lr_decay_epochs = 30

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터 증강 설정
def get_transforms(model_name='resnet'):
    if 'inception' in model_name.lower():  # Inception 계열 모델인 경우
        resize_size = 299  # Inception-ResNet-v2의 경우 299x299
    else:
        resize_size = 224  # 기본 ResNet은 224x224

    train_transforms = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomResizedCrop(resize_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transforms, val_transforms
