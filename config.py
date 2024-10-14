import torch
from torchvision import transforms

# Hyperparameters
initial_lr = 0.001
batch_size = 32
momentum = 0.9
weight_decay = 0.0001
num_epochs = 100
lr_decay_epochs = 30

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# cifar = {'mean' : [0.4914, 0.4822, 0.4465], 'std' : [0.2023, 0.1994, 0.2010]}

animals = {'mean' : [0.5177, 0.5003, 0.4126], 'std' : [0.2133, 0.2130, 0.2149]}

def get_transforms():

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=animals['mean'], std=animals['std'])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=animals['mean'], std=animals['std'])
    ])

    return train_transforms, val_transforms