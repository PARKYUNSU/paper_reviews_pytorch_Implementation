import torch
from torchvision import transforms
from PIL import Image

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pascal VOC
pascal = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

# Hyperparameters
initial_lr = 0.001
batch_size = 32
momentum = 0.9
weight_decay = 0.0001
num_epochs = 50
lr_decay_epochs = 30

def get_transforms(input_size=224):
    
    image_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pascal['mean'], std=pascal['std'])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    return image_transform, mask_transform
