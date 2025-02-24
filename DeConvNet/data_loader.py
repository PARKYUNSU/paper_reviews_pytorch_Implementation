import os
import torch
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader

def get_voc_dataloader(batch_size=4):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])
    
    # Load VOC dataset
    dataset = VOCSegmentation(root='/kaggle/working/', year='2012', image_set='train', download=True,
                              transform=transform, target_transform=target_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
