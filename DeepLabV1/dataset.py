import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as transf_F
import pandas as pd

class_dict_path = '/kaggle/input/camvid/CamVid/class_dict.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_dict: pd.DataFrame = pd.read_csv(class_dict_path)

rgb_to_label_dict: dict[tuple, int] = {
    (row['r'], row['g'], row['b']): idx
    for idx, row in class_dict.iterrows()
}

label_to_rgb_dict: dict[int, tuple] = {
    idx: (row['r'], row['g'], row['b'])
    for idx, row in class_dict.iterrows()
}

def rgb_to_label(image: torch.Tensor) -> torch.Tensor:
    width, height, _ = image.shape
    label_image = torch.zeros(width, height, device=device)

    image = (image * 255).int()

    for rgb, label in rgb_to_label_dict.items():
        rgb_tensor = torch.tensor(rgb, device=device)
        mask = torch.all(image == rgb_tensor, dim=-1)
        label_image[mask] = label
        
    return label_image

def label_to_rgb_tensor(label_tensor: torch.Tensor) -> torch.Tensor:
    height, width = label_tensor.shape
    rgb_image = torch.zeros(3, height, width, dtype=torch.uint8)

    for label, rgb in label_to_rgb_dict.items():
        mask = (label_tensor == label)
        rgb_image[0][mask] = rgb[0]  # Red
        rgb_image[1][mask] = rgb[1]  # Green
        rgb_image[2][mask] = rgb[2]  # Blue

    return rgb_image

def read_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    image = transforms.ToTensor()(image)
    return image

class CamVidDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, augment: bool = False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.Resize((360, 480)),
        ])
        self.img_files = os.listdir(img_dir)
        self.label_files = os.listdir(label_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        label_file = img_file.replace(".png", "_L.png")

        img_path = os.path.join(self.img_dir, img_file)
        label_path = os.path.join(self.label_dir, label_file)

        img = read_image(img_path).to(device)
        label = read_image(label_path).to(device)

        img = self.transform(img)
        label = self.transform(label)
        
        # augmentation
        if self.augment:
            if torch.rand(1) > 0.5:
                img = transf_F.hflip(img)
                label = transf_F.hflip(label)
            
            img = transf_F.pad(img, (10, 10, 10, 10))
            label = transf_F.pad(label, (10, 10, 10, 10))
        
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(360, 480))
            img = transf_F.crop(img, i, j, h, w)
            label = transf_F.crop(label, i, j, h, w)
            
            img = transforms.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0.2)(img)
        
        
        label = rgb_to_label(label.permute(1, 2, 0))  # (C, H, W) -> (H, W, C)

        return img, label.long()