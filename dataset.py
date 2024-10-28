import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as transf_F
import pandas as pd

class CamVidDataset(Dataset):
    def __init__(self, img_dir: str, label_dir: str, augment: bool = False, rgb_to_label_dict: dict = None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.augment = augment
        if rgb_to_label_dict is None:
            raise ValueError("rgb_to_label_dict must be provided and cannot be None.")
        self.rgb_to_label_dict = rgb_to_label_dict  # 변환 사전 저장
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

        img = read_image(img_path)
        label = read_image(label_path)

        img = self.transform(img)
        label = self.transform(label)

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

        # rgb_to_label 함수 호출 시 변환 사전을 전달
        label = rgb_to_label(label.permute(1, 2, 0), self.rgb_to_label_dict)

        return img, label.long()

def read_image(image_path: str) -> torch.Tensor:
    image = Image.open(image_path)
    return transforms.ToTensor()(image)

def load_class_dict(class_dict_path: str) -> pd.DataFrame:
    return pd.read_csv(class_dict_path)

def create_conversion_dicts(class_dict: pd.DataFrame):
    rgb_to_label_dict = {
        (row['r'], row['g'], row['b']): idx
        for idx, row in class_dict.iterrows()
    }

    label_to_rgb_dict = {
        idx: (row['r'], row['g'], row['b'])
        for idx, row in class_dict.iterrows()
    }

    return rgb_to_label_dict, label_to_rgb_dict

def rgb_to_label(image: torch.Tensor, rgb_to_label_dict: dict) -> torch.Tensor:
    width, height, _ = image.shape
    label_image = torch.zeros(width, height, device=image.device)

    image = (image * 255).int()

    for rgb, label in rgb_to_label_dict.items():
        rgb_tensor = torch.tensor(rgb, device=image.device)
        mask = torch.all(image == rgb_tensor, dim=-1)
        label_image[mask] = label

    return label_image

def label_to_rgb_tensor(label_tensor: torch.Tensor, label_to_rgb_dict: dict) -> torch.Tensor:
    height, width = label_tensor.shape
    rgb_image = torch.zeros(3, height, width, dtype=torch.uint8)

    for label, rgb in label_to_rgb_dict.items():
        mask = (label_tensor == label)
        rgb_image[0][mask] = rgb[0]
        rgb_image[1][mask] = rgb[1]
        rgb_image[2][mask] = rgb[2]

    return rgb_image
