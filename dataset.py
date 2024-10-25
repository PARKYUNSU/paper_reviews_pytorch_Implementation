import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

VOC_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]

VOC_COLORMAP = [
    [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
    [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
    [0, 192, 0], [128, 192, 0], [0, 64, 128]
]

class VOCDataset(Dataset):
    def __init__(self, root, is_train=True, transform=None):
        self.root = root
        self.transform = transform
        img_root = os.path.join(self.root, "ImageSets", "Segmentation", "train.txt" if is_train else "val.txt")
        with open(img_root, 'r') as f:
            self.img_names = f.read().splitlines()

    def __len__(self):
        return len(self.img_names)

    def colormap2label(self, mask):
        """Convert RGB mask to label map."""
        label_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        for idx, color in enumerate(VOC_COLORMAP):
            label_map[np.all(mask == color, axis=-1)] = idx
        return label_map
        
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img = cv2.imread(os.path.join(self.root, "JPEGImages", img_name + ".jpg"))
        mask = cv2.imread(os.path.join(self.root, "SegmentationClass", img_name + ".png"))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)  # Convert mask to RGB
        
        # Convert mask from RGB to label map
        mask = self.colormap2label(mask)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)  # Pass as named arguments
            img = augmented['image']
            mask = augmented['mask']

        return img, mask