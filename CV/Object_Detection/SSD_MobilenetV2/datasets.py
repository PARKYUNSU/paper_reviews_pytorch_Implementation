import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class PascalVOCSegmentationDataset(Dataset):
    def __init__(self, data_folder, split='train', transform=None):
        """
        Pascal VOC 세그멘테이션 데이터셋 초기화
        param data_folder: Pascal VOC 데이터셋이 저장된 폴더
        param split: 데이터셋 분할 (train/val/test)
        param transform: 이미지에 적용할 변환 함수
        """
        self.split = split
        self.transform = transform

        # JSON 파일로부터 이미지 및 마스크 경로를 불러옴
        with open(os.path.join(data_folder, f'{split.upper()}_images.json'), 'r') as f:
            self.image_paths = json.load(f)
        with open(os.path.join(data_folder, f'{split.upper()}_masks.json'), 'r') as f:
            self.mask_paths = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 이미지 및 마스크 경로
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 이미지 및 마스크 로드
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 마스크는 단일 채널 (L 모드)

        # 변환 적용
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def collate_fn(self, batch):
        """
        배치 내에 서로 다른 크기의 이미지를 처리하기 위한 커스텀 collate 함수.
        """
        images = [b[0] for b in batch]
        masks = [b[1] for b in batch]

        # 이미지를 하나의 텐서로 스택
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)

        return images, masks
