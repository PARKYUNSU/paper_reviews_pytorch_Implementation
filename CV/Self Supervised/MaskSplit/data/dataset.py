import cv2
import numpy as np
import torch
import argparse
import random
from typing import List

from torch.utils.data import Dataset, DataLoader, RandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .classes import get_pascal_splits


def get_train_loader(args):
    """Pascal VOC용 학습 데이터 로더 생성"""
    assert args.train_split in [0, 1, 2, 3]

    train_transform = A.Compose([
        A.RandomResizedCrop(height=args.image_size, width=args.image_size, scale=(0.5, 1.2)),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(limit=30),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    split_classes = get_pascal_splits()
    class_list = split_classes[args.train_split]['train']

    train_data = PascalDataset(transform=train_transform, class_list=class_list, data_list_path=args.train_list)
    return DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)


def get_test_loader(args):
    """Pascal VOC용 테스트 데이터 로더 생성"""
    assert args.test_split in [0, 1, 2, 3, -1]

    val_transform = A.Compose([
        A.Resize(height=args.image_size, width=args.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    split_classes = get_pascal_splits()
    class_list = split_classes[args.test_split]['val']

    val_data = PascalDataset(transform=val_transform, class_list=class_list, data_list_path=args.val_list)
    val_sampler = RandomSampler(val_data, replacement=True, num_samples=2000)
    
    return DataLoader(val_data, batch_size=16, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

def copy_paste_loader(args: argparse.Namespace,
                      normalize: bool = True) -> torch.utils.data.DataLoader:
    """
        Build the train loader. This is a standard loader (not episodic)
    """
    assert args.train_split in [0, 1, 2, 3]

    if args.sup_aug:
        augs_sup = A.Compose([A.RandomResizedCrop(scale=(0.5, 1.2),
                                                  height=args.image_size,
                                                  width=args.image_size),
                              A.HorizontalFlip(),
                              A.VerticalFlip(),
                              A.ColorJitter(),
                              A.GaussianBlur(p=0.5),
                              A.ToGray(p=0.4),
                              A.Rotate(p=0.5),
                              A.Resize(height=args.image_size, width=args.image_size)])
    else:
        augs_sup = A.Compose([A.Resize(height=args.image_size, width=args.image_size)])
    if args.query_aug:
        augs_query = A.Compose([A.RandomResizedCrop(scale=(0.5, 1.2),
                                                    height=args.image_size,
                                                    width=args.image_size),
                                A.HorizontalFlip(),
                                A.VerticalFlip(),
                                A.ColorJitter(),
                                A.GaussianBlur(p=0.5),
                                A.ToGray(p=0.4),
                                A.Rotate(p=0.5)])
    else:
        augs_query = A.Compose([A.Resize(height=args.image_size, width=args.image_size)])

    split_classes = get_split_classes(args)
    if args.use_all_classes:
        class_list = split_classes[args.train_name][args.train_split]['unsup']
    else:
        class_list = split_classes[args.train_name][args.train_split]['train']

    # ===================== Build loader =====================
    train_data = EpisodicDataMaskSplit(transform=augs_query,
                                       sup_transform=augs_sup,
                                       class_list=class_list,
                                       data_list_path=args.train_list,
                                       args=args,
                                       normalize=normalize)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True
                                               )
    return train_loader

class EpisodicDataMaskSplit(Dataset):
    def __init__(self,
                 transform,
                 sup_transform,
                 class_list: List[int],
                 data_list_path: str,
                 args: argparse.Namespace,
                 normalize=True):

        self.shot = args.shot
        self.data_root = args.data_root
        self.class_list = class_list
        self.data_list = make_dataset2(args.data_root, data_list_path, self.class_list)
        self.transform = transform
        self.sup_transform = sup_transform
        self.normalize = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    ToTensorV2()]) if normalize else None
        self.resizedcrop = A.RandomResizedCrop(scale=(0.8, 1.2),
                                               height=args.image_size,
                                               width=args.image_size)
        self.args = args

    def __len__(self):
        return len(self.data_list)

    def masksplit(self, image, label, vertical=True):
        masked_pixels_on_columns = np.sum(label, axis=0 if vertical else 1)
        half_the_masked_pixels = np.sum(label) / 2
        pixel_count_cumulative_sum = np.cumsum(masked_pixels_on_columns)
        comask = abs(pixel_count_cumulative_sum - half_the_masked_pixels)
        half_index = np.argmin(comask)

        shift_val = np.random.randint(low=self.args.vcrop_range[0], high=self.args.vcrop_range[1])

        indices = np.where(label == 1)
        if vertical:
            y_max = image.shape[0]-1
            half_first = (half_index + shift_val, 0)
            half_second = (half_index - shift_val, y_max)
        else:
            x_max = image.shape[1] - 1
            half_first = (0, half_index - shift_val)
            half_second = (x_max, half_index + shift_val)

        y_coords = indices[0]
        x_coords = indices[1]
        perp_vec = (half_second[1] - half_first[1], half_second[0] - half_first[0])

        lower_half = np.where(((x_coords - half_second[0])*perp_vec[0] + (y_coords - half_second[1])*perp_vec[1]) < 0)
        upper_half = np.where(((x_coords - half_second[0])*perp_vec[0] + (y_coords - half_second[1])*perp_vec[1]) > 0)
        
        lower_half_indices = (indices[0][lower_half], indices[1][lower_half])
        upper_half_indices = (indices[0][upper_half], indices[1][upper_half])

        split_p = np.random.rand()
        q_label = np.zeros(label.shape)
        s_label = np.zeros(label.shape)
        if split_p > 0.5 or (not self.args.alternate):
            s_indices, q_indices = lower_half_indices, upper_half_indices
        else:
            s_indices, q_indices = upper_half_indices, lower_half_indices
        s_label[s_indices] = 1
        q_label[q_indices] = 1
        if self.args.vcrop_ignore_support:
            q_label[s_indices] = 255
        else:
            q_label[s_indices] = 1
        return q_label, s_label

    def __getitem__(self, index):
        # ========= Read query image + Chose class =========================
        image_path, label_path, saliency_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)
        label[label != 255] = 0
        label[label == 255] = 1

        vsplit_prob = np.random.rand()
        hsplit_prob = 1-vsplit_prob
        if vsplit_prob < self.args.vsplit_prob and self.args.vsplit:
            q_label, s_label = self.masksplit(image, label, vertical=True)
        elif hsplit_prob <= self.args.hsplit_prob and self.args.hsplit:
            q_label, s_label = self.masksplit(image, label, vertical=False)
        else:
            s_label = np.copy(label)
            q_label = np.copy(label)
        if (q_label.sum() < 2*16*16) or (s_label.sum() < 2*16*16):
            return self.__getitem__(np.random.randint(low=0, high=self.__len__()))

        transformed_query = self.transform(image=image, mask=q_label)  
        transformed_support = self.sup_transform(image=image, mask=s_label)  
        qry_img, target = transformed_query["image"], transformed_query["mask"]

        spprt_img, spprt_mask = transformed_support["image"], transformed_support["mask"]

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        if self.normalize is not None:  # variables are torch tensor
            augmented_image_normalized = self.normalize(image=qry_img, mask=target)
            qry_img, target = augmented_image_normalized["image"], augmented_image_normalized["mask"]

            sup_augmented_image_normalized = self.normalize(image=spprt_img, mask=spprt_mask)
            spprt_img, spprt_mask = sup_augmented_image_normalized["image"], sup_augmented_image_normalized["mask"]

            qry_img = qry_img.float()
            target = target.long()
            spprt_img = spprt_img.float().unsqueeze(0)
            spprt_mask = spprt_mask.long().unsqueeze(0)

        subcls_list = []
        support_image_path_list = []
        return qry_img, target, spprt_img, spprt_mask, subcls_list, support_image_path_list, [image_path]
    
class PascalDataset(Dataset):
    """Pascal VOC 데이터셋 클래스"""
    def __init__(self, transform, class_list, data_list_path):
        self.class_list = class_list
        self.data_list = self.load_data_list(data_list_path)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path, label_path = self.data_list[index]
        
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # 유효한 클래스만 남김
        label_class = np.unique(label).tolist()
        label_class = [c for c in label_class if c in self.class_list]
        assert len(label_class) > 0, "이미지에 포함된 클래스가 없음!"

        # 랜덤 클래스를 선택하여 라벨링
        chosen_class = np.random.choice(label_class)
        label[label != chosen_class] = 0
        label[label == chosen_class] = 1

        # 변환 적용
        transformed = self.transform(image=image, mask=label)
        image, label = transformed["image"], transformed["mask"]

        return image, label

    @staticmethod
    def load_data_list(data_list_path):
        """데이터 목록을 로드"""
        with open(data_list_path, "r") as f:
            lines = f.readlines()
        
        return [tuple(line.strip().split()) for line in lines]