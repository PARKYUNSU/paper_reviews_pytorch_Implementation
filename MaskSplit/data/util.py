import os
import cv2
import numpy as np
from typing import List, Tuple
from collections import defaultdict


def is_image_file(filename):
    """파일이 이미지 파일인지 확인"""
    return filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))


def make_dataset(data_root: str, data_list: str, class_list: List[int]) -> List[Tuple[str, str]]:
    """Pascal VOC 데이터셋 경로 로드"""
    if not os.path.isfile(data_list):
        raise RuntimeError(f"파일이 존재하지 않음: {data_list}")

    image_label_list = []
    with open(data_list, "r") as f:
        lines = f.readlines()

    for line in lines:
        img_path, label_path = line.strip().split()
        img_path = os.path.join(data_root, img_path)
        label_path = os.path.join(data_root, label_path)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_classes = np.unique(label).tolist()
        label_classes = [c for c in label_classes if c in class_list]

        if len(label_classes) > 0:
            image_label_list.append((img_path, label_path))

    return image_label_list