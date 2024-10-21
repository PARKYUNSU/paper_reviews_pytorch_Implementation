import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms

def intersection_and_union(pred, label, num_classes):
    pred = np.argmax(pred, axis=1)  # 예측된 클래스
    pred = pred.flatten()  # 1차원 배열로 변환
    label = label.flatten()  # 1차원 배열로 변환

    intersection = np.histogram2d(label, pred, bins=num_classes, range=[[0, num_classes], [0, num_classes]])[0]
    area_pred = np.histogram(pred, bins=num_classes, range=(0, num_classes))[0]
    area_label = np.histogram(label, bins=num_classes, range=(0, num_classes))[0]

    union = area_pred + area_label - intersection

    return intersection, union

def resize_labels(labels, size):
    """
    레이블 크기를 조정하기 위한 함수 (nearest interpolation 사용).
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels

def build_metrics(model, batch, device, num_classes):
    """
    손실, 정확도 및 mIoU 계산.
    """
    CEL = nn.CrossEntropyLoss(ignore_index=255).to(device)

    image_ids, images, labels = batch
    labels = resize_labels(labels, size=(41, 41)).to(device)  # 레이블 크기 조정
    logits = model(images.to(device))

    # 손실 계산 (CrossEntropy Loss)
    loss_seg = CEL(logits, labels)

    # 예측된 클래스 계산
    preds = torch.argmax(logits, dim=1)
    
    # 정확도 계산
    accuracy = float(torch.eq(preds, labels).sum().cpu()) / (len(image_ids) * logits.shape[2] * logits.shape[3])

    # mIoU 계산
    preds_np = preds.detach().cpu().numpy()
    labels_np = labels.cpu().numpy()
    intersection, union = intersection_and_union(preds_np, labels_np, num_classes)
    iou = intersection / (union + 1e-10)
    mIoU = np.mean(iou)

    return loss_seg, accuracy, mIoU


class SegmentationTransform:
    def __init__(self, size):
        self.image_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()  # 마스크는 Normalize 하지 않음
        ])

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = torch.squeeze(mask, dim=0)  # 마스크는 (1, H, W)이므로 (H, W)로 변환
        return image, mask


import os
import json

def create_segmentation_data_lists(voc07_path, voc12_path, output_folder):
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    # 훈련 데이터 준비
    train_images = list()
    train_masks = list()

    for path in [voc07_path, voc12_path]:
        with open(os.path.join(path, 'ImageSets/Segmentation/trainval.txt')) as f:
            ids = f.read().splitlines()

        for id in ids:
            # 이미지 경로와 세그멘테이션 마스크 경로 저장
            image_path = os.path.join(path, 'JPEGImages', id + '.jpg')
            mask_path = os.path.join(path, 'SegmentationClass', id + '.png')

            # 이미지 및 마스크 경로 추가
            train_images.append(image_path)
            train_masks.append(mask_path)

    # JSON 파일로 저장
    assert len(train_images) == len(train_masks)

    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_masks.json'), 'w') as j:
        json.dump(train_masks, j)

    print(f'\n{len(train_images)} training images and masks have been saved to {os.path.abspath(output_folder)}.')

    # 테스트 데이터 준비 (VOC 2007)
    test_images = list()
    test_masks = list()

    with open(os.path.join(voc07_path, 'ImageSets/Segmentation/test.txt')) as f:
        ids = f.read().splitlines()

    for id in ids:
        image_path = os.path.join(voc07_path, 'JPEGImages', id + '.jpg')
        mask_path = os.path.join(voc07_path, 'SegmentationClass', id + '.png')

        test_images.append(image_path)
        test_masks.append(mask_path)

    # JSON 파일로 저장
    assert len(test_images) == len(test_masks)

    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_masks.json'), 'w') as j:
        json.dump(test_masks, j)

    print(f'\n{len(test_images)} test images and masks have been saved to {os.path.abspath(output_folder)}.')