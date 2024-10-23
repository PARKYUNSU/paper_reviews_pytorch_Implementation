import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import os
import json

def intersection_and_union(pred, label, num_classes):
    pred = pred.flatten()  # 1차원 배열로 변환
    label = label.flatten()  # 1차원 배열로 변환

    # 두 배열의 크기가 동일한지 확인
    if pred.size != label.size:
        raise ValueError(f"Prediction and label sizes do not match: {pred.size}, {label.size}")

    # IoU 계산
    intersection = np.histogram2d(label, pred, bins=num_classes, range=[[0, num_classes], [0, num_classes]])[0]
    area_pred = np.histogram(pred, bins=num_classes, range=(0, num_classes))[0]
    area_label = np.histogram(label, bins=num_classes, range=(0, num_classes))[0]

    union = area_pred + area_label - intersection

    return intersection, union


def resize_labels(labels, size):
    """ 레이블 크기를 조정하기 위한 함수 (nearest interpolation 사용). """
    new_labels = []
    for label in labels:
        label = label.cpu().numpy().astype(np.uint8)
        
        # 배열의 차원이 3차원 이상일 경우, 불필요한 차원을 제거
        if label.ndim > 2:
            label = np.squeeze(label)
        
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    
    new_labels = np.array(new_labels)
    new_labels = torch.LongTensor(new_labels)  # 마스크는 LongTensor로 변환
    return new_labels


def build_metrics(model, batch, device, num_classes, class_weights=None):
    # 클래스 가중치가 없으면 기본값으로 설정
    if class_weights is None:
        CEL = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
    else:
        CEL = torch.nn.CrossEntropyLoss(ignore_index=255, weight=class_weights).to(device)

    images, labels = batch
    labels = resize_labels(labels, size=(224, 224)).to(device)  # 레이블 크기를 모델 출력 크기와 맞춤
    logits = model(images.to(device))

    # 손실 계산 (CrossEntropy Loss)
    loss_seg = CEL(logits, labels)

    # 예측된 클래스 계산
    preds = torch.argmax(logits, dim=1)  # (batch_size, H, W)

    # 정확도 계산
    valid_mask = labels != 255  # ignore_index인 255를 제외한 마스크
    correct = (preds == labels) & valid_mask  # 올바르게 예측한 픽셀
    accuracy = correct.sum().item() / valid_mask.sum().item()  # 유효한 픽셀에 대한 정확도

    # mIoU 계산
    preds_np = preds.detach().cpu().numpy()  # (batch_size, H, W)
    labels_np = labels.cpu().numpy()  # (batch_size, H, W)

    # 예측값과 레이블 크기를 1차원 배열로 변환하여 처리
    preds_np = preds_np.flatten()
    labels_np = labels_np.flatten()

    # 교차 및 합집합 계산
    intersection, union = intersection_and_union(preds_np, labels_np, num_classes)
    
    # IoU 계산
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
            transforms.Resize(size, interpolation=Image.NEAREST),
            transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.uint8))),  # 마스크를 LongTensor로 변환
        ])

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask


def create_segmentation_data_lists(voc07_path, voc12_path, output_folder):
    voc07_path = os.path.abspath(voc07_path)
    voc12_path = os.path.abspath(voc12_path)

    # 훈련 데이터 준비
    train_images = []
    train_masks = []

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
    test_images = []
    test_masks = []

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


def save_segmentation_result(model, image, device, filename="segmentation_result.pth"):
    """세그멘테이션 결과를 .pth 파일로 저장합니다."""
    model.eval()  # 모델을 평가 모드로 전환
    with torch.no_grad():
        image = image.to(device)
        logits = model(image.unsqueeze(0))  # 배치 차원을 추가
        pred = torch.argmax(logits, dim=1).cpu().numpy()  # 예측된 클래스 맵
    
    # 세그멘테이션 결과를 .pth로 저장
    torch.save(pred, filename)
    print(f"Segmentation result saved as {filename}")