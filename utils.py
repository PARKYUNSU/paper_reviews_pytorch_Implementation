from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms

# COCO 라벨 매핑 (category_id -> 0-79)
COCO_CATEGORY_MAPPING = {
    id: i for i, id in enumerate(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
            67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90
        ]
    )
}

def collate_fn(batch):
    """
    DataLoader에서 배치를 구성하기 위해 사용되는 함수.
    Args:
        batch (list): [(image, target), ...] 형태의 데이터.
    Returns:
        tuple: (images, targets)로 묶어서 반환.
    """
    images = [item[0] for item in batch]
    targets = []

    for target in [item[1] for item in batch]:
        mapped_target = [
            {
                **obj,
                "category_id": COCO_CATEGORY_MAPPING[obj["category_id"]]
            }
            for obj in target if obj["category_id"] in COCO_CATEGORY_MAPPING
        ]
        targets.append(mapped_target)

    return images, targets

def get_coco_data_loaders(batch_size=128, root_path="/kaggle/input/2017-2017"):
    # COCO 경로 설정
    train_img_path = f"{root_path}/train2017/train2017/"
    val_img_path = f"{root_path}/val2017/val2017/"
    train_anno_path = f"{root_path}/annotations_trainval2017/annotations/instances_train2017.json"
    val_anno_path = f"{root_path}/annotations_trainval2017/annotations/instances_val2017.json"

    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 크기 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

    # COCO 데이터셋 로드
    train_dataset = CocoDetection(root=train_img_path, annFile=train_anno_path, transform=transform)
    val_dataset = CocoDetection(root=val_img_path, annFile=val_anno_path, transform=transform)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    return train_loader, val_loader