from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader

def get_coco_data_loaders(batch_size=32, root_path="/kaggle/input/2017-2017"):
    # COCO 데이터 경로
    train_img_path = f"{root_path}/train2017/train2017/"
    val_img_path = f"{root_path}/val2017/val2017/"
    train_anno_path = f"{root_path}/annotations_trainval2017/annotations/instances_train2017.json"
    val_anno_path = f"{root_path}/annotations_trainval2017/annotations/instances_val2017.json"

    # COCO 데이터셋용 변환 (학습)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),         # 이미지 크기 조정
        transforms.RandomCrop((224, 224)),    # 랜덤 크롭
        transforms.RandomHorizontalFlip(),    # 좌우 반전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색상 변형
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

    # COCO 데이터셋용 변환 (검증)
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),  # 이미지 크기 조정
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

    # COCO 데이터셋 로드
    train_dataset = CocoDetection(root=train_img_path, annFile=train_anno_path, transform=transform_train)
    val_dataset = CocoDetection(root=val_img_path, annFile=val_anno_path, transform=transform_val)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    return train_loader, val_loader