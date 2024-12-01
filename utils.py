from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms

def collate_fn(batch):
    images = [item[0] for item in batch]  # 이미지 리스트
    targets = [item[1] for item in batch]  # 주석 리스트
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
