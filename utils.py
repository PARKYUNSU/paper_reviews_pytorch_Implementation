from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_data_loaders(batch_size=128):
    # CIFAR-10 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    # CIFAR-10 데이터셋 로드
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader