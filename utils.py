from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128):
    # CIFAR-10 정규화 값
    cifar10_mean = [0.4914, 0.4822, 0.4465]
    cifar10_std = [0.2023, 0.1994, 0.2010]

    # 데이터 증강 (학습용)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 랜덤 크롭
        transforms.RandomHorizontalFlip(),     # 좌우 반전
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 색상 변형
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)  # CIFAR-10 정규화
    ])

    # 데이터 변환 (테스트용)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)  # CIFAR-10 정규화
    ])

    # CIFAR-10 데이터셋 로드
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader