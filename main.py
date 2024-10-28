import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import *
from model import VGG16_FCN, init_weights
from train import train_model, plot_metrics, visualize_segmentation  # 시각화 함수 추가
from evaluation import evaluate_model  # calculate_metrics 제거
import os

# 데이터셋 경로 설정
base_dir = '/kaggle/input/camvid/CamVid/'
train_dir = base_dir + 'train'
train_labels_dir = base_dir + 'train_labels'
val_dir = base_dir + 'val'
val_labels_dir = base_dir + 'val_labels'
test_dir = base_dir + 'test'
test_labels_dir = base_dir + 'test_labels'

# 파일 리스트 가져오기
train_files = os.listdir(train_dir)
train_labels_files = os.listdir(train_labels_dir)
val_files = os.listdir(val_dir)
val_labels_files = os.listdir(val_labels_dir)
test_files = os.listdir(test_dir)
test_labels_files = os.listdir(test_labels_dir)

# 파일 정렬
train_files.sort()
train_labels_files.sort()
val_files.sort()
val_labels_files.sort()
test_files.sort()
test_labels_files.sort()

# 데이터셋 파일 수 검증
assert len(train_files) == len(train_labels_files) == 369
assert len(val_files) == len(val_labels_files) == 100
assert len(test_files) == len(test_labels_files) == 232

for i in train_files:
    assert i[:-4] + '_L.png' in train_labels_files, f'{i} not found'
for i in val_files:
    assert i[:-4] + '_L.png' in val_labels_files, f'{i} not found'
for i in test_files:
    assert i[:-4] + '_L.png' in test_labels_files, f'{i} not found'

print('ok')

# 클래스 딕셔너리 로드 및 변환 사전 생성
class_dict_path = "/kaggle/input/camvid/CamVid/class_dict.csv"
class_dict = load_class_dict(class_dict_path)
rgb_to_label_dict, label_to_rgb_dict = create_conversion_dicts(class_dict)

# 데이터셋 생성 시 rgb_to_label_dict 전달
train_dataset = CamVidDataset(img_dir=train_dir, label_dir=train_labels_dir, augment=True, rgb_to_label_dict=rgb_to_label_dict)
val_dataset = CamVidDataset(img_dir=val_dir, label_dir=val_labels_dir, rgb_to_label_dict=rgb_to_label_dict)
test_dataset = CamVidDataset(img_dir=test_dir, label_dir=test_labels_dir, rgb_to_label_dict=rgb_to_label_dict)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16_FCN(num_classes=32).to(device)
model.apply(init_weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 모델 학습 및 조기 종료 설정
model, history = train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=100,
    device=device,
    patience=5,
    delta=0.01
)

# 학습 결과 시각화
plot_metrics(history)

# 테스트 데이터셋을 사용한 모델 평가
print("Evaluating on test set...")
avg_pixel_acc, accuracy, iou, precision, recall, f1 = evaluate_model(model, test_loader, device, num_classes=32)