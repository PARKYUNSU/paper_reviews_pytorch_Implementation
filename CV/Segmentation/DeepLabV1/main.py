import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import os
from dataset import *
from model import DeepLabv1
from train import train_model, plot_metrics
from eval import evaluate_model
import matplotlib.pyplot as plt


def visualize_result(image_path, model, iter_max, bi_w, bi_xy_std, bi_rgb_std, pos_w, pos_xy_std, save_path=None):
    # 모델 CRF 후처리 적용
    result = model.inference(
        image_path,
        iter_max=iter_max,
        bi_w=bi_w,
        bi_xy_std=bi_xy_std,
        bi_rgb_std=bi_rgb_std,
        pos_w=pos_w,
        pos_xy_std=pos_xy_std,
    )

    # 결과를 PIL 이미지로 변환
    result_image = Image.fromarray(np.uint8(result * (255 / model.num_classes)))

    # 결과 저장
    if save_path:
        result_image.save(save_path)
        print(f"후처리 결과가 저장되었습니다: {save_path}")

    # 원본 이미지 및 결과 시각화
    original_image = Image.open(image_path)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(result_image, cmap="jet")
    axs[1].set_title("CRF-Processed Result")
    axs[1].axis("off")
    plt.show()


def main():

    # 데이터셋 경로
    base_dir = '/kaggle/input/camvid/CamVid/'
    train_dir = base_dir + 'train'
    train_labels_dir = base_dir + 'train_labels'
    val_dir = base_dir + 'val'
    val_labels_dir = base_dir + 'val_labels'
    test_dir = base_dir + 'test'
    test_labels_dir = base_dir + 'test_labels'

    # 파일 리스트
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

    print('파일 검증 완료')

    # 데이터셋 로드
    train_dataset = CamVidDataset(img_dir=train_dir, label_dir=train_labels_dir)
    val_dataset = CamVidDataset(img_dir=val_dir, label_dir=val_labels_dir)
    test_dataset = CamVidDataset(img_dir=test_dir, label_dir=test_labels_dir)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_file = None
    model = DeepLabv1(num_classes=32, init_weights=True if weight_file is None else False).to(device)
    if weight_file:
        model.load_state_dict(torch.load(weight_file))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 모델 학습
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        num_epochs=200,
        device=device
    )

    # 학습 결과 시각화
    plot_metrics(history)

    # 테스트 데이터셋을 사용한 모델 평가
    # 평가 시 CRF 파라미터를 설정하여 함수 호출
    crf_params = (10, 3, 3, 5, 140, 5)  # iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std
    avg_pixel_acc, accuracy, iou, precision, recall, f1 = evaluate_model(
        model, test_loader, device, num_classes=32, use_crf=True, crf_params=crf_params)
    # CRF 파라미터 설정
    iter_max = 10
    pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std = 3, 3, 5, 140, 5

    # 이미지 경로 설정 (원하는 테스트 이미지의 경로)
    test_image_path = "/kaggle/input/camvid/CamVid/test/0001TP_008550.png"
    save_path = "/kaggle/working/crf_processed_result.png"

    # CRF 후처리 결과 시각화 및 저장
    print("CRF 후처리 결과 시각화 및 저장 중...")
    visualize_result(test_image_path, model, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std, save_path=save_path)

if __name__ == "__main__":
    main()