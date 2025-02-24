import argparse
import os
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from model import simpleNetwork


def parse_arguments():
    """인퍼런스 실행을 위한 설정 파싱"""
    parser = argparse.ArgumentParser(description="Inference Configuration")
    parser.add_argument("--sup_im_path", type=str, required=True, help="Path to support image")
    parser.add_argument("--query_im_path", type=str, required=True, help="Path to query image")
    parser.add_argument("--sup_mask_path", type=str, required=True, help="Path to support mask")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--use_cuda", action="store_true", help="Enable CUDA for inference")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument("--image_size", type=int, default=400, help="Resize image size")
    return parser.parse_args()


def load_and_preprocess_images(opt):
    """이미지 및 마스크 로드 및 전처리"""
    # 이미지 불러오기 (RGB 변환)
    sup_im = cv2.cvtColor(cv2.imread(opt.sup_im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
    smask = cv2.imread(opt.sup_mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    query = cv2.cvtColor(cv2.imread(opt.query_im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)

    # 이미지 변환 (Normalize & Tensor 변환 포함)
    transform = A.Compose([
        A.Resize(height=opt.image_size, width=opt.image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    transformed_sup = transform(image=sup_im, mask=smask)
    sup_im, smask = transformed_sup["image"], transformed_sup["mask"]
    query = transform(image=query)["image"]

    return sup_im.unsqueeze(0).unsqueeze(0), smask.unsqueeze(0).unsqueeze(0), query.unsqueeze(0)


def visualize_results(opt, sup_im, smask, query, preds):
    """인퍼런스 결과 시각화"""
    figure, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))

    # 원본 이미지 로드
    sup_im_orig = cv2.cvtColor(cv2.imread(opt.sup_im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    smask_orig = cv2.imread(opt.sup_mask_path, cv2.IMREAD_GRAYSCALE)
    query_orig = cv2.cvtColor(cv2.imread(opt.query_im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # 시각화
    ax[0].imshow(sup_im_orig)
    ax[0].set_title("Support Image")
    ax[1].imshow(smask_orig, cmap="gray")
    ax[1].set_title("Support Mask")
    ax[2].imshow(query_orig)
    ax[2].set_title("Query Image")
    ax[3].imshow(preds, cmap="gray")
    ax[3].set_title("Prediction")

    plt.tight_layout()
    plt.savefig("inference.png")
    plt.show()


if __name__ == "__main__":
    opt = parse_arguments()

    # 모델 로드
    model = simpleNetwork.load_from_checkpoint(
        checkpoint_path=opt.checkpoint,
        map_location="cpu"
    ).eval()

    # 데이터 전처리
    sup_im, smask, query = load_and_preprocess_images(opt)

    # CUDA 사용 여부 설정
    if opt.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        sup_im, smask, query = sup_im.to(device), smask.to(device), query.to(device)
    else:
        device = torch.device("cpu")

    # 모델 추론
    preds = model(sup_im, smask, query)
    preds = torch.argmax(preds, dim=1).squeeze(0).detach().cpu().numpy()

    # 결과 시각화
    if opt.visualize:
        visualize_results(opt, sup_im, smask, query, preds)