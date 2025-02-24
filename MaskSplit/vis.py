from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List


def make_episode_visualization(img_s, img_q, gt_s, gt_q, preds, save_path):

    # Preliminary checks
    assert len(img_s.shape) == 4, f"Support shape expected : K x 3 x H x W or K x H x W x 3. Currently: {img_s.shape}"
    assert len(img_q.shape) == 3, f"Query shape expected : 3 x H x W or H x W x 3. Currently: {img_q.shape}"
    assert len(preds.shape) == 4, f"Predictions shape expected : T x num_classes x H x W. Currently: {preds.shape}"
    assert len(gt_s.shape) == 3, f"Support GT shape expected : K x H x W. Currently: {gt_s.shape}"
    assert len(gt_q.shape) == 2, f"Query GT shape expected : H x W. Currently: {gt_q.shape}"

    # Support / Query 이미지 변환
    if img_s.shape[1] == 3:
        img_s = np.transpose(img_s, (0, 2, 3, 1))
    if img_q.shape[0] == 3:
        img_q = np.transpose(img_q, (1, 2, 0))

    assert img_s.shape[-3:-1] == img_q.shape[-3:-1] == gt_s.shape[-2:] == gt_q.shape

    # 저장 경로 생성
    output_dir = os.path.join("qualitative_results", save_path)
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 정규화 해제
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_s = (img_s * std) + mean
    img_q = (img_q * std) + mean
    img_s = img_s[0]

    # 예측값 계산
    mask_pred = preds.argmax(1)[0]

    # 255 값은 0으로 변경
    gt_s = gt_s.copy()
    gt_q = gt_q.copy()
    gt_s[gt_s == 255] = 0
    gt_q[gt_q == 255] = 0

    # 컬러맵 리스트 설정
    cmap_list = ['hsv', 'cool', 'jet']

    make_plot(img_q, gt_q, os.path.join(output_dir, f"{save_path}_qry.png"), cmap_list)
    make_plot(img_s, gt_s[0], os.path.join(output_dir, f"{save_path}_sup.png"), cmap_list)
    make_plot(img_q, mask_pred, os.path.join(output_dir, f"{save_path}_pred.png"), cmap_list)


def make_plot(img, mask, save_path, cmap_name='hsv'):
    sizes = np.shape(img)
    if sizes[1] == 0:
        sizes = (256, 256, 3)

    fig = plt.figure(figsize=(4. * sizes[0] / max(sizes[1], 1), 4), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, interpolation='none')

    cmap = plt.get_cmap(cmap_name)
    alphas = Normalize(0, .3, clip=True)(mask)
    alphas = np.clip(alphas, 0., 0.5)
    colors = Normalize()(mask)
    colors = cmap(colors)
    colors[..., -1] = alphas
    ax.imshow(colors, cmap=cmap)

    plt.savefig(save_path)
    plt.close()