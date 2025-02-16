from pytorch_lightning import Trainer, seed_everything
import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random

from dataset.dataset import load_test_data
from model import simpleNetwork
from util import load_config


def parse_args():
    """테스트 실행을 위한 설정 파싱"""
    parser = argparse.ArgumentParser(description="Testing Configuration")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    cfg = load_config(args.config)  # 설정 파일 로드
    return cfg


if __name__ == "__main__":
    opt = parse_args()

    # 랜덤 시드 설정 (재현 가능성 확보)
    seed_value = 202
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    seed_everything(seed_value)

    model = simpleNetwork.load_from_checkpoint(
        checkpoint_path=opt.ckpt_used,
        map_location="cpu",
        visualize=opt.visualize
    ).eval()

    trainer = Trainer(gpus=opt.gpus, benchmark=True)

    for _ in range(5):
        trainer.test(model, dataloaders=[load_test_data(opt)])