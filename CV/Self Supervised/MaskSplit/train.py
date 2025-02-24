from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import os

from data.dataset import get_train_loader, get_test_loader, unsupervised_loader, multi_test_loader
from pytorch_lightning.callbacks import ModelCheckpoint
from model import simpleNetwork
from util import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Training Configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    return cfg


if __name__ == "__main__":
    opt = parse_args()
    model = simpleNetwork(opt)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name=opt.model_name)

    if opt.strategy == "unsupervised_all":
        checkpoint_callbacks = []
        for i in range(4):
            checkpoint_dir = os.path.join(opt.ckpt_path, f"{opt.model_name}_{i}")
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_callbacks.append(
                ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    save_top_k=2,
                    save_weights_only=False,
                    mode='max',
                    every_n_epochs=1,
                    monitor=f"val_miou_{i}",
                )
            )

        trainer = Trainer(
            max_epochs=opt.epochs,
            gpus=opt.gpus,
            profiler="simple",
            benchmark=True,
            callbacks=checkpoint_callbacks
        )
    else:
        checkpoint_dir = os.path.join(opt.ckpt_path, opt.model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=2,
            save_weights_only=False,
            mode='max',
            every_n_epochs=1,
            monitor="val_miou"
        )

        trainer = Trainer(
            max_epochs=opt.epochs,
            gpus=opt.gpus,
            profiler="simple",
            benchmark=True,
            callbacks=[checkpoint_callback],
            logger=logger
        )

    # 학습 데이터 로더 선택
    if opt.strategy == "unsupervised_fbf":
        trainer.fit(model, unsupervised_loader(opt), get_test_loader(opt))
    elif opt.strategy == "supervised":
        trainer.fit(model, get_train_loader(opt), get_test_loader(opt))
    elif opt.strategy == "unsupervised_all":
        assert opt.use_all_classes is True
        trainer.fit(model, unsupervised_loader(opt), multi_test_loader(opt))