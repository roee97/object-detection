import os
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning.callbacks import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from models.yolo import YOLO
from utils.losses import yolo_loss
from utils.progress_bar import LitProgressBar
from utils.data_loading import get_coco_dataloader, get_coco_dataset
from utils.visualization import image_bbox_vis, batch_bbox_vis


class COCOModel(pl.LightningModule):
    """
    COCO Model
    """

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.data_path = hparams.data_path
        self.batch_size = hparams.batch_size
        self.learning_rate = hparams.lr
        self.net = YOLO()

        self.train_dataset = get_coco_dataset(train=True)
        self.val_dataset = get_coco_dataset(train=False)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        images, targets = batch

        preds = self.net(images)

        if batch_nb % 300 == 0 or (batch_nb < 300 and batch_nb % 50 == 0) or batch_nb == 10:
            bbox_vis = batch_bbox_vis(images.cpu(), preds.cpu())
            grid = torchvision.utils.make_grid(bbox_vis)
            self.logger.experiment.add_image('images', grid, batch_nb)

        bboxes = [t['bboxes'] for t in targets]
        loss, precision, recall = yolo_loss(preds, bboxes)

        log_dict = {'train_loss': loss, 'precision': precision, 'recall': recall}
        return {'loss': loss, 'log': log_dict, 'progress_bar': log_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        preds = self.net(images)

        bboxes = [t['bboxes'] for t in targets]
        loss, precision, recall = yolo_loss(preds, bboxes)

        log_dict = {'val_loss_step': loss, 'val_precision_step': precision, 'val_recall_step': recall}
        return {'val_loss': loss, 'val_precision': precision, 'val_recall': recall, 'progress_bar': log_dict}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        precision = np.array([x['val_precision'] for x in outputs]).mean()
        recall = np.array([x['val_recall'] for x in outputs]).mean()

        log_dict = {'val_loss': loss, 'val_precision': precision, 'val_recall': recall}
        return {'log': log_dict, 'val_loss': log_dict['val_loss'], 'progress_bar': log_dict}

    def configure_optimizers(self):
        print('LR: ', self.learning_rate)
        # optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        # scheduler = ExponentialLR(optimizer, 0.95)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        return get_coco_dataloader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return get_coco_dataloader(self.val_dataset, batch_size=self.batch_size * 2)


def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = COCOModel(hparams)

    # ------------------------
    # 2 SET LOGGER
    # ------------------------
    logger = False
    if hparams.log_tb:
        logger = TensorBoardLogger('tb_logs/', name='my_model')

        # optional: log model topology
        # logger.watch(model.net)

    bar = LitProgressBar()

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        logger=logger,
        max_epochs=hparams.epochs,
        accumulate_grad_batches=hparams.grad_batches,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_amp else 32,
        callbacks=[bar],
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path where dataset is stored")
    parser.add_argument("--gpus", type=int, default=-1, help="number of available GPUs")
    parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    parser.add_argument('--use_amp', action='store_true', help='if true uses 16 bit precision')
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--grad_batches", type=int, default=1, help="number of batches to accumulate")
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs to train")
    parser.add_argument("--log_tb", action='store_false', help="log training on TensorBoard")

    hparams = parser.parse_args()

    main(hparams)
