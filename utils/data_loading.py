import os

import numpy as np
import torch
from albumentations import (
    HorizontalFlip, Normalize, RandomResizedCrop, Compose
)
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms
from albumentations.core.composition import BboxParams
from torch.utils.data import DataLoader
from torchvision import datasets


def coco_detection_collate_fn(samples):
    images, bboxes = [s[0] for s in samples], [s[1] for s in samples]

    images = torch.stack(images)

    return images, bboxes


# Transforms


class TransformWrapper:
    def __init__(self, transform_list):
        self.transform = Compose(transform_list, bbox_params=BboxParams('coco', ['labels']))

    def __call__(self, img, tgt, *args, **kwargs):
        try:
            res = self.transform(**{
                'image': np.array(img),
                'bboxes': np.array([o['bbox'] for o in tgt]),
                'labels': np.array([o['category_id'] for o in tgt])
            })
        except Exception as e:
            res = self.transform(**{
                'image': np.array(img),
                'bboxes': np.array([]),
                'labels': np.array([])
            })

        image = res['image']
        del res['image']

        return image, res


NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZE_STD = (0.2023, 0.1994, 0.2010)
NORMALIZE = Normalize(NORMALIZE_MEAN, NORMALIZE_STD)

INV_NORMALIZE = transforms.Normalize(
    mean=[-m / s for m, s in zip(NORMALIZE_MEAN, NORMALIZE_STD)],
    std=[1 / s for s in NORMALIZE_STD]
)

BASE_TRANSFORM_LIST = [
    RandomResizedCrop(416, 416, scale=(0.8, 1)),
    NORMALIZE,
    ToTensorV2(),
]

VAL_TRANSFORM_LIST = [] + BASE_TRANSFORM_LIST

TRAIN_TRANSFORM_LIST = [] + BASE_TRANSFORM_LIST

VAL_TRANSFORM = TransformWrapper(VAL_TRANSFORM_LIST)
TRAIN_TRANSFORM = TransformWrapper(TRAIN_TRANSFORM_LIST)


# public methods

def get_coco_dataset(base_path='../data/', train=True):
    # Paths

    train_data_path = os.path.join(base_path, 'train2017')
    val_data_path = os.path.join(base_path, 'val2017')

    ann_path = os.path.join(base_path, 'annotations')

    train_ann_path = os.path.join(ann_path, 'instances_train2017.json')
    val_ann_path = os.path.join(ann_path, 'instances_val2017.json')

    data_path = train_data_path if train else val_data_path
    ann_path = train_ann_path if train else val_ann_path
    transform = TRAIN_TRANSFORM if train else VAL_TRANSFORM

    coco_val_dataset = datasets.CocoDetection(root=data_path,
                                              annFile=ann_path,
                                              transforms=transform)
    return coco_val_dataset


def get_coco_dataloader(dataset, batch_size=32, shuffle=False):
    coco_cal_dataloader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     collate_fn=coco_detection_collate_fn,
                                     num_workers=4)

    return coco_cal_dataloader
