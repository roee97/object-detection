import numpy as np
import torch
from albumentations import convert_bbox_to_albumentations, convert_bbox_from_albumentations
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score

from utils.bbox import coco_to_yolo


def yolo_loss(bbox_preds, bboxes, image_shape=(416, 416)):
    per_image = [yolo_loss_image(bbox_preds[i], bboxes[i], image_shape) for i in range(len(bboxes))]
    loss = sum([l[0] for l in per_image]) / len(bboxes)
    precision = sum([l[1] for l in per_image]) / len(bboxes)
    recall = sum([l[2] for l in per_image]) / len(bboxes)
    return loss, precision, recall


def yolo_loss_image(bbox_preds, bboxes, image_shape=(416, 416)):
    device = bbox_preds.device

    rows, cols = image_shape
    pred_rows, pred_cols = bbox_preds.shape[-2:]

    yolo_bboxes = [coco_to_yolo(bbox) for bbox in bboxes]

    loss = 0.0

    # TODO: implement multiple anchors
    anchor_assignment = torch.zeros((pred_rows, pred_cols), device=device)

    # loop over bboxes in image
    for j in range(len(yolo_bboxes)):
        bbox = yolo_bboxes[j]
        gt_yolo_x, gt_yolo_y = bbox[0] * pred_rows, bbox[1] * pred_cols
        gt_yolo_w, gt_yolo_h = bbox[2] * image_shape[0], bbox[3] * image_shape[1]
        gt_yolo_col, gt_yolo_row = int(gt_yolo_x), int(gt_yolo_y)

        if anchor_assignment[gt_yolo_row, gt_yolo_col] == 0:
            anchor_assignment[gt_yolo_row, gt_yolo_col] = 1

            gt_yolo_coord = torch.tensor([gt_yolo_x, gt_yolo_y, gt_yolo_w, gt_yolo_h], device=device)
            pred_yolo_coord = bbox_preds[:4, gt_yolo_row, gt_yolo_col]

            regression_loss = ((gt_yolo_coord - pred_yolo_coord).pow(2)).sum().sqrt()

            loss += 0.2 * regression_loss

    pos_weight = 10
    pos_weight = torch.ones_like(anchor_assignment.view(-1)) * pos_weight
    objectness_loss = F.binary_cross_entropy_with_logits(bbox_preds[4, :, :].view(-1),
                                                         anchor_assignment.view(-1),
                                                         reduction='sum',
                                                         pos_weight=pos_weight)
    loss += objectness_loss

    # TODO: implement this outside (in train.py)
    objectness_preds = (bbox_preds[4, :, :] > 0.5).detach().cpu().view(-1)
    objectness_gt = anchor_assignment.view(-1).cpu()

    precision = precision_score(objectness_gt, objectness_preds)
    recall = recall_score(objectness_gt, objectness_preds)

    # temp. debugging code to verify orientation
    # import matplotlib.pyplot as plt
    # from matplotlib import patches
    # fig, ax = plt.subplots(1)
    #
    # ax.imshow(anchor_assignment)
    #
    # fig, ax = plt.subplots(1)
    #
    # ax.imshow(bbox_preds[4, :, :].detach())

    return loss, precision, recall
