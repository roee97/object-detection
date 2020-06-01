import torch
from albumentations import convert_bbox_to_albumentations, convert_bbox_from_albumentations


def coco_to_yolo(bbox, image_shape=(416, 416)):
    rows, cols = image_shape

    alb_bboxes = convert_bbox_to_albumentations(bbox, 'coco', rows, cols)
    yolo_bboxes = convert_bbox_from_albumentations(alb_bboxes, 'yolo', rows, cols)

    return yolo_bboxes


def yolo_to_coco(yolo_bbox, img_shape=(416, 416)):
    # For one image
    pred_w, pred_h = yolo_bbox.shape[-2:]
    img_w, img_h = img_shape

    coco_bbox = yolo_bbox.clone()

    coco_bbox[0, :, :] = yolo_bbox[0, :, :] * (img_w / pred_w) - yolo_bbox[2, :, :] / 2
    coco_bbox[1, :, :] = yolo_bbox[1, :, :] * (img_h / pred_h) - yolo_bbox[3, :, :] / 2
    coco_bbox[0, :, :] = coco_bbox[0, :, :].clamp(0, img_w).int()
    coco_bbox[1, :, :] = coco_bbox[1, :, :].clamp(0, img_h).int()

    coco_bbox = coco_bbox.view(4, -1)

    coco_bbox_tuples = [tuple(coco_bbox[:, i].tolist()) for i in range(coco_bbox.shape[1])]

    # bs = yolo_bbox.shape[0]
    # coco_bbox = coco_bbox.reshape(bs, 4, -1)

    return coco_bbox_tuples
