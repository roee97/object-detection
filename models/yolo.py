import torch
from torch import nn

from backbones.resnet import ResNet
from heads.yolo_detector import YOLODetectorHead
from models.object_detector import ObjectDetector
from torchvision import models


class YOLO(ObjectDetector):
    def __init__(self, backbone=None):
        if backbone is None:
            # backbone = ResNet().features
            # backbone.n_features = 1024

            # backbone = models.mobilenet_v2(pretrained=True).features
            # backbone.n_features = 1280

            resnet = models.resnet18(pretrained=True)
            modules = list(resnet.children())[:-2]
            backbone = nn.Sequential(*modules)
            backbone.n_features = 512

            for param in backbone.parameters():
                param.requires_grad = True

        yolo_detector_head = YOLODetectorHead(in_channels=backbone.n_features)

        super(YOLO, self).__init__(backbone=backbone, detector_head=yolo_detector_head)
