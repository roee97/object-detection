import torch
from torch import nn
import torch.nn.functional as F


class YOLODetectorHead(nn.Module):
    def __init__(self, in_channels=512, n_anchors=1):
        super(YOLODetectorHead, self).__init__()
        # TODO: add classification and multiple anchors
        self.out_channels = n_anchors * (4 + 1)  # bbox (4) + objectness (1)

        self.predictor = nn.Conv2d(in_channels, self.out_channels, 1, bias=True)

        self._anchors = torch.tensor([50, 50])  # TODO: implement multiple anchors
        self.register_buffer('anchors', self._anchors)

    def forward(self, features):
        bbox_preds = self.predictor(F.relu(features))

        # x, y (in yolo grid)
        pred_w, pred_h = bbox_preds.shape[2:]
        bbox_preds[:, :2, :, :] = torch.sigmoid(bbox_preds[:, :2, :, :])

        x_grid = torch.arange(pred_w, device=bbox_preds.device).repeat(pred_h, 1)
        y_grid = torch.arange(pred_h, device=bbox_preds.device).unsqueeze(1).repeat(1, pred_w)

        bbox_preds[:, 0, :, :] = (bbox_preds[:, 0, :, :] + x_grid)
        bbox_preds[:, 1, :, :] = (bbox_preds[:, 1, :, :] + y_grid)

        # h, w
        anchors = dict(self.named_buffers())['anchors']
        bbox_preds[:, 2:4, :, :] = anchors.view(1, -1, 1, 1) * torch.exp(bbox_preds[:, 2:4, :, :])

        # objectness (stays as a logit)

        return bbox_preds
