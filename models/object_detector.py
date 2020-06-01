import os
import random
import sys

import numpy as np
import torch
import torch.autograd as autograd  # computation graph
from torch import Tensor  # tensor node in the computation graph
import torch.nn as nn  # neural networks
import torch.nn.functional as F  # layers, activations and more
import torch.optim as optim  # optimizers e.g. gradient descent, ADAM, etc.
from torch.jit import script, trace  # hybrid frontend decorator and tracing jit


## TODO: make more general, with more heads and more types of heads
class ObjectDetector(nn.Module):
    def __init__(self, backbone, detector_head):
        super(ObjectDetector, self).__init__()
        self.backbone = backbone
        self.detector_head = detector_head

    def forward(self, x):
        features = self.backbone(x)
        detections = self.detector_head(features)
        return detections
