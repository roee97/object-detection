import torch
import torch.nn as nn
import torch.nn.functional as F


class ResConv(nn.Module):
    def __init__(self, n_channels, first_downscale=False):
        super(ResConv, self).__init__()
        # self.conv1 = nn.Conv2d(n_channels, n_channels // 2, 1)
        # self.conv2 = nn.Conv2d(n_channels // 2, n_channels, 3, padding=1)
        self.first_downscale = first_downscale
        first_stride = 1
        first_channels = n_channels
        if first_downscale:
            first_channels = n_channels // 2
            first_stride = 2
        self.conv1 = nn.Conv2d(first_channels, n_channels, 3, padding=1, stride=first_stride)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if not self.first_downscale:
            y += x
        y = F.relu(y)
        return F.dropout(y, 0)


class ResConvBlock(nn.Sequential):
    def __init__(self, n_channels, n_convs):
        layers = [ResConv(n_channels, first_downscale=True)] + \
                 [ResConv(n_channels) for _ in range(n_convs - 1)]

        super(ResConvBlock, self).__init__(*layers)


class ConvClassfierHead(nn.Module):
    def __init__(self, in_features, n_classes):
        super(ConvClassfierHead, self).__init__()
        self.fc1 = nn.Linear(in_features, 1000)
        # self.fc2 = nn.Linear(in_features, in_features)
        self.fc3 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x = torch.mean(x, [2, 3])
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x)) + x
        x = self.fc3(x)
        return x


class ResNet(nn.Module):
    def __init__(self, layers=(2, 2, 2, 2), n_classes=10, base_channels=64):
        super(ResNet, self).__init__()

        rc_blocks = [(base_channels * 2 ** i, layers[i]) for i in range(4)]

        self.features_layers = [nn.Conv2d(3, rc_blocks[0][0] // 2, 7, padding=0, stride=2)] + \
                               [nn.ReLU()] + \
                               [ResConvBlock(n_channels, n_convs) for n_channels, n_convs in rc_blocks]

        self.features = nn.Sequential(*self.features_layers)
        self.classifier = [ConvClassfierHead(rc_blocks[-1][0], n_classes)]

    def forward(self, x):
        return self.sequential(x)
