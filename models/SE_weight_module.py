import torch
import torch.nn as nn
import math

class SqueezeExcitation(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out_channels = math.ceil(channels / reduction)
        self.conv1 = nn.Conv2d(channels, out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        weight = self.sigmoid(out)

        return weight