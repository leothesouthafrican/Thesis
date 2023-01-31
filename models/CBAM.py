import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Linear(in_channels, in_channels // reduction_ratio, bias=False)
        self.fc4 = nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.global_avg_pool(x)
        y = y.view(b, c)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)

        z = self.channel_avg_pool(x)
        z = z.view(b, c)
        z = self.fc3(z)
        z = self.fc4(z)
        z = self.sigmoid2(z).view(b, c, 1, 1)

        att = y * z
        x = x * att

        return x