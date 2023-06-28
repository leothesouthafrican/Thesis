import torch
import torch.nn as nn
import torch.nn.functional as F

class LR_Spatial(nn.Module):

    def __init__(self, num_in, kernel_breadth=11, stride=1):
        super(LR_Spatial, self).__init__()
        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)

        self.short_conv = nn.Sequential(
            nn.Conv2d(num_in, num_in, kernel_size=1, stride=stride,
                    padding=kernel_breadth//2, groups=1),
            nn.Conv2d(num_in, num_in, kernel_size=(1, kernel_breadth), stride=1,
                    padding=(0, 11//2), groups=num_in),
            nn.Conv2d(num_in, num_in, kernel_size=(kernel_breadth, 1), stride=1,
                    padding=(11//2, 0), groups=num_in)
        )

    def forward(self, x):
        """ ghost module forward """
        input_size = x.shape[-2:]  # Get the original input size
        res = self.avgpool2d(x)
        res = self.short_conv(res)
        res = nn.ReLU6(inplace=True)(res)

        # Upsample res before multiplying
        res_up = F.interpolate(res, size=input_size, mode="bilinear", align_corners=True)
        out = res_up * F.interpolate(res, size=input_size, mode="bilinear", align_corners=True)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        channel_attention_scale = self.sigmoid(avg_out + max_out)

        return channel_attention_scale * x
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), f'kernel size must be 3 or 7 but got {kernel_size}'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        conv = self.conv(concat)
        spatial_attention_scale = self.sigmoid(conv)

        return spatial_attention_scale * x
    
class _CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.lr = LR_Spatial(in_channels)
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        b, c, h, w = x.shape

        # Apply LR_Spatial if the input is large enough
        if h >= 56 and w >= 56:
            x = self.lr(x)

        # Apply Channel Attention
        f_prime = self.ca(x)

        # Apply Spatial Attention if the input is large enough
        if h > 7 and w > 7:
            f_double_prime = self.sa(f_prime)
        else:
            f_double_prime = f_prime

        return f_double_prime
    
if __name__ == '__main__':
    cbam = _CBAM(16)
    x = torch.randn(1, 16, 56, 56)
    y = cbam(x)
    print(y.shape)