import torch
import torch.nn as nn

# CBAM_Light.py

# ChannelAttention and SpatialAttention modules in this file 
# are identical to the ones found in CBAM.py.

# The main distinction is in the _CBAM block:
# The application of spatial attention is conditional based on 
# the height and width of the input feature map.
# Specifically:
# - If both the height and width of the input feature map (h and w) 
#   are greater than 7, both channel and spatial attentions are applied.
# - Otherwise, only channel attention is applied.

# This conditional application results in a "lighter" variant of CBAM
# as it might avoid the spatial attention (which involves a convolution) 
# on smaller feature maps, thus potentially saving computational resources.

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
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        b, c, h, w = x.shape

        f_prime = self.ca(x)
        if h > 7 and w > 7:
            f_double_prime = self.sa(f_prime)
        else:
            f_double_prime = f_prime

        return f_double_prime
    
if __name__ == '__main__':
    cbam = _CBAM(7)
    x = torch.randn(1, 7, 56, 56)
    y = cbam(x)
    print(y.shape)