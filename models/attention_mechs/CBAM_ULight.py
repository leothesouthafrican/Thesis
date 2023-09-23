import torch
import torch.nn as nn

# CBAM_UltraLight

# Once again, the ChannelAttention and SpatialAttention implementations
# in this file are consistent with those found in CBAM.py and CBAM_Light.py.

# The key difference in this variant lies in the _CBAM block:
# The criterion for the application of spatial attention here is 
# more stringent than in CBAM_Light.py.
# Namely:
# - Spatial attention is applied only if both the height and width 
#   of the input feature map (h and w) exceed 14.

# This renders the module "ultra light" since it imposes an even stricter 
# condition on the application of the spatial attention mechanism.
# This can lead to further computational savings, especially on smaller feature maps.

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
        if h > 14 and w > 14:
            f_double_prime = self.sa(f_prime)
        else:
            f_double_prime = f_prime

        return f_double_prime