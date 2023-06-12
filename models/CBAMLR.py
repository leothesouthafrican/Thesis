import torch
import torch.nn as nn


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
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_attention_scale = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))

        return spatial_attention_scale * x
    
class LR_CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.LR_Conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, (1, 11), stride=1, padding = (0, 5), groups=in_channels, bias=True),
            nn.Conv2d(in_channels, in_channels, (11, 1), stride=1, padding = (5, 0), groups=in_channels, bias=True)
        )
        self.pwl = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        if h > 11 and w > 11:
            x = self.LR_Conv(x)
            x = self.pwl(x)

        return x

    
class _CBAM_LR(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.lr = LR_CBAM(in_channels)

    def forward(self, x):
        f_prime = self.ca(x)
        f_double_prime = self.lr(f_prime)

        return f_double_prime
    
if __name__ == '__main__':
    import time
    
    start = time.time()

    # Instantiate your model
    in_channels = 72
    model = _CBAM_LR(in_channels)

    # Create a dummy input tensor
    x = torch.randn(1, in_channels, 16, 16)

    # Run the forward pass
    out = model(x)

    print(out.shape)

    end = time.time()

    print(f"Runtime of the program is {end - start}")

