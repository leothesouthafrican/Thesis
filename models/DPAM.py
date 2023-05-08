import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        """
        Channel Attention module.
        
        Args:
            in_channels (int): Number of input channels.
            reduction_ratio (int, optional): Reduction ratio for the number of channels in the intermediate layers. Default is 16.
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Linear = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the Channel Attention module.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Channel attention scale.
        """
        avg_out = self.Linear(self.avg_pool(x))
        channel_attention_scale = self.sigmoid(avg_out)

        return channel_attention_scale

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, dilation_rates = [1,3]):
        """
        Spatial Attention module.
        
        Args:
            in_channels (int): Number of input channels.
            dilation_rates (List[int], optional): Dilation rates for the dilated convolutions. Default is [1, 3].
        """
        super().__init__()
        self.dilation_rates = dilation_rates
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels = in_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        for i in range(len(dilation_rates)):
            setattr(self, 'dilation_conv_{}'.format(i + 1),
                    nn.Conv2d(in_channels // 2, out_channels=in_channels // 2, kernel_size=3, stride=1,
                              padding=dilation_rates[i], dilation=dilation_rates[i], groups=in_channels // 2, bias=False))

        self.relu = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv2d(in_channels // 2, out_channels = in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        """
        Forward pass for the Spatial Attention module.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying spatial attention.
        """
        # Reduce channels with 1x1 convolution
        out = self.conv1x1_1(x)

        # Apply dilated convolutions and sum the outputs
        out_dilated = sum(getattr(self, f'dilation_conv_{i+1}')(out) for i in range(len(self.dilation_rates)))
        out = self.relu(out_dilated)

        # Increase channels back to the original number with 1x1 convolution
        out = self.conv1x1_2(out)

        return out

class _DPAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, dilation_rates = [1,3]):
        """
        Dual-path Attention Module (DPAM) combining Channel and Spatial Attention.
        
        Args:
            in_channels (int): Number of input channels.
            reduction_ratio (int, optional): Reduction ratio for the number of channels in the intermediate layers. Default is 16.
            dilation_rates             dilation_rates (List[int], optional): Dilation rates for the dilated convolutions. Default is [1, 3].
        """
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(in_channels, dilation_rates)

    def forward(self, x):
        """
        Forward pass for the Dual-path Attention Module (DPAM).
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying both channel and spatial attention.
        """
        # Apply channel attention
        f_prime = self.ca(x) * x

        # Apply spatial attention
        f_double_prime = self.sa(f_prime) * f_prime

        return f_double_prime + x
    
if __name__ == "__main__":
    # Test Channel Attention
    x = torch.randn(1, 64, 56, 56)
    ca = ChannelAttention(64)
    print(ca(x).shape)

    # Test Spatial Attention
    x = torch.randn(1, 64, 56, 56)
    sa = SpatialAttention(64)
    print(sa(x).shape)

    # Test DPAM
    x = torch.randn(1, 64, 56, 56)
    dpam = _DPAM(64)
    print(dpam(x).shape)

    

