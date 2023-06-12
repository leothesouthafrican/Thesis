import torch
import torch.nn as nn
import time
from SE_weight_module import SqueezeExcitation

def LCM_64(n):
    return ((n + 63) // 64) * 64

class PyConv(nn.Module):
    def __init__(self, in_channels, stride = 1):
        super().__init__()
        self.in_channels = in_channels
        
        # Define the expansion channels
        self.exp_channels = LCM_64(in_channels)

        # Define the initial conv_kernels and self.pyconv_groups
        self.conv_kernels = [3, 7]


        self.exp_conv = nn.Sequential(

            nn.Conv2d(in_channels, self.exp_channels, kernel_size = 1, stride = stride, padding = "same", bias = False),
            nn.BatchNorm2d(self.exp_channels),
            nn.ReLU(inplace = True)
        )

        # Define the number of output channels for each convolution
        self.exp_channels_out = self.exp_channels // len(self.conv_kernels)

        # Define the squeeze and excitation layer
        self.se = SqueezeExcitation(self.exp_channels_out)

        #For each conv_kernel, define a convolution layer
        for i in range(len(self.conv_kernels)):
            setattr(self, f"conv_{i}", nn.Sequential(
                nn.Conv2d(self.exp_channels, self.exp_channels_out, kernel_size = self.conv_kernels[i], stride = 1, padding = "same", groups = self.exp_channels_out, bias = False),
                nn.BatchNorm2d(self.exp_channels_out),
                nn.ReLU(inplace = True)
            ))

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(self.exp_channels, self.in_channels, kernel_size = 1, stride = 1, padding = "same", bias = False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace = True)
        )



    def forward(self, x):
        # Apply each of the convolutions to the input feature
        expanded_x = self.exp_conv(x)
        b , c, h, w = expanded_x.shape

        output_features = [getattr(self, f"conv_{i}")(expanded_x) for i in range(len(self.conv_kernels))]
        feats = torch.cat(output_features, dim=1)
        feats = feats.view(b, len(self.conv_kernels), c//len(self.conv_kernels), h, w)

        x_se = [self.se(x) for x in output_features]
        x_se = torch.cat(x_se, dim=1)
        x_se = x_se.view(b, len(self.conv_kernels), c//len(self.conv_kernels), 1, 1)

        # Compute the attention weights
        weights = torch.softmax(x_se, dim=1)

        # Multiply the attention weights with the output features
        weighted_feats = torch.mul(weights, feats).view(b, -1, h, w)

        # Reduce the number of channels back to the original
        weighted_feats = self.reduce_conv(weighted_feats)

        return weighted_feats

if __name__ == "__main__":
    start = time.time()
    in_channels = 72
    x = torch.randn(1, in_channels, 224, 224)
    model = PyConv(in_channels)
    out = model(x)
    end = time.time()
    print(f"Time taken: {end - start:.4f} secs")

    from thop import profile

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PyConv(in_channels).to(device)
    input = torch.randn(1, in_channels, 224, 224).to(device)

    macs, params = profile(model, inputs=(input, ))
    print(f"FLOPs: {macs * 2:,}, Params: {params:,}")

