import torch
import torch.nn as nn
import time

def LCM_64(n):
    return ((n + 63) // 64) * 64

class PyConv(nn.Module):
    def __init__(self, in_channels, stride = 1):
        super().__init__()
        self.in_channels = in_channels
        
        # Define the expansion channels
        self.exp_channels = LCM_64(in_channels)

        # Define the initial conv_kernels and pyconv_groups
        conv_kernels = [3, 5, 7, 9]

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=in_channels, kernel_size = conv_kernels[0], stride = 1, padding = "same"),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=in_channels, kernel_size = conv_kernels[1], stride = 1, padding = "same"),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        self.third_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=in_channels, kernel_size = conv_kernels[2], stride = 1, padding = "same"),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )

        self.fourth_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=in_channels, kernel_size = conv_kernels[3], stride = 1, padding = "same"),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace = True)
        )


    def forward(self, x):
        # Apply each of the convolutions to the input feature

        x1 = self.first_conv(x)
        x2 = self.second_conv(x)
        x3 = self.third_conv(x)
        x4 = self.fourth_conv(x)

        return x4


if __name__ == "__main__":
    start = time.time()
    in_channels = 72
    x = torch.randn(1, in_channels, 224, 224)
    model = PyConv(in_channels)
    out = model(x)
    end = time.time()
    print(out.shape)
    print(f"Time taken: {end - start:.4f} secs")

    from thop import profile

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PyConv(in_channels).to(device)
    input = torch.randn(1, in_channels, 224, 224).to(device)

    macs, params = profile(model, inputs=(input, ))
    print(f"FLOPs: {macs * 2:,}, Params: {params:,}")