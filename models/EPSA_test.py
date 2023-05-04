import torch
import torch.nn as nn
import math
from SE_weight_module import SqueezeExcitation

class PSA(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernels=[3,5], stride=1):
        super().__init__()

        self.split_channel = out_channels // len(conv_kernels)
        self.conv_kernels = conv_kernels
        self.se = SqueezeExcitation(self.split_channel)

        for i in range(len(conv_kernels)): 
            setattr(self, f"conv_{i+1}", nn.Conv2d(in_channels, self.split_channel, kernel_size=conv_kernels[i],
                                    padding=(conv_kernels[i]//2), stride=stride, groups=int(2**((conv_kernels[i]-1)/2))))

    def forward(self, x):
        batch_size = x.shape[0]
        # Create a list of output features from each convolutional layer
        output_features = [getattr(self, f"conv_{i+1}")(x) for i in range(len(self.conv_kernels))]
        print(len(output_features))
        print(f"Dimensions of first cat: {output_features[0].shape}")
        feats = torch.cat(output_features, dim=1)
        print(f"Pre-view shape: {feats.shape}")
        feats = feats.view(batch_size, len(self.conv_kernels), self.split_channel, feats.shape[2], feats.shape[3])
        print(f"Post-view shape: {feats.shape}")

        x_se = [self.se(x) for x in output_features]
        print(f"Shape of x_se: len(x_se) = {len(x_se)})")
        print(f"Shape of first x_se: {x_se[0].shape}")
        x_se = torch.cat(x_se, dim=1)
        print(f"Shape of x_se after cat: {x_se.shape}")
        x_se = x_se.view(batch_size, len(self.conv_kernels), self.split_channel, x_se.shape[2], x_se.shape[3])
        print(f"Shape of x_se after view: {x_se.shape}")

        # Compute the attention weights
        weights = torch.softmax(x_se, dim=1)
        print(f"Shape of weights: {weights.shape}")

        # Multiply the attention weights with the output features
        weighted_feats = torch.mul(weights, feats)
        print(f"Shape of weighted_feats: {weighted_feats.shape}")

        return weighted_feats.view(batch_size, -1, weighted_feats.shape[3], weighted_feats.shape[4])

class PSA_Block(nn.Module):
    def __init__(self,PSA_block, in_channels, out_channels, conv_kernels=[3,5], stride=1):
        super().__init__()
        self.psa = PSA_block(in_channels, out_channels, conv_kernels, stride)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.conv1x1(x)
        out = self.bn(out)
        out = self.relu(out)

        out = self.psa(out)
        out = self.bn(out)
        out = self.relu(out)

        out = self.conv1x1(out)
        out = self.bn(out)
        out = self.relu(out)

        return out

if __name__ == "__main__":
    x = torch.randn(1, 32, 224, 224)
    psa = PSA_Block(PSA, 32, 32)
    out = psa(x)
    print(f"Shape of output: {out.shape}")
