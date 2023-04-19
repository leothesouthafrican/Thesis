import torch
import torch.nn as nn
import math
from SE_weight_module import SqueezeExcitation

def conv(in_channels, out_channels, kernel_size, padding, stride = 1, exp=1, groups=1):
    print("in_channels: ", in_channels)
    print("out_channels: ", out_channels)

    print("groups: ", groups)
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride,
                    padding=padding, dilation=1, groups=groups, bias = False)

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias = False)

def find_common_divisors(in_channels, out_channels):
    divisors = []

    for i in range(1, min(in_channels, out_channels) + 1):
        if in_channels % i == 0 and out_channels % i == 0:
            divisors.append(i)

    if len(divisors) >= 4:
        result = [1]
        result.extend(divisors[-3:])
        return result
    else:
        raise ValueError("Could not find enough divisors to divide in_channels and out_channels.")

class SPC(nn.Module):

    def __init__(self, in_channels, conv_kernels=[3,5,7,9], stride=1, reduction_rate = 1):
        super(SPC, self).__init__()

        out_channels = in_channels // reduction_rate

        conv_groups = find_common_divisors(in_channels, out_channels)
        print("conv_groups: ", conv_groups)

        self.conv_1 = conv(in_channels, out_channels, kernel_size=conv_kernels[0], 
                        padding=(conv_kernels[0]//2), stride = stride, groups=conv_groups[0])
        self.conv_2 = conv(in_channels, out_channels, kernel_size=conv_kernels[1], padding=(conv_kernels[1]//2),
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(in_channels, out_channels, kernel_size=conv_kernels[2], padding=(conv_kernels[2]//2),
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(in_channels, out_channels, kernel_size=conv_kernels[3], padding=(conv_kernels[3]//2),
                            stride=stride, groups=conv_groups[3])
        self.se = SqueezeExcitation(out_channels)
        self.split_channel = out_channels
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        batch_size = x.shape[0]

        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1,x2,x3,x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors

        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        out = conv1x1(in_channels=4*self.split_channel, out_channels=self.split_channel)(out)

        return out