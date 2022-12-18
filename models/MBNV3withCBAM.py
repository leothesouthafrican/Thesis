#MobileNet V3 Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import models.functions as f
from models.CBAM import CBAM as cbam

#defining hswift and hsigmoid activation functions
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True): 
        super(h_sigmoid, self).__init__() 
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6 #defining the hard sigmoid activation function

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x) #defining the hard swish activation function
                                    #using the hard sigmoid function

#defining the squeeze and excitation block
class SqueezeBlock(nn.Module): #defining the squeeze and excitation block
    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.flatten = nn.AdaptiveAvgPool2d(1) #reducing the dimensionality
        self.dense = nn.Sequential(
            nn.Linear(exp_size, exp_size // divide), #excitation block
            nn.ReLU(inplace=True),
            nn.Linear(exp_size // divide, exp_size), #bringing back the dimensionality
            h_sigmoid() #using the hard sigmoid activation function
        )

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = self.flatten(x).view(batch, -1)
        out = self.dense(out) 
        out = out.view(batch, channels, 1, 1) 

        return out * x

#defining the residual block
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size, stride, NL, SE, CBAM, exp_size):
        super(InvertedResidualBlock, self).__init__()
        self.out_channels = out_channels
        self.NL = NL
        self.SE = SE
        self.CBAM = cbam
        padding = (kernal_size - 1) // 2 #padding to maintain the dimensionality

        self.use_connect = stride == 1 and in_channels == out_channels #using the residual connection 
                                                                        #if the stride is 1 and the input
                                                                        # and output channels are the same

        if self.NL == 'HS':
            activation = h_swish #using the hard swish activation function
        else:
            activation = nn.ReLU #using the relu activation function

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(exp_size),
            activation(inplace=True)
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernal_size, stride=stride, padding=padding, groups=exp_size),
            nn.BatchNorm2d(exp_size),
        )

        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)

        if self.CBAM:
            self.CBAM = cbam(exp_size)

        self.point_conv = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=1, stride=1, padding=0), #pointwise convolution
            nn.BatchNorm2d(out_channels), #batch normalization
            activation(inplace=True) #activation function
        )

    def forward(self, x):
        # MobileNetV3 residual block
        out = self.conv(x) 
        out = self.depth_conv(out)

        if self.SE:
            out = self.squeeze_block(out)

        if self.CBAM:
            out = self.CBAM(out)

        out = self.point_conv(out)

        if self.use_connect:
            return x + out
        else:
            return out

    
#defining the MobileNetV3 model
class MobileNetV3CBAM(nn.Module):
    def __init__(self, mode, num_classes=10, mu=1.0, dropout=0.2):
        super(MobileNetV3CBAM, self).__init__()
        self.num_classes = num_classes

        if mode == 'small':
            self.cfg = [
                #in, out, k, s, nl, se, CBAM, exp
                [16, 16, 3, 2, "RE", True, False, 16],
                [16, 24, 3, 2, "RE", False, False, 72],
                [24, 24, 3, 1, "RE", False, False, 88],
                [24, 40, 5, 2, "RE", True, False, 96],
                [40, 40, 5, 1, "RE", True, False, 240],
                [40, 40, 5, 1, "RE", True, False, 240],
                [40, 48, 5, 1, "HS", True, False, 120],
                [48, 48, 5, 1, "HS", True, True, 144],
                [48, 96, 5, 2, "HS", True, True, 288],
                [96, 96, 5, 1, "HS", True, True, 576],
                [96, 96, 5, 1, "HS", True, True, 576],
            ]

            first_conv_out = f._make_divisible(16 * mu)

            self.first_conv = nn.Sequential(
                nn.Conv2d(3, first_conv_out, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(first_conv_out),
                h_swish(inplace=True)
            )

            self.layers = []
            for i, (in_channels, out_channels, kernal_size, stride, NL, SE,CBAM, exp_size) in enumerate(self.cfg):
                in_channels = f._make_divisible(in_channels * mu)
                out_channels = f._make_divisible(out_channels * mu)
                exp_size = f._make_divisible(exp_size * mu)
                self.layers.append(InvertedResidualBlock(in_channels, out_channels, kernal_size, stride, NL, SE,CBAM, exp_size))
            self.layers = nn.Sequential(*self.layers)

            conv1_in = f._make_divisible(96 * mu) # making the input channels divisible
            conv1_out = f._make_divisible(576 * mu) # making the output channels divisible
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(conv1_in, conv1_out, kernel_size=3, stride=1, padding=1, groups=conv1_in),
                SqueezeBlock(conv1_out),
                nn.BatchNorm2d(conv1_out),
                h_swish(inplace=True))

            conv_2_in = f._make_divisible(576 * mu) # making the output channels divisible
            conv_2_out = f._make_divisible(1280 * mu) # making the output channels divisible
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(conv_2_in, conv_2_out, kernel_size=1, stride=1, padding=0),
                h_swish(inplace=True)
            )

            conv_3_in = f._make_divisible(1280 * mu) # making the output channels divisible
            self.out_conv3 = nn.Sequential(
                nn.Conv2d(conv_3_in, self.num_classes, kernel_size=1, stride=1, padding=0)
            )

        elif mode == 'large':
            self.cfg = [
                #in, out, k, s, nl, se, CBAM,  exp
                [16, 16, 3, 1, "RE", False, False, 16],
                [16, 24, 3, 2, "RE", False, False, 64],
                [24, 24, 3, 1, "RE", False, False, 72],
                [24, 40, 5, 2, "RE", True, False,72],
                [40, 40, 5, 1, "RE", True, False,120],
                [40, 40, 5, 1, "RE", True, False,120],
                [40, 80, 3, 2, "HS", False, False,240],
                [80, 80, 3, 1, "HS", False, False,200],
                [80, 80, 3, 1, "HS", False, False,184],
                [80, 80, 3, 1, "HS", False, False,184],
                [80, 112, 3, 1, "HS", True, False,480],
                [112, 112, 3, 1, "HS", True, True,672],
                [112, 160, 5, 2, "HS", True,True,672],
                [160, 160, 5, 1, "HS", True, True,672],
                [160, 160, 5, 1, "HS", True, True,960],
            ]

            first_conv_out = f._make_divisible(16 * mu)
            self.first_conv = nn.Sequential(
                nn.Conv2d(3, first_conv_out, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(first_conv_out),
                h_swish(inplace=True)
            )

            self.layers = []
            for i, (in_channels, out_channels, kernal_size, stride, NL, SE,CBAM, exp_size) in enumerate(self.cfg):
                in_channels = f._make_divisible(in_channels * mu)
                out_channels = f._make_divisible(out_channels * mu)
                exp_size = f._make_divisible(exp_size * mu)
                self.layers.append(InvertedResidualBlock(in_channels, out_channels, kernal_size, stride, NL, SE,CBAM, exp_size))
            self.layers = nn.Sequential(*self.layers)

            conv1_in = f._make_divisible(160 * mu) # making the input channels divisible
            conv1_out = f._make_divisible(960 * mu) # making the output channels divisible
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(conv1_in, conv1_out, kernel_size=3, stride=1, padding=1, groups=conv1_in),
                SqueezeBlock(conv1_out),
                nn.BatchNorm2d(conv1_out),
                h_swish(inplace=True))

            conv_2_in = f._make_divisible(960 * mu) # making the output channels divisible
            conv_2_out = f._make_divisible(1280 * mu) # making the output channels divisible
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(conv_2_in, conv_2_out, kernel_size=1, stride=1, padding=0),
                h_swish(inplace=True)
            )

            conv_3_in = f._make_divisible(1280 * mu) # making the output channels divisible
            self.out_conv3 = nn.Sequential(
                nn.Conv2d(conv_3_in, self.num_classes, kernel_size=1, stride=1, padding=0)
            )

        self.apply(f._weights_init)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layers(x)
        x = self.out_conv1(x)
        x = self.out_conv2(x)
        x = self.out_conv3(x)
        x = x.view(x.size(0), -1)
        return x