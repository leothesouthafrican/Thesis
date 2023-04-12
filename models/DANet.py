import torch
import torch.nn as nn
import torch.nn.functional as F
import time

#Implementing the DANet Model in two parts

#Part 1: The Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SAM, self).__init__()

        #key and query convolutions
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//reduction, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//reduction, kernel_size=1)

        #value convolution
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        #softmax
        self.softmax = nn.Softmax(dim=-1)

        #alpha
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        #Get the batch size
        batch_size, C, H, W = x.size() #batch_size, C, H, W -> (1, 64, 32, 32)

        #Get the key and query values
        key = self.key_conv(x).view(batch_size, -1, H*W) #batch_size, C//reduction, H*W -> (1, 4, 1024)

        query = self.query_conv(x).view(batch_size, -1, H*W).permute(0, 2, 1) #batch_size, C//reduction, H*W -> (1, 1024, 4)

        #Get the energy which is the dot product of query and key
        energy = torch.bmm(query, key) #batch_size, H*W, H*W -> (1, 1024, 1024)

        #Get the attention
        attention = self.softmax(energy) #batch_size, H*W, H*W -> (1, 1024, 1024)

        #Get the value
        value = self.value_conv(x).view(batch_size, -1, H*W) #batch_size, C, H*W -> (1, 64, 1024)

        #Get the output
        out = torch.bmm(value, attention.permute(0, 2, 1)) #batch_size, C, H*W -> (1, 64, 1024)
        out = out.view(batch_size, C, H, W) #batch_size, C, H, W

        #Get the output
        out = self.alpha * out #batch_size, C, H, W -> (1, 64, 32, 32)

        return out
    
#Part 2: The Channel Attention Module
class CAM(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(CAM, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        #Get the batch size
        batch_size, channels, height, width = x.size() #batch_size, C, H, W -> (1, 3, 3, 3

        query = x.view(batch_size, channels, -1) #batch_size, C, H*W -> (1, 3, 9)

        key = x.view(batch_size, channels, -1).permute(0, 2, 1) #batch_size, H*W, C -> (1, 9, 3)

        #Get the energy which is the dot product of key and query
        energy = torch.bmm(query, key) #batch_size, C, C -> (1, 3, 3)

        #Get the energy new
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy #batch_size, C, C -> (1, 3, 3) this is the max value of the energy matrix
        
        #Get the attention
        attention = self.softmax(energy_new) #batch_size, C, C -> (1, 3, 3)

        #Get the value
        value = x.view(batch_size, channels, -1) #batch_size, C, H*W -> (1, 3, 9)

        #Get the output
        out = torch.bmm(attention, value) #batch_size, C, H*W -> (1, 3, 9)

        #Get the output
        out = out.view(batch_size, channels, height, width) #batch_size, C, H, W -> (1, 3, 3, 3)

        #Get the output
        out = self.beta * out #batch_size, C, H, W -> (1, 3, 3, 3)

        return out

#Part 3: The DANet Module
class _DANet(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(_DANet, self).__init__()
        self.sam = SAM(in_channels, reduction)
        self.cam = CAM(in_channels)

    def forward(self, x):
        out = self.sam(x) + self.cam(x)
        return out











