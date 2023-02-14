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

        query = self.query_conv(x).view(batch_size, -1, H*W) #batch_size, C//reduction, H*W -> (1, 4, 1024)

        #Get the energy which is the dot product of key and query
        energy = torch.bmm(key.permute(0, 2, 1), query) #batch_size, H*W, H*W -> (1, 1024, 1024)

        #Get the attention
        attention = self.softmax(energy) #batch_size, H*W, H*W -> (1, 1024, 1024)

        #Get the value
        value = self.value_conv(x).view(batch_size, -1, H*W) #batch_size, C, H*W -> (1, 64, 1024)

        #Get the output
        out = torch.bmm(value, attention.permute(0, 2, 1)) #batch_size, C, H*W -> (1, 64, 1024)
        out = out.view(batch_size, C, H, W) #batch_size, C, H, W

        #Get the output
        out = self.alpha * out + x #batch_size, C, H, W -> (1, 64, 32, 32)

        return out
    
#Part 2: The Channel Attention Module
class CAM(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(CAM, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        #Get the batch size
        batch_size, _, height, width = x.size() #batch_size, C, H, W -> (1, 64, 32, 32)

        #Get the key and query values
        key = x.view(batch_size, -1, height * width) #batch_size, C, H*W -> (1, 64, 1024)
        query = x.view(batch_size, -1, height * width).permute(0, 2, 1) #batch_size, H*W, C -> (1, 1024, 64)

        attention = torch.bmm(key, query) #batch_size, C, C -> (1, 64, 64)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention #batch_size, C, C -> (1, 64, 64)
        attention = self.softmax(attention_new) #batch_size, C, C -> (1, 64, 64)

        feat_e = torch.bmm(attention, key).view(batch_size, -1, height, width) #batch_size, C, H, W -> (1, 64, 32, 32)
        out = self.beta * feat_e + x #batch_size, C, H, W -> (1, 64, 32, 32)

        return out


if __name__ == "__main__":
    
    device = torch.device("cpu")
    #time to test the model
    start =time.time()
    x = torch.randn(300, 64, 32, 32).to(device)
    model = SAM(64).to(device)
    out = model(x)

    end = time.time()
    print(f"SAM: {out.shape} in {end-start} seconds on {device}")

    start =time.time()
    x = torch.randn(300, 64, 32, 32).to(device)
    model = CAM(64).to(device)
    out = model(x)

    end = time.time()
    print(f"CAM: {out.shape} in {end-start} seconds on {device}")


