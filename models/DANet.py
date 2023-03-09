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
        out = self.alpha * out #batch_size, C, H, W -> (1, 64, 32, 32)

        #Add the original input
        out = out + x #batch_size, C, H, W -> (1, 64, 32, 32)

        return out
    
#Part 2: The Channel Attention Module
class CAM(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(CAM, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

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

        out = self.beta * feat_e #batch_size, C, H, W -> (1, 64, 32, 32)
        
        #add the original input
        out = out + x #batch_size, C, H, W -> (1, 64, 32, 32)
        
        return out

#Part 3: The DANet Module
class DANet(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(DANet, self).__init__()
        self.sam = SAM(in_channels, reduction)
        self.cam = CAM(in_channels)

    def forward(self, x):
        out = self.sam(x) + self.cam(x)
        return out
    
if __name__ == "__main__":
    
        import torch
        from PIL import Image
        import torchvision.transforms as transforms
        import matplotlib.pyplot as plt
        model = DANet(64)

        def image_to_64Ctensor(path, size=(128, 128)):
            image = Image.open(path).convert('RGB')
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize(size)])
            tensor = transform(image).unsqueeze(0)

            conv2d = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, padding=0)

            return conv2d(tensor)

        cam = CAM(64)
        sam = SAM(64)

        #









