import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class CAM_from_DANet(nn.Module):
    
    def __init__(self, in_channels, **kwargs):
        super(CAM_from_DANet, self).__init__()
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

        return self.beta * out #batch_size, C, H, W -> (1, 3, 3, 3)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)
    
class _CBAM_CAM_DA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.ca = CAM_from_DANet(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        f_prime = self.ca(x)
        f_double_prime =self.sa(f_prime) * f_prime
        return f_double_prime
    
