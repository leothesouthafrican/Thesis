import torch
import torch.nn as nn

class ChannelAttentionModule(nn.Module): 
    def __init__(self, in_channels, ratio = 8): #passing the number of input channels and the ratio for dimensionality reduction
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1) # Adaptive average pooling layer
        self.max_pool = nn.AdaptiveMaxPool2d(1) # Adaptive max pooling layer

        self.sharedMLP = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias = False), # Linear layer with dimensionality reduction
            nn.ReLU(inplace = True), # ReLU activation
            nn.Linear(in_channels // ratio, in_channels, bias = False) # Linear layer with dimensionality increase
        )

        self.sigmoid = nn.Sigmoid() # Sigmoid activation

    def forward(self, x):
        x1 = self.avg_pool(x).squeeze(-1).squeeze(-1) # Average pooling and flattening
        x1 = self.sharedMLP(x1) # Shared MLP and reshaping

        x2 = self.max_pool(x).squeeze(-1).squeeze(-1) # Max pooling and flattening
        x2 = self.sharedMLP(x2) # Shared MLP and reshaping

        #Now we add the two outputs and apply the sigmoid activation
        feats = self.sigmoid(x1 + x2).unsqueeze(-1).unsqueeze(-1) # Sigmoid activation and reshaping

        refined_features = x * feats # Element-wise multiplication

        return refined_features

class SpatialAttentionModule(nn.Module):

    def __init__(self, kernel_size = 7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size = kernel_size, padding = kernel_size // 2, bias = False) # Convolutional layer
        self.sigmoid = nn.Sigmoid() # Sigmoid activation

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True) # Average pooling on the first dimension (the channel dimension)
        max_out, _ = torch.max(x, dim = 1, keepdim = True) # Max pooling on the first dimension (the channel dimension)
        #now to concatenate the two outputs
        feats = torch.cat([avg_out, max_out], dim = 1) # Concatenating the two outputs on the channel dimension

        feats = self.conv(feats) # Convolutional layer
        feats = self.sigmoid(feats) # Sigmoid activation
    
        #Now we multiply the input with the output of the convolutional layer
        refined_features = x * feats # Element-wise multiplication
        
        return refined_features

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio = 8, kernel_size = 7):
        super().__init__()

        self.CAM = ChannelAttentionModule(in_channels, ratio) # Channel attention module
        self.SAM = SpatialAttentionModule(kernel_size) # Spatial attention module

    def forward(self, x):
        x = self.CAM(x) # Channel attention module
        x = self.SAM(x) # Spatial attention module

        return x