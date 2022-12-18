import torch
import torch.nn as nn
from CBAM import CBAM

class ConvolutionalNetCBAM(nn.Module):
    def __init__(self,input_channels = 3, num_classes=10, image_size=28):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=1, padding=2), #input: 28x28x3 output: 28x28x32
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #input: 28x28x64 output: 28x28x128
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.cbam1 = CBAM(128)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), #input: 28x28x128 output: 28x28x32
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1), #input: 28x28x128 output: 28x28x32
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1), #input: 28x28x32 output: 28x28x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) #input: 28x28x16 output: 14x14x16

        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1), #input: 14x14x16 output: 14x14x8
            nn.BatchNorm2d(4),
            nn.ReLU())

        self.fc1 = nn.Linear(4 * image_size // 2 * image_size // 2, 64) #input: 14x14x4 output: 64
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out_1 = self.conv2(out)
        out = self.cbam1(out_1)
        #add skip connection
        out = out + out_1
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# Pass random data through the network to check the output shape
x = torch.randn(1, 3, 28, 28)
model = ConvolutionalNetCBAM()
print(model(x).shape)
