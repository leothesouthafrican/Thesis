import torch
import torch.nn as nn
import math

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def depthwise_conv(inp, oup, kernel_size, stride, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y

class ControlModule(nn.Module):
    def __init__ (self, input_channels, output_channels, kernel_size, stride, relu = True):
        super(ControlModule, self).__init__()
        self.output_channels = output_channels
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, kernel_size // 2, bias = False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace = True) if relu else nn.Sequential(),
        )
    def forward(self, x):
        out = self.conv(x)
        return out
    
class DepthwiseSeperableModule(nn.Module):
    def __init__ (self, input_channels, output_channels, kernel_size, stride, relu = True):
        super(DepthwiseSeperableModule, self).__init__()
        self.output_channels = output_channels
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, kernel_size // 2, groups = input_channels, bias = False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace = True) if relu else nn.Sequential(),
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace = True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out

class GhostModule(nn.Module):
    def __init__ (self, input_channels, output_channels, kernel_size = 1, ratio = 2, dw_size = 3, stride = 1, relu = True):
        super(GhostModule, self).__init__()
        self.output_channels = output_channels
        
        init_channels = math.ceil(output_channels / ratio) # the number of channels for each ghost module
        new_channels = init_channels * (ratio - 1) 

        self.primary_conv = nn.Sequential(
            nn.Conv2d(input_channels, init_channels, kernel_size, stride, kernel_size // 2, bias = False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace = True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(in_channels= init_channels, out_channels= new_channels, kernel_size= dw_size, stride= 1, padding= dw_size // 2, groups= init_channels, bias= False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace = True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim = 1)
        return out[:, :self.output_channels, :, :]
    
class GhostBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, module_type):
        super(GhostBlock, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential()
        new_layers = []
        # pw
        if module_type == 'control':
            new_layers.append(ControlModule(inp, hidden_dim, kernel_size=1, stride=1, relu=True))
        elif module_type == 'depthwise_seperable':
            new_layers.append(DepthwiseSeperableModule(inp, hidden_dim, kernel_size=1, stride=1, relu=True))
        elif module_type == 'ghost':
            new_layers.append(GhostModule(inp, hidden_dim, kernel_size=1, relu=True))
        else:
            raise ValueError('Invalid module type')
        
        # dw
        new_layers.append(depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride)) if stride==2 else None

        # Squeeze-and-Excite
        new_layers.append(SELayer(hidden_dim)) if use_se else None

        # pw-linear to match dimensions
        new_layers.append(GhostModule(hidden_dim, oup, kernel_size=1, relu=False))

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, relu=False),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        self.conv = nn.Sequential(*new_layers)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
    

class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1, module_type='ghost'):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            nn.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True)
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBlock
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se, module_type=module_type))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        input_channel = output_channel

        output_channel = 1280
        self.classifier = nn.Sequential(
            nn.Linear(input_channel, output_channel, bias=False),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.squeeze(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ghost_net(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs_large = [
        # k, t, c, SE, s k = kernel_size, t = exp_size, c = output_channel, SE = use_se, s = stride
        [3,  16,  16, 0, 1],
        [3,  48,  24, 0, 2],
        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],
        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],
        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 0, 1],
        [5, 960, 160, 1, 1]
    ]

    cfgs_small = [
        # k, t, c, SE, s k = kernel_size, t = exp_size, c = output_channel, SE = use_se, s = stride
        [3,  16,  16, 1, 2],
        [3,  72,  24, 0, 2],
        [3,  88,  24, 0, 1],
        [5,  96,  40, 1, 2],
        [5, 240,  40, 1, 1],
        [5, 240,  40, 1, 1],
        [5, 120,  48, 1, 1],
        [5, 144,  48, 1, 1],
        [5, 288,  96, 1, 2],
        [5, 576,  96, 1, 1],
        [5, 576,  96, 1, 1],
    ]


    return GhostNet(cfgs_small, **kwargs)

import torch

if __name__ == '__main__':
    
    from torchsummary import summary
    from thop import profile

    def measure_flops_and_time(device, model):
        input_size = (3, 224, 224)
        model = model.to(device)
        device_str = str(device)  # Convert torch.device object to string

        input_data = torch.randn(1, *input_size).to(device)
        
        # Use thop to get FLOPs and number of parameters
        flops, params = profile(model, inputs=(input_data, ), verbose=False)

        # Print the results
        print(f"Total FLOPs: {flops}")
        print(f"Total Parameters: {params}")

        return flops
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ghost_net(num_classes=1000, width_mult=1, module_type='ghost')

    flops = measure_flops_and_time(device, model)
    print(flops)
