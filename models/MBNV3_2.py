import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, MobileNet_V3_Large_Weights as weights

from pytorch_grad_cam import GradCAM as GCAM
#from pytorch_grad_cam.utils.model_targets import TargetCategory
from pytorch_grad_cam.utils.image import show_cam_on_image


class MBNV3GradCAM(nn.Module):
    def __init__(self, num_classes, model, weights, device, module = None):
        super().__init__()
        self.model = model(weights = weights)
        self.model.classifier[3] = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)
        self.module = module
        self.device = device

    def _weights_init(self, m):
        torch.manual_seed(42)
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)

    def build(self):
        if self.module != None:
            counter = 0
            for i in range(len(self.model.features)):
                try:
                    if type(self.model.features[i].block[2]) == torchvision.ops.misc.SqueezeExcitation:
                        prev_out_channels = self.model.features[i].block[0].out_channels
                        self.model.features[i].block[2] = self.module(prev_out_channels)
                        counter += 1
                except:
                    pass
            print(f"{counter} SE blocks replaced with {self.module}")

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True

        self.model.classifier[3].apply(self._weights_init)

        #init the weights of the inserted module layers or the SE block
        if self.module != None:
            for i in range(len(self.model.features)):
                try:
                    if type(self.model.features[i].block[2]) == self.module:
                        #init the weights of the inserted module layers
                        self.model.features[i].block[2].apply(self._weights_init)
                        for param in self.model.features[i].block[2].parameters():
                            param.requires_grad = True
                except:
                    pass
            print("Weights of the inserted module layers initialized and weights trainable")

        else:
            for i in range(len(self.model.features)):
                try:
                    if type(self.model.features[i].block[2]) == torchvision.ops.misc.SqueezeExcitation:
                        #init the weights of the SE block
                        self.model.features[i].block[2].apply(self._weights_init)
                        for param in self.model.features[i].block[2].parameters():
                            param.requires_grad = True
                except:
                    pass
            print("Weights of the SE block initialized and weights trainable")

        return self.model.to(self.device)




    


