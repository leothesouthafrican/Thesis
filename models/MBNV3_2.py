import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, MobileNet_V3_Large_Weights as weights


class MBNV3Creator(nn.Module):

    """_summary_: This class is used to create a MobileNetV3 model 
    with a custom Squeeze and Excitation block or a custom module.
    """
    def __init__(self, model, num_classes, weights, device, module = None):
        super().__init__()
        self.model = model(weights = weights)
        self.model.classifier[3] = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)
        self.module = module
        self.device = device

    # Initialize the weights of the inserted module layers or the SE block
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
        """_summary_: This function is used to replace the SE block with the inserted module.

        Returns:
            _type_: torch.nn.Module
        """
        # If a custom module is inserted, replace all SE blocks with the inserted module
        if self.module != None:
            counter = 0
            for i in range(len(self.model.features)):
                try:
                    if type(self.model.features[i].block[2]) == torchvision.ops.misc.SqueezeExcitation:
                        prev_out_channels = self.model.features[i].block[0].out_channels
                        self.model.features[i].block[2] = self.module(prev_out_channels)
                        #set the requires_grad attribute of the inserted module layers to True
                        for param in self.model.features[i].block[2].parameters():
                            param.requires_grad = True
                        counter += 1

                    if type(self.model.features[i].block[1]) == torchvision.ops.misc.SqueezeExcitation:
                        prev_out_channels = self.model.features[i].block[0].out_channels
                        self.model.features[i].block[1] = self.module(prev_out_channels)
                        counter += 1
                        for param in self.model.features[i].block[1].parameters():
                            param.requires_grad = True
                except:
                    pass
            print(f"{counter} SE blocks replaced with {self.module}")

        # Set the weights of the entire model to not trainable
        for param in self.model.parameters():
            param.requires_grad = False
        # Set the weights of the last layer to trainable
        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True
        # initialize the weights of the last layer
        self.model.classifier[-1].apply(self._weights_init)

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
            # if no custom module is inserted, init the weights of the SE block
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

        model = self.model.to(self.device)

        return model