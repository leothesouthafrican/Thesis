import torch
import numpy as np
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, MobileNet_V3_Large_Weights as weights_large, MobileNet_V3_Small_Weights as weights_small


class MBNV3Creator(nn.Module):

    """_summary_: This class is used to create a MobileNetV3 model 
    with a custom Squeeze and Excitation block or a custom module.
    """
    def __init__(self, model, num_classes, weights, device, module = None, manual_block_insertion = None):
        super().__init__()
        self.model = model(weights = weights)
        self.weights = weights
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True).apply(self._weights_init)
        self.module = module
        self.device = device
        self.layers = None
        self.model_variant = "small" if "small" in str(model) else "large"

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

    def define_layers_insertion(self):
        """_summary_: This function is used to define all the possible layers where the custom module
        or the SE block will be inserted.
        """
        se_layers = []
        for i in range (len(self.model.features)):
            try:
                if type(self.model.features[i].block[1]) == torchvision.ops.misc.SqueezeExcitation or \
                    type(self.model.features[i].block[2]) == torchvision.ops.misc.SqueezeExcitation:
                    se_layers.append(i)
            except:
                pass
        
        self.layers = se_layers
        print(f"SE layers: {se_layers}")
        return se_layers
    
    def insert_modules(self):
        """_summary_: This function is used to change the SE block with the custom module.
        """

        if self.model_variant == "small" and self.layers[0] == 1:
            prev_out_channels = self.model.features[1].block[0].out_channels
            self.model.features[1].block[1] = self.module(prev_out_channels) if self.module != None else self.model.features[1].block[1]
            self.model.features[1].block[1].apply(self._weights_init)

            for i in self.layers[1:]:
                prev_out_channels = self.model.features[i].block[0].out_channels
                self.model.features[i].block[2] = self.module(prev_out_channels) if self.module != None else self.model.features[i].block[2]
                self.model.features[i].block[2].apply(self._weights_init)
        else:
            for i in self.layers:
                prev_out_channels = self.model.features[i].block[0].out_channels
                self.model.features[i].block[2] = self.module(prev_out_channels) if self.module != None else self.model.features[i].block[2]
                self.model.features[i].block[2].apply(self._weights_init)

        print(f"{self.module if self.module != None else 'SE'} inserted in the following layers: {self.layers}")
        print(f"Weights initialized for {self.module if self.module != None else 'SE'} inserted in the following layers: {self.layers} as well as the last layer.")
        return self.model
    
    def set_grad(self):
        """_summary_: This function is used to set the gradient of the inserted module layers or the SE block.
        """
        #set all grads in the model to false
        for param in self.model.parameters():
            param.requires_grad = False

        #set the grad of the last layer to true
        for param in self.model.classifier[-1].parameters():
            param.requires_grad = True

        if self.model_variant == "small":
            for param in self.model.features[1].block[1].parameters():
                param.requires_grad = True

        for i in self.layers:
            for param in self.model.features[i].block[2].parameters():
                param.requires_grad = True

        print(f"Grads set to True for {self.layers} and the last layer. (False for the rest)")

        return self.model
    
    def build(self, manual_block_insertion:list = None, ):
        """_summary_: This function is used to build the model.
        """

        #check if the manual_block_insertion is not None
        if manual_block_insertion != None:
            #if yes, set the se.layers to the manual_block_insertion
            self.layers = manual_block_insertion
        else:
            #if not, define the layers where the SE block will be inserted
            self.define_layers_insertion()
        #insert the custom module
        self.insert_modules()

        #freeze and unfreeze the weights if weights are provided if not it means that we need to train the entire model
        if self.weights != None:
            self.set_grad()

        return self.model

