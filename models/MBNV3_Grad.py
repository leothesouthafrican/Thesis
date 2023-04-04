from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models import MobileNet_V3_Small_Weights as small_weights, MobileNet_V3_Large_Weights as large_weights
import torch.nn.functional as F
import torch
import torch.nn as nn


class MBNV3_Grad(nn.Module):
    def __init__(self, model, module, output_size):
        super(MBNV3_Grad, self).__init__()

        self.model = model
        self.module = module
        self.output_size = output_size
        self.layers = [1, 4, 5, 6, 7, 8, 9, 10, 11] if self.model == "small" else [4, 5, 6, 11, 12, 13, 14, 15]

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

    def set_grad(self):
        """_summary_: This function is used to set the gradient of the inserted module layers or the SE block.
        """
        #set all grads in the model to false
        for param in self.mbnv3.parameters():
            param.requires_grad = False

        #set the grad of the last layer to true
        for param in self.mbnv3.classifier[-1].parameters():
            param.requires_grad = True

        if self.model == "small":
            for param in self.mbnv3.features[1].block[1].parameters():
                param.requires_grad = True

            for i in self.layers[1:]:
                for param in self.mbnv3.features[i].block[2].parameters():
                    param.requires_grad = True
        else:
            for i in self.layers:
                for param in self.mbnv3.features[i].block[2].parameters():
                    param.requires_grad = True


        return self.mbnv3

    def create_custom_model(self, device):
        """_summary_: This function is used to create a custom model by inserting the custom module """
        if self.model == "small":
            # setting the base model
            self.mbnv3 = mobilenet_v3_small(weights= small_weights.IMAGENET1K_V1)
            # setting the custom module in specific layers for small variant
            if self.model == "small" and self.layers[0] == 1:
                prev_out_channels = self.mbnv3.features[1].block[0].out_channels
                self.mbnv3.features[1].block[1] = self.module(prev_out_channels) if self.module != None else self.mbnv3.features[1].block[1]
                self.mbnv3.features[1].block[1].apply(self._weights_init)

            for i in self.layers[1:]:
                prev_out_channels = self.mbnv3.features[i].block[0].out_channels
                self.mbnv3.features[i].block[2] = self.module(prev_out_channels) if self.module != None else self.mbnv3.features[i].block[2]
                self.mbnv3.features[i].block[2].apply(self._weights_init)
        else:
            # setting the base model in specific layers for large variant
            self.mbnv3 = mobilenet_v3_large(weights= large_weights.DEFAULT)
            for i in self.layers:
                prev_out_channels = self.mbnv3.features[i].block[0].out_channels
                self.mbnv3.features[i].block[2] = self.module(prev_out_channels) if self.module != None else self.mbnv3.features[i].block[2]
                self.mbnv3.features[i].block[2].apply(self._weights_init)

        self.mbnv3.classifier[-1] = nn.Linear(self.mbnv3.classifier[-1].in_features, self.output_size)
        self.mbnv3.classifier[-1].apply(self._weights_init)

        # move the model to GPU
        self.mbnv3 = self.mbnv3.to(device)

        self.set_grad()

        return self.mbnv3

    def restructure_model(self):
        # dissect the network to access its last convolutional layer
        self.features = self.mbnv3.features[:13] if self.model == "small" else self.mbnv3.features[:17]  
        # adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)  
        # get the classifier of the model
        self.classifier = self.mbnv3.classifier
        # placeholder for the gradients
        self.gradients = None

        # delete the original model
        del self.mbnv3

        return self.features, self.avgpool, self.classifier
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features(x)
        
    def forward(self, x, device):
        
        self.create_custom_model(device) # create the custom model
        self.restructure_model() # restructure the model
        
        x = self.features(x) # extract the features

        # register the hook
        h = x.register_hook(self.activations_hook) # register the hook
    
        # adaptively average pool the features
        x = self.avgpool(x) # adaptive average pooling

        x = x.squeeze() # flatten the output of the adaptive average pooling
        x = self.classifier(x) # get the class probabilities from the classifier

        return x