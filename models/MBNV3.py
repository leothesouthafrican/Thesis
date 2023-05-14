import torch
import torch.nn as nn
import torchvision
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, MobileNet_V3_Large_Weights as weights_large, MobileNet_V3_Small_Weights as weights_small


class MBNV3_Creator(nn.Module):
    """
    This class is used to create a MobileNetV3 model with a custom Squeeze and Excitation block or a custom module.
    """

    def __init__(self, num_classes, device, module=None):
        super().__init__()
        self.model = mobilenet_v3_small(weights=None)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes, bias=True)
        self.module = module
        self.device = device
        self.module_layers = [1, 4, 5, 6, 7, 8, 9, 10, 11]

    def _weights_init(self):
        """
        Xavier initialize the weights of the inserted module layers or the SE block.
        """
        torch.manual_seed(42)
        if isinstance(self.model, nn.Conv2d):
            torch.nn.init.xavier_uniform_(self.model.weight)
            if self.model.bias is not None:
                torch.nn.init.zeros_(self.model.bias)
                print(f"nn.Conv2d weights initialized with Xavier initialization.")
        elif isinstance(self.model, nn.BatchNorm2d):
            self.model.weight.data.fill_(1)
            self.model.bias.data.zero_()
            print(f"nn.BatchNorm2d weights initialized with Xavier initialization.")
        elif isinstance(self.model, nn.Linear):
            n = self.model.weight.size(1)
            self.model.weight.data.normal_(0, 0.01)
            self.model.bias.data.zero_()
            print(f"nn.Linear weights initialized with Xavier initialization.")

        print(f"Model weights initialized with Xavier initialization.")

    def insert_modules(self):
        """
        This function is used to change the SE block with the custom module.
        """

        prev_out_channels = self.model.features[1].block[0].out_channels
        self.model.features[1].block[1] = self.module(prev_out_channels) if self.module != None else self.model.features[1].block[1]
        
        for i in self.module_layers[1:]:
            prev_out_channels = self.model.features[i].block[0].out_channels
            self.model.features[i].block[2] = self.module(prev_out_channels) if self.module != None else self.model.features[i].block[2]

        print(f"Model modified with custom module: {self.module} in layers: {self.module_layers}.") if self.module != None else print(f"SE block in layers: {self.module_layers}.")

    def set_all_parameters_trainable(self):
        """
        This function is used to set all the parameters of the model trainable.
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def build_model(self):
        """
        This function is used to build the model.
        """
        self.model = self.model.to(self.device)
        self.insert_modules()
        self._weights_init()
        self.set_all_parameters_trainable()
        return self.model
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MBNV3_Creator(num_classes=1000, device=device, module=None)
    model = model.build_model()
    print(model.features[-1])

    #print the classifier
    print(model.classifier)
