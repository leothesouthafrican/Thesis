import cv2
import numpy as np
import os
from PIL import Image
import math
from pytorch_grad_cam import GradCAM
import torch
import torch.nn as nn

class VisualizeAttention:
    def __init__(self, model, path, hyper_params, transform, experiment):
        """
        Initialize the VisualizeAttention class.

        Args:
        - model (nn.Module): The PyTorch model whose attention maps we want to visualize.
        - path (str): Path to the image file or folder containing images.
        - target_layer (int): Index of the target layer in the model for which the attention maps are to be generated.
        - hyper_params (dict): Dictionary of hyperparameters.
        - transform (torchvision.transforms): Image transformations to be applied before processing.
        - experiment (comet_ml.Experiment): Comet.ml experiment object for logging images.
        """
        self.model = model.to(hyper_params["device"])
        self.path = path
        self.img_size = hyper_params["img_size"]
        self.transform = transform
        self.device = hyper_params["device"]
        self.experiment = experiment
        self.target_layer = self.find_last_conv_layer()

        if os.path.isfile(path):
            self.image_path = path
            print(f"found file")
        elif os.path.isdir(path):
            self.image_path = None
        else:
            raise ValueError("Invalid path provided. Please provide a path to an image or a folder containing images.")
        
    #Find the last conv layer
    def find_last_conv_layer(self):
        last_conv_layer = None
        for layer in reversed(list(self.model.modules())):
            if isinstance(layer, nn.Conv2d):
                last_conv_layer = layer
                break
        return last_conv_layer

    def attention_map(self):
        """
        Generate the attention map for the image.

        Returns:
        - grayscale_cam (np.ndarray): Attention map as a NumPy array.
        """
        # Load image and convert to RGB
        img = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, (self.img_size, self.img_size))

        # Transform image to tensor
        img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0)

        # Move tensor to device
        img = img.to(self.device)

        # Compute GradCAM
        cam = GradCAM(model=self.model, target_layers=[self.target_layer])
        targets = None
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        return grayscale_cam

    def image_to_numpy_array(self):
        """
        Load the image and convert it to a NumPy array.

        Returns:
        - image_array (np.ndarray): Image as a NumPy array.
        """
        # Read the image
        image = cv2.imread(self.image_path)
        # Resize the image
        image = cv2.resize(image, (self.img_size, self.img_size))
        # Convert the image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert the image to a NumPy array
        image_array = np.array(image)

        return image_array

    def image_map_overlay(self, image_array, grayscale_cam):
        """
        Overlay the attention map on the original image.

        Args:
        - image_array (np.ndarray): Image as a NumPy array.
        - grayscale_cam (np.ndarray): Attention map as a NumPy array.

        Returns:
        - cam (np.ndarray): Combined image and attention map as a NumPy array.
        """
        # Apply color map to grayscale attention map
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # Convert image_array to float32 in the range [0, 1]
        array_float = np.float32(image_array) / 255
        # Blend the heatmap and the original image using an alpha value
        alpha = 0.5
        cam = (1 - alpha) * array_float + alpha * heatmap
        # Scale the result back to the [0, 255] range
        cam = np.uint8(255 * cam)
        # Convert the resulting CAM image to a PIL Image
        cam_pil = Image.fromarray(cam)
        # Log the image to Comet
        self.experiment.log_image(cam_pil, name="CAM_overlay")

        return cam

    def log_images(self):
        """
        Log the images with overlaid attention maps to the Comet experiment.

        This method assumes that the `path` provided during initialization is a folder containing images.
        """
        img_files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for idx, img_file in enumerate(img_files):
            img_path = os.path.join(self.path, img_file)
            self.image_path = img_path
            image_array = self.image_to_numpy_array()
            grayscale_cam = self.attention_map()
            self.cam = self.image_map_overlay(image_array, grayscale_cam)