import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
import os
from PIL import Image
import math

class VisualizeAttention:
    def __init__(self, model, path, target_layer, hyper_params, transform, experiment):
        self.model = model.to(hyper_params["device"])
        self.path = path
        self.target_layer = self.model.features[target_layer]
        self.img_size = hyper_params["img_size"]
        self.transform = transform
        self.device = hyper_params["device"]
        self.experiment = experiment  # Add this line
        
        if os.path.isfile(path):
            self.image_path = path
        elif os.path.isdir(path):
            self.image_path = None
        else:
            raise ValueError("Invalid path provided. Please provide a path to an image or a folder containing images.")
    
    def attention_map(self):
        #image to tensor
        img = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_rgb, (self.img_size, self.img_size))
        #transform to tensor
        # Convert the OpenCV image to a PIL image
        img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0)
        #move to gpu
        img = img.to(self.device)
        #gradcam
        cam = GradCAM(model=self.model, target_layers=self.target_layer)
        #set target class
        targets = None
        grayscale_cam = cam(input_tensor=img, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        return grayscale_cam

    def image_to_numpy_array(self):
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
        # Assuming grayscale_cam and image_array are already available
        heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        # Convert image_array to float32 in the range [0, 1]
        image_array_float = np.float32(image_array) / 255
        # Blend the heatmap and the original image using an alpha value
        alpha = 0.5
        cam = (1 - alpha) * image_array_float + alpha * heatmap
        # Scale the result back to the [0, 255] range
        cam = np.uint8(255 * cam)
        # Convert the resulting CAM image to a PIL Image
        cam_pil = Image.fromarray(cam)
        # Log the image to Comet
        self.experiment.log_image(cam_pil, name="CAM_overlay")

        return cam
    
    def log_images(self):

        img_files = [f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for idx, img_file in enumerate(img_files):
            img_path = os.path.join(self.path, img_file)
            self.image_path = img_path
            image_array = self.image_to_numpy_array()
            grayscale_cam = self.attention_map()
            self.cam = self.image_map_overlay(image_array, grayscale_cam)

