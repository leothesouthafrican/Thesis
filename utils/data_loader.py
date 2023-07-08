import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ImageNetDataset(Dataset):
    def __init__(self, csv_file, root_dir, synset_mapping, transform=None, train=True):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.synset_mapping = synset_mapping  # mapping from synset string to numerical label
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Correctly form the path to the image
        if self.train:
            img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0][:9], self.df.iloc[idx, 0] + '.JPEG')
        else:
            img_name =  os.path.join(self.root_dir, self.df.iloc[idx, 0] + '.JPEG')

        image = Image.open(img_name).convert('RGB')
        label_bounding_box = self.df.iloc[idx, 1]
        label, _ = label_bounding_box.split(" ", 1)
        label = self.synset_mapping[label]  # Convert string label to numerical label

        if self.transform:
            image = self.transform(image)

        return image, label