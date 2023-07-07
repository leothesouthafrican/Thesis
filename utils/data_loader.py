import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ImageNetDataset(Dataset):
    def __init__(self, csv_file, root_dir, train=True, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

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

        if self.transform:
            image = self.transform(image)

        return image, label
    
def load_synset_mapping(synset_mapping_file):
    """Loads a synset mapping file"""
    synset_mapping = {}
    with open(synset_mapping_file) as f:
        lines = f.readlines()
    for idx, line in enumerate(lines):
        line = line.strip()  # Remove trailing newline
        synset, class_name = line.split(" ", 1)  # Split only once
        synset_mapping[int(idx)] = class_name  # Map numerical index to class name
    return synset_mapping


def load_index_mapping(synset_mapping):
    """Creates a mapping from synset to numerical index"""
    index_mapping = {synset: idx for idx, synset in enumerate(synset_mapping.keys())}
    return index_mapping



