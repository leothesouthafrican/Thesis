#import dataset
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

class mixedSets(Dataset):
    def __init__(self, data_path, transform):
        self.data_path = data_path
        self.transform = transform
        self.data = datasets.ImageFolder(root=data_path, transform=transform)
        self.classes = self.data.classes
        self.class_to_idx = self.data.class_to_idx
        self.samples = self.data.samples
        self.targets = self.data.targets
        self.imgs = self.data.imgs
        self.train_indices, self.val_indices, self.test_indices = self.split_data()

    def split_data(self):
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
        for train_index, test_index in sss.split(self.samples, self.targets):
            train_indices, test_indices = train_index, test_index
        train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42, stratify=[self.targets[i] for i in train_indices])
        return sorted(train_indices), sorted(val_indices), sorted(test_indices)

    def get_train_dataset(self):
        train_dataset = torch.utils.data.Subset(self.data, self.train_indices)
        train_dataset.transform = self.transform
        return train_dataset

    def get_val_dataset(self):
        val_dataset = torch.utils.data.Subset(self.data, self.val_indices)
        val_dataset.transform = self.transform
        return val_dataset

    def get_test_dataset(self):
        test_dataset = torch.utils.data.Subset(self.data, self.test_indices)
        test_dataset.transform = self.transform
        return test_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]