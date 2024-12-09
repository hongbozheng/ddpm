from typing import Dict

import torch
import os
from PIL import Image
from torch.utils.data import Dataset

class MedicalMNIST(Dataset):
    def __init__(self, path: str, transform=None) -> None:
        self.image_paths = []
        self.labels = []
        self.cls2idx = {}
        self.transform = transform

        self._load_data(path=path)

        return

    def _load_data(self, path: str) -> None:
        """
        Loads all image paths and labels into memory.
        """
        class_names = os.listdir(path=path)

        for i, class_name in enumerate(class_names):
            class_path = os.path.join(path, class_name)
            if os.path.isdir(class_path):
                self.cls2idx[class_name] = i
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    if image_path.lower().endswith('.jpeg'):
                        self.image_paths.append(image_path)
                        self.labels.append(i)
        return

    def __getitem__(self, idx: int) -> Dict:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(fp=image_path)

        image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.int64)
        item = {'image': image, 'label': label}

        return item

    def __len__(self):
        return len(self.image_paths)
