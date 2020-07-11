import os
from glob import glob
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PreprocessingDataset(Dataset):

    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        # List images in folder by pattern
        pattern = os.path.join(self.images_dir, "*.jpg")
        # Use glob over iglob to sort and calculate length
        self.images_paths = sorted(glob(pattern))

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images_paths[idx]
        img_fn = Path(img_name).stem
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {"image": img, "id": img_fn, "idx": idx}
