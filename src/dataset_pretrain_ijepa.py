import glob
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.dataset_pretrain import unseen_classes


class PhotoOnlyIJEPADataset(torch.utils.data.Dataset):
    """
    Photo-only dataset for original-style I-JEPA pretraining.
    Returns a single augmented image per sample.
    """

    def __init__(self, opts, transform, mode="train"):
        self.opts = opts
        self.transform = transform
        self.all_categories = self._load_categories()
        self.all_categories = self._select_categories(self.all_categories, mode)
        self.all_photos_path = self._collect_photo_paths(self.all_categories)

    def __len__(self):
        return len(self.all_photos_path)

    def __getitem__(self, index):
        img_path = self.all_photos_path[index]
        category = img_path.split(os.path.sep)[-2]
        filename = os.path.basename(img_path)

        img = ImageOps.pad(
            Image.open(img_path).convert("RGB"),
            size=(self.opts.max_size, self.opts.max_size),
        )
        img_tensor = self.transform(img)
        return img_tensor, category, filename

    def _load_categories(self):
        categories = os.listdir(os.path.join(self.opts.data_dir, "sketch"))
        categories = [c for c in categories if c != ".ipynb_checkpoints"]
        return sorted(categories)

    def _select_categories(self, categories, mode):
        if self.opts.data_split > 0:
            categories = categories.copy()
            np.random.shuffle(categories)
            split_idx = int(len(categories) * self.opts.data_split)
            if mode == "train":
                return categories[:split_idx]
            return categories[split_idx:]

        if mode == "train":
            return sorted(list(set(categories) - set(unseen_classes)))
        return unseen_classes

    def _collect_photo_paths(self, categories):
        all_paths = []
        for category in categories:
            all_paths.extend(glob.glob(os.path.join(self.opts.data_dir, "photo", category, "*.jpg")))
        return sorted(all_paths)

    @staticmethod
    def data_transform(opts):
        # I-JEPA favors semantic-preserving augmentation.
        return transforms.Compose(
            [
                transforms.Resize((opts.max_size, opts.max_size)),
                transforms.RandomResizedCrop(
                    size=opts.max_size,
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
