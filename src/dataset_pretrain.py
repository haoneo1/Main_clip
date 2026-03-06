import os
import glob
import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

unseen_classes = [
    "bat", "cabin", "cow", "dolphin", "door", "giraffe", "helicopter",
    "mouse", "pear", "raccoon", "rhinoceros", "saw", "scissors",
    "seagull", "skyscraper", "songbird", "sword", "tree",
    "wheelchair", "windmill", "window",
]


class PadToSquare:
    """
    Pad PIL image to square while keeping aspect ratio.
    """
    def __init__(self, fill=(255, 255, 255)):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        max_side = max(w, h)
        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left
        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top
        return transforms.functional.pad(
            img,
            padding=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill
        )


class MultiModalJEPADataset(torch.utils.data.Dataset):
    """
    JEPA pretraining dataset with both photo and sketch.
    Each sample returns:
        img_view1, img_view2, sk_view1, sk_view2, category
    """

    def __init__(self, opts, transform_img, transform_sk=None, mode='train', return_orig=False):
        self.opts = opts
        self.transform_img = transform_img
        self.transform_sk = transform_sk if transform_sk is not None else transform_img
        self.return_orig = return_orig

        sketch_root = os.path.join(self.opts.data_dir, 'sketch')
        self.all_categories = os.listdir(sketch_root)
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
        self.all_categories = sorted(self.all_categories)

        # split categories
        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            split_idx = int(len(self.all_categories) * self.opts.data_split)
            if mode == 'train':
                self.all_categories = self.all_categories[:split_idx]
            else:
                self.all_categories = self.all_categories[split_idx:]
        else:
            if mode == 'train':
                self.all_categories = sorted(list(set(self.all_categories) - set(unseen_classes)))
            else:
                self.all_categories = sorted(unseen_classes)

        self.photo_dict = {}
        self.sketch_dict = {}
        valid_categories = []

        for category in self.all_categories:
            photo_list = sorted(glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpg')))
            photo_list += sorted(glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.png')))

            sketch_list = sorted(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png')))
            sketch_list += sorted(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.jpg')))

            if len(photo_list) > 0 and len(sketch_list) > 0:
                self.photo_dict[category] = photo_list
                self.sketch_dict[category] = sketch_list
                valid_categories.append(category)

        self.all_categories = sorted(valid_categories)

        # build explicit sample list: each photo is one anchor sample
        self.samples = []
        for category in self.all_categories:
            for img_path in self.photo_dict[category]:
                self.samples.append((category, img_path))

        self.length = len(self.samples)

    def __len__(self):
        return self.length

    def _load_rgb(self, path):
        return Image.open(path).convert('RGB')

    def __getitem__(self, index):
        category, img_path = self.samples[index]
        sk_path = random.choice(self.sketch_dict[category])

        img = self._load_rgb(img_path)
        sk = self._load_rgb(sk_path)

        img_view1 = self.transform_img(img)
        img_view2 = self.transform_img(img)

        sk_view1 = self.transform_sk(sk)
        sk_view2 = self.transform_sk(sk)

        if self.return_orig:
            return img_view1, img_view2, sk_view1, sk_view2, category, img, sk
        else:
            return img_view1, img_view2, sk_view1, sk_view2, category

    @staticmethod
    def data_transform(opts):
        transform = transforms.Compose([
            PadToSquare(fill=(255, 255, 255)),
            transforms.RandomResizedCrop(
                size=opts.max_size,
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05
                )
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform