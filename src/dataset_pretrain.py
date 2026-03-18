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


class MultiModalJEPADataset(torch.utils.data.Dataset):
    """
    Each sample returns:
        img_global1, img_global2, img_local,
        sk_global1, sk_global2, sk_local,
        category
    """

    def __init__(self, opts, transforms_dict, mode='train', return_orig=False):
        self.opts = opts
        self.return_orig = return_orig

        self.img_global_tf = transforms_dict["img_global"]
        self.img_local_tf = transforms_dict["img_local"]
        self.sk_global_tf = transforms_dict["sk_global"]
        self.sk_local_tf = transforms_dict["sk_local"]

        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')
        self.all_categories = sorted(self.all_categories)

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
            sketch_list = sorted(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png')))
            if len(sketch_list) == 0:
                sketch_list = sorted(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.jpg')))

            if len(photo_list) > 0 and len(sketch_list) > 0:
                self.photo_dict[category] = photo_list
                self.sketch_dict[category] = sketch_list
                valid_categories.append(category)

        self.all_categories = valid_categories
        self.length = sum(len(self.photo_dict[c]) for c in self.all_categories)

    def __len__(self):
        return self.length

    def _load_and_pad(self, path):
        return ImageOps.pad(
            Image.open(path).convert('RGB'),
            size=(self.opts.max_size, self.opts.max_size)
        )

    def __getitem__(self, index):
        category = self.all_categories[index % len(self.all_categories)]

        img_path = random.choice(self.photo_dict[category])
        sk_path = random.choice(self.sketch_dict[category])

        img = self._load_and_pad(img_path)
        sk = self._load_and_pad(sk_path)

        img_global1 = self.img_global_tf(img)
        img_global2 = self.img_global_tf(img)
        img_local = self.img_local_tf(img)

        sk_global1 = self.sk_global_tf(sk)
        sk_global2 = self.sk_global_tf(sk)
        sk_local = self.sk_local_tf(sk)

        if self.return_orig:
            return (
                img_global1, img_global2, img_local,
                sk_global1, sk_global2, sk_local,
                category, img, sk
            )
        else:
            return (
                img_global1, img_global2, img_local,
                sk_global1, sk_global2, sk_local,
                category
            )

    @staticmethod
    def data_transform(opts):
        normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

        img_global = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.RandomResizedCrop(
                size=opts.max_size,
                scale=(0.65, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                    hue=0.08
                )
            ], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

        img_local = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.RandomResizedCrop(
                size=opts.max_size,
                scale=(0.35, 0.7),
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.25,
                    contrast=0.25,
                    saturation=0.25,
                    hue=0.08
                )
            ], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

        sk_global = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.RandomResizedCrop(
                size=opts.max_size,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

        sk_local = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.RandomResizedCrop(
                size=opts.max_size,
                scale=(0.45, 0.8),
                ratio=(0.8, 1.25)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ])

        return {
            "img_global": img_global,
            "img_local": img_local,
            "sk_global": sk_global,
            "sk_local": sk_local,
        }