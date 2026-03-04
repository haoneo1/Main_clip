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
    JEPA pretraining dataset with both photo and sketch.
    Each sample returns:
        img_view1, img_view2, sk_view1, sk_view2, category
    """

    def __init__(self, opts, transform_img, transform_sk=None, mode='train', return_orig=False):
        self.opts = opts
        self.transform_img = transform_img
        self.transform_sk = transform_sk if transform_sk is not None else transform_img
        self.return_orig = return_orig

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

        # length by photos for enough iterations; category sampled by index
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
            transforms.Resize((opts.max_size, opts.max_size)),
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