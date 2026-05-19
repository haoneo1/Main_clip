import os
import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

unseen_classes = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]


class PhotoOnlyJEPADataset(torch.utils.data.Dataset):
    """
    只用于 photo-only JEPA 预训练：
    - 只读取 photo
    - 只使用 base classes（默认 mode='train'）
    - 每次返回同一张 photo 的两个增强视图
    """

    def __init__(self, opts, transform_view1, transform_view2=None, mode='train', return_orig=False):
        self.opts = opts
        self.transform_view1 = transform_view1
        self.transform_view2 = transform_view2 if transform_view2 is not None else transform_view1
        self.return_orig = return_orig

        self.all_categories = self._load_categories()
        self.all_categories = self._select_categories(self.all_categories, mode)

        self.all_photos_path = self._collect_photo_paths(self.all_categories)

    def __len__(self):
        return len(self.all_photos_path)

    def __getitem__(self, index):
        img_path = self.all_photos_path[index]
        category = img_path.split(os.path.sep)[-2]
        filename = os.path.basename(img_path)

        img_data = ImageOps.pad(
            Image.open(img_path).convert('RGB'),
            size=(self.opts.max_size, self.opts.max_size)
        )

        # 同一张图，两次随机增强
        img_view1 = self.transform_view1(img_data)
        img_view2 = self.transform_view2(img_data)

        if self.return_orig:
            return img_view1, img_view2, category, filename, img_data
        else:
            return img_view1, img_view2, category, filename

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
        """
        给 JEPA 预训练用的数据增强。
        注意：必须带随机性，否则同一张图两次输出完全一样，JEPA 就没意义了。
        建议先用“温和增强”，不要太激进。
        """
        transform = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.RandomResizedCrop(
                size=opts.max_size,
                scale=(0.8, 1.0),
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


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    transform = PhotoOnlyJEPADataset.data_transform(opts)
    dataset = PhotoOnlyJEPADataset(opts, transform, mode='train', return_orig=True)

    print("num photos:", len(dataset))
    print("num categories:", len(dataset.all_categories))
    print("categories:", dataset.all_categories[:10])

    for data in tqdm.tqdm(dataset):
        img_view1, img_view2, category, filename, img_data = data
        # 这里只是测试 dataset 是否能正常跑通
        break