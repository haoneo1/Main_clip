import os
import glob
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
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

    def __init__(self, opts, transform_img=None, transform_sk=None, mode='train', return_orig=False):
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

    @staticmethod
    def _random_resized_crop(img, out_size, scale, ratio=(0.9, 1.1)):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)
        crop = TF.crop(img, i, j, h, w)
        crop = TF.resize(crop, [out_size, out_size], interpolation=transforms.InterpolationMode.BICUBIC)
        return crop

    @staticmethod
    def _fg_and_edge_ratio(img):
        arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        fg_ratio = float((arr < 0.95).mean())

        gx = np.abs(arr[:, 1:] - arr[:, :-1])
        gy = np.abs(arr[1:, :] - arr[:-1, :])
        edge_map = np.zeros_like(arr, dtype=np.float32)
        edge_map[:, 1:] += gx
        edge_map[1:, :] += gy
        edge_ratio = float((edge_map > 0.08).mean())
        return fg_ratio, edge_ratio

    def _sample_sketch_local(self, sk):
        for _ in range(self.opts.max_local_retry):
            crop = self._random_resized_crop(
                sk,
                out_size=self.opts.local_size_sk,
                scale=(self.opts.sk_local_scale_min, self.opts.sk_local_scale_max),
            )
            fg_ratio, edge_ratio = self._fg_and_edge_ratio(crop)
            if fg_ratio >= self.opts.min_fg_ratio_sk_local and edge_ratio >= self.opts.min_edge_ratio_sk_local:
                return crop
        # fallback: keep larger semantic region
        return self._random_resized_crop(
            sk,
            out_size=self.opts.local_size_sk,
            scale=(max(0.6, self.opts.sk_local_scale_min), self.opts.sk_global_scale_max),
        )

    def _normalize_tensor(self, img):
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return tensor

    def _sample_global_views(self, image, is_sketch):
        views = []
        num = self.opts.num_global_sk if is_sketch else self.opts.num_global_ph
        out_size = self.opts.global_size
        if is_sketch:
            scale = (self.opts.sk_global_scale_min, self.opts.sk_global_scale_max)
        else:
            scale = (self.opts.ph_global_scale_min, self.opts.ph_global_scale_max)
        for _ in range(num):
            crop = self._random_resized_crop(image, out_size=out_size, scale=scale)
            views.append(self._normalize_tensor(crop))
        return torch.stack(views, dim=0)

    def _sample_local_views(self, image, is_sketch):
        views = []
        if is_sketch:
            num = self.opts.num_local_sk
            for _ in range(num):
                crop = self._sample_sketch_local(image)
                views.append(self._normalize_tensor(crop))
        else:
            num = self.opts.num_local_ph
            scale = (self.opts.ph_local_scale_min, self.opts.ph_local_scale_max)
            for _ in range(num):
                crop = self._random_resized_crop(image, out_size=self.opts.local_size_ph, scale=scale)
                if random.random() < 0.5:
                    crop = ImageOps.grayscale(crop).convert("RGB")
                views.append(self._normalize_tensor(crop))
        return torch.stack(views, dim=0)

    def __getitem__(self, index):
        category = self.all_categories[index % len(self.all_categories)]

        img_path = random.choice(self.photo_dict[category])
        sk_path = random.choice(self.sketch_dict[category])

        img = self._load_and_pad(img_path)
        sk = self._load_and_pad(sk_path)

        img_global_views = self._sample_global_views(img, is_sketch=False)
        img_local_views = self._sample_local_views(img, is_sketch=False)
        sk_global_views = self._sample_global_views(sk, is_sketch=True)
        sk_local_views = self._sample_local_views(sk, is_sketch=True)

        if self.return_orig:
            return img_global_views, img_local_views, sk_global_views, sk_local_views, category, img, sk
        else:
            return img_global_views, img_local_views, sk_global_views, sk_local_views, category

    @staticmethod
    def data_transform(opts):
        # Kept for compatibility with existing training script.
        # Multi-view sampling is handled in __getitem__ with per-modality configs.
        return None