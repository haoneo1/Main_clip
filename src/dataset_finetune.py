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


class Sketchy(torch.utils.data.Dataset):
    """
    返回：
      cross-domain triplet:   (sk, img_pos, img_neg)
      sketch intra triplet:   (sk_pos, sk_neg)   (anchor 仍然是 sk)
      photo  intra triplet:   (img_pos2, img_neg2) (anchor 仍然是 img_pos)
    最终 tuple：
      (sk, img_pos, img_neg, sk_pos, sk_neg, img_pos2, img_neg2, category, filename)
    """

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):
        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        self.all_categories = os.listdir(os.path.join(self.opts.data_dir, 'sketch'))
        if '.ipynb_checkpoints' in self.all_categories:
            self.all_categories.remove('.ipynb_checkpoints')

        if self.opts.data_split > 0:
            np.random.shuffle(self.all_categories)
            if used_cat is None:
                self.all_categories = self.all_categories[:int(len(self.all_categories) * self.opts.data_split)]
            else:
                self.all_categories = list(set(self.all_categories) - set(used_cat))
        else:
            if mode == 'train':
                self.all_categories = list(set(self.all_categories) - set(unseen_classes))
            else:
                self.all_categories = unseen_classes

        self.all_sketches_path = []
        self.all_sketches_by_cat = {}
        self.all_photos_path = {}

        for category in self.all_categories:
            sk_list = glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png'))
            ph_list = glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpg'))

            # 如果某类照片为空会直接导致采样崩溃，提前过滤/报错更好
            if len(sk_list) == 0:
                continue
            if len(ph_list) == 0:
                raise RuntimeError(f"[Sketchy] photo list is empty for category: {category}")

            self.all_sketches_path.extend(sk_list)
            self.all_sketches_by_cat[category] = sk_list
            self.all_photos_path[category] = ph_list

        # 保险：如果被 continue 过滤掉某些类别导致 all_sketches_path 为空
        if len(self.all_sketches_path) == 0:
            raise RuntimeError("[Sketchy] all_sketches_path is empty. Check data_dir structure.")

    def __len__(self):
        return len(self.all_sketches_path)

    def _load_rgb_pad(self, path: str):
        return ImageOps.pad(
            Image.open(path).convert('RGB'),
            size=(self.opts.max_size, self.opts.max_size)
        )

    def __getitem__(self, index):
        sk_path = self.all_sketches_path[index]
        category = sk_path.split(os.path.sep)[-2]
        filename = os.path.basename(sk_path)

        # negative category
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)
        neg_cat = np.random.choice(neg_classes)

        # ===== cross-domain =====
        img_pos_path = np.random.choice(self.all_photos_path[category])
        img_neg_path = np.random.choice(self.all_photos_path[neg_cat])

        # ===== sketch intra-domain =====
        sk_pos_candidates = self.all_sketches_by_cat[category]
        if len(sk_pos_candidates) > 1:
            # 尽量避免取到同一张
            sk_pos_path = np.random.choice([p for p in sk_pos_candidates if p != sk_path])
        else:
            sk_pos_path = sk_path  # 兜底

        sk_neg_path = np.random.choice(self.all_sketches_by_cat[neg_cat])

        # ===== photo intra-domain =====
        img_pos_candidates = self.all_photos_path[category]
        if len(img_pos_candidates) > 1:
            # 尽量避免取到 img_pos_path
            img_pos2_path = np.random.choice([p for p in img_pos_candidates if p != img_pos_path])
        else:
            img_pos2_path = img_pos_path

        img_neg2_path = np.random.choice(self.all_photos_path[neg_cat])

        # load
        sk_data = self._load_rgb_pad(sk_path)
        img_pos_data = self._load_rgb_pad(img_pos_path)
        img_neg_data = self._load_rgb_pad(img_neg_path)

        sk_pos_data = self._load_rgb_pad(sk_pos_path)
        sk_neg_data = self._load_rgb_pad(sk_neg_path)

        img_pos2_data = self._load_rgb_pad(img_pos2_path)
        img_neg2_data = self._load_rgb_pad(img_neg2_path)

        # transform
        sk = self.transform(sk_data)
        img_pos = self.transform(img_pos_data)
        img_neg = self.transform(img_neg_data)

        sk_pos = self.transform(sk_pos_data)
        sk_neg = self.transform(sk_neg_data)

        img_pos2 = self.transform(img_pos2_data)
        img_neg2 = self.transform(img_neg2_data)

        if self.return_orig:
            return (
                sk, img_pos, img_neg, sk_pos, sk_neg, img_pos2, img_neg2, category, filename,
                sk_data, img_pos_data, img_neg_data, sk_pos_data, sk_neg_data, img_pos2_data, img_neg2_data
            )
        else:
            return (sk, img_pos, img_neg, sk_pos, sk_neg, img_pos2, img_neg2, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=False)

    for _ in tqdm.tqdm(dataset_val):
        pass

    print("[OK] dataset sampling works.")