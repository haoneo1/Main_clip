import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps


# Sketchy 常用 unseen classes
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


class Sketchy(Dataset):
    """
    用于 ZS-SBIR 的 retrieval dataset
    兼容:
        1) Triplet loss
        2) CLIP text classification loss

    返回:
        sk_tensor, img_tensor, neg_tensor, category, filename

    其中:
        - sk_tensor: sketch query
        - img_tensor: positive photo
        - neg_tensor: negative photo
        - category: 类别名字符串，用于 class loss
        - filename: sketch 文件名，便于调试
    """

    def __init__(self, opts, transform=None, mode='train', used_cat=None, return_orig=False):
        super().__init__()

        assert mode in ['train', 'val', 'test'], f"Unsupported mode: {mode}"

        self.opts = opts
        self.transform = transform if transform is not None else self.data_transform(opts)
        self.mode = mode
        self.return_orig = return_orig

        self.sketch_root = os.path.join(self.opts.data_dir, 'sketch')
        self.photo_root = os.path.join(self.opts.data_dir, 'photo')

        # --------------------------------------------------
        # 1) 收集全部类别
        # --------------------------------------------------
        self.all_categories = sorted([
            c for c in os.listdir(self.sketch_root)
            if os.path.isdir(os.path.join(self.sketch_root, c))
            and c != '.ipynb_checkpoints'
        ])
        # --------------------------------------------------
        # 2) 类别划分
        # --------------------------------------------------
        if self.opts.data_split > 0:
            # 这是你自定义随机切分类别的模式
            cats = self.all_categories.copy()
            random.shuffle(cats)

            if used_cat is None:
                keep_n = max(1, int(len(cats) * self.opts.data_split))
                self.all_categories = sorted(cats[:keep_n])
            else:
                self.all_categories = sorted(list(set(cats) - set(used_cat)))
        else:
            # 标准 seen / unseen 划分
            if mode == 'train':
                self.all_categories = sorted(list(set(self.all_categories) - set(unseen_classes)))
            else:
                self.all_categories = sorted([c for c in unseen_classes if c in self.all_categories])

        if len(self.all_categories) == 0:
            raise RuntimeError(f"No categories left after split for mode={mode}")

        # --------------------------------------------------
        # 3) 建立 category -> image list / sketch list
        # --------------------------------------------------
        self.all_sketches_path = []
        self.all_photos_path = {}
        self.class_to_idx = {}

        valid_categories = []

        for idx, category in enumerate(self.all_categories):
            sketch_dir = os.path.join(self.sketch_root, category)
            photo_dir = os.path.join(self.photo_root, category)

            sketch_list = sorted(glob.glob(os.path.join(sketch_dir, '*.png')))
            photo_list = sorted(glob.glob(os.path.join(photo_dir, '*.jpg')))

            # 有些数据也可能是 png/jpeg，顺手兼容
            if len(photo_list) == 0:
                photo_list.extend(sorted(glob.glob(os.path.join(photo_dir, '*.png'))))
                photo_list.extend(sorted(glob.glob(os.path.join(photo_dir, '*.jpeg'))))

            if len(sketch_list) == 0:
                print(f"[WARN] No sketches found for class: {category}")
                continue

            if len(photo_list) == 0:
                print(f"[WARN] No photos found for class: {category}")
                continue

            self.all_sketches_path.extend(sketch_list)
            self.all_photos_path[category] = photo_list
            valid_categories.append(category)

        self.all_categories = sorted(valid_categories)
        self.class_to_idx = {c: i for i, c in enumerate(self.all_categories)}

        if len(self.all_categories) == 0:
            raise RuntimeError("No valid categories with both sketches and photos were found.")

        if len(self.all_sketches_path) == 0:
            raise RuntimeError("No sketch samples found.")

        # 训练时需要至少两个类别，才能采负样本
        if self.mode == 'train' and len(self.all_categories) < 2:
            raise RuntimeError("Triplet training requires at least 2 categories.")

        print(f"[Sketchy-{self.mode}] #classes = {len(self.all_categories)}, #sketches = {len(self.all_sketches_path)}")

    def __len__(self):
        return len(self.all_sketches_path)

    def _load_rgb_pad(self, path):
        img = Image.open(path).convert('RGB')
        img = ImageOps.pad(img, size=(self.opts.max_size, self.opts.max_size))
        return img

    def _sample_negative_class(self, pos_category):
        neg_classes = [c for c in self.all_categories if c != pos_category]
        if len(neg_classes) == 0:
            raise RuntimeError(f"No negative classes available for category: {pos_category}")
        return random.choice(neg_classes)

    def __getitem__(self, index):
        # ----------------------------------------
        # 1) sketch 路径与类别
        # ----------------------------------------
        sk_path = self.all_sketches_path[index]
        category = os.path.basename(os.path.dirname(sk_path))
        filename = os.path.basename(sk_path)

        # ----------------------------------------
        # 2) 正样本 / 负样本照片
        # ----------------------------------------
        img_path = random.choice(self.all_photos_path[category])

        if self.mode == 'train':
            neg_category = self._sample_negative_class(category)
            neg_path = random.choice(self.all_photos_path[neg_category])
        else:
            # val/test 阶段也仍然返回 neg_tensor，保证和训练接口一致
            # 这样 model 里 batch[:4] 不用改
            neg_category = self._sample_negative_class(category) if len(self.all_categories) > 1 else category
            neg_path = random.choice(self.all_photos_path[neg_category])

        # ----------------------------------------
        # 3) 读图
        # ----------------------------------------
        sk_data = self._load_rgb_pad(sk_path)
        img_data = self._load_rgb_pad(img_path)
        neg_data = self._load_rgb_pad(neg_path)

        # ----------------------------------------
        # 4) transform
        # ----------------------------------------
        sk_tensor = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)

        # category 保持字符串形式，最适合你现在的 class loss 写法
        if self.return_orig:
            return (
                sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data
            )
        else:
            return (
                sk_tensor, img_tensor, neg_tensor, category, filename
            )

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    from PIL import Image
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)

    dataset_train = Sketchy(
        opts,
        dataset_transforms,
        mode='train',
        return_orig=True
    )

    dataset_val = Sketchy(
        opts,
        dataset_transforms,
        mode='val',
        used_cat=dataset_train.all_categories,
        return_orig=True
    )

    os.makedirs('output', exist_ok=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        (
            sk_tensor, img_tensor, neg_tensor, category, filename,
            sk_data, img_data, neg_data
        ) = data

        canvas = Image.new('RGB', (opts.max_size * 3, opts.max_size))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]

        canvas.save(f'output/{idx:05d}_{category}_{filename}.jpg')
        idx += 1