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

# unseen_classes = ["airplane"]

class Sketchy(torch.utils.data.Dataset):

    def __init__(self, opts, transform, mode='train', used_cat=None, return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        categories = self._load_categories()
        self.all_categories = self._select_categories(categories, mode, used_cat)
        self.all_sketches_path, self.all_photos_path = self._build_index(self.all_categories)

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)

        neg_classes = [c for c in self.all_categories if c != category]
        pos_img_path = np.random.choice(self.all_photos_path[category])
        neg_img_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data = self._load_padded_image(filepath)
        img_data = self._load_padded_image(pos_img_path)
        neg_data = self._load_padded_image(neg_img_path)

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    def _load_categories(self):
        categories = os.listdir(os.path.join(self.opts.data_dir, "sketch"))
        categories = [c for c in categories if c != ".ipynb_checkpoints"]
        return sorted(categories)

    def _select_categories(self, categories, mode, used_cat):
        if self.opts.data_split > 0:
            categories = categories.copy()
            np.random.shuffle(categories)
            if used_cat is None:
                split_idx = int(len(categories) * self.opts.data_split)
                return categories[:split_idx]
            return sorted(list(set(categories) - set(used_cat)))

        if mode == "train":
            return sorted(list(set(categories) - set(unseen_classes)))
        return unseen_classes

    def _build_index(self, categories):
        sketch_paths = []
        photo_paths = {}
        for category in categories:
            sketch_paths.extend(
                glob.glob(os.path.join(self.opts.data_dir, "sketch", category, "*.png"))
            )
            photo_paths[category] = glob.glob(
                os.path.join(self.opts.data_dir, "photo", category, "*.jpg")
            )
        return sketch_paths, photo_paths

    def _load_padded_image(self, path):
        return ImageOps.pad(
            Image.open(path).convert("RGB"),
            size=(self.opts.max_size, self.opts.max_size),
        )

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return dataset_transforms


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(opts, dataset_transforms, mode='val', used_cat=dataset_train.all_categories, return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
