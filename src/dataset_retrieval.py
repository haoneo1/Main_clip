import os
import glob
import random
from typing import List, Dict

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def is_image_file(path: str) -> bool:
    return path.lower().endswith(IMG_EXTENSIONS)


def list_image_files(folder: str) -> List[str]:
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)


def sketch_name_to_instance_id(sketch_path: str) -> str:
    """
    sketch 文件名示例:
        2kn308a2ca10_1.png
        2kn308a2ca10_2.png
        123-din-s_3.png

    对应 photo 文件名:
        2kn308a2ca10.png
        123-din-s.png

    规则:
        去掉扩展名后，再去掉最后一个 "_数字"
    """
    stem = os.path.splitext(os.path.basename(sketch_path))[0]

    if "_" in stem:
        prefix, suffix = stem.rsplit("_", 1)
        if suffix.isdigit():
            return prefix

    # 若不满足 *_数字 格式，则退化为原 stem
    return stem


class FGSBIRBaseDataset(Dataset):
    def __init__(self, opts, transform=None, return_orig=False):
        super().__init__()
        self.opts = opts
        self.return_orig = return_orig
        self.transform = transform if transform is not None else self.data_transform(opts)

    def _load_rgb_pad(self, path: str):
        img = Image.open(path).convert("RGB")
        img = ImageOps.pad(img, size=(self.opts.max_size, self.opts.max_size))
        return img

    @staticmethod
    def data_transform(opts):
        return transforms.Compose([
            transforms.Resize((opts.max_size, opts.max_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class FGSBIRTrainDataset(FGSBIRBaseDataset):
    """
    训练集:
        返回:
            sk_tensor, pos_tensor, neg_tensor, instance_id, filename

        含义:
            - sk_tensor: sketch query
            - pos_tensor: 对应正样本 photo
            - neg_tensor: 不同 instance 的负样本 photo
            - instance_id: 实例 ID，如 220-blanc
            - filename: sketch 文件名
    """
    def __init__(self, opts, transform=None, return_orig=False):
        super().__init__(opts, transform=transform, return_orig=return_orig)

        self.sketch_dir = self.opts.train_sk_dir
        self.photo_dir = self.opts.train_ph_dir

        if not os.path.isdir(self.sketch_dir):
            raise FileNotFoundError(f"train sketch dir not found: {self.sketch_dir}")
        if not os.path.isdir(self.photo_dir):
            raise FileNotFoundError(f"train photo dir not found: {self.photo_dir}")

        # 1) photo: instance_id -> photo_path
        self.photo_map = self._build_photo_map(self.photo_dir)

        # 2) sketch-photo 配对
        self.samples = self._build_paired_samples(self.sketch_dir, self.photo_map)

        # 3) instance -> photo_path
        self.instance_to_photo: Dict[str, str] = {}
        for s in self.samples:
            self.instance_to_photo[s["instance_id"]] = s["photo_path"]

        self.instance_ids = sorted(list(self.instance_to_photo.keys()))

        if len(self.instance_ids) < 2:
            raise RuntimeError("Triplet training requires at least 2 different instances.")

        print(
            f"[FGSBIR-Train] dataset={self.opts.dataset}, "
            f"#instances={len(self.instance_ids)}, #sketches={len(self.samples)}"
        )

    def _build_photo_map(self, photo_dir: str) -> Dict[str, str]:
        photo_files = list_image_files(photo_dir)
        if len(photo_files) == 0:
            raise RuntimeError(f"No photo files found in: {photo_dir}")

        photo_map = {}
        for p in photo_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            if stem in photo_map:
                raise RuntimeError(f"Duplicate photo stem detected: {stem}")
            photo_map[stem] = p
        return photo_map

    def _build_paired_samples(self, sketch_dir: str, photo_map: Dict[str, str]):
        sketch_files = list_image_files(sketch_dir)
        if len(sketch_files) == 0:
            raise RuntimeError(f"No sketch files found in: {sketch_dir}")

        samples = []
        missing = 0

        for sk_path in sketch_files:
            instance_id = sketch_name_to_instance_id(sk_path)

            if instance_id not in photo_map:
                print(f"[WARN] No matched photo for sketch: {sk_path}")
                missing += 1
                continue

            samples.append({
                "sketch_path": sk_path,
                "photo_path": photo_map[instance_id],
                "instance_id": instance_id,
                "filename": os.path.basename(sk_path),
            })

        if len(samples) == 0:
            raise RuntimeError(
                f"No valid sketch-photo pairs found in {sketch_dir}. "
                f"Please check filename matching."
            )

        if missing > 0:
            print(f"[WARN] {missing} sketches have no matched photo and were skipped.")

        return samples

    def _sample_negative_instance(self, pos_instance_id: str) -> str:
        neg_ids = [iid for iid in self.instance_ids if iid != pos_instance_id]
        if len(neg_ids) == 0:
            raise RuntimeError(f"No negative instance available for {pos_instance_id}")
        return random.choice(neg_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        sk_path = sample["sketch_path"]
        pos_path = sample["photo_path"]
        instance_id = sample["instance_id"]
        filename = sample["filename"]

        neg_instance_id = self._sample_negative_instance(instance_id)
        neg_path = self.instance_to_photo[neg_instance_id]

        sk_img = self._load_rgb_pad(sk_path)
        pos_img = self._load_rgb_pad(pos_path)
        neg_img = self._load_rgb_pad(neg_path)

        sk_tensor = self.transform(sk_img)
        pos_tensor = self.transform(pos_img)
        neg_tensor = self.transform(neg_img)

        if self.return_orig:
            return (
                sk_tensor, pos_tensor, neg_tensor, instance_id, filename,
                sk_img, pos_img, neg_img
            )
        else:
            return sk_tensor, pos_tensor, neg_tensor, instance_id, filename


class FGSBIRQueryDataset(FGSBIRBaseDataset):
    """
    验证 / 测试 query sketch:
        返回:
            sk_tensor, instance_id, filename

    用于:
        query sketch -> 到整个 gallery photo 库里检索
    """
    def __init__(self, opts, split="test", transform=None, return_orig=False):
        super().__init__(opts, transform=transform, return_orig=return_orig)

        assert split in ["train", "test"], f"Unsupported split: {split}"

        self.split = split
        self.sketch_dir = self.opts.train_sk_dir if split == "train" else self.opts.test_sk_dir
        self.photo_dir = self.opts.train_ph_dir if split == "train" else self.opts.test_ph_dir

        if not os.path.isdir(self.sketch_dir):
            raise FileNotFoundError(f"sketch dir not found: {self.sketch_dir}")
        if not os.path.isdir(self.photo_dir):
            raise FileNotFoundError(f"photo dir not found: {self.photo_dir}")

        self.photo_map = self._build_photo_map(self.photo_dir)
        self.samples = self._build_query_samples(self.sketch_dir, self.photo_map)

        print(
            f"[FGSBIR-Query-{split}] dataset={self.opts.dataset}, "
            f"#queries={len(self.samples)}"
        )

    def _build_photo_map(self, photo_dir: str) -> Dict[str, str]:
        photo_files = list_image_files(photo_dir)
        photo_map = {}
        for p in photo_files:
            stem = os.path.splitext(os.path.basename(p))[0]
            photo_map[stem] = p
        return photo_map

    def _build_query_samples(self, sketch_dir: str, photo_map: Dict[str, str]):
        sketch_files = list_image_files(sketch_dir)

        samples = []
        for sk_path in sketch_files:
            instance_id = sketch_name_to_instance_id(sk_path)
            if instance_id not in photo_map:
                print(f"[WARN] No matched photo for query sketch: {sk_path}")
                continue

            samples.append({
                "sketch_path": sk_path,
                "instance_id": instance_id,
                "gt_photo_path": photo_map[instance_id],
                "filename": os.path.basename(sk_path),
            })

        if len(samples) == 0:
            raise RuntimeError(f"No valid query samples found in {sketch_dir}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        sk_path = sample["sketch_path"]
        instance_id = sample["instance_id"]
        filename = sample["filename"]
        gt_photo_path = sample["gt_photo_path"]

        sk_img = self._load_rgb_pad(sk_path)
        sk_tensor = self.transform(sk_img)

        if self.return_orig:
            gt_img = self._load_rgb_pad(gt_photo_path)
            return sk_tensor, instance_id, filename, sk_img, gt_img
        else:
            return sk_tensor, instance_id, filename


class FGSBIRGalleryDataset(FGSBIRBaseDataset):
    """
    验证 / 测试 gallery photo:
        返回:
            img_tensor, instance_id, filename
    """
    def __init__(self, opts, split="test", transform=None, return_orig=False):
        super().__init__(opts, transform=transform, return_orig=return_orig)

        assert split in ["train", "test"], f"Unsupported split: {split}"

        self.split = split
        self.photo_dir = self.opts.train_ph_dir if split == "train" else self.opts.test_ph_dir

        if not os.path.isdir(self.photo_dir):
            raise FileNotFoundError(f"photo dir not found: {self.photo_dir}")

        self.samples = self._build_gallery_samples(self.photo_dir)

        print(
            f"[FGSBIR-Gallery-{split}] dataset={self.opts.dataset}, "
            f"#gallery={len(self.samples)}"
        )

    def _build_gallery_samples(self, photo_dir: str):
        photo_files = list_image_files(photo_dir)
        if len(photo_files) == 0:
            raise RuntimeError(f"No gallery photos found in {photo_dir}")

        samples = []
        for p in photo_files:
            instance_id = os.path.splitext(os.path.basename(p))[0]
            samples.append({
                "photo_path": p,
                "instance_id": instance_id,
                "filename": os.path.basename(p),
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        photo_path = sample["photo_path"]
        instance_id = sample["instance_id"]
        filename = sample["filename"]

        img = self._load_rgb_pad(photo_path)
        img_tensor = self.transform(img)

        if self.return_orig:
            return img_tensor, instance_id, filename, img
        else:
            return img_tensor, instance_id, filename


if __name__ == "__main__":
    import tqdm
    from PIL import Image
    from experiments.options import opts

    train_set = FGSBIRTrainDataset(opts, return_orig=True)
    test_query_set = FGSBIRQueryDataset(opts, split="test", return_orig=True)
    test_gallery_set = FGSBIRGalleryDataset(opts, split="test", return_orig=True)

    os.makedirs("output_debug_train", exist_ok=True)
    os.makedirs("output_debug_query", exist_ok=True)
    os.makedirs("output_debug_gallery", exist_ok=True)

    # 1) 检查训练 triplet
    for idx in tqdm.tqdm(range(min(20, len(train_set)))):
        (
            sk_tensor, pos_tensor, neg_tensor, instance_id, filename,
            sk_img, pos_img, neg_img
        ) = train_set[idx]

        canvas = Image.new("RGB", (opts.max_size * 3, opts.max_size))
        canvas.paste(sk_img, (0, 0))
        canvas.paste(pos_img, (opts.max_size, 0))
        canvas.paste(neg_img, (opts.max_size * 2, 0))
        canvas.save(f"output_debug_train/{idx:03d}_{instance_id}_{filename}.jpg")

    # 2) 检查 query
    for idx in tqdm.tqdm(range(min(20, len(test_query_set)))):
        sk_tensor, instance_id, filename, sk_img, gt_img = test_query_set[idx]

        canvas = Image.new("RGB", (opts.max_size * 2, opts.max_size))
        canvas.paste(sk_img, (0, 0))
        canvas.paste(gt_img, (opts.max_size, 0))
        canvas.save(f"output_debug_query/{idx:03d}_{instance_id}_{filename}.jpg")

    # 3) 检查 gallery
    for idx in tqdm.tqdm(range(min(20, len(test_gallery_set)))):
        img_tensor, instance_id, filename, img = test_gallery_set[idx]
        img.save(f"output_debug_gallery/{idx:03d}_{instance_id}_{filename}")