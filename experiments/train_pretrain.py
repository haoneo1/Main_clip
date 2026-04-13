import os
import sys
from pathlib import Path

# Allow `python experiments/train_pretrain.py` (repo root must be on PYTHONPATH).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_pretrain import PretrainModel
from src.dataset_pretrain import MultiModalJEPADataset
from experiments.options import opts


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(opts.seed, workers=True)

    opts.train_stage = "pretrain"

    if not opts.exp_name.endswith("_pretrain"):
        opts.exp_name = f"{opts.exp_name}_pretrain"

    opts.save_root = "/home/mig/Documents/SBIR_Data/saved_models"
    save_dir = os.path.join(opts.save_root, opts.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    dataset_transforms = MultiModalJEPADataset.data_transform(opts)

    train_dataset = MultiModalJEPADataset(
        opts,
        transform_img=dataset_transforms,
        transform_sk=dataset_transforms,
        mode='train',
        return_orig=False
    )

    total_views = opts.num_global_sk + opts.num_local_sk + opts.num_global_ph + opts.num_local_ph
    if opts.batch_size * total_views > 192:
        old_bs = opts.batch_size
        opts.batch_size = max(8, 192 // max(total_views, 1))
        print(
            f"[JEPA PRETRAIN] auto-adjust batch_size {old_bs} -> {opts.batch_size} "
            f"(total_views={total_views}) to reduce OOM risk."
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(opts.workers > 0),
    )

    print(f"[JEPA PRETRAIN] num samples: {len(train_dataset)}")
    print(f"[JEPA PRETRAIN] num categories: {len(train_dataset.all_categories)}")
    print(f"[JEPA PRETRAIN] first 10 categories: {train_dataset.all_categories[:10]}")
    print(f"[JEPA PRETRAIN] save_dir: {save_dir}")

    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss_epoch',
        dirpath=save_dir,
        filename='pretrain-{epoch:02d}-{train_loss_epoch:.4f}',
        mode='min',
        save_top_k=1,
        save_last=True,
    )

    ckpt_path = os.path.join(save_dir, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print(f"[JEPA PRETRAIN] resuming training from {ckpt_path}")

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=opts.precision,
        min_epochs=1,
        max_epochs=opts.max_epochs,
        benchmark=False,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=100,
    )

    model = PretrainModel(opts)

    print('[JEPA PRETRAIN] beginning multimodal LeJEPA-like pretraining...')
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)