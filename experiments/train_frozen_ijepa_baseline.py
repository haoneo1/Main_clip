import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from experiments.options import opts
from src.dataset_finetune import Sketchy
from src.model_frozen_ijepa import FrozenIJEPAFinetuneModel


def build_dataloaders():
    dataset_transforms = Sketchy.data_transform(opts)
    train_dataset = Sketchy(
        opts,
        dataset_transforms,
        mode="train",
        return_orig=False,
    )
    val_dataset = Sketchy(
        opts,
        dataset_transforms,
        mode="val",
        used_cat=train_dataset.all_categories,
        return_orig=False,
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
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(opts.workers > 0),
    )
    return train_dataset, val_dataset, train_loader, val_loader


def main():
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(opts.seed, workers=True)

    opts.train_stage = "frozen_ijepa_finetune"
    if not opts.exp_name.endswith("_frozen_ijepa"):
        opts.exp_name = f"{opts.exp_name}_frozen_ijepa"

    opts.save_root = "/home/mig/Documents/SBIR_Data/saved_models"
    save_dir = os.path.join(opts.save_root, opts.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()

    print(f"[Frozen-IJEPA] train samples: {len(train_dataset)}")
    print(f"[Frozen-IJEPA] val samples: {len(val_dataset)}")
    print(f"[Frozen-IJEPA] seen classes: {len(train_dataset.all_categories)}")
    print(f"[Frozen-IJEPA] frozen_backbone_arch: {opts.frozen_backbone_arch}")
    print(f"[Frozen-IJEPA] frozen_backbone_ckpt: {opts.frozen_backbone_ckpt}")
    print(f"[Frozen-IJEPA] freeze_backbone: {opts.freeze_backbone}")
    print(f"[Frozen-IJEPA] adapter_hidden_dim: {opts.adapter_hidden_dim}")
    print(f"[Frozen-IJEPA] proj_dim: {opts.proj_dim}")
    print(f"[Frozen-IJEPA] frozen_lr: {opts.frozen_lr}")

    logger = TensorBoardLogger("tb_logs", name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="mAP",
        dirpath=save_dir,
        filename="best-{epoch:02d}-{mAP:.4f}",
        mode="max",
        save_top_k=1,
        save_last=opts.save_last_ckpt,
        save_weights_only=True,
    )

    early_stop = EarlyStopping(
        monitor="mAP",
        mode="max",
        patience=opts.early_stop_patience,
        min_delta=1e-4,
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        min_epochs=1,
        max_epochs=opts.max_epochs,
        benchmark=False,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    model = FrozenIJEPAFinetuneModel(opts)
    print("[Frozen-IJEPA] begin training...")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
