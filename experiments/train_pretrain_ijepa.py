import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from experiments.options import opts
from src.dataset_pretrain_ijepa import PhotoOnlyIJEPADataset
from src.model_pretrain_ijepa import IJEPAPretrainModel


def build_train_loader():
    transforms = PhotoOnlyIJEPADataset.data_transform(opts)
    dataset = PhotoOnlyIJEPADataset(opts, transforms, mode="train")
    loader = DataLoader(
        dataset=dataset,
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=opts.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(opts.workers > 0),
    )
    return dataset, loader


def main():
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(opts.seed, workers=True)

    opts.train_stage = "pretrain"
    if not opts.exp_name.endswith("_ijepa_pretrain"):
        opts.exp_name = f"{opts.exp_name}_ijepa_pretrain"

    opts.save_root = "/home/mig/Documents/SBIR_Data/saved_models"
    save_dir = os.path.join(opts.save_root, opts.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    dataset, train_loader = build_train_loader()
    print(f"[I-JEPA PRETRAIN] photos: {len(dataset)}")
    print(f"[I-JEPA PRETRAIN] categories: {len(dataset.all_categories)}")
    print(f"[I-JEPA PRETRAIN] arch: {opts.ijepa_backbone_arch}")

    logger = TensorBoardLogger("tb_logs", name=opts.exp_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss_epoch",
        dirpath=save_dir,
        filename="ijepa-{epoch:02d}-{train_loss_epoch:.4f}",
        mode="min",
        save_top_k=1,
        save_last=opts.pretrain_save_last_ckpt,
        save_weights_only=True,
    )

    trainer = Trainer(
        accelerator=opts.pretrain_accelerator,
        devices=opts.pretrain_devices,
        min_epochs=1,
        max_epochs=opts.max_epochs,
        benchmark=False,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=opts.pretrain_log_every_n_steps,
    )

    model = IJEPAPretrainModel(opts)
    print("[I-JEPA PRETRAIN] begin training...")
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
