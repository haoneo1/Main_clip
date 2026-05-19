import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model_pretrain import PretrainModel
from src.dataset_pretrain import PhotoOnlyJEPADataset
from experiments.options import opts


def build_train_loader():
    dataset_transforms = PhotoOnlyJEPADataset.data_transform(opts)
    train_dataset = PhotoOnlyJEPADataset(
        opts,
        transform_view1=dataset_transforms,
        transform_view2=dataset_transforms,
        mode="train",
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
    return train_dataset, train_loader


def resolve_resume_ckpt(save_dir):
    if getattr(opts, "pretrain_resume_ckpt", ""):
        ckpt = opts.pretrain_resume_ckpt
        if os.path.exists(ckpt):
            print(f"[JEPA PRETRAIN] resuming from explicit ckpt: {ckpt}")
            return ckpt
        print(f"[JEPA PRETRAIN] explicit ckpt not found: {ckpt}")

    ckpt_path = os.path.join(save_dir, "last.ckpt")
    if os.path.exists(ckpt_path):
        print(f"[JEPA PRETRAIN] resuming training from {ckpt_path}")
        return ckpt_path
    return None


def main():
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(opts.seed, workers=True)

    opts.train_stage = "pretrain"
    if not opts.exp_name.endswith("_pretrain"):
        opts.exp_name = f"{opts.exp_name}_pretrain"

    opts.save_root = "/home/mig/Documents/SBIR_Data/saved_models"
    save_dir = os.path.join(opts.save_root, opts.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    train_dataset, train_loader = build_train_loader()

    print(f"[JEPA PRETRAIN] num photos: {len(train_dataset)}")
    print(f"[JEPA PRETRAIN] num categories: {len(train_dataset.all_categories)}")
    print(f"[JEPA PRETRAIN] first 10 categories: {train_dataset.all_categories[:10]}")

    logger = TensorBoardLogger("tb_logs", name=opts.exp_name)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss_epoch",
        dirpath=save_dir,
        filename="pretrain-{epoch:02d}-{train_loss_epoch:.4f}",
        mode="min",
        save_top_k=1,
        save_last=opts.pretrain_save_last_ckpt,
        save_weights_only=True,
    )

    ckpt_path = resolve_resume_ckpt(save_dir)

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

    model = PretrainModel(opts)
    print("[JEPA PRETRAIN] beginning training... good luck...")
    trainer.fit(model, train_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()