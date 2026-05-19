import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.model_finetune import FinetuneModel
from src.dataset_finetune import Sketchy
from experiments.options import opts


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


def maybe_load_finetune_weights(model, save_dir):
    finetune_ckpt = os.path.join(save_dir, "last.ckpt")
    if not os.path.exists(finetune_ckpt):
        print("[FINETUNE] no finetune checkpoint found, training from scratch (without pretrain ckpt).")
        return False

    print(f"[FINETUNE] loading finetune model weights from: {finetune_ckpt}")
    ckpt = torch.load(finetune_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[FINETUNE] missing keys:", missing)
    print("[FINETUNE] unexpected keys:", unexpected)
    return True


def maybe_load_init_weights(model):
    init_ckpt = getattr(opts, "finetune_init_ckpt", "")
    if not init_ckpt:
        return False
    if not os.path.exists(init_ckpt):
        print(f"[FINETUNE] finetune_init_ckpt not found: {init_ckpt}")
        return False

    print(f"[FINETUNE] loading init model weights from pretrain ckpt: {init_ckpt}")
    ckpt = torch.load(init_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("[FINETUNE] init missing keys:", missing)
    print("[FINETUNE] init unexpected keys:", unexpected)
    return True


def main():
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(opts.seed, workers=True)

    opts.train_stage = "triplet_finetune"
    opts.pretrain_ckpt = None

    if not opts.exp_name.endswith("_finetune"):
        opts.exp_name = f"{opts.exp_name}_finetune"

    opts.save_root = "/home/mig/Documents/SBIR_Data/saved_models"
    save_dir = os.path.join(opts.save_root, opts.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()

    print(f"[FINETUNE] train samples: {len(train_dataset)}")
    print(f"[FINETUNE] val samples: {len(val_dataset)}")
    print(f"[FINETUNE] seen classes: {len(train_dataset.all_categories)}")
    print(f"[FINETUNE] first 10 seen classes: {train_dataset.all_categories[:10]}")
    print(f"[FINETUNE] val classes: {len(val_dataset.all_categories)}")
    print(f"[FINETUNE] first 10 val classes: {val_dataset.all_categories[:10]}")

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

    opts.seen_class_names = train_dataset.all_categories
    model = FinetuneModel(opts)
    resumed = maybe_load_finetune_weights(model, save_dir)
    if not resumed:
        maybe_load_init_weights(model)
    print("[FINETUNE] beginning a NEW fine-tuning stage from loaded weights...")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()