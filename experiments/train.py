import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.model_LN_prompt import Model
from src.dataset_retrieval import (
    FGSBIRTrainDataset,
    FGSBIRQueryDataset,
    FGSBIRGalleryDataset,
)
from experiments.options import opts


if __name__ == "__main__":
    # =========================
    # 1) Seed
    # =========================
    seed = getattr(opts, "seed", 42)
    pl.seed_everything(seed, workers=True)

    # =========================
    # 2) Save dir
    # =========================
    save_dir = os.path.join(opts.save_root, opts.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 80)
    print("FG-SBIR Training")
    print(f"Dataset      : {opts.dataset}")
    print(f"Data dir     : {opts.data_dir}")
    print(f"Train SK dir : {opts.train_sk_dir}")
    print(f"Train PH dir : {opts.train_ph_dir}")
    print(f"Test  SK dir : {opts.test_sk_dir}")
    print(f"Test  PH dir : {opts.test_ph_dir}")
    print(f"Save dir     : {save_dir}")
    print("=" * 80)

    # =========================
    # 3) Dataset / DataLoader
    # =========================
    dataset_transforms = FGSBIRTrainDataset.data_transform(opts)

    train_dataset = FGSBIRTrainDataset(
        opts,
        transform=dataset_transforms,
        return_orig=False,
    )

    # 验证阶段必须拆成 query 和 gallery
    # 这里直接用 test split 做验证；如果你后面单独切 val，也只需要把 split 改掉
    val_query_dataset = FGSBIRQueryDataset(
        opts,
        split="test",
        transform=dataset_transforms,
        return_orig=False,
    )

    val_gallery_dataset = FGSBIRGalleryDataset(
        opts,
        split="test",
        transform=dataset_transforms,
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

    val_query_loader = DataLoader(
        dataset=val_query_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(opts.workers > 0),
    )

    val_gallery_loader = DataLoader(
        dataset=val_gallery_dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(opts.workers > 0),
    )

    print(f"[Data] train sketches = {len(train_dataset)}")
    print(f"[Data] val queries    = {len(val_query_dataset)}")
    print(f"[Data] val gallery    = {len(val_gallery_dataset)}")

    # =========================
    # 4) Logger
    # =========================
    logger = TensorBoardLogger("tb_logs", name=opts.exp_name)

    # =========================
    # 5) Checkpoint / Early Stop
    # =========================
    # FG-SBIR 更推荐监控 rank1 或 top10，而不是 val_loss
    checkpoint_callback = ModelCheckpoint(
        monitor="rank1",
        dirpath=save_dir,
        filename="best-{epoch:02d}-{rank1:.4f}-{top10:.4f}",
        mode="max",
        save_top_k=1,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor="rank1",
        mode="max",
        patience=20,
        min_delta=1e-4,
    )

    ckpt_path = os.path.join(save_dir, "last.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print(f"[Resume] resuming training from {ckpt_path}")

    # =========================
    # 6) Trainer
    # =========================
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
    )

    # =========================
    # 7) Model
    # =========================
    model = Model()

    # 可选：让 sketch prompt 从 image prompt 初始化，通常更稳
    if hasattr(model, "init_sk_prompt_from_img_prompt"):
        model.init_sk_prompt_from_img_prompt()

    print("beginning FG-SBIR training... good luck...")

    # 注意：
    # val_dataloaders 必须是一个 list:
    #   [val_query_loader, val_gallery_loader]
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=[val_query_loader, val_gallery_loader],
        ckpt_path=ckpt_path,
    )