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


if __name__ == '__main__':
    # =========================
    # 0) Performance hint for Tensor Cores
    # =========================
    torch.set_float32_matmul_precision("high")

    # 你现在想重新开一段新的 fine-tune
    opts.max_epochs = 100

    # =========================
    # 1) Seed
    # =========================
    pl.seed_everything(opts.seed, workers=True)

    # =========================
    # 2) Stage / experiment name
    # =========================
    opts.train_stage = "triplet_finetune"
    opts.pretrain_ckpt = '/home/mig/Documents/SBIR_Data/saved_models/SSL_same_class_jepa_pretrain/last.ckpt'

    if not opts.exp_name.endswith("_finetune"):
        opts.exp_name = f"{opts.exp_name}_finetune"

    # =========================
    # 2.1) Save root / save dir
    # =========================
    opts.save_root = "/home/mig/Documents/SBIR_Data/saved_models"
    save_dir = os.path.join(opts.save_root, opts.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 3) Dataset / DataLoader
    # =========================
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(
        opts,
        dataset_transforms,
        mode='train',
        return_orig=False
    )

    val_dataset = Sketchy(
        opts,
        dataset_transforms,
        mode='val',
        used_cat=train_dataset.all_categories,
        return_orig=False
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

    print(f"[FINETUNE] train samples: {len(train_dataset)}")
    print(f"[FINETUNE] val samples: {len(val_dataset)}")
    print(f"[FINETUNE] seen classes: {len(train_dataset.all_categories)}")
    print(f"[FINETUNE] first 10 seen classes: {train_dataset.all_categories[:10]}")
    print(f"[FINETUNE] val classes: {len(val_dataset.all_categories)}")
    print(f"[FINETUNE] first 10 val classes: {val_dataset.all_categories[:10]}")

    # =========================
    # 4) Logger
    # =========================
    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    # =========================
    # 5) Checkpoint / EarlyStopping
    # =========================
    checkpoint_callback = ModelCheckpoint(
        monitor='mAP',
        dirpath=save_dir,
        filename='best-{epoch:02d}-{mAP:.4f}',
        mode='max',
        save_top_k=1,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor='mAP',
        mode='max',
        patience=100,
        min_delta=1e-4,
    )

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
        num_sanity_val_steps=0,
    )

    # =========================
    # 7) Build model
    # =========================
    opts.seen_class_names = train_dataset.all_categories
    model = FinetuneModel(opts)

    # =========================
    # 8) Load weights only
    # 优先级：
    #   A. 如果已有 fine-tune 的 last.ckpt -> 只加载模型权重
    #   B. 否则加载 JEPA pretrain 权重
    #   C. 否则从头开始
    # =========================
    finetune_ckpt = os.path.join(save_dir, 'last.ckpt')

    if os.path.exists(finetune_ckpt):
        print(f"[FINETUNE] loading finetune model weights from: {finetune_ckpt}")
        ckpt = torch.load(finetune_ckpt, map_location='cpu')
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[FINETUNE] missing keys:", missing)
        print("[FINETUNE] unexpected keys:", unexpected)

    else:
        pretrain_ckpt = opts.pretrain_ckpt

        if pretrain_ckpt is not None and os.path.exists(pretrain_ckpt):
            print(f"[FINETUNE] loading JEPA pretrain weights from: {pretrain_ckpt}")
            ckpt = torch.load(pretrain_ckpt, map_location='cpu')
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("[FINETUNE] missing keys:", missing)
            print("[FINETUNE] unexpected keys:", unexpected)

            model.init_sk_prompt_from_img_prompt()
            print("[FINETUNE] initialized sk_prompt from img_prompt")
        else:
            print("[FINETUNE] no valid checkpoint found, training from scratch.")

    # =========================
    # 9) Fit
    # 注意：这里不要再传 ckpt_path
    # =========================
    print('[FINETUNE] beginning a NEW fine-tuning stage from loaded weights...')
    trainer.fit(model, train_loader, val_loader)