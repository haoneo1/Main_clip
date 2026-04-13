import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from src.model_finetune import FinetuneModel
from src.dataset_finetune import Sketchy
from experiments.options import opts


if __name__ == '__main__':
    # =========================
    # 0) Performance hint for Tensor Cores
    # =========================
    torch.set_float32_matmul_precision("high")

    # 早停会在 mAP 不再提升时结束；max_epochs 仅用 CLI 上限（不再强制 ≥80）

    # =========================
    # 1) Seed
    # =========================
    pl.seed_everything(opts.seed, workers=True)

    # =========================
    # 2) Stage / experiment name / pretrain ckpt
    # =========================
    opts.train_stage = "triplet_finetune"

    opts.save_root = "/home/mig/Documents/SBIR_Data/saved_models"
    exp_base = str(opts.exp_name).removesuffix("_finetune")
    if not (getattr(opts, "pretrain_ckpt", None) or "").strip():
        opts.pretrain_ckpt = os.path.join(
            opts.save_root, f"{exp_base}_pretrain", "last.ckpt"
        )
    print(f"[FINETUNE] will load pretrain from (if not resuming finetune): {opts.pretrain_ckpt}")
    # 从 JEPA 重新 fine-tune、且目录里已有 last.ckpt 时：加 --finetune_resume false
    # opts.finetune_resume = False

    if not opts.exp_name.endswith("_finetune"):
        opts.exp_name = f"{opts.exp_name}_finetune"

    # =========================
    # 2.1) Save dir
    # =========================
    save_dir = os.path.join(opts.save_root, opts.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # =========================
    # 3) Dataset / DataLoader
    # =========================
    train_tf = Sketchy.data_transform(opts, train=True)
    val_tf = Sketchy.data_transform(opts, train=False)

    train_dataset = Sketchy(
        opts,
        train_tf,
        mode='train',
        return_orig=False
    )

    val_dataset = Sketchy(
        opts,
        val_tf,
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

    es_patience = int(getattr(opts, "finetune_early_stop_patience", 15))
    early_stop = EarlyStopping(
        monitor='mAP',
        mode='max',
        patience=es_patience,
        min_delta=1e-4,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # =========================
    # 6) Trainer
    # =========================
    use_fp32 = getattr(opts, "finetune_full_precision", True)
    prec = "32-true" if use_fp32 else getattr(opts, "precision", "16-mixed")
    grad_clip = float(getattr(opts, "finetune_gradient_clip", 1.0))
    use_cos = getattr(opts, "finetune_use_cosine_lr", True) and not getattr(
        opts, "use_finetune_lr_plateau", False
    )
    print(
        f"[FINETUNE] precision={prec}, gradient_clip_val={grad_clip if grad_clip > 0 else 0} "
        f"(finetune_clip_LN_lr={getattr(opts, 'finetune_clip_LN_lr', opts.clip_LN_lr)}, "
        f"finetune_prompt_lr={getattr(opts, 'finetune_prompt_lr', opts.prompt_lr)}, cosine={use_cos})"
    )

    trainer_kwargs = dict(
        accelerator="gpu",
        devices=1,
        precision=prec,
        min_epochs=1,
        max_epochs=opts.max_epochs,
        benchmark=False,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_callback, early_stop, lr_monitor],
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )
    if grad_clip > 0:
        trainer_kwargs["gradient_clip_val"] = grad_clip
    trainer = Trainer(**trainer_kwargs)

    # =========================
    # 7) Build model
    # =========================
    opts.seen_class_names = train_dataset.all_categories
    model = FinetuneModel(opts)

    # =========================
    # 8) Load weights only
    # 优先级：
    #   A. finetune_resume=True 且 save_dir/last.ckpt 存在 -> 续训 fine-tune
    #   B. 否则：finetune_resume=False 时显式忽略 last.ckpt，改从 pretrain_ckpt 加载
    #   C. 无预训练路径则从头
    # 无需再手动删除 last.ckpt：用 --finetune_resume false 即可从 JEPA 重新 fine-tune
    # =========================
    finetune_ckpt = os.path.join(save_dir, 'last.ckpt')
    finetune_resume = getattr(opts, "finetune_resume", True)
    loaded_finetune = False

    if finetune_resume and os.path.exists(finetune_ckpt):
        print(f"[FINETUNE] loading finetune model weights from: {finetune_ckpt}")
        ckpt = torch.load(finetune_ckpt, map_location='cpu')
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("[FINETUNE] missing keys:", missing)
        print("[FINETUNE] unexpected keys:", unexpected)
        loaded_finetune = True
    elif os.path.exists(finetune_ckpt) and not finetune_resume:
        print(
            f"[FINETUNE] finetune_resume=False: skipping {finetune_ckpt}, "
            "loading from pretrain_ckpt instead."
        )

    if not loaded_finetune:
        pretrain_ckpt = opts.pretrain_ckpt

        if pretrain_ckpt is not None and os.path.exists(pretrain_ckpt):
            print(f"[FINETUNE] loading JEPA pretrain weights from: {pretrain_ckpt}")
            ckpt = torch.load(pretrain_ckpt, map_location='cpu')
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("[FINETUNE] missing keys:", missing)
            print("[FINETUNE] unexpected keys:", unexpected)

            if getattr(opts, "finetune_init_sk_from_img", False):
                model.init_sk_prompt_from_img_prompt()
                print("[FINETUNE] initialized sk_prompt from img_prompt")
            else:
                print(
                    "[FINETUNE] keeping sk_prompt from pretrain "
                    "(set --finetune_init_sk_from_img true to copy from img_prompt)"
                )
        else:
            print("[FINETUNE] no valid checkpoint found, training from scratch.")

    # =========================
    # 9) Fit
    # 注意：这里不要再传 ckpt_path
    # =========================
    print('[FINETUNE] beginning a NEW fine-tuning stage from loaded weights...')
    trainer.fit(model, train_loader, val_loader)
    if checkpoint_callback.best_model_path:
        print(f"[FINETUNE] best checkpoint (by val mAP): {checkpoint_callback.best_model_path}")