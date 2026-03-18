import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.model_LN_prompt import Model
from src.dataset_retrieval import Sketchy
from experiments.options import opts


if __name__ == '__main__':
    # =========================
    # 1) Seed
    # =========================
    seed = getattr(opts, "seed", 42)
    pl.seed_everything(seed, workers=True)

    # 如需更强确定性可打开：
    # import torch
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    # =========================
    # 2) Dataset / DataLoader
    # =========================
    dataset_transforms = Sketchy.data_transform(opts)

    train_dataset = Sketchy(opts, dataset_transforms, mode='train', return_orig=False)
    val_dataset = Sketchy(
        opts,
        dataset_transforms,
        mode='val',
        used_cat=train_dataset.all_categories,
        return_orig=False
    )

    # 给 text classification loss 用
    opts.seen_class_names = [str(x) for x in train_dataset.all_categories]
    print("seen classes:", opts.seen_class_names)

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

    # =========================
    # 3) Logger
    # =========================
    logger = TensorBoardLogger('tb_logs', name=opts.exp_name)

    # =========================
    # 4) Checkpoint / Early Stop
    # =========================
    checkpoint_callback = ModelCheckpoint(
        monitor='mAP',
        dirpath=f'saved_models/{opts.exp_name}',
        filename="best-{epoch:02d}-{mAP:.4f}",
        mode='max',
        save_top_k=1,
        save_last=True,
    )

    early_stop = EarlyStopping(
        monitor='mAP',
        mode='max',
        patience=20,
        min_delta=1e-4,
    )

    ckpt_path = os.path.join('saved_models', opts.exp_name, 'last.ckpt')
    if not os.path.exists(ckpt_path):
        ckpt_path = None
    else:
        print(f"resuming training from {ckpt_path}")

    # =========================
    # 5) Trainer
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
    # 6) Model
    # =========================
    model = Model()

    print('beginning training...good luck...')
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)