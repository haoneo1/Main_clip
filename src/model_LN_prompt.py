import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision
import pytorch_lightning as pl

from src.clip import clip
from experiments.options import opts


def freeze_model(m):
    m.requires_grad_(False)


def freeze_all_but_bn(m):
    # NOTE: 原作者函数名叫 bn，但这里其实是 LayerNorm
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts

        # IMPORTANT:
        # clip.load 内部会使用 self.device；Lightning 会在后续把 module move 到 GPU。
        # 这里保持你原有写法不变（不改名字/结构）。
        self.clip, _ = clip.load("ViT-B/32", device=self.device)
        self.clip.apply(freeze_all_but_bn)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2
        )

        self.best_metric = -1e3

        # --- Lightning 2.x validation aggregation buffers (replaces validation_epoch_end) ---
        self._val_query_feats = []
        self._val_gallery_feats = []
        self._val_categories = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.clip.parameters(), "lr": self.opts.clip_LN_lr},
                {"params": [self.sk_prompt] + [self.img_prompt], "lr": self.opts.prompt_lr},
            ]
        )
        return optimizer

    def forward(self, data, dtype="image"):
        if dtype == "image":
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1)
            )
        else:
            feat = self.clip.encode_image(
                data, self.sk_prompt.expand(data.shape[0], -1, -1)
            )
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype="image")
        sk_feat = self.forward(sk_tensor, dtype="sketch")
        neg_feat = self.forward(neg_tensor, dtype="image")

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # --------- Lightning 2.x compatible validation aggregation (do not keep validation_epoch_end) ---------
    def on_validation_epoch_start(self):
        self._val_query_feats = []
        self._val_gallery_feats = []
        self._val_categories = []

    def validation_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype="image")
        sk_feat = self.forward(sk_tensor, dtype="sketch")
        neg_feat = self.forward(neg_tensor, dtype="image")

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)

        # 名字不改：val_loss
        # on_epoch=True 确保 ModelCheckpoint(monitor='val_loss') 监控到 epoch-level 指标
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # 缓存特征与类别，用于 epoch 末计算 mAP
        # 放到 CPU，避免验证集大时占用过多 GPU 显存
        self._val_query_feats.append(sk_feat.detach().cpu())
        self._val_gallery_feats.append(img_feat.detach().cpu())

        # category 可能是 list/tuple 或 tensor；统一扁平化保存
        if isinstance(category, (list, tuple)):
            self._val_categories.extend(list(category))
        else:
            try:
                self._val_categories.extend(category.detach().cpu().tolist())
            except Exception:
                self._val_categories.append(category)

        return loss

    def on_validation_epoch_end(self):
        Len = len(self._val_query_feats)
        if Len == 0:
            return

        query_feat_all = torch.cat(self._val_query_feats, dim=0)      # CPU
        gallery_feat_all = torch.cat(self._val_gallery_feats, dim=0)  # CPU
        all_category = np.array(self._val_categories)

        # mAP category-level SBIR Metrics
        gallery = gallery_feat_all
        ap = torch.zeros(len(query_feat_all), dtype=torch.float32)

        for idx, sk_feat in enumerate(query_feat_all):
            category = all_category[idx]
            distance = -1 * self.distance_fn(sk_feat.unsqueeze(0), gallery)  # CPU tensor [N]

            target = torch.zeros(len(gallery), dtype=torch.bool)
            target[np.where(all_category == category)] = True

            ap[idx] = retrieval_average_precision(distance, target)

        mAP = torch.mean(ap)

        # 名字不改：mAP
        self.log("mAP", mAP, on_step=False, on_epoch=True, prog_bar=True)

        # 如果你的 ModelCheckpoint filename 里用了 {top10:.2f} 且你“不改名字”，
        # 这里额外 log 一个 top10，令 top10 == mAP，避免保存 checkpoint 时缺字段报错
        self.log("top10", mAP, on_step=False, on_epoch=True, prog_bar=False)

        if self.global_step > 0:
            self.best_metric = self.best_metric if (self.best_metric > mAP.item()) else mAP.item()

        print("mAP: {}, Best mAP: {}".format(mAP.item(), self.best_metric))

        # 清理缓存
        self._val_query_feats.clear()
        self._val_gallery_feats.clear()
        self._val_categories.clear()
