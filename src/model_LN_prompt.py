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


def _to_key_list(category):
    """
    把 category 统一规整为 Python 的 str list（可比较 key）。
    支持: torch.Tensor / np.ndarray / list / tuple / 单个 python 标量 / 单个字符串
    """
    if isinstance(category, torch.Tensor):
        c = category.detach().cpu()
        if c.numel() == 1:
            return [str(c.item())]
        return [str(v) for v in c.view(-1).tolist()]

    if isinstance(category, np.ndarray):
        c = category
        if c.size == 1:
            return [str(c.item())]
        return [str(v) for v in c.reshape(-1).tolist()]

    if isinstance(category, (list, tuple)):
        out = []
        for x in category:
            if isinstance(x, torch.Tensor):
                xx = x.detach().cpu()
                out.append(str(xx.item()) if xx.numel() == 1 else str(xx.view(-1)[0].item()))
            elif isinstance(x, np.ndarray):
                out.append(str(x.item()) if x.size == 1 else str(x.reshape(-1)[0]))
            else:
                out.append(str(x))
        return out

    return [str(category)]


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts

        # IMPORTANT:
        # clip.load 内部会使用 self.device；Lightning 会在后续把 module move 到 GPU。
        self.clip, _ = clip.load("ViT-B/32", device=self.device)
        self.clip.apply(freeze_all_but_bn)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        # Triplet loss
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2
        )

        self.best_metric = -1e3

        # --- Lightning 2.x validation aggregation buffers ---
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

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # 缓存到 CPU
        self._val_query_feats.append(sk_feat.detach().cpu())
        self._val_gallery_feats.append(img_feat.detach().cpu())

        # 关键：把 category 规整为 str list
        cats = _to_key_list(category)
        self._val_categories.extend(cats)

        return loss

    def on_validation_epoch_end(self):
        if len(self._val_query_feats) == 0:
            return

        query_feat_all = torch.cat(self._val_query_feats, dim=0)      # [Q, D] CPU
        gallery_feat_all = torch.cat(self._val_gallery_feats, dim=0)  # [G, D] CPU

        # category 统一为 str array（避免 dtype=object 乱套）
        all_category = np.asarray(self._val_categories, dtype=str)

        # --- 诊断：NaN ---
        q_nan = torch.isnan(query_feat_all).any().item()
        g_nan = torch.isnan(gallery_feat_all).any().item()
        if q_nan or g_nan:
            print(f"[WARN] NaN detected in feats: query_nan={q_nan}, gallery_nan={g_nan}")

        # --- normalize ---
        query = F.normalize(query_feat_all, dim=1)
        gallery = F.normalize(gallery_feat_all, dim=1)

        # --- 诊断：每个 query 在 gallery 里的正样本数 ---
        pos_counts = []
        for i in range(len(all_category)):
            pos_counts.append(int(np.sum(all_category == all_category[i])))
        print(
            f"[VAL] pos_count min/mean/max = {min(pos_counts)}/{(sum(pos_counts)/len(pos_counts)):.2f}/{max(pos_counts)}"
        )

        # --- mAP ---
        ap = torch.zeros(len(query), dtype=torch.float32)

        for idx in range(len(query)):
            # cosine similarity score: larger = more similar
            scores = torch.matmul(gallery, query[idx])  # [G]

            # target 用字符串比较得到 numpy bool，再转 torch.bool
            target_np = (all_category == all_category[idx])
            target = torch.from_numpy(target_np).to(dtype=torch.bool)

            if target.sum().item() == 0:
                ap[idx] = 0.0
                continue

            ap[idx] = retrieval_average_precision(scores, target)

        mAP = ap.mean()

        self.log("mAP", mAP, on_step=False, on_epoch=True, prog_bar=True)
        self.log("top10", mAP, on_step=False, on_epoch=True, prog_bar=False)

        if self.global_step > 0:
            self.best_metric = max(self.best_metric, float(mAP.item()))

        print(f"[VAL] mAP: {mAP.item():.6f}, Best mAP: {self.best_metric:.6f}")

        self._val_query_feats.clear()
        self._val_gallery_feats.clear()
        self._val_categories.clear()
