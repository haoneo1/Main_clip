import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import retrieval_average_precision

from src.clip import clip
from experiments.options import opts


def freeze_model(m):
    m.requires_grad_(False)


def freeze_all_but_ln(m):
    """
    只保留 LayerNorm 可训练，其余参数冻结。
    """
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


def _to_key_list(x):
    """
    把 batch 里的 instance_id/category 统一转成 Python str list。
    支持:
        - torch.Tensor
        - np.ndarray
        - list / tuple
        - 单个字符串或标量
    """
    if isinstance(x, torch.Tensor):
        xx = x.detach().cpu()
        if xx.numel() == 1:
            return [str(xx.item())]
        return [str(v) for v in xx.view(-1).tolist()]

    if isinstance(x, np.ndarray):
        if x.size == 1:
            return [str(x.item())]
        return [str(v) for v in x.reshape(-1).tolist()]

    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            if isinstance(v, torch.Tensor):
                vv = v.detach().cpu()
                out.append(str(vv.item()) if vv.numel() == 1 else str(vv.view(-1)[0].item()))
            elif isinstance(v, np.ndarray):
                out.append(str(v.item()) if v.size == 1 else str(v.reshape(-1)[0]))
            else:
                out.append(str(v))
        return out

    return [str(x)]


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.opts = opts

        # --------------------------------------------------
        # CLIP
        # --------------------------------------------------
        # 先在 CPU 上加载，Lightning 会自动搬到 GPU
        self.clip, _ = clip.load("ViT-B/32", device="cpu")
        self.clip.apply(freeze_all_but_ln)
        self.clip.train()

        # --------------------------------------------------
        # Prompt
        # --------------------------------------------------
        self.sk_prompt = nn.Parameter(
            torch.randn(self.opts.n_prompts, self.opts.prompt_dim)
        )
        self.img_prompt = nn.Parameter(
            torch.randn(self.opts.n_prompts, self.opts.prompt_dim)
        )

        # --------------------------------------------------
        # Triplet loss
        # --------------------------------------------------
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn,
            margin=0.2,
        )

        self.best_metric = self.opts.best_metric_init

        # --------------------------------------------------
        # validation / test 缓存
        # --------------------------------------------------
        self._val_query_feats = []
        self._val_query_ids = []
        self._val_gallery_feats = []
        self._val_gallery_ids = []

        self._test_query_feats = []
        self._test_query_ids = []
        self._test_gallery_feats = []
        self._test_gallery_ids = []

    # ======================================================
    # prompt helper
    # ======================================================
    def init_sk_prompt_from_img_prompt(self):
        """
        可选手动调用：
            model.init_sk_prompt_from_img_prompt()
        让 sketch prompt 从 image prompt 初始化，通常更稳。
        """
        with torch.no_grad():
            self.sk_prompt.copy_(self.img_prompt)

    # ======================================================
    # encoder
    # ======================================================
    def encode_image_branch(self, data):
        return self.clip.encode_image(
            data,
            self.img_prompt.expand(data.shape[0], -1, -1)
        )

    def encode_sketch_branch(self, data):
        return self.clip.encode_image(
            data,
            self.sk_prompt.expand(data.shape[0], -1, -1)
        )

    def forward(self, data, dtype="image"):
        if dtype == "image":
            return self.encode_image_branch(data)
        elif dtype == "sketch":
            return self.encode_sketch_branch(data)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    # ======================================================
    # optimizer
    # ======================================================
    def configure_optimizers(self):
        clip_trainable_params = [p for p in self.clip.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(
            [
                {"params": clip_trainable_params, "lr": self.opts.clip_LN_lr},
                {"params": [self.sk_prompt, self.img_prompt], "lr": self.opts.prompt_lr},
            ]
        )
        return optimizer

    # ======================================================
    # training
    # 训练 batch:
    #   sk_tensor, pos_tensor, neg_tensor, instance_id, filename
    # ======================================================
    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, instance_id, filename = batch

        img_feat = self.forward(img_tensor, dtype="image")
        sk_feat = self.forward(sk_tensor, dtype="sketch")
        neg_feat = self.forward(neg_tensor, dtype="image")

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        total_loss = triplet_loss

        self.log("train_triplet", triplet_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        self.log(
            "train_sk_norm",
            sk_feat.norm(dim=1).mean().detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_img_norm",
            img_feat.norm(dim=1).mean().detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_neg_norm",
            neg_feat.norm(dim=1).mean().detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        return total_loss

    # ======================================================
    # shared eval utils
    # ======================================================
    def _reset_val_buffers(self):
        self._val_query_feats = []
        self._val_query_ids = []
        self._val_gallery_feats = []
        self._val_gallery_ids = []

    def _reset_test_buffers(self):
        self._test_query_feats = []
        self._test_query_ids = []
        self._test_gallery_feats = []
        self._test_gallery_ids = []

    def _compute_retrieval_metrics(self, query_feat_all, query_ids, gallery_feat_all, gallery_ids, prefix="val"):
        """
        query_feat_all: [Q, D]
        gallery_feat_all: [G, D]
        query_ids: list[str]
        gallery_ids: list[str]
        """
        if len(query_ids) == 0 or len(gallery_ids) == 0:
            print(f"[{prefix.upper()}] Empty query/gallery, skip metric computation.")
            return

        query = F.normalize(query_feat_all, dim=1)
        gallery = F.normalize(gallery_feat_all, dim=1)

        query_ids_np = np.asarray(query_ids, dtype=str)
        gallery_ids_np = np.asarray(gallery_ids, dtype=str)

        q_nan = torch.isnan(query).any().item()
        g_nan = torch.isnan(gallery).any().item()
        if q_nan or g_nan:
            print(f"[WARN][{prefix}] NaN detected in features: query_nan={q_nan}, gallery_nan={g_nan}")

        ap_list = []
        rank1_list = []
        top10_list = []

        num_gallery = gallery.shape[0]
        topk = min(10, num_gallery)

        for i in range(query.shape[0]):
            scores = torch.matmul(gallery, query[i])  # [G]

            target_np = (gallery_ids_np == query_ids_np[i])
            target = torch.from_numpy(target_np).to(dtype=torch.bool, device=scores.device)

            # 正常来说每个 query 在 gallery 里应该至少有 1 个正样本
            if target.sum().item() == 0:
                ap_list.append(torch.tensor(0.0))
                rank1_list.append(torch.tensor(0.0))
                top10_list.append(torch.tensor(0.0))
                continue

            ap = retrieval_average_precision(scores, target)
            ap_list.append(ap)

            ranked_idx = torch.argsort(scores, descending=True)

            rank1_hit = target[ranked_idx[0]].float()
            rank1_list.append(rank1_hit)

            top10_hit = target[ranked_idx[:topk]].any().float()
            top10_list.append(top10_hit)

        mAP = torch.stack(ap_list).mean()
        rank1 = torch.stack(rank1_list).mean()
        top10 = torch.stack(top10_list).mean()

        self.log(f"{prefix}_mAP", mAP, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_rank1", rank1, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}_top10", top10, on_step=False, on_epoch=True, prog_bar=True)

        # 为了兼容你原来的 checkpoint 监控字段
        if prefix == "val":
            self.log("mAP", mAP, on_step=False, on_epoch=True, prog_bar=False)
            self.log("rank1", rank1, on_step=False, on_epoch=True, prog_bar=False)
            self.log("top10", top10, on_step=False, on_epoch=True, prog_bar=False)

            if self.global_step > 0:
                self.best_metric = max(self.best_metric, float(top10.item()))

        print(
            f"[{prefix.upper()}] mAP={mAP.item():.6f}, "
            f"rank1={rank1.item():.6f}, top10={top10.item():.6f}"
        )

    # ======================================================
    # validation
    #
    # 需要两个 val dataloader:
    # dataloader_idx == 0 -> query loader
    # dataloader_idx == 1 -> gallery loader
    # ======================================================
    def on_validation_epoch_start(self):
        self._reset_val_buffers()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            # query loader:
            # sk_tensor, instance_id, filename
            sk_tensor, instance_id, filename = batch

            sk_feat = self.forward(sk_tensor, dtype="sketch")

            self._val_query_feats.append(sk_feat.detach().cpu())
            self._val_query_ids.extend(_to_key_list(instance_id))

        elif dataloader_idx == 1:
            # gallery loader:
            # img_tensor, instance_id, filename
            img_tensor, instance_id, filename = batch

            img_feat = self.forward(img_tensor, dtype="image")

            self._val_gallery_feats.append(img_feat.detach().cpu())
            self._val_gallery_ids.extend(_to_key_list(instance_id))

        else:
            raise ValueError(f"Unexpected dataloader_idx: {dataloader_idx}")

    def on_validation_epoch_end(self):
        if len(self._val_query_feats) == 0 or len(self._val_gallery_feats) == 0:
            print("[VAL] query/gallery buffers are empty, skip.")
            return

        query_feat_all = torch.cat(self._val_query_feats, dim=0)      # [Q, D]
        gallery_feat_all = torch.cat(self._val_gallery_feats, dim=0)  # [G, D]

        self._compute_retrieval_metrics(
            query_feat_all=query_feat_all,
            query_ids=self._val_query_ids,
            gallery_feat_all=gallery_feat_all,
            gallery_ids=self._val_gallery_ids,
            prefix="val",
        )

        self._reset_val_buffers()

    # ======================================================
    # test
    #
    # 同样需要两个 test dataloader:
    # dataloader_idx == 0 -> query loader
    # dataloader_idx == 1 -> gallery loader
    # ======================================================
    def on_test_epoch_start(self):
        self._reset_test_buffers()

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            sk_tensor, instance_id, filename = batch
            sk_feat = self.forward(sk_tensor, dtype="sketch")

            self._test_query_feats.append(sk_feat.detach().cpu())
            self._test_query_ids.extend(_to_key_list(instance_id))

        elif dataloader_idx == 1:
            img_tensor, instance_id, filename = batch
            img_feat = self.forward(img_tensor, dtype="image")

            self._test_gallery_feats.append(img_feat.detach().cpu())
            self._test_gallery_ids.extend(_to_key_list(instance_id))

        else:
            raise ValueError(f"Unexpected dataloader_idx: {dataloader_idx}")

    def on_test_epoch_end(self):
        if len(self._test_query_feats) == 0 or len(self._test_gallery_feats) == 0:
            print("[TEST] query/gallery buffers are empty, skip.")
            return

        query_feat_all = torch.cat(self._test_query_feats, dim=0)
        gallery_feat_all = torch.cat(self._test_gallery_feats, dim=0)

        self._compute_retrieval_metrics(
            query_feat_all=query_feat_all,
            query_ids=self._test_query_ids,
            gallery_feat_all=gallery_feat_all,
            gallery_ids=self._test_gallery_ids,
            prefix="test",
        )

        self._reset_test_buffers()
        