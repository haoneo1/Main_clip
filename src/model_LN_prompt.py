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


def freeze_all_but_ln(m):
    """
    只保留 LayerNorm 可训练，其余层冻结。
    """
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


def _to_key_list(category):
    """
    把 category 统一规整为 Python 的 str list（可比较 key）。
    支持:
    - torch.Tensor
    - np.ndarray
    - list / tuple
    - 单个 python 标量
    - 单个字符串
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

        # ---------------------------
        # CLIP
        # ---------------------------
        # 这里先在 CPU 上加载，Lightning 后续会自动把整个模型搬到 GPU
        self.clip, _ = clip.load("ViT-B/32", device="cpu")
        self.clip.apply(freeze_all_but_ln)
        self.clip.train()

        # ---------------------------
        # Prompt Engineering
        # ---------------------------
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        # ---------------------------
        # Triplet loss
        # ---------------------------
        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn,
            margin=0.2
        )

        self.best_metric = self.opts.best_metric_init

        # ---------------------------
        # feature dim
        # ---------------------------
        if hasattr(self.clip, "visual") and hasattr(self.clip.visual, "output_dim"):
            feat_dim = int(self.clip.visual.output_dim)
        else:
            feat_dim = 512

        self.feat_dim = feat_dim

        # =========================================================
        # CLIP text classification loss config
        # =========================================================
        self.use_clip_cls = self.opts.use_clip_cls
        self.lambda_cls = self.opts.lambda_cls
        self.cls_tau = self.opts.cls_tau
        self.use_clip_logit_scale = self.opts.use_clip_logit_scale

        # 训练集 seen classes 名字列表
        self.seen_class_names = [str(x) for x in list(getattr(self.opts, "seen_class_names", []))]

        # 多模板 prompt
        self.cls_templates = list(
            getattr(
                self.opts,
                "cls_templates",
                [
                    "a photo of a {}",
                    "a photo of the {}",
                ],
            )
        )

        # category -> label index
        self.class_to_idx = {
            str(name): idx for idx, name in enumerate(self.seen_class_names)
        }

        # 用 buffer 保存文本分类器原型 [Ns, D]
        self.register_buffer(
            "text_classifier",
            torch.empty(0, feat_dim, dtype=torch.float32),
            persistent=False,
        )

        # classification loss 只把 text encoder 当固定语义锚点
        if self.use_clip_cls:
            self.freeze_text_tower()

        # ---------------------------
        # validation aggregation buffers
        # ---------------------------
        self._val_query_feats = []
        self._val_gallery_feats = []
        self._val_categories = []

    # =========================================================
    # freeze helpers
    # =========================================================
    def freeze_text_tower(self):
        """
        冻结 CLIP text tower，避免 classification loss 反向更新文本编码器。
        """
        if hasattr(self.clip, "transformer"):
            for p in self.clip.transformer.parameters():
                p.requires_grad_(False)

        if hasattr(self.clip, "token_embedding"):
            if hasattr(self.clip.token_embedding, "weight") and self.clip.token_embedding.weight is not None:
                self.clip.token_embedding.weight.requires_grad_(False)

        if hasattr(self.clip, "positional_embedding"):
            if isinstance(self.clip.positional_embedding, torch.Tensor):
                self.clip.positional_embedding.requires_grad_(False)

        if hasattr(self.clip, "ln_final"):
            for p in self.clip.ln_final.parameters():
                p.requires_grad_(False)

        if hasattr(self.clip, "text_projection"):
            if isinstance(self.clip.text_projection, torch.Tensor):
                self.clip.text_projection.requires_grad_(False)

    # =========================================================
    # text classifier helpers
    # =========================================================
    def build_text_classifier(self):
        """
        用 CLIP text encoder 构建 seen classes 的文本原型。
        每个类别用多个 prompt template 编码后取均值。
        """
        if not self.use_clip_cls:
            return

        if len(self.seen_class_names) == 0:
            raise ValueError(
                "use_clip_cls=True, but seen_class_names is empty. "
                "Please provide opts.seen_class_names."
            )

        text_features = []

        was_training = self.clip.training
        self.clip.eval()

        with torch.no_grad():
            for class_name in self.seen_class_names:
                cname = str(class_name).replace("_", " ")
                prompts = [tmpl.format(cname) for tmpl in self.cls_templates]

                tokens = clip.tokenize(prompts).to(self.device)
                txt_feat = self.clip.encode_text(tokens)   # [T, D]
                txt_feat = F.normalize(txt_feat, dim=-1)

                class_feat = txt_feat.mean(dim=0)          # 多模板平均
                class_feat = F.normalize(class_feat, dim=0)

                text_features.append(class_feat)

        text_features = torch.stack(text_features, dim=0).to(self.device)  # [Ns, D]
        self.text_classifier = text_features

        if was_training:
            self.clip.train()

        print(f"[Model] Built text_classifier with shape = {tuple(self.text_classifier.shape)}")

    def _get_cls_logit_scale(self):
        """
        classification logits 的尺度：
        - 若 use_clip_logit_scale=True，则使用 CLIP 自带的 logit_scale.exp()
        - 否则使用 1 / tau
        """
        if self.use_clip_logit_scale and hasattr(self.clip, "logit_scale"):
            return self.clip.logit_scale.exp().clamp(max=100.0)
        else:
            return torch.tensor(1.0 / self.cls_tau, device=self.device)

    def _category_to_label_tensor(self, category):
        """
        将 batch 中的 category 转成 [B] 的 long tensor labels。
        """
        keys = _to_key_list(category)
        labels = []

        for k in keys:
            if k not in self.class_to_idx:
                raise KeyError(
                    f"Category '{k}' not found in seen_class_names. "
                    f"Please check your dataset category format and opts.seen_class_names."
                )
            labels.append(self.class_to_idx[k])

        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def _clip_text_cls_loss(self, feat, labels):
        """
        feat: [B, D]
        labels: [B]
        """
        if self.text_classifier.numel() == 0:
            raise RuntimeError(
                "text_classifier is empty. Please call build_text_classifier() first."
            )

        feat = F.normalize(feat, dim=-1)
        text_w = F.normalize(self.text_classifier, dim=-1)  # [Ns, D]

        scale = self._get_cls_logit_scale()
        logits = scale * torch.matmul(feat, text_w.t())     # [B, Ns]

        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        return loss, logits, acc

    # =========================================================
    # prompt helpers
    # =========================================================
    def init_sk_prompt_from_img_prompt(self):
        """
        可手动调用：
            model.init_sk_prompt_from_img_prompt()
        让 sketch prompt 从 image prompt 初始化，通常更稳。
        """
        with torch.no_grad():
            self.sk_prompt.copy_(self.img_prompt)

    # =========================================================
    # encoders
    # =========================================================
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
        else:
            return self.encode_sketch_branch(data)

    # =========================================================
    # Lightning hooks
    # =========================================================
    def on_fit_start(self):
        """
        Lightning 已经把模型放到正确 device 上后，构建 text classifier。
        """
        if self.use_clip_cls:
            self.build_text_classifier()

    # =========================================================
    # optimizer
    # =========================================================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.clip.parameters(), "lr": self.opts.clip_LN_lr},
                {"params": [self.sk_prompt, self.img_prompt], "lr": self.opts.prompt_lr},
            ]
        )
        return optimizer

    # =========================================================
    # training
    # =========================================================
    def training_step(self, batch, batch_idx):
        # sk_tensor, img_tensor, neg_tensor, category, filename
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]

        img_feat = self.forward(img_tensor, dtype="image")
        sk_feat = self.forward(sk_tensor, dtype="sketch")
        neg_feat = self.forward(neg_tensor, dtype="image")

        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        total_loss = triplet_loss

        self.log("train_triplet", triplet_loss, on_step=True, on_epoch=True, prog_bar=False)

        if self.use_clip_cls:
            labels = self._category_to_label_tensor(category)

            sk_cls_loss, _, sk_acc = self._clip_text_cls_loss(sk_feat, labels)
            img_cls_loss, _, img_acc = self._clip_text_cls_loss(img_feat, labels)

            cls_loss = sk_cls_loss + img_cls_loss
            total_loss = triplet_loss + self.lambda_cls * cls_loss

            self.log("train_cls_sk", sk_cls_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_cls_img", img_cls_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_cls_total", cls_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_acc_sk", sk_acc, on_step=True, on_epoch=True, prog_bar=False)
            self.log("train_acc_img", img_acc, on_step=True, on_epoch=True, prog_bar=False)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_sk_norm", sk_feat.norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_img_norm", img_feat.norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_neg_norm", neg_feat.norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    # =========================================================
    # validation
    # =========================================================
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

        if self.use_clip_cls and self.text_classifier.numel() > 0:
            try:
                labels = self._category_to_label_tensor(category)
                sk_cls_loss, _, sk_acc = self._clip_text_cls_loss(sk_feat, labels)
                img_cls_loss, _, img_acc = self._clip_text_cls_loss(img_feat, labels)

                self.log("val_cls_sk", sk_cls_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_cls_img", img_cls_loss, on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_acc_sk", sk_acc, on_step=False, on_epoch=True, prog_bar=False)
                self.log("val_acc_img", img_acc, on_step=False, on_epoch=True, prog_bar=False)
            except Exception as e:
                print(f"[WARN] val classification logging skipped: {e}")

        self._val_query_feats.append(sk_feat.detach().cpu())
        self._val_gallery_feats.append(img_feat.detach().cpu())

        cats = _to_key_list(category)
        self._val_categories.extend(cats)

        return loss

    def on_validation_epoch_end(self):
        if len(self._val_query_feats) == 0:
            return

        query_feat_all = torch.cat(self._val_query_feats, dim=0)      # [Q, D]
        gallery_feat_all = torch.cat(self._val_gallery_feats, dim=0)  # [G, D]

        all_category = np.asarray(self._val_categories, dtype=str)

        q_nan = torch.isnan(query_feat_all).any().item()
        g_nan = torch.isnan(gallery_feat_all).any().item()
        if q_nan or g_nan:
            print(f"[WARN] NaN detected in feats: query_nan={q_nan}, gallery_nan={g_nan}")

        query = F.normalize(query_feat_all, dim=1)
        gallery = F.normalize(gallery_feat_all, dim=1)

        pos_counts = []
        for i in range(len(all_category)):
            pos_counts.append(int(np.sum(all_category == all_category[i])))

        print(
            f"[VAL] pos_count min/mean/max = "
            f"{min(pos_counts)}/{(sum(pos_counts) / len(pos_counts)):.2f}/{max(pos_counts)}"
        )

        ap = torch.zeros(len(query), dtype=torch.float32)

        for idx in range(len(query)):
            scores = torch.matmul(gallery, query[idx])  # [G]
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