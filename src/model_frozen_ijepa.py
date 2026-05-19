import os
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision

from src.model_base import _to_key_list


IJEPA_ARCH_TO_TIMM = {
    # I-JEPA official model table
    "ijepa_vit_h14_224_in1k": "vit_huge_patch14_224",
    "ijepa_vit_h16_448_in1k": "vit_huge_patch16_448",
    "ijepa_vit_h14_224_in22k": "vit_huge_patch14_224",
    "ijepa_vit_g16_224_in22k": "vit_giant_patch16_224",
}


def _unwrap_state_dict(ckpt, preferred_encoder_key="auto"):
    if not isinstance(ckpt, dict):
        return ckpt

    if preferred_encoder_key and preferred_encoder_key != "auto":
        v = ckpt.get(preferred_encoder_key)
        if isinstance(v, dict):
            return v

    # Prefer target encoder first for JEPA usage.
    for key in (
        "target_encoder",
        "encoder",
        "state_dict",
        "model",
        "module",
        "model_state_dict",
    ):
        v = ckpt.get(key)
        if isinstance(v, dict):
            return v
    return ckpt


def _strip_prefixes(state_dict, prefixes):
    out = OrderedDict()
    for k, v in state_dict.items():
        new_key = k
        for p in prefixes:
            if new_key.startswith(p):
                new_key = new_key[len(p) :]
        out[new_key] = v
    return out


def _resize_pos_embed_2d(src_pos_embed, target_pos_embed):
    """
    Resize patch position embeddings with bicubic interpolation.
    Both tensors are expected as [1, N, C] (no cls token).
    """
    if src_pos_embed.ndim != 3 or target_pos_embed.ndim != 3:
        return None
    if src_pos_embed.shape[0] != 1 or target_pos_embed.shape[0] != 1:
        return None
    if src_pos_embed.shape[2] != target_pos_embed.shape[2]:
        return None

    src_n = src_pos_embed.shape[1]
    tgt_n = target_pos_embed.shape[1]
    src_hw = int(src_n**0.5)
    tgt_hw = int(tgt_n**0.5)
    if src_hw * src_hw != src_n or tgt_hw * tgt_hw != tgt_n:
        return None

    src = src_pos_embed.reshape(1, src_hw, src_hw, -1).permute(0, 3, 1, 2)
    resized = F.interpolate(src, size=(tgt_hw, tgt_hw), mode="bicubic", align_corners=False)
    resized = resized.permute(0, 2, 3, 1).reshape(1, tgt_n, -1)
    return resized


def _adapt_pos_embed_for_model(src_pos_embed, model_pos_embed):
    """
    Handle common JEPA<->timm position-embedding mismatches:
    - no cls token vs cls token
    - different patch-grid sizes (interpolate patch pos)
    """
    if src_pos_embed.shape == model_pos_embed.shape:
        return src_pos_embed

    model_has_cls = model_pos_embed.shape[1] > 0
    src_n = src_pos_embed.shape[1]
    model_n = model_pos_embed.shape[1]

    # Case A: checkpoint has no cls token, model has cls token
    if src_n == model_n - 1:
        cls_pos = model_pos_embed[:, :1, :]
        return torch.cat([cls_pos, src_pos_embed], dim=1)

    # Case B: checkpoint has cls token, model has no cls token
    if src_n == model_n + 1:
        return src_pos_embed[:, 1:, :]

    # Case C: both have cls token but patch-grid size differs
    if model_has_cls and src_n > 1 and model_n > 1:
        src_cls, src_patch = src_pos_embed[:, :1, :], src_pos_embed[:, 1:, :]
        tgt_cls, tgt_patch = model_pos_embed[:, :1, :], model_pos_embed[:, 1:, :]
        resized_patch = _resize_pos_embed_2d(src_patch, tgt_patch)
        if resized_patch is not None:
            return torch.cat([tgt_cls, resized_patch], dim=1)

    # Case D: no cls token on both sides and patch-grid differs
    resized = _resize_pos_embed_2d(src_pos_embed, model_pos_embed)
    if resized is not None:
        return resized

    return None


class FrozenIJEPAFinetuneModel(pl.LightningModule):
    """
    Frozen visual-backbone retrieval baseline.
    - Backbone is fully frozen (ViT-B/16 by default)
    - Trainable modules: sketch adapter, image adapter, projection heads
    - Losses: triplet + optional sigreg
    """

    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        self.backbone = self._build_backbone()
        self._maybe_load_backbone_ckpt()
        self._set_backbone_frozen(getattr(self.opts, "freeze_backbone", True))

        self.feat_dim = int(self.backbone.num_features)
        hidden = int(getattr(self.opts, "adapter_hidden_dim", 512))
        proj_dim = int(getattr(self.opts, "proj_dim", 512))

        self.sk_adapter = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.feat_dim),
        )
        self.img_adapter = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.feat_dim),
        )

        self.sk_proj = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, proj_dim),
        )
        self.img_proj = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, proj_dim),
        )

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = torch.nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=0.2
        )

        self.use_sigreg = bool(getattr(self.opts, "use_sigreg", False))
        self.lambda_sigreg = float(getattr(self.opts, "lambda_sigreg", 0.0))
        self.sigreg_gamma = float(getattr(self.opts, "sigreg_gamma", 1.0))
        self.sigreg_eps = float(getattr(self.opts, "sigreg_eps", 1e-4))
        self.best_metric = float(getattr(self.opts, "best_metric_init", -1e3))

        self._val_query_feats = []
        self._val_gallery_feats = []
        self._val_categories = []

    def _build_backbone(self):
        arch = getattr(self.opts, "frozen_backbone_arch", "ijepa_vit_h14_224_in1k").lower()
        model_name = IJEPA_ARCH_TO_TIMM.get(arch, arch)
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for frozen I-JEPA backbones. Install with: pip install timm"
            ) from exc

        try:
            backbone = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="")
        except Exception as exc:
            raise ValueError(
                f"Failed to create backbone '{model_name}' (from frozen_backbone_arch='{arch}'). "
                "Please check supported timm model names in your environment."
            ) from exc
        return backbone

    def _set_backbone_frozen(self, freeze_backbone):
        if not freeze_backbone:
            return
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def _maybe_load_backbone_ckpt(self):
        ckpt_path = getattr(self.opts, "frozen_backbone_ckpt", "")
        if not ckpt_path:
            print("[Frozen-IJEPA] No frozen_backbone_ckpt provided. Using randomly initialized timm backbone.")
            return
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"frozen_backbone_ckpt not found: {ckpt_path}")

        print(f"[Frozen-IJEPA] loading backbone checkpoint: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")
        state_dict = _unwrap_state_dict(
            raw,
            preferred_encoder_key=getattr(self.opts, "ijepa_encoder_key", "auto"),
        )
        if not isinstance(state_dict, dict):
            raise RuntimeError("Checkpoint does not contain a valid state dict.")

        state_dict = _strip_prefixes(
            state_dict,
            prefixes=(
                "module.",
                "backbone.",
                "model.",
                "encoder.",
                "target_encoder.",
                "context_encoder.",
                "backbone.",
                "student_encoder.",
            ),
        )

        # Keep only keys relevant to the target backbone and adapt known mismatches.
        model_state = self.backbone.state_dict()
        filtered = OrderedDict()
        skipped_mismatch = []
        for k, v in state_dict.items():
            if k not in model_state:
                continue
            tgt = model_state[k]
            if v.shape == tgt.shape:
                filtered[k] = v
                continue
            if k == "pos_embed":
                adapted = _adapt_pos_embed_for_model(v, tgt)
                if adapted is not None and adapted.shape == tgt.shape:
                    filtered[k] = adapted
                    print(
                        f"[Frozen-IJEPA] adapted pos_embed from {tuple(v.shape)} to {tuple(tgt.shape)}."
                    )
                    continue
            skipped_mismatch.append((k, tuple(v.shape), tuple(tgt.shape)))

        # Some I-JEPA checkpoints do not carry cls_token; keep model init token explicitly.
        if "cls_token" in model_state and "cls_token" not in filtered:
            filtered["cls_token"] = model_state["cls_token"]
            print("[Frozen-IJEPA] cls_token missing in ckpt; using model-initialized cls_token.")

        missing, unexpected = self.backbone.load_state_dict(filtered, strict=False)
        print(f"[Frozen-IJEPA] load_state_dict strict=False; missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("[Frozen-IJEPA] first missing keys:", missing[:10])
        if len(unexpected) > 0:
            print("[Frozen-IJEPA] first unexpected keys:", unexpected[:10])
        if len(skipped_mismatch) > 0:
            print("[Frozen-IJEPA] skipped mismatched keys (first 10):", skipped_mismatch[:10])
        if bool(getattr(self.opts, "strict_backbone_load", True)) and (
            len(missing) > 0 or len(unexpected) > 0 or len(skipped_mismatch) > 0
        ):
            raise RuntimeError(
                "strict_backbone_load=True and checkpoint is not an exact match. "
                "Use a matching architecture/checkpoint pair or set --strict_backbone_load False."
            )

    def _encode_backbone(self, x):
        feat = self.backbone.forward_features(x)
        # For timm ViT: forward_features usually returns tokens [B, N, C].
        if feat.ndim == 3:
            return feat[:, 0]
        return feat

    def _encode_image(self, x):
        if getattr(self.opts, "freeze_backbone", True):
            with torch.no_grad():
                z = self._encode_backbone(x)
        else:
            z = self._encode_backbone(x)
        z = self.img_adapter(z)
        z = self.img_proj(z)
        return F.normalize(z, dim=-1)

    def _encode_sketch(self, x):
        if getattr(self.opts, "freeze_backbone", True):
            with torch.no_grad():
                z = self._encode_backbone(x)
        else:
            z = self._encode_backbone(x)
        z = self.sk_adapter(z)
        z = self.sk_proj(z)
        return F.normalize(z, dim=-1)

    def _sigreg_loss(self, feat):
        feat_centered = feat - feat.mean(dim=0, keepdim=True)
        std = torch.sqrt(feat_centered.var(dim=0, unbiased=False) + self.sigreg_eps)
        var_loss = F.relu(self.sigreg_gamma - std).mean()

        if feat_centered.shape[0] <= 1:
            cov_loss = torch.zeros((), device=feat.device, dtype=feat.dtype)
        else:
            cov = (feat_centered.t() @ feat_centered) / (feat_centered.shape[0] - 1)
            diag = torch.eye(cov.shape[0], device=cov.device, dtype=torch.bool)
            cov_loss = cov.masked_select(~diag).pow(2).mean()
        return var_loss + cov_loss

    def _encode_triplet_batch(self, batch):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        sk_feat = self._encode_sketch(sk_tensor)
        img_feat = self._encode_image(img_tensor)
        neg_feat = self._encode_image(neg_tensor)
        return sk_feat, img_feat, neg_feat, category

    def configure_optimizers(self):
        params = [
            {"params": self.sk_adapter.parameters(), "lr": self.opts.frozen_lr},
            {"params": self.img_adapter.parameters(), "lr": self.opts.frozen_lr},
            {"params": self.sk_proj.parameters(), "lr": self.opts.frozen_lr},
            {"params": self.img_proj.parameters(), "lr": self.opts.frozen_lr},
        ]
        if not getattr(self.opts, "freeze_backbone", True):
            params.append({"params": self.backbone.parameters(), "lr": self.opts.clip_LN_lr})

        optimizer = torch.optim.AdamW(
            params,
            weight_decay=float(getattr(self.opts, "frozen_weight_decay", 1e-4)),
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        sk_feat, img_feat, neg_feat, _ = self._encode_triplet_batch(batch)
        triplet_loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        total_loss = triplet_loss
        self.log("train_triplet", triplet_loss, on_step=False, on_epoch=True, prog_bar=False)

        if self.use_sigreg and self.lambda_sigreg > 0.0:
            sk_sigreg = self._sigreg_loss(sk_feat)
            img_sigreg = self._sigreg_loss(img_feat)
            sigreg_loss = 0.5 * (sk_sigreg + img_sigreg)
            total_loss = total_loss + self.lambda_sigreg * sigreg_loss
            self.log("train_sigreg", sigreg_loss, on_step=False, on_epoch=True, prog_bar=False)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def on_validation_epoch_start(self):
        self._val_query_feats = []
        self._val_gallery_feats = []
        self._val_categories = []

    def validation_step(self, batch, batch_idx):
        sk_feat, img_feat, neg_feat, category = self._encode_triplet_batch(batch)
        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_query_feats.append(sk_feat.detach().cpu())
        self._val_gallery_feats.append(img_feat.detach().cpu())
        self._val_categories.extend(_to_key_list(category))
        return loss

    def on_validation_epoch_end(self):
        if len(self._val_query_feats) == 0:
            return

        query_feat_all = torch.cat(self._val_query_feats, dim=0)
        gallery_feat_all = torch.cat(self._val_gallery_feats, dim=0)
        all_category = np.asarray(self._val_categories, dtype=str)

        ap = torch.zeros(len(query_feat_all), dtype=torch.float32)
        for idx in range(len(query_feat_all)):
            scores = torch.matmul(gallery_feat_all, query_feat_all[idx])
            target_np = all_category == all_category[idx]
            target = torch.from_numpy(target_np).to(dtype=torch.bool)
            if target.sum().item() == 0:
                ap[idx] = 0.0
                continue
            ap[idx] = retrieval_average_precision(scores, target)

        mAP = ap.mean()
        self.log("mAP", mAP, on_step=False, on_epoch=True, prog_bar=True)
        if self.global_step > 0:
            self.best_metric = max(self.best_metric, float(mAP.item()))
        print(f"[VAL] mAP: {mAP.item():.6f}, Best mAP: {self.best_metric:.6f}")
