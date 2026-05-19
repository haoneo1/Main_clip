import copy
import math
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_2d_sincos_pos_embed(grid_h, grid_w, dim, device):
    if dim % 4 != 0:
        raise ValueError("positional embedding dimension must be divisible by 4")
    half_dim = dim // 2
    quarter = dim // 4

    grid_y = torch.arange(grid_h, dtype=torch.float32, device=device)
    grid_x = torch.arange(grid_w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(grid_y, grid_x, indexing="ij")
    yy = yy.reshape(-1, 1)
    xx = xx.reshape(-1, 1)

    omega = torch.arange(quarter, dtype=torch.float32, device=device) / float(quarter)
    omega = 1.0 / (10000**omega)
    out_y = yy * omega[None, :]
    out_x = xx * omega[None, :]

    pos = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=1)
    return pos


class IJEPAPredictor(nn.Module):
    def __init__(self, dim, nhead, depth, mlp_ratio):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.out_norm(x)


class IJEPAPretrainModel(pl.LightningModule):
    """
    I-JEPA style pretraining:
    - context encoder (trainable)
    - target encoder (EMA, no grad)
    - block-based target masks
    - latent token prediction
    """

    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for I-JEPA pretraining. Install with: pip install timm") from exc

        arch = getattr(self.opts, "ijepa_backbone_arch", "vit_base_patch16_224")
        self.context_encoder = timm.create_model(arch, pretrained=False, num_classes=0, global_pool="")
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.target_encoder.eval()

        self.embed_dim = int(self.context_encoder.num_features)
        self.grid_h, self.grid_w = self.context_encoder.patch_embed.grid_size
        self.num_patches = self.grid_h * self.grid_w

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.predictor = IJEPAPredictor(
            dim=self.embed_dim,
            nhead=int(getattr(self.opts, "ijepa_pred_heads", 8)),
            depth=int(getattr(self.opts, "ijepa_pred_depth", 2)),
            mlp_ratio=float(getattr(self.opts, "ijepa_pred_mlp_ratio", 4.0)),
        )
        self.momentum = float(getattr(self.opts, "ijepa_momentum", 0.996))

    def _encode_patch_tokens(self, model, x):
        feat = model.forward_features(x)
        if feat.ndim != 3:
            raise RuntimeError("Expected token output [B, N(+1), C] from ViT forward_features.")
        if feat.shape[1] == self.num_patches + 1:
            feat = feat[:, 1:, :]
        elif feat.shape[1] != self.num_patches:
            raise RuntimeError(f"Unexpected token length: got {feat.shape[1]}, expected {self.num_patches} or {self.num_patches+1}.")
        return feat

    def _sample_target_mask(self):
        n_targets = int(getattr(self.opts, "ijepa_num_targets", 4))
        min_scale = float(getattr(self.opts, "ijepa_target_scale_min", 0.15))
        max_scale = float(getattr(self.opts, "ijepa_target_scale_max", 0.25))

        all_idx = []
        for _ in range(n_targets):
            area = random.uniform(min_scale, max_scale) * self.num_patches
            side = max(1, int(math.sqrt(area)))
            h = min(side, self.grid_h)
            w = min(side, self.grid_w)
            top = random.randint(0, self.grid_h - h)
            left = random.randint(0, self.grid_w - w)

            idx = []
            for r in range(top, top + h):
                start = r * self.grid_w + left
                idx.extend(range(start, start + w))
            all_idx.extend(idx)

        all_idx = sorted(set(all_idx))
        return torch.tensor(all_idx, dtype=torch.long, device=self.device)

    def _sample_context_idx(self, target_idx):
        keep_ratio = float(getattr(self.opts, "ijepa_context_keep_ratio", 0.5))
        full = torch.arange(self.num_patches, device=self.device)
        mask = torch.ones(self.num_patches, dtype=torch.bool, device=self.device)
        mask[target_idx] = False
        candidates = full[mask]
        n_keep = max(1, int(candidates.numel() * keep_ratio))
        perm = torch.randperm(candidates.numel(), device=self.device)[:n_keep]
        return candidates[perm]

    def _predict_targets(self, context_tokens, context_idx, target_idx):
        # context_tokens: [Nc, C]
        # Build sequence: (context + pos_ctx) + (mask_token + pos_tgt)
        pos = _build_2d_sincos_pos_embed(self.grid_h, self.grid_w, self.embed_dim, device=self.device)
        pos_ctx = pos[context_idx]
        pos_tgt = pos[target_idx]

        ctx_in = context_tokens + pos_ctx
        # mask_token is [1, 1, C], expand to [Nt, C] before adding target positional embedding.
        tgt_in = self.mask_token[0].expand(target_idx.numel(), -1) + pos_tgt
        seq = torch.cat([ctx_in, tgt_in], dim=0).unsqueeze(0)  # [1, Nc+Nt, C]
        out = self.predictor(seq)[0]
        pred_tgt = out[-target_idx.numel() :]
        return pred_tgt

    @torch.no_grad()
    def _ema_update_target(self):
        m = self.momentum
        for q, k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            k.data.mul_(m).add_((1.0 - m) * q.data)

    def training_step(self, batch, batch_idx):
        x = batch[0]
        with torch.no_grad():
            target_tokens_all = self._encode_patch_tokens(self.target_encoder, x)
        context_tokens_all = self._encode_patch_tokens(self.context_encoder, x)

        losses = []
        for b in range(x.shape[0]):
            target_idx = self._sample_target_mask()
            context_idx = self._sample_context_idx(target_idx)

            ctx_tokens = context_tokens_all[b, context_idx, :]
            tgt_tokens = target_tokens_all[b, target_idx, :]
            pred_tokens = self._predict_targets(ctx_tokens, context_idx, target_idx)

            pred_tokens = F.normalize(pred_tokens, dim=-1)
            tgt_tokens = F.normalize(tgt_tokens, dim=-1)
            loss = (1.0 - F.cosine_similarity(pred_tokens, tgt_tokens, dim=-1)).mean()
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_jepa_pred", total_loss.detach(), on_step=True, on_epoch=True, prog_bar=False)
        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._ema_update_target()

    def configure_optimizers(self):
        lr = float(getattr(self.opts, "ijepa_lr", 1e-4))
        wd = float(getattr(self.opts, "ijepa_weight_decay", 0.04))
        optimizer = torch.optim.AdamW(
            [
                {"params": self.context_encoder.parameters(), "lr": lr, "weight_decay": wd},
                {"params": self.predictor.parameters(), "lr": lr, "weight_decay": wd},
                {"params": [self.mask_token], "lr": lr, "weight_decay": 0.0},
            ]
        )
        return optimizer
