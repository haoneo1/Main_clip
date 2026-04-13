import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_base import BasePromptModel
from src.retrieval_losses import category_list_to_mask, supervised_infonce


class PretrainModel(BasePromptModel):
    """
    More LeJEPA-like pretraining:
    - no predictor head
    - one shared projection space
    - multi-view + multi-modal JEPA loss
    - regularization term to avoid collapse
    """

    def __init__(self, opts):
        super().__init__(opts)

        hidden_dim = self.opts.jepa_hidden_dim
        if hidden_dim <= 0:
            hidden_dim = self.feat_dim

        proj_dim = getattr(self.opts, "jepa_proj_dim", self.feat_dim)

        # projector only, no predictor
        self.projector = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.lambda_jepa = getattr(self.opts, "lambda_jepa", 1.0)
        self.lambda_reg = getattr(self.opts, "lambda_reg", 0.1)

        # optional weights for different alignments
        self.lambda_intra = getattr(self.opts, "lambda_intra", 1.0)
        self.lambda_cross = getattr(self.opts, "lambda_cross", 1.0)

        self.proj_dim = proj_dim

    # -------------------------
    # helpers
    # -------------------------
    def _project(self, z):
        return self.projector(z)

    def _cosine_to_center(self, p, center):
        p = F.normalize(p, dim=-1)
        center = F.normalize(center, dim=-1)
        return (1.0 - F.cosine_similarity(p, center, dim=-1)).mean()

    def _pair_cosine_loss(self, p1, p2):
        p1 = F.normalize(p1, dim=-1)
        p2 = F.normalize(p2, dim=-1)
        return (1.0 - F.cosine_similarity(p1, p2, dim=-1)).mean()

    def _cosine_to_center_multi(self, p_views, center):
        # p_views: [B, V, D], center: [B, D]
        p_views = F.normalize(p_views, dim=-1)
        center = F.normalize(center, dim=-1).unsqueeze(1)
        sim = (p_views * center).sum(dim=-1)
        return (1.0 - sim).mean()

    def _clip_input_resolution(self):
        """CLIP ViT positional embeddings are fixed to (input_resolution // patch) ** 2 + 1 tokens."""
        visual = self.clip.visual
        r = getattr(visual, "input_resolution", 224)
        if isinstance(r, torch.Tensor):
            r = int(r.item())
        return int(r)

    def _encode_multi_views(self, x, branch="image"):
        # x: [B, V, C, H, W] -> [B, V, D]
        bsz, n_views, c, h, w = x.shape
        x = x.reshape(bsz * n_views, c, h, w)
        target = self._clip_input_resolution()
        if h != target or w != target:
            x = F.interpolate(
                x, size=(target, target), mode="bicubic", align_corners=False
            )
        if branch == "image":
            z = self.encode_image_branch(x)
        else:
            z = self.encode_sketch_branch(x)
        p = self._project(z).reshape(bsz, n_views, -1)
        return z.reshape(bsz, n_views, -1), p

    def _variance_regularizer(self, x, eps=1e-4):
        """
        VICReg-style variance floor regularizer.
        Not official SIGReg, but a simple anti-collapse surrogate.
        """
        std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(1.0 - std))

    def _covariance_regularizer(self, x):
        """
        Encourage decorrelation across dimensions.
        """
        x = x - x.mean(dim=0, keepdim=True)
        n = x.shape[0]
        if n <= 1:
            return x.new_tensor(0.0)
        cov = (x.T @ x) / (n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag.pow(2).sum() / x.shape[1])

    def _regularization_loss(self, p_all):
        """
        Practical surrogate for SIGReg if you don't want to add external lejepa pkg yet.
        """
        var_loss = self._variance_regularizer(p_all)
        cov_loss = self._covariance_regularizer(p_all)
        return var_loss + cov_loss

    # -------------------------
    # multi-modal LeJEPA-style loss
    # -------------------------
    def _multimodal_jepa_loss(
        self, img_globals, img_locals, sk_globals, sk_locals, categories=None
    ):
        # backbone + projector for multi-views
        z_img_g, p_img_g = self._encode_multi_views(img_globals, branch="image")
        z_img_l, p_img_l = self._encode_multi_views(img_locals, branch="image")
        z_sk_g, p_sk_g = self._encode_multi_views(sk_globals, branch="sketch")
        z_sk_l, p_sk_l = self._encode_multi_views(sk_locals, branch="sketch")

        # modality-specific global centers
        mu_p = p_img_g.mean(dim=1)  # [B, D]
        mu_s = p_sk_g.mean(dim=1)   # [B, D]

        # intra-modal: all views -> own global center
        intra_photo = (
            self._cosine_to_center_multi(p_img_g, mu_p) +
            self._cosine_to_center_multi(p_img_l, mu_p)
        ) / 2.0
        intra_sketch = (
            self._cosine_to_center_multi(p_sk_g, mu_s) +
            self._cosine_to_center_multi(p_sk_l, mu_s)
        ) / 2.0

        # cross-modal global-global
        cross_global = self._pair_cosine_loss(mu_s, mu_p)

        # cross-modal local -> opposite global center
        cross_local2global = (
            self._cosine_to_center_multi(p_sk_l, mu_p) +
            self._cosine_to_center_multi(p_img_l, mu_s)
        ) / 2.0

        # weighted objective (first stable version)
        lambda_intra_sk = getattr(self.opts, "lambda_intra_sk", 0.5)
        lambda_intra_ph = getattr(self.opts, "lambda_intra_ph", 0.5)
        lambda_cross_global = getattr(self.opts, "lambda_cross_global", 0.3)
        lambda_cross_local2global = getattr(self.opts, "lambda_cross_local2global", 0.2)

        jepa_loss = (
            lambda_intra_sk * intra_sketch +
            lambda_intra_ph * intra_photo +
            lambda_cross_global * cross_global +
            lambda_cross_local2global * cross_local2global
        )

        # anti-collapse regularization (optional, lightweight)
        p_all = torch.cat(
            [
                p_img_g.reshape(-1, p_img_g.shape[-1]),
                p_img_l.reshape(-1, p_img_l.shape[-1]),
                p_sk_g.reshape(-1, p_sk_g.shape[-1]),
                p_sk_l.reshape(-1, p_sk_l.shape[-1]),
            ],
            dim=0,
        )
        reg_loss = self._regularization_loss(p_all)

        total_loss = self.lambda_jepa * jepa_loss + self.lambda_reg * reg_loss
        align_loss = jepa_loss.new_tensor(0.0)
        lam_align = float(getattr(self.opts, "lambda_pretrain_sk_img_align", 0.0))
        if (
            lam_align > 0
            and categories is not None
            and len(categories) == mu_s.shape[0]
            and mu_s.shape[0] > 1
        ):
            mask = category_list_to_mask(categories, mu_s.device)
            tau = float(getattr(self.opts, "pretrain_infonce_tau", 0.07))
            align_loss = supervised_infonce(mu_s, mu_p, mask, tau=tau)
            total_loss = total_loss + lam_align * align_loss

        stats = {
            "z_img_g": z_img_g,
            "z_img_l": z_img_l,
            "z_sk_g": z_sk_g,
            "z_sk_l": z_sk_l,
            "p_img_g": p_img_g,
            "p_img_l": p_img_l,
            "p_sk_g": p_sk_g,
            "p_sk_l": p_sk_l,
            "intra_photo": intra_photo.detach(),
            "intra_sketch": intra_sketch.detach(),
            "cross_global": cross_global.detach(),
            "cross_local2global": cross_local2global.detach(),
            "reg_loss": reg_loss.detach(),
            "align_loss": align_loss.detach(),
        }

        return total_loss, stats

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.clip.parameters(), "lr": self.opts.clip_LN_lr},
                {"params": [self.img_prompt, self.sk_prompt], "lr": self.opts.prompt_lr},
                {"params": self.projector.parameters(), "lr": self.opts.jepa_lr},
            ]
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        img_globals, img_locals, sk_globals, sk_locals = batch[:4]
        categories = batch[4] if len(batch) > 4 else None

        loss, stats = self._multimodal_jepa_loss(
            img_globals, img_locals, sk_globals, sk_locals, categories=categories
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_intra_photo", stats["intra_photo"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_intra_sketch", stats["intra_sketch"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_cross_global", stats["cross_global"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_cross_local2global", stats["cross_local2global"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_reg_loss", stats["reg_loss"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_align_loss", stats["align_loss"], on_step=True, on_epoch=True, prog_bar=False)

        self.log("train_img_feat_norm", stats["z_img_g"].norm(dim=-1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_sk_feat_norm", stats["z_sk_g"].norm(dim=-1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_img_proj_norm", stats["p_img_g"].norm(dim=-1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_sk_proj_norm", stats["p_sk_g"].norm(dim=-1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)

        return loss