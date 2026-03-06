import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_base import BasePromptModel


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

        self.projector = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        self.lambda_jepa = getattr(self.opts, "lambda_jepa", 1.0)
        self.lambda_reg = getattr(self.opts, "lambda_reg", 0.1)
        self.lambda_intra = getattr(self.opts, "lambda_intra", 1.0)
        self.lambda_cross = getattr(self.opts, "lambda_cross", 1.0)

        self.proj_dim = proj_dim

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

    def _variance_regularizer(self, x, eps=1e-4):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(1.0 - std))

    def _covariance_regularizer(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        n = x.shape[0]
        if n <= 1:
            return x.new_tensor(0.0)
        cov = (x.T @ x) / (n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag.pow(2).sum() / x.shape[1])

    def _regularization_loss(self, p_all):
        var_loss = self._variance_regularizer(p_all)
        cov_loss = self._covariance_regularizer(p_all)
        return var_loss + cov_loss

    def _multimodal_jepa_loss(self, img_view1, img_view2, sk_view1, sk_view2):
        z_img1 = self.encode_image_branch(img_view1)
        z_img2 = self.encode_image_branch(img_view2)
        z_sk1 = self.encode_sketch_branch(sk_view1)
        z_sk2 = self.encode_sketch_branch(sk_view2)

        p_img1 = self._project(z_img1)
        p_img2 = self._project(z_img2)
        p_sk1 = self._project(z_sk1)
        p_sk2 = self._project(z_sk2)

        center = (p_img1 + p_img2 + p_sk1 + p_sk2) / 4.0

        center_loss = (
            self._cosine_to_center(p_img1, center) +
            self._cosine_to_center(p_img2, center) +
            self._cosine_to_center(p_sk1, center) +
            self._cosine_to_center(p_sk2, center)
        ) / 4.0

        intra_loss = (
            self._pair_cosine_loss(p_img1, p_img2) +
            self._pair_cosine_loss(p_sk1, p_sk2)
        ) / 2.0

        cross_loss = (
            self._pair_cosine_loss(p_img1, p_sk1) +
            self._pair_cosine_loss(p_img1, p_sk2) +
            self._pair_cosine_loss(p_img2, p_sk1) +
            self._pair_cosine_loss(p_img2, p_sk2)
        ) / 4.0

        jepa_loss = center_loss + self.lambda_intra * intra_loss + self.lambda_cross * cross_loss

        p_all = torch.cat([p_img1, p_img2, p_sk1, p_sk2], dim=0)
        reg_loss = self._regularization_loss(p_all)

        total_loss = self.lambda_jepa * jepa_loss + self.lambda_reg * reg_loss

        stats = {
            "z_img1": z_img1,
            "z_img2": z_img2,
            "z_sk1": z_sk1,
            "z_sk2": z_sk2,
            "p_img1": p_img1,
            "p_img2": p_img2,
            "p_sk1": p_sk1,
            "p_sk2": p_sk2,
            "center_loss": center_loss.detach(),
            "intra_loss": intra_loss.detach(),
            "cross_loss": cross_loss.detach(),
            "reg_loss": reg_loss.detach(),
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
        img_view1, img_view2, sk_view1, sk_view2 = batch[:4]

        # channels_last 只对 4D image tensors 有意义
        img_view1 = img_view1.contiguous(memory_format=torch.channels_last)
        img_view2 = img_view2.contiguous(memory_format=torch.channels_last)
        sk_view1 = sk_view1.contiguous(memory_format=torch.channels_last)
        sk_view2 = sk_view2.contiguous(memory_format=torch.channels_last)

        loss, stats = self._multimodal_jepa_loss(
            img_view1, img_view2, sk_view1, sk_view2
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_center_loss", stats["center_loss"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_intra_loss", stats["intra_loss"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_cross_loss", stats["cross_loss"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_reg_loss", stats["reg_loss"], on_step=True, on_epoch=True, prog_bar=False)

        self.log("train_img_feat_norm", stats["z_img1"].norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_sk_feat_norm", stats["z_sk1"].norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_img_proj_norm", stats["p_img1"].norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_sk_proj_norm", stats["p_sk1"].norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)

        return loss