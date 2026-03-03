import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_base import BasePromptModel


class PretrainModel(BasePromptModel):
    def __init__(self, opts):
        super().__init__(opts)

        hidden_dim = self.opts.jepa_hidden_dim
        if hidden_dim <= 0:
            hidden_dim = self.feat_dim

        self.pred_img2img = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.feat_dim),
        )

        self.use_detach_jepa_target = self.opts.use_detach_jepa_target
        self.lambda_jepa_pred = self.opts.lambda_jepa_pred

    def _cosine_pred_loss(self, pred, target):
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()

    def _photo_jepa_loss(self, img_view1, img_view2):
        z1 = self.encode_image_branch(img_view1)
        z2 = self.encode_image_branch(img_view2)

        pred_z2 = self.pred_img2img(z1)

        target = z2.detach() if self.use_detach_jepa_target else z2
        loss = self._cosine_pred_loss(pred_z2, target)

        return loss, z1, z2, pred_z2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.clip.parameters(), "lr": self.opts.clip_LN_lr},
                {"params": [self.img_prompt], "lr": self.opts.prompt_lr},
                {"params": self.pred_img2img.parameters(), "lr": self.opts.jepa_lr},
            ]
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        img_view1, img_view2 = batch[:2]

        loss, z1, z2, pred_z2 = self._photo_jepa_loss(img_view1, img_view2)
        loss = self.lambda_jepa_pred * loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_jepa_pred", loss.detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_img_norm_v1", z1.norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_img_norm_v2", z2.norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_pred_norm", pred_z2.norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False)

        return loss
    