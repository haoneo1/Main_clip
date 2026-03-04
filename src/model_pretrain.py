import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_base import BasePromptModel


class PretrainModel(BasePromptModel):
    def __init__(self, opts):
        super().__init__(opts)

        self.lambda_sigreg = float(getattr(opts, "lambda_sigreg", 0.1))
        self.num_slices = int(getattr(opts, "sigreg_num_slices", 256))
        self.num_t_points = int(getattr(opts, "sigreg_num_t_points", 17))
        self.sigreg_t_max = float(getattr(opts, "sigreg_t_max", 5.0))

    def _lejepa_pred_loss(self, z1, z2):
        # two-view 版本：center = 两个视图 embedding 的均值
        center = 0.5 * (z1 + z2)
        loss1 = (z1 - center).pow(2).mean()
        loss2 = (z2 - center).pow(2).mean()
        return 0.5 * (loss1 + loss2)

    def _sigreg_loss(self, z):
        """
        z: [B, D]
        基于论文伪代码的简化单卡版 SIGReg
        """
        B, D = z.shape
        device = z.device
        dtype = z.dtype

        # 采样随机切片方向 A: [D, M]
        A = torch.randn(D, self.num_slices, device=device, dtype=dtype)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-12)

        # 积分点 t: [T]
        t = torch.linspace(
            -self.sigreg_t_max,
            self.sigreg_t_max,
            self.num_t_points,
            device=device,
            dtype=dtype,
        )

        # 标准高斯 N(0,1) 的理论 characteristic function
        exp_f = torch.exp(-0.5 * t ** 2)   # [T]

        # 投影后数据: [B, M]
        z_proj = z @ A

        # x_t: [B, M, T]
        x_t = z_proj.unsqueeze(-1) * t.view(1, 1, -1)

        # empirical characteristic function: [M, T]
        ecf = torch.exp(1j * x_t).mean(dim=0)

        # weighted L2 distance
        err = (ecf - exp_f.view(1, -1)).abs().square() * exp_f.view(1, -1)

        # 对 t 做梯形积分，再对 slices 求均值
        loss_per_slice = torch.trapz(err, t, dim=1) * B
        return loss_per_slice.mean().real

    def _total_loss(self, img_view1, img_view2):
        z1 = self.encode_image_branch(img_view1)
        z2 = self.encode_image_branch(img_view2)

        pred_loss = self._lejepa_pred_loss(z1, z2)
        sigreg_loss = 0.5 * (self._sigreg_loss(z1) + self._sigreg_loss(z2))

        total_loss = (1.0 - self.lambda_sigreg) * pred_loss + self.lambda_sigreg * sigreg_loss
        return total_loss, pred_loss, sigreg_loss, z1, z2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.clip.parameters(), "lr": self.opts.clip_LN_lr},
                {"params": [self.img_prompt], "lr": self.opts.prompt_lr},
            ],
            weight_decay=getattr(self.opts, "weight_decay", 5e-4),
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        img_view1, img_view2 = batch[:2]

        total_loss, pred_loss, sigreg_loss, z1, z2 = self._total_loss(img_view1, img_view2)
        batch_size = img_view1.size(0)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_pred", pred_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train_sigreg", sigreg_loss.detach(), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train_img_norm_v1", z1.norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.log("train_img_norm_v2", z2.norm(dim=1).mean().detach(), on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)

        return total_loss