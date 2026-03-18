import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model_base import BasePromptModel


class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PredictorHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PretrainModel(BasePromptModel):
    """
    Pretraining objective:
      1) DINO-like global alignment
      2) JEPA-like local prediction
      3) variance/covariance regularization

    Assumptions:
      - self.clip exists
      - self.img_prompt / self.sk_prompt are learnable prompts
      - self.feat_dim exists
      - CLIP is ViT-based and supports encode_image(image, prompt=...)
    """

    def __init__(self, opts):
        super().__init__(opts)

        hidden_dim = getattr(self.opts, "jepa_hidden_dim", self.feat_dim)
        proj_dim = getattr(self.opts, "jepa_proj_dim", self.feat_dim)
        pred_hidden_dim = getattr(self.opts, "jepa_pred_hidden_dim", hidden_dim)

        self.proj_dim = proj_dim

        # online heads
        self.projector = MLPHead(self.feat_dim, hidden_dim, proj_dim)
        self.predictor = PredictorHead(proj_dim, pred_hidden_dim, proj_dim)

        # EMA teacher
        self.teacher_clip = copy.deepcopy(self.clip)
        self.teacher_projector = copy.deepcopy(self.projector)

        # img_prompt / sk_prompt 大概率是 Parameter，不是 Module
        self.teacher_img_prompt = nn.Parameter(self.img_prompt.detach().clone(), requires_grad=False)
        self.teacher_sk_prompt = nn.Parameter(self.sk_prompt.detach().clone(), requires_grad=False)

        self.teacher_clip.eval()
        self.teacher_projector.eval()
        for p in self.teacher_clip.parameters():
            p.requires_grad = False
        for p in self.teacher_projector.parameters():
            p.requires_grad = False

        # loss weights
        self.lambda_dino_global = getattr(self.opts, "lambda_dino_global", 1.0)
        self.lambda_dino_intra = getattr(self.opts, "lambda_dino_intra", 0.5)
        self.lambda_jepa_local = getattr(self.opts, "lambda_jepa_local", 1.0)
        self.lambda_reg = getattr(self.opts, "lambda_reg", 0.05)

        # EMA momentum
        self.teacher_momentum = getattr(self.opts, "teacher_momentum", 0.996)

    # =========================================================
    # prompt helpers
    # =========================================================
    def _expand_prompt(self, prompt, batch_size, device, dtype):
        """
        prompt can be:
          - [N, D]
          - [1, N, D]
          - [B, N, D]
        """
        if prompt is None:
            return None

        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)
        elif prompt.dim() == 3:
            if prompt.shape[0] == 1:
                prompt = prompt.expand(batch_size, -1, -1)
            elif prompt.shape[0] != batch_size:
                raise ValueError(
                    f"Prompt batch dim mismatch: prompt.shape={prompt.shape}, batch_size={batch_size}"
                )
        else:
            raise ValueError(f"Unsupported prompt shape: {prompt.shape}")

        return prompt.to(device=device, dtype=dtype)

    # =========================================================
    # online encode
    # =========================================================
    def _encode_image_online(self, x):
        prompt = self._expand_prompt(
            self.img_prompt,
            batch_size=x.shape[0],
            device=x.device,
            dtype=self.clip.dtype
        )
        z = self.clip.encode_image(x, prompt=prompt)
        return z.float()

    def _encode_sketch_online(self, x):
        prompt = self._expand_prompt(
            self.sk_prompt,
            batch_size=x.shape[0],
            device=x.device,
            dtype=self.clip.dtype
        )
        z = self.clip.encode_image(x, prompt=prompt)
        return z.float()

    # =========================================================
    # teacher encode
    # =========================================================
    @torch.no_grad()
    def _encode_image_teacher(self, x):
        prompt = self._expand_prompt(
            self.teacher_img_prompt,
            batch_size=x.shape[0],
            device=x.device,
            dtype=self.teacher_clip.dtype
        )
        z = self.teacher_clip.encode_image(x, prompt=prompt)
        return z.float()

    @torch.no_grad()
    def _encode_sketch_teacher(self, x):
        prompt = self._expand_prompt(
            self.teacher_sk_prompt,
            batch_size=x.shape[0],
            device=x.device,
            dtype=self.teacher_clip.dtype
        )
        z = self.teacher_clip.encode_image(x, prompt=prompt)
        return z.float()

    # =========================================================
    # projector / predictor
    # =========================================================
    def _project(self, z):
        return self.projector(z)

    def _predict(self, p):
        return self.predictor(p)

    @torch.no_grad()
    def _teacher_project(self, z):
        return self.teacher_projector(z)

    # =========================================================
    # EMA update
    # =========================================================
    @torch.no_grad()
    def _ema_update_teacher(self):
        self._ema_update_module(self.clip, self.teacher_clip, self.teacher_momentum)
        self._ema_update_module(self.projector, self.teacher_projector, self.teacher_momentum)
        self._ema_update_param(self.img_prompt, self.teacher_img_prompt, self.teacher_momentum)
        self._ema_update_param(self.sk_prompt, self.teacher_sk_prompt, self.teacher_momentum)

    @staticmethod
    @torch.no_grad()
    def _ema_update_module(student_module, teacher_module, m):
        for p_s, p_t in zip(student_module.parameters(), teacher_module.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

        for b_s, b_t in zip(student_module.buffers(), teacher_module.buffers()):
            b_t.data.copy_(b_s.data)

    @staticmethod
    @torch.no_grad()
    def _ema_update_param(student_param, teacher_param, m):
        teacher_param.data.mul_(m).add_(student_param.data, alpha=1.0 - m)

    # =========================================================
    # loss helpers
    # =========================================================
    def _cosine_loss(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (1.0 - (x * y).sum(dim=-1)).mean()

    def _variance_regularizer(self, x, eps=1e-4):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
        return torch.mean(F.relu(1.0 - std))

    def _covariance_regularizer(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        n, d = x.shape
        if n <= 1:
            return x.new_tensor(0.0)
        cov = (x.T @ x) / (n - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return off_diag.pow(2).sum() / d

    def _regularization_loss(self, p_all):
        return self._variance_regularizer(p_all) + self._covariance_regularizer(p_all)

    # =========================================================
    # pretraining loss
    # =========================================================
    def _pretrain_loss(
        self,
        img_global1, img_global2, img_local,
        sk_global1, sk_global2, sk_local
    ):
        # -------------------------
        # online features
        # -------------------------
        z_img_g1 = self._encode_image_online(img_global1)
        z_img_g2 = self._encode_image_online(img_global2)
        z_img_l = self._encode_image_online(img_local)

        z_sk_g1 = self._encode_sketch_online(sk_global1)
        z_sk_g2 = self._encode_sketch_online(sk_global2)
        z_sk_l = self._encode_sketch_online(sk_local)

        p_img_g1 = self._project(z_img_g1)
        p_img_g2 = self._project(z_img_g2)
        p_img_l = self._project(z_img_l)

        p_sk_g1 = self._project(z_sk_g1)
        p_sk_g2 = self._project(z_sk_g2)
        p_sk_l = self._project(z_sk_l)

        q_img_g1 = self._predict(p_img_g1)
        q_sk_g1 = self._predict(p_sk_g1)

        # -------------------------
        # teacher targets
        # -------------------------
        with torch.no_grad():
            t_z_img_g2 = self._encode_image_teacher(img_global2)
            t_z_sk_g2 = self._encode_sketch_teacher(sk_global2)
            t_z_img_l = self._encode_image_teacher(img_local)
            t_z_sk_l = self._encode_sketch_teacher(sk_local)

            t_p_img_g2 = self._teacher_project(t_z_img_g2)
            t_p_sk_g2 = self._teacher_project(t_z_sk_g2)
            t_p_img_l = self._teacher_project(t_z_img_l)
            t_p_sk_l = self._teacher_project(t_z_sk_l)

        # -------------------------
        # 1) DINO-like global alignment
        # -------------------------
        # cross-modal global
        loss_global_cross = 0.5 * (
            self._cosine_loss(q_img_g1, t_p_sk_g2.detach()) +
            self._cosine_loss(q_sk_g1, t_p_img_g2.detach())
        )

        # intra-modal multi-view
        loss_global_intra = 0.5 * (
            self._cosine_loss(q_img_g1, t_p_img_g2.detach()) +
            self._cosine_loss(q_sk_g1, t_p_sk_g2.detach())
        )

        # -------------------------
        # 2) JEPA-like local prediction
        # -------------------------
        # image global context -> sketch local target
        # sketch global context -> image local target
        loss_local_jepa = 0.5 * (
            self._cosine_loss(q_img_g1, t_p_sk_l.detach()) +
            self._cosine_loss(q_sk_g1, t_p_img_l.detach())
        )

        # -------------------------
        # 3) anti-collapse regularization
        # -------------------------
        p_all = torch.cat([p_img_g1, p_img_g2, p_img_l, p_sk_g1, p_sk_g2, p_sk_l], dim=0)
        loss_reg = self._regularization_loss(p_all)

        total_loss = (
            self.lambda_dino_global * loss_global_cross +
            self.lambda_dino_intra * loss_global_intra +
            self.lambda_jepa_local * loss_local_jepa +
            self.lambda_reg * loss_reg
        )

        stats = {
            "loss_global_cross": loss_global_cross.detach(),
            "loss_global_intra": loss_global_intra.detach(),
            "loss_local_jepa": loss_local_jepa.detach(),
            "loss_reg": loss_reg.detach(),
            "img_feat_norm": z_img_g1.norm(dim=1).mean().detach(),
            "sk_feat_norm": z_sk_g1.norm(dim=1).mean().detach(),
            "img_proj_norm": p_img_g1.norm(dim=1).mean().detach(),
            "sk_proj_norm": p_sk_g1.norm(dim=1).mean().detach(),
        }
        return total_loss, stats

    # =========================================================
    # optim
    # =========================================================
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.clip.parameters(), "lr": self.opts.clip_LN_lr},
                {"params": [self.img_prompt, self.sk_prompt], "lr": self.opts.prompt_lr},
                {"params": self.projector.parameters(), "lr": self.opts.jepa_lr},
                {"params": self.predictor.parameters(), "lr": self.opts.jepa_lr},
            ],
            weight_decay=getattr(self.opts, "weight_decay", 1e-5),
        )
        return optimizer

    # =========================================================
    # training
    # =========================================================
    def training_step(self, batch, batch_idx):
        img_global1, img_global2, img_local, sk_global1, sk_global2, sk_local = batch[:6]

        loss, stats = self._pretrain_loss(
            img_global1, img_global2, img_local,
            sk_global1, sk_global2, sk_local
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_global_cross", stats["loss_global_cross"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_global_intra", stats["loss_global_intra"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_local_jepa", stats["loss_local_jepa"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_reg", stats["loss_reg"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_img_feat_norm", stats["img_feat_norm"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_sk_feat_norm", stats["sk_feat_norm"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_img_proj_norm", stats["img_proj_norm"], on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_sk_proj_norm", stats["sk_proj_norm"], on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._ema_update_teacher()