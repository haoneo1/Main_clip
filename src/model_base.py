import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.clip import clip


def freeze_model(m):
    m.requires_grad_(False)


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.requires_grad_(False)


def _to_key_list(category):
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


class BasePromptModel(pl.LightningModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        self.clip, _ = clip.load("ViT-B/32", device=self.device)
        self.clip.apply(freeze_all_but_bn)
        self.clip.train()

        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        if hasattr(self.clip, "visual") and hasattr(self.clip.visual, "output_dim"):
            feat_dim = int(self.clip.visual.output_dim)
        else:
            feat_dim = 512

        self.feat_dim = feat_dim

    def encode_image_branch(self, data):
        return self.clip.encode_image(
            data, self.img_prompt.expand(data.shape[0], -1, -1)
        )

    def encode_sketch_branch(self, data):
        return self.clip.encode_image(
            data, self.sk_prompt.expand(data.shape[0], -1, -1)
        )

    def forward(self, data, dtype="image"):
        if dtype == "image":
            return self.encode_image_branch(data)
        else:
            return self.encode_sketch_branch(data)

    def init_sk_prompt_from_img_prompt(self):
        with torch.no_grad():
            self.sk_prompt.copy_(self.img_prompt)