"""
Retrieval-oriented losses shared by pretrain (optional bridge) and finetune.
"""

import numpy as np
import torch
import torch.nn.functional as F


def category_list_to_mask(categories, device: torch.device) -> torch.Tensor:
    """Batch of category strings -> [B,B] bool, True where same class."""
    cat = np.asarray([str(c) for c in categories])
    m = cat[:, None] == cat[None, :]
    return torch.tensor(m, dtype=torch.bool, device=device)


def label_tensor_to_mask(labels: torch.Tensor) -> torch.Tensor:
    """Integer labels [B] -> [B,B] bool."""
    return labels.unsqueeze(0) == labels.unsqueeze(1)


def supervised_infonce(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    positive_mask: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """
    Symmetric multi-positive InfoNCE: align feat_a[i] with all feat_b[j] where mask[i,j].
    """
    feat_a = F.normalize(feat_a, dim=-1)
    feat_b = F.normalize(feat_b, dim=-1)
    logits = (feat_a @ feat_b.T) / tau
    loss_ab = _multipos_infonce_rows(logits, positive_mask)
    loss_ba = _multipos_infonce_rows(logits.T, positive_mask.T)
    return 0.5 * (loss_ab + loss_ba)


def _multipos_infonce_rows(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pos_logits = logits.masked_fill(~mask, float("-inf"))
    row_lse_pos = torch.logsumexp(pos_logits, dim=1)
    row_lse_all = torch.logsumexp(logits, dim=1)
    return -(row_lse_pos - row_lse_all).mean()
