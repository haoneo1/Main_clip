# I-JEPA Pretrain Trials Summary (auto2s_ijepa_24h)

> Pipeline 已完成 3 个 trial 的 pretrain；finetune 未执行（路径 bug，已在 `auto_two_stage_pipeline.py` 修复）。

## 共同 pretrain 设置

| 项 | 值 |
|---|---|
| 脚本 | `experiments/train_pretrain_ijepa.py` |
| backbone | `vit_base_patch16_224` |
| max_epochs | 25 |
| batch_size | 128 |
| 保存目录后缀 | `{exp_name}_ijepa_pretrain` |

## 3 个 trial 结果

| Trial | exp_name | pretrain 有效超参 | best train_loss | 完成 epoch | checkpoint 目录 |
|---:|---|---|---:|---:|---|
| 1 | `auto2s_ijepa_24h_t001` | `ijepa_lr=5e-5` | **0.0162** | 25/25 | `/home/mig/Documents/SBIR_Data/saved_models/auto2s_ijepa_24h_t001_ijepa_pretrain` |
| 2 | `auto2s_ijepa_24h_t002` | `ijepa_lr=5e-5` | 0.0619 | ~4–5（中断） | `/home/mig/Documents/SBIR_Data/saved_models/auto2s_ijepa_24h_t002_ijepa_pretrain` |
| 3 | `auto2s_ijepa_24h_t003` | `ijepa_lr=5e-5` | 0.0306 | 25/25 | `/home/mig/Documents/SBIR_Data/saved_models/auto2s_ijepa_24h_t003_ijepa_pretrain` |

### 对应 finetune 侧配置（尚未跑 finetune）

| Trial | lambda_cls | use_sigreg | lambda_sigreg | clip_LN_lr | prompt_lr | n_prompts | cls_tau |
|---:|---:|---|---:|---:|---|---:|---:|
| 1 | 0.1 | True | 0.05 | 1e-6 | 5e-5 | 3 | 0.05 |
| 2 | 0.1 | True | 0.1 | 1e-6 | 5e-5 | 3 | 0.05 |
| 3 | 0.1 | True | 0.2 | 1e-6 | 5e-5 | 3 | 0.05 |

说明：`jepa_hidden_dim` / `lambda_jepa_pred` 仅 legacy pretrain 使用；I-JEPA pretrain 实际只受 `ijepa_lr` 等 I-JEPA 参数影响。

## 结论

- **Pretrain 最优（loss 最低）**：trial 1（0.0162）
- **完整跑满 25 epoch**：trial 1、trial 3
- **trial 2** 可能在中途被下一轮抢占/中断，仅保留到 epoch 4 左右的 best ckpt
- **Finetune 状态**：0/3 完成（pipeline 原查找 `*_pretrain/last.ckpt`，与实际 `*_ijepa_pretrain/last.ckpt` 不一致）

## 修复说明（已完成）

`auto_two_stage_pipeline.py` 现已按 `pretrain_runner` 自动选择目录：

- `legacy` → `{exp_name}_pretrain`
- `ijepa` → `{exp_name}_ijepa_pretrain`

后续重跑 pipeline 或手动 finetune 时，应能正常衔接 pretrain checkpoint。
