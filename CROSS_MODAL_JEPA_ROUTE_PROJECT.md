# Cross-Modal JEPA SBIR 路线探索文档（面向当前项目）

> 目标：基于 `cross_modal_jepa_sbir_route.md` 与当前代码现状，给出一条可以直接落地执行的路线。  
> 范围：优先保证可跑通、可对比、可复现实验，再逐步升级到双向跨模态 JEPA。

---

## 1) 当前项目基线状态（先对齐现实）

### 已有能力

- 预训练：`experiments/train_pretrain.py` + `src/model_pretrain.py`
  - 已有 JEPA 预测头 `pred_img2img`
  - 当前是 **photo -> photo（同模态）latent prediction**
- 微调：`experiments/train_finetune.py` + `src/model_finetune.py`
  - 主损失为 `triplet`
  - 可选 `CLIP text分类损失`（`use_clip_cls` / `lambda_cls` / `cls_tau`）
  - 可选 `sigreg`（`use_sigreg` / `lambda_sigreg`）
- 自动调参：`experiments/auto_iterate_map.py`
  - 已有全局扫参和本地精修策略

### 当前缺口（与跨模态 JEPA 主线对比）

- 还没有 `s->p` / `p->s` 的跨模态 predictor
- 还没有 class prototype target 机制
- 还没有 patch-level JEPA
- 最近 `stage2` 失败点是环境启动方式（`ModuleNotFoundError: src`）

---

## 2) 路线总览（可行且低风险）

建议采用 5 个阶段，按“最小改动 -> 主线创新 -> 扩展增强”推进：

1. **Stage A：稳定基线 + 跑通环境**
2. **Stage B：Text Anchor 与现有 finetune 优化**
3. **Stage C：单向跨模态 JEPA（s->p）**
4. **Stage D：双向跨模态 JEPA（s->p + p->s）**
5. **Stage E：Patch-level JEPA（可选，第二轮）**

说明：对第一版论文，Stage C/D 才是核心贡献；Stage E 是加分项。

---

## 3) 阶段化执行计划（含超参数）

## Stage A：稳定基线 + 跑通环境（1 周）

### 目标

- 确保 pretrain 和 finetune 均可稳定运行
- 形成可复现实验入口（统一命令模板）

### 必做动作

- 所有训练命令统一带 `PYTHONPATH=/home/mig/Documents/SBIR/Main_clip`
- 先固定 1 组稳定参数，跑完完整训练

### 推荐命令模板

```bash
PYTHONPATH=/home/mig/Documents/SBIR/Main_clip \
CUDA_VISIBLE_DEVICES=0 conda run -n pyenv python experiments/train_pretrain.py \
  --exp_name stageA_pretrain --max_epochs 40 --batch_size 128 \
  --jepa_hidden_dim 512 --lambda_jepa_pred 1.0 --jepa_lr 1e-4
```

```bash
PYTHONPATH=/home/mig/Documents/SBIR/Main_clip \
CUDA_VISIBLE_DEVICES=0 conda run -n pyenv python experiments/train_finetune.py \
  --exp_name stageA_finetune --finetune_init_ckpt /home/mig/Documents/SBIR_Data/saved_models/stageA_pretrain_pretrain/last.ckpt \
  --max_epochs 60 --early_stop_patience 12 --batch_size 64 \
  --use_clip_cls True --lambda_cls 0.2 --use_sigreg True --lambda_sigreg 0.1 \
  --clip_LN_lr 1e-6 --prompt_lr 5e-5 --n_prompts 3 --cls_tau 0.07
```

### Stage A 超参数（锁定区间）

| 参数 | 推荐值 | 备注 |
|---|---:|---|
| `max_epochs` (pretrain) | 40 | 与当前 stage2 对齐 |
| `max_epochs` (finetune) | 60 | 与当前脚本对齐 |
| `batch_size` pretrain | 128 | 4090 可承受 |
| `batch_size` finetune | 64 | 稳定优先 |
| `clip_LN_lr` | `1e-6` | 先用当前有效区 |
| `prompt_lr` | `5e-5` | 先用当前有效区 |
| `lambda_cls` | `0.2` | 来自现有最佳附近 |
| `lambda_sigreg` | `0.1` | 来自现有最佳附近 |
| `n_prompts` | 3 或 6 | 先 3，后 6 对照 |
| `cls_tau` | 0.05 ~ 0.07 | 先 0.07 |

成功标准：至少拿到 1 组无报错完整曲线和可用 checkpoint。

---

## Stage B：Text Anchor 与 finetune 结构优化（1 周）

### 目标

- 在不改主模型拓扑前提下，先稳住 zero-shot 语义锚点
- 给后续 JEPA 损失引入更稳 target

### 改动建议

- 保持 `triplet + clip_cls (+sigreg)` 主线
- 优先扩大 text template（你现在已有 `cls_templates` 入口）
- 对比 `use_clip_logit_scale=True/False`

### Stage B 超参数

| 参数 | 建议 |
|---|---|
| `lambda_cls` | `0.1 / 0.2 / 0.3` |
| `use_sigreg` | `True` 优先 |
| `lambda_sigreg` | `0.05 / 0.1 / 0.2` |
| `clip_LN_lr` | `5e-7 / 1e-6 / 2e-6` |
| `prompt_lr` | `5e-5 / 1e-4` |
| `n_prompts` | `3 / 6` |
| `cls_tau` | `0.05 / 0.07 / 0.1` |

成功标准：相对 Stage A 最好结果，`mAP` 稳定提升 `>= 0.5 ~ 1.0` 点。

---

## Stage C：单向跨模态 JEPA（s->p，主推）2 周

### 目标

- 在现有 `FinetuneModel` 上新增 `Q_sp`，验证 sketch latent 是否可预测 photo prototype latent

### 最小实现路径

- 在 `src/model_finetune.py` 新增：
  - `pred_s2p`（MLP predictor）
  - `lambda_jepa_sp`、`use_jepa_sp`
  - `photo class prototype` 缓存（每 epoch 离线更新一次）
- 损失改为：
  - `L = L_triplet + lambda_cls*L_cls + lambda_sigreg*L_sigreg + lambda_sp*L_sp`

### Stage C 超参数（核心）

| 参数 | 首选值 | 扫描范围 |
|---|---:|---|
| `lambda_sp` | `0.1` | `0.05 / 0.1 / 0.2` |
| `sp_predictor_hidden` | `512` | `512 / 768` |
| `sp_target_type` | `class_prototype` | `class_prototype / instance` |
| `sp_warmup_epochs` | `5` | `3 / 5 / 8` |
| `prototype_update` | `epoch_offline` | `epoch_offline / ema` |

建议：先固定 `lambda_cls=0.2`, `lambda_sigreg=0.1`，只扫描 `lambda_sp`。

成功标准：相对 Stage B 最优，主指标提升 `>= 1.0 ~ 2.0` 点。

---

## Stage D：双向跨模态 JEPA（s->p + p->s）2 周

### 目标

- 加入 `Q_ps` 验证双向是否优于单向
- 如果不稳定，保留非对称方案（只保留 s->p）

### 最小实现路径

- 在 `src/model_finetune.py` 增加：
  - `pred_p2s`
  - `lambda_jepa_ps`、`use_jepa_ps`
  - sketch prototype 缓存
- 总损失：
  - `L = L_base + lambda_sp*L_sp + lambda_ps*L_ps`

### Stage D 超参数（核心）

| 参数 | 首选值 | 扫描范围 |
|---|---:|---|
| `lambda_sp` | `0.1` | `0.05 / 0.1 / 0.2` |
| `lambda_ps` | `0.05` | `0.02 / 0.05 / 0.1` |
| `lambda_ps_start_epoch` | `8` | `5 / 8 / 10` |
| `detach_jepa_target` | `True` | `True` 固定 |
| `predictor_share` | `False` | `False` 固定 |

关键策略：`lambda_ps < lambda_sp`，并延迟启用 `p->s`。

成功标准：双向优于单向；若不成立，论文可采用 Asymmetric 主线。

---

## Stage E：Patch-level JEPA（可选，第二轮）2~3 周

### 目标

- 从 CLS/prototype 升级到 token 级别预测
- 增强 “JEPA/world-model style” 叙事

### 超参数（建议起点）

| 参数 | 建议值 |
|---|---|
| `image_size` | 224 |
| `patch_size` | 16 |
| `target_block_ratio` | `0.15 ~ 0.30` |
| `sketch_mask_ratio` | `0.30 ~ 0.50` |
| `photo_mask_ratio` | `0.50 ~ 0.70` |
| `predictor_depth` | 2 |
| `predictor_heads` | 8 |
| `ema_decay` | `0.996 ~ 0.999` |

说明：若 Stage D 结果还不稳，不建议立即进入 Stage E。

---

## 4) 建议的实验矩阵（第一轮）

按最小可行矩阵运行：

1. Baseline (`triplet + cls + sigreg`)
2. Baseline + `s->p`
3. Baseline + `p->s`
4. Baseline + `s->p + p->s`
5. (可选) `s->p + p->s` + prototype 策略切换（instance vs class）

每组固定 3 个 seed：`42 / 43 / 44`。

---

## 5) 自动化搜索建议（与你现有脚本兼容）

你已有 `auto_iterate_map.py`，建议分两层：

- **层 1（现有）**：扫描 `lambda_cls, lambda_sigreg, clip_LN_lr, prompt_lr, n_prompts, cls_tau`
- **层 2（新增 JEPA 后）**：在层1最优附近再扫描：
  - `lambda_sp`（主）
  - `lambda_ps`（次）
  - `sp_warmup_epochs`
  - `prototype_update_mode`

推荐先加 `lambda_sp` 到 `auto_iterate_map.py`，确认单向收益后再加 `lambda_ps`。

---

## 6) 里程碑与停机规则

### 里程碑

- M1：Stage A 跑通并有可复现 best ckpt
- M2：Stage B 获得稳定改进
- M3：Stage C 达到 `+1 ~ +2 mAP`（相对 Stage B）
- M4：Stage D 若双向优于单向则定主线，否则定非对称主线

### 停机规则

- 连续 3 次 trial `return_code != 0`：立即停，先修环境
- JEPA loss 连续下降但 mAP 不变：减小 `lambda_sp/lambda_ps`
- 加入 `p->s` 后明显退化：保留 `s->p` 主线

---

## 7) 第一版推荐默认配置（直接可用）

```text
Backbone: CLIP ViT-B/32 (沿用当前)
Batch: 64
Epoch: 60
Base loss: triplet + clip_cls + sigreg
lambda_cls: 0.2
lambda_sigreg: 0.1
clip_LN_lr: 1e-6
prompt_lr: 5e-5
n_prompts: 3
cls_tau: 0.07
lambda_sp: 0.1
lambda_ps: 0.05 (Stage D 再开)
sp_warmup_epochs: 5
prototype target: class prototype
```

---

## 8) 结论（项目级执行建议）

- 你的项目当前最合理主线是：**Stage A -> B -> C -> D**
- 第一轮不要直接做 patch-level，先把 `prototype-level cross-modal JEPA` 做出稳定增益
- 从论文风险角度，**单向 s->p 成功** 就已经足够形成有价值的方法主线
- 双向 `p->s` 建议定位为“增强项”，不是硬性必须成功项

---

## 9) 已落地自动化脚本（两阶段批量）

已新增脚本：`experiments/auto_two_stage_pipeline.py`，用于批量执行：

```text
pretrain -> finetune -> 解析best mAP -> 写实时状态markdown
```

特点：

- 自动注入 `PYTHONPATH`，规避 `ModuleNotFoundError: src`
- 支持 `smoke/main/explore` 三种网格规模
- 支持时间预算 `--hours`
- 支持失败自动停止（连续失败阈值）
- 支持复用已有 pretrain checkpoint（节省算力）
- 支持删除非最佳 trial checkpoint

推荐启动命令（先 smoke 验证）：

```bash
python experiments/auto_two_stage_pipeline.py \
  --conda_env pyenv \
  --grid_profile smoke \
  --hours 3 \
  --base_exp auto2s_smoke
```

正式批量命令（main 网格）：

```bash
python experiments/auto_two_stage_pipeline.py \
  --conda_env pyenv \
  --grid_profile main \
  --hours 24 \
  --base_exp auto2s_main \
  --delete_non_best false
```

状态文件默认输出：

```text
experiments/auto_two_stage_status.md
```

