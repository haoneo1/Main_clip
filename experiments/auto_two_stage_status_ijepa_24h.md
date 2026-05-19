# Auto Two-Stage Pipeline Status

- Updated at: `2026-05-19T17:35:00`
- Last trial id: `1`
- Last phase: `pretrain`
- Grid profile: `main`
- Pretrain runner: `ijepa`
- Total candidate configs: `12`

## Last Result
- exp_name: `auto2s_t001`
- status: `running_pretrain`
- pretrain_rc: `None`
- pretrain_loss: `None`
- pretrain_ckpt_dir: `/home/mig/Documents/SBIR_Data/saved_models/auto2s_t001_ijepa_pretrain`
- finetune_rc: `None`
- mAP: `None`
- cfg: `{'stage': 's1_coarse', 'jepa_hidden_dim': 512, 'lambda_jepa_pred': 0.5, 'jepa_lr': 5e-05, 'ijepa_lr': 5e-05, 'ijepa_target_scale_min': 0.15, 'ijepa_target_scale_max': 0.25, 'ijepa_context_keep_ratio': 0.4, 'ijepa_pred_depth': 2, 'ijepa_pred_heads': 8, 'ijepa_pred_mlp_ratio': 4.0, 'ijepa_momentum': 0.996, 'ijepa_num_targets': 4, 'ijepa_backbone_arch': 'vit_base_patch16_224', 'lambda_cls': 0.0, 'use_sigreg': True, 'lambda_sigreg': 0.1, 'clip_LN_lr': 1e-06, 'prompt_lr': 5e-05, 'n_prompts': 3, 'cls_tau': 0.05, 'use_clip_cls': False}`

## Best Finetune So Far
- trial_id: `None`
- exp_name: `None`
- mAP: `-1.0`
- cfg: `None`

## Best Pretrain So Far
- trial_id: `None`
- exp_name: `None`
- train_loss_epoch: `None`
- cfg: `None`
