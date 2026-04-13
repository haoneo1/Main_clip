import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Sketch-based OD')



# --------------------
# DataLoader Options
# --------------------



# ----------------------
# Training Stage
# ----------------------
parser.add_argument(
    '--train_stage',
    type=str,
    default='pretrain',
    choices=['pretrain', 'finetune']
)
# ----------------------
# 常规设置
# ----------------------

parser.add_argument('--data_dir', type=str, default='/home/mig/Documents/SBIR_Data/Sketchy/')
#parser.add_argument('--data_dir', type=str, default='/home/bingxing2/home/scx9951/SBIR_Data/Sketchy/')#超算数据路径
parser.add_argument('--exp_name', type=str, default='SSL_v2_local')
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--precision', type=str, default='16-mixed')



# ----------------------
# Pretrain setting
# ----------------------
parser.add_argument('--jepa_hidden_dim', type=int, default=512)#pre head的隐藏层
parser.add_argument('--use_detach_jepa_target', type=str2bool, default=True)#决定 JEPA 的目标分支是否停止梯度，通常设 True 更稳
parser.add_argument('--lambda_jepa_pred', type=float, default=1.0)#决定 JEPA 预测损失的权重大小，当前一般设 1.0
parser.add_argument('--jepa_proj_dim', type=int, default=512)

# Multi-view settings (first stable version for ZS-SBIR)
parser.add_argument('--num_global_sk', type=int, default=2)
parser.add_argument('--num_local_sk', type=int, default=4)
parser.add_argument('--num_global_ph', type=int, default=2)
parser.add_argument('--num_local_ph', type=int, default=4)
parser.add_argument('--global_size', type=int, default=224)
parser.add_argument('--local_size_sk', type=int, default=128)
parser.add_argument('--local_size_ph', type=int, default=112)

# Crop ranges
parser.add_argument('--sk_global_scale_min', type=float, default=0.65)
parser.add_argument('--sk_global_scale_max', type=float, default=1.00)
parser.add_argument('--sk_local_scale_min', type=float, default=0.45)
parser.add_argument('--sk_local_scale_max', type=float, default=0.70)
parser.add_argument('--ph_global_scale_min', type=float, default=0.60)
parser.add_argument('--ph_global_scale_max', type=float, default=1.00)
parser.add_argument('--ph_local_scale_min', type=float, default=0.30)
parser.add_argument('--ph_local_scale_max', type=float, default=0.55)

# Sketch local crop constraints
parser.add_argument('--min_fg_ratio_sk_local', type=float, default=0.12)
parser.add_argument('--min_edge_ratio_sk_local', type=float, default=0.04)
parser.add_argument('--max_local_retry', type=int, default=10)




parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
# Finetune-only LRs (pretrain still uses clip_LN_lr / prompt_lr above)
parser.add_argument(
    '--finetune_clip_LN_lr',
    type=float,
    default=8e-7,
    help='Finetune: LR for CLIP visual (+LN). Lower than pretrain reduces drift from JEPA init.',
)
parser.add_argument(
    '--finetune_prompt_lr',
    type=float,
    default=4e-5,
    help='Finetune: LR for sk/img prompts.',
)
parser.add_argument('--jepa_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# ----------------------
# Triplet / finetune setting
# ----------------------
parser.add_argument('--best_metric_init', type=float, default=-1e3) #用来存储best map
parser.add_argument('--triplet_margin', type=float, default=0.2)
parser.add_argument(
    '--finetune_weight_decay',
    type=float,
    default=0.01,
    help='Finetune: AdamW weight decay on CLIP visual params (prompts stay 0).',
)
parser.add_argument(
    '--use_finetune_lr_plateau',
    type=str2bool,
    default=False,
    help='If True, use AdamW + ReduceLROnPlateau on mAP (overrides cosine finetune scheduler).',
)
parser.add_argument(
    '--finetune_use_cosine_lr',
    type=str2bool,
    default=True,
    help='If True (and not plateau), AdamW + cosine epoch schedule for finetune.',
)
parser.add_argument(
    '--finetune_cosine_eta_min',
    type=float,
    default=1e-8,
    help='Finetune: cosine scheduler minimum LR.',
)
parser.add_argument(
    '--finetune_full_precision',
    type=str2bool,
    default=True,
    help='Finetune: use 32-true precision (often slightly better mAP than AMP).',
)
parser.add_argument(
    '--finetune_gradient_clip',
    type=float,
    default=1.0,
    help='Finetune: clip global grad norm; 0 disables.',
)
parser.add_argument(
    '--finetune_init_sk_from_img',
    type=str2bool,
    default=False,
    help='If True, copy img_prompt to sk_prompt after loading pretrain (usually hurts if JEPA learned separate prompts).',
)
parser.add_argument(
    '--cls_label_smoothing',
    type=float,
    default=0.05,
    help='Label smoothing for finetune CLIP-classifier CE (finetune only).',
)
parser.add_argument(
    '--pretrain_ckpt',
    type=str,
    default='',
    help='JEPA pretrain .ckpt for finetune. Empty = {save_root}/{exp_name}_pretrain/last.ckpt in train_finetune.',
)
parser.add_argument(
    '--finetune_resume',
    type=str2bool,
    default=True,
    help='If True, load save_dir/last.ckpt when present. If False, skip last.ckpt and load pretrain_ckpt.',
)
parser.add_argument(
    '--lambda_infonce',
    type=float,
    default=0.05,
    help='Finetune: symmetric supervised batch InfoNCE (sk-img, same class as positives). Set 0 to disable.',
)
parser.add_argument('--finetune_infonce_tau', type=float, default=0.07)
parser.add_argument(
    '--lambda_pretrain_sk_img_align',
    type=float,
    default=0.08,
    help='Pretrain: extra supervised InfoNCE on global centers mu_sk vs mu_img in proj space. Set 0 to disable.',
)
parser.add_argument('--pretrain_infonce_tau', type=float, default=0.07)
parser.add_argument(
    '--finetune_early_stop_patience',
    type=int,
    default=15,
    help='EarlyStopping on val mAP: stop after this many epochs without improvement.',
)
parser.add_argument('--lr_plateau_factor', type=float, default=0.5)
parser.add_argument('--lr_plateau_patience', type=int, default=8)
parser.add_argument('--lr_plateau_min_lr', type=float, default=1e-8)


# ----------------------
# CClip text setting
# ----------------------
parser.add_argument('--use_clip_cls', type=str2bool, default=True)

parser.add_argument('--cls_tau', type=float, default=0.07)
parser.add_argument('--use_clip_logit_scale', type=str2bool, default=True)

parser.add_argument(
    '--lambda_cls',
    type=float,
    default=0.15,
    help='Weight for CLIP text classifier loss in finetune (lower reduces overfit vs triplet).',
)
parser.add_argument('--lambda_jepa', type=float, default=1.0)
parser.add_argument('--lambda_reg', type=float, default=0.03)
parser.add_argument('--lambda_intra_sk', type=float, default=0.5)
parser.add_argument('--lambda_intra_ph', type=float, default=0.5)
parser.add_argument('--lambda_cross_global', type=float, default=0.3)
parser.add_argument('--lambda_cross_local2global', type=float, default=0.2)


# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

opts = parser.parse_args()
