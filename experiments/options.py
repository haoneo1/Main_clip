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

# ----------------------
# 常规设置
# ----------------------
parser.add_argument('--data_dir', type=str, default='/home/mig/Documents/SBIR_Data/Sketchy/')
parser.add_argument('--exp_name', type=str, default='clip_triplet_cls')
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--max_epochs', type=int, default=50)

# ----------------------
# Learning rate
# ----------------------
parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)

# ----------------------
# Dataset / split
# ----------------------
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# ----------------------
# Triplet loss setting
# ----------------------
parser.add_argument('--best_metric_init', type=float, default=-1e3)

# ----------------------
# CLIP text classification loss
# ----------------------
parser.add_argument('--use_clip_cls', type=str2bool, default=True)
parser.add_argument('--cls_tau', type=float, default=0.07)
parser.add_argument('--use_clip_logit_scale', type=str2bool, default=True)
parser.add_argument('--lambda_cls', type=float, default=0.3)

# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

opts = parser.parse_args()

# -------------------------------------------------
# 运行时动态补充
# -------------------------------------------------
# 训练脚本里会根据 train_dataset 自动写入
opts.seen_class_names = []

# 论文里更贴近的是 photo template
opts.cls_templates = [
    "a photo of a {}",
    "a photo of the {}",
]