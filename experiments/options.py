import argparse
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Fine-Grained SBIR")

# ----------------------
# 路径设置
# ----------------------
parser.add_argument(
    "--data_root",
    type=str,
    default="/home/mig/Documents/SBIR_Data/FG_shoes_chair",
    help="包含 ChairV2 / ShoeV2 的总目录",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="ChairV2",
    choices=["ChairV2", "ShoeV2"],
    help="选择使用哪个数据集",
)
parser.add_argument(
    "--save_root",
    type=str,
    default="/home/mig/Documents/SBIR_Data/saved_models/",
)
parser.add_argument("--exp_name", type=str, default="fg_sbir_tri")

# ----------------------
# 常规训练参数
# ----------------------
parser.add_argument("--max_size", type=int, default=224)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--max_epochs", type=int, default=100)

# ----------------------
# Learning rate
# ----------------------
parser.add_argument("--clip_lr", type=float, default=1e-4)
parser.add_argument("--clip_LN_lr", type=float, default=1e-6)
parser.add_argument("--prompt_lr", type=float, default=1e-4)
parser.add_argument("--linear_lr", type=float, default=1e-4)

# ----------------------
# Retrieval / Triplet
# ----------------------
parser.add_argument("--best_metric_init", type=float, default=-1e3)
parser.add_argument(
    "--neg_in_batch_only",
    type=str2bool,
    default=False,
    help="暂时保留字段；当前 dataloader 默认显式采样 neg",
)

# ----------------------
# CLIP classification
# FG-SBIR 第一版建议关闭
# ----------------------
parser.add_argument("--use_clip_cls", type=str2bool, default=True)
parser.add_argument("--cls_tau", type=float, default=0.07)
parser.add_argument("--use_clip_logit_scale", type=str2bool, default=True)
parser.add_argument("--lambda_cls", type=float, default=0.0)

# ----------------------
# ViT Prompt
# ----------------------
parser.add_argument("--prompt_dim", type=int, default=768)
parser.add_argument("--n_prompts", type=int, default=3)

opts = parser.parse_args()

# ----------------------
# 动态生成数据路径
# ----------------------
opts.data_dir = os.path.join(opts.data_root, opts.dataset)

opts.train_sk_dir = os.path.join(opts.data_dir, "train_SK")
opts.train_ph_dir = os.path.join(opts.data_dir, "train_PH")
opts.test_sk_dir = os.path.join(opts.data_dir, "test_SK")
opts.test_ph_dir = os.path.join(opts.data_dir, "test_PH")

# 先保留接口兼容
opts.seen_class_names = []
opts.cls_templates = []