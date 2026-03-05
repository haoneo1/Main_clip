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
parser.add_argument('--exp_name', type=str, default='SSL_jepa_loss')
parser.add_argument('--max_size', type=int, default=224)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--max_epochs', type=int, default=50)



# ----------------------
# Pretrain setting
# ----------------------
parser.add_argument('--jepa_hidden_dim', type=int, default=512)#pre head的隐藏层
parser.add_argument('--use_detach_jepa_target', type=str2bool, default=True)#决定 JEPA 的目标分支是否停止梯度，通常设 True 更稳
parser.add_argument('--lambda_jepa_pred', type=float, default=1.0)#决定 JEPA 预测损失的权重大小，当前一般设 1.0




parser.add_argument('--clip_lr', type=float, default=1e-4)
parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
parser.add_argument('--prompt_lr', type=float, default=1e-4)
parser.add_argument('--jepa_lr', type=float, default=1e-4)
parser.add_argument('--linear_lr', type=float, default=1e-4)
parser.add_argument('--nclass', type=int, default=10)
parser.add_argument('--data_split', type=float, default=-1.0)

# ----------------------
# Triplet loss setting
# ----------------------
parser.add_argument('--best_metric_init', type=float, default=-1e3) #用来存储best map


# ----------------------
# CClip text setting
# ----------------------
parser.add_argument('--use_clip_cls', type=str2bool, default=False)

parser.add_argument('--cls_tau', type=float, default=0.07)
parser.add_argument('--use_clip_logit_scale', type=str2bool, default=True)

#loss weight
parser.add_argument('--lambda_cls', type=float, default=0.3)


# ----------------------
# ViT Prompt Parameters
# ----------------------
parser.add_argument('--prompt_dim', type=int, default=768)
parser.add_argument('--n_prompts', type=int, default=3)

opts = parser.parse_args()
