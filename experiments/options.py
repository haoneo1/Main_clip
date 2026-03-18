import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1', 'y'):
        return True
    if v.lower() in ('no', 'false', 'f', '0', 'n'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def build_parser():
    parser = argparse.ArgumentParser(description='Sketch-based OD')

    # ======================
    # Training Stage
    # ======================
    parser.add_argument(
        '--train_stage',
        type=str,
        default='pretrain',
        choices=['pretrain', 'finetune'],
        help='Training stage: pretrain or finetune'
    )

    # ======================
    # General Settings
    # ======================
    parser.add_argument('--data_dir', type=str, default='/home/mig/Documents/SBIR_Data/Sketchy/')
    # parser.add_argument('--data_dir', type=str, default='/home/bingxing2/home/scx9951/SBIR_Data/Sketchy/')
    parser.add_argument('--exp_name', type=str, default='SSL_cross-ph+sk_dino+jepa')
    parser.add_argument('--max_size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--data_split', type=float, default=-1.0)
    parser.add_argument('--nclass', type=int, default=10)

    # ======================
    # Optimizer / LR Settings
    # ======================
    parser.add_argument('--clip_lr', type=float, default=1e-4)
    parser.add_argument('--clip_LN_lr', type=float, default=1e-6)
    parser.add_argument('--prompt_lr', type=float, default=1e-4)
    parser.add_argument('--jepa_lr', type=float, default=1e-4)
    parser.add_argument('--linear_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # ======================
    # Pretrain / JEPA-like Settings
    # ======================
    parser.add_argument('--jepa_hidden_dim', type=int, default=1024, help='Hidden dim of projector MLP')
    parser.add_argument('--jepa_proj_dim', type=int, default=512, help='Output dim of projector')
    parser.add_argument('--jepa_pred_hidden_dim', type=int, default=1024, help='Hidden dim of predictor MLP')

    parser.add_argument('--use_detach_jepa_target', type=str2bool, default=True,
                        help='Whether to stop gradient on JEPA target branch')
    parser.add_argument('--lambda_jepa_pred', type=float, default=1.0,
                        help='Weight of JEPA prediction loss')

    parser.add_argument('--lambda_dino_global', type=float, default=1.0,
                        help='Weight of cross-modal global DINO-like loss')
    parser.add_argument('--lambda_dino_intra', type=float, default=0.5,
                        help='Weight of intra-modal multi-view consistency loss')
    parser.add_argument('--lambda_jepa_local', type=float, default=1.0,
                        help='Weight of cross-modal local JEPA-like loss')
    parser.add_argument('--lambda_reg', type=float, default=0.05,
                        help='Weight of variance/covariance regularization')

    parser.add_argument('--teacher_momentum', type=float, default=0.996,
                        help='EMA momentum for teacher update')

    # ======================
    # Triplet / Retrieval Settings
    # ======================
    parser.add_argument('--best_metric_init', type=float, default=-1e3,
                        help='Initial value for best retrieval metric')
    parser.add_argument('--triplet_margin', type=float, default=0.2,
                        help='Margin for triplet loss')

    # ======================
    # CLIP Text / Classification Settings
    # ======================
    parser.add_argument('--use_clip_cls', type=str2bool, default=True)
    parser.add_argument('--cls_tau', type=float, default=0.07)
    parser.add_argument('--use_clip_logit_scale', type=str2bool, default=True)
    parser.add_argument('--lambda_cls', type=float, default=0.3)

    # ======================
    # ViT Prompt Settings
    # ======================
    parser.add_argument('--prompt_dim', type=int, default=768)
    parser.add_argument('--n_prompts', type=int, default=3)

    return parser


parser = build_parser()
opts = parser.parse_args()