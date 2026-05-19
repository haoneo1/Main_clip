import argparse


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if value.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="Sketch-based OD")

    # ----------------------
    # Training stage
    # ----------------------
    parser.add_argument(
        "--train_stage",
        type=str,
        default="pretrain",
        choices=["pretrain", "finetune"],
    )

    # ----------------------
    # General config
    # ----------------------
    parser.add_argument("--data_dir", type=str, default="/home/mig/Documents/SBIR_Data/Sketchy/")
    parser.add_argument("--exp_name", type=str, default="SSL_jepa_loss")
    parser.add_argument("--max_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--save_last_ckpt", type=str2bool, default=True)
    parser.add_argument("--finetune_init_ckpt", type=str, default="")
    parser.add_argument("--data_split", type=float, default=-1.0)

    # ----------------------
    # JEPA pretrain
    # ----------------------
    parser.add_argument("--jepa_hidden_dim", type=int, default=512)
    parser.add_argument("--use_detach_jepa_target", type=str2bool, default=True)
    parser.add_argument("--lambda_jepa_pred", type=float, default=1.0)
    parser.add_argument("--pretrain_resume_ckpt", type=str, default="")
    parser.add_argument("--pretrain_save_last_ckpt", type=str2bool, default=True)
    parser.add_argument("--pretrain_accelerator", type=str, default="gpu")
    parser.add_argument("--pretrain_devices", type=int, default=1)
    parser.add_argument("--pretrain_log_every_n_steps", type=int, default=100)
    parser.add_argument("--ijepa_backbone_arch", type=str, default="vit_base_patch16_224")
    parser.add_argument("--ijepa_num_targets", type=int, default=4)
    parser.add_argument("--ijepa_target_scale_min", type=float, default=0.15)
    parser.add_argument("--ijepa_target_scale_max", type=float, default=0.25)
    parser.add_argument("--ijepa_context_keep_ratio", type=float, default=0.5)
    parser.add_argument("--ijepa_pred_depth", type=int, default=2)
    parser.add_argument("--ijepa_pred_heads", type=int, default=8)
    parser.add_argument("--ijepa_pred_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--ijepa_momentum", type=float, default=0.996)
    parser.add_argument("--ijepa_lr", type=float, default=1e-4)
    parser.add_argument("--ijepa_weight_decay", type=float, default=0.04)

    # ----------------------
    # Learning rates------------
    # ----------------------
    parser.add_argument("--clip_lr", type=float, default=1e-4)
    parser.add_argument("--clip_LN_lr", type=float, default=1e-6)
    parser.add_argument("--prompt_lr", type=float, default=1e-4)
    parser.add_argument("--jepa_lr", type=float, default=1e-4)
    parser.add_argument("--linear_lr", type=float, default=1e-4)

    # ----------------------
    # Misc training knobs
    # ----------------------
    parser.add_argument("--nclass", type=int, default=10)
    parser.add_argument("--best_metric_init", type=float, default=-1e3)

    # ----------------------
    # CLIP text classification
    # ----------------------
    parser.add_argument("--use_clip_cls", type=str2bool, default=True)
    parser.add_argument("--cls_tau", type=float, default=0.07)
    parser.add_argument("--use_clip_logit_scale", type=str2bool, default=True)
    parser.add_argument("--lambda_cls", type=float, default=0.3)

    # ----------------------
    # SigReg (finetune)
    # ----------------------
    parser.add_argument("--use_sigreg", type=str2bool, default=True)
    parser.add_argument("--lambda_sigreg", type=float, default=0.1)
    parser.add_argument("--sigreg_gamma", type=float, default=1.0)
    parser.add_argument("--sigreg_eps", type=float, default=1e-4)

    # ----------------------
    # Prompt config
    # ----------------------
    parser.add_argument("--prompt_dim", type=int, default=768)
    parser.add_argument("--n_prompts", type=int, default=3)

    # ----------------------
    # Frozen I-JEPA baseline
    # ----------------------
    parser.add_argument("--frozen_backbone_arch", type=str, default="ijepa_vit_h14_224_in1k")
    parser.add_argument("--frozen_backbone_ckpt", type=str, default="")
    parser.add_argument(
        "--strict_backbone_load",
        type=str2bool,
        default=True,
        help="Require zero missing/unexpected keys when loading frozen backbone ckpt.",
    )
    parser.add_argument(
        "--ijepa_encoder_key",
        type=str,
        default="auto",
        help="Checkpoint sub-key to load backbone weights from (e.g., auto/target_encoder/encoder/state_dict).",
    )
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--proj_dim", type=int, default=512)
    parser.add_argument("--frozen_lr", type=float, default=1e-4)
    parser.add_argument("--frozen_weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)
    return parser


opts = build_parser().parse_args()
