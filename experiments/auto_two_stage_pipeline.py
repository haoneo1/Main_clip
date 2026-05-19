import argparse
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from itertools import product


BEST_CKPT_RE = re.compile(r"mAP[=\-](\d+\.\d+)")
PRETRAIN_LOSS_RE = re.compile(r"train_loss_epoch[=\-](\d+\.\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated two-stage pretrain->finetune experiment pipeline."
    )
    parser.add_argument("--hours", type=float, default=24.0, help="Total runtime budget in hours.")
    parser.add_argument(
        "--status_file",
        type=str,
        default="experiments/auto_two_stage_status.md",
        help="Live status markdown path (relative to workspace).",
    )
    parser.add_argument("--python_bin", type=str, default="python", help="Python executable to use.")
    parser.add_argument("--conda_env", type=str, default="", help="Optional conda environment.")
    parser.add_argument(
        "--save_root",
        type=str,
        default="/home/mig/Documents/SBIR_Data/saved_models",
        help="Checkpoint root directory.",
    )
    parser.add_argument("--base_exp", type=str, default="auto2s", help="Experiment name prefix.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--pretrain_runner",
        type=str,
        default="legacy",
        choices=["legacy", "ijepa"],
        help="Pretrain script to use: legacy=train_pretrain.py, ijepa=train_pretrain_ijepa.py",
    )
    parser.add_argument(
        "--grid_profile",
        type=str,
        default="main",
        choices=["smoke", "main", "explore"],
        help="Hyperparameter grid profile size.",
    )
    parser.add_argument(
        "--trial_config_json",
        type=str,
        default="",
        help="Optional JSON file path for explicit trial configs.",
    )
    parser.add_argument("--pretrain_max_epochs", type=int, default=40)
    parser.add_argument("--finetune_max_epochs", type=int, default=60)
    parser.add_argument("--finetune_patience", type=int, default=12)
    parser.add_argument("--pretrain_batch_size", type=int, default=128)
    parser.add_argument("--finetune_batch_size", type=int, default=64)
    parser.add_argument(
        "--reuse_pretrain_if_exists",
        type=str,
        default="true",
        help="Skip pretrain if last.ckpt exists for this trial.",
    )
    parser.add_argument(
        "--delete_non_best",
        type=str,
        default="false",
        help="Delete non-best trial checkpoint directories.",
    )
    parser.add_argument(
        "--stop_after_consecutive_failures",
        type=int,
        default=3,
        help="Stop search after N consecutive failed trials.",
    )
    parser.add_argument(
        "--dynamic_stage23_from_s1",
        type=str,
        default="true",
        help="If true, run Stage-1 first, auto-build Stage-2/3 from Stage-1 top configs.",
    )
    parser.add_argument(
        "--dynamic_stage1_topk",
        type=int,
        default=3,
        help="Top-K Stage-1 configs used to build dynamic Stage-2.",
    )
    return parser.parse_args()


def to_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y", "t"}


def pretrain_suffix(pretrain_runner):
    if pretrain_runner == "ijepa":
        return "ijepa_pretrain"
    return "pretrain"


def pretrain_dir_for(save_root, exp_name, pretrain_runner):
    return os.path.join(save_root, f"{exp_name}_{pretrain_suffix(pretrain_runner)}")


def build_trial_grid(profile):
    if profile == "smoke":
        jepa_hidden_dims = [512]
        lambda_jepa_preds = [1.0]
        jepa_lrs = [1e-4]
        lambda_cls_vals = [0.2]
        use_sigreg_vals = [True]
        lambda_sigreg_vals = [0.1]
        clip_ln_lrs = [1e-6]
        prompt_lrs = [5e-5]
        n_prompts_vals = [3]
        cls_tau_vals = [0.07]
    elif profile == "main":
        jepa_hidden_dims = [512, 768]
        lambda_jepa_preds = [0.5, 1.0]
        jepa_lrs = [5e-5, 1e-4]
        lambda_cls_vals = [0.1, 0.2, 0.3]
        use_sigreg_vals = [True, False]
        lambda_sigreg_vals = [0.05, 0.1, 0.2]
        clip_ln_lrs = [1e-6, 2e-6]
        prompt_lrs = [5e-5, 1e-4]
        n_prompts_vals = [3, 6]
        cls_tau_vals = [0.05, 0.07, 0.1]
    else:
        jepa_hidden_dims = [256, 512, 768]
        lambda_jepa_preds = [0.3, 0.5, 1.0]
        jepa_lrs = [3e-5, 5e-5, 1e-4]
        lambda_cls_vals = [0.0, 0.1, 0.2, 0.3]
        use_sigreg_vals = [True, False]
        lambda_sigreg_vals = [0.03, 0.1, 0.2]
        clip_ln_lrs = [5e-7, 1e-6, 2e-6]
        prompt_lrs = [2e-5, 5e-5, 1e-4]
        n_prompts_vals = [2, 3, 6]
        cls_tau_vals = [0.05, 0.07, 0.1, 0.15]

    grid = []
    for (
        jepa_hidden_dim,
        lambda_jepa_pred,
        jepa_lr,
        lambda_cls,
        use_sigreg,
        lambda_sigreg,
        clip_ln_lr,
        prompt_lr,
        n_prompts,
        cls_tau,
    ) in product(
        jepa_hidden_dims,
        lambda_jepa_preds,
        jepa_lrs,
        lambda_cls_vals,
        use_sigreg_vals,
        lambda_sigreg_vals,
        clip_ln_lrs,
        prompt_lrs,
        n_prompts_vals,
        cls_tau_vals,
    ):
        if not use_sigreg and lambda_sigreg != lambda_sigreg_vals[0]:
            continue
        if lambda_cls == 0.0 and cls_tau != cls_tau_vals[0]:
            continue
        grid.append(
            {
                "jepa_hidden_dim": jepa_hidden_dim,
                "lambda_jepa_pred": lambda_jepa_pred,
                "jepa_lr": jepa_lr,
                "lambda_cls": lambda_cls,
                "use_sigreg": use_sigreg,
                "lambda_sigreg": lambda_sigreg,
                "clip_LN_lr": clip_ln_lr,
                "prompt_lr": prompt_lr,
                "n_prompts": n_prompts,
                "cls_tau": cls_tau,
                "use_clip_cls": lambda_cls > 0.0,
            }
        )
    return grid


def load_trial_grid_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("trial_config_json must be a JSON array of config objects.")

    normalized = []
    required = [
        "jepa_hidden_dim",
        "lambda_jepa_pred",
        "jepa_lr",
        "lambda_cls",
        "use_sigreg",
        "lambda_sigreg",
        "clip_LN_lr",
        "prompt_lr",
        "n_prompts",
        "cls_tau",
    ]
    for i, cfg in enumerate(data, start=1):
        if not isinstance(cfg, dict):
            raise ValueError(f"trial_config_json item #{i} is not an object.")
        for k in required:
            if k not in cfg:
                raise ValueError(f"trial_config_json item #{i} missing key: {k}")
        cfg = cfg.copy()
        cfg["use_clip_cls"] = bool(cfg.get("use_clip_cls", float(cfg["lambda_cls"]) > 0.0))
        normalized.append(cfg)
    return normalized


def split_stage1_and_rest(grid):
    stage1, rest = [], []
    for cfg in grid:
        stage = str(cfg.get("stage", "")).lower()
        if stage in {"s1", "s1_coarse", "stage1", "coarse"}:
            stage1.append(cfg)
        else:
            rest.append(cfg)
    return stage1, rest


def rank_pretrain_records(records):
    ranked = []
    for rec in records:
        if rec.get("pretrain_loss") is None:
            continue
        map_score = rec.get("mAP")
        if map_score is None:
            map_score = -1.0
        ranked.append(
            (
                float(rec["pretrain_loss"]),
                -float(map_score),
                rec["cfg"],
            )
        )
    ranked.sort(key=lambda x: (x[0], x[1]))
    return [cfg for _, _, cfg in ranked]


def make_stage2_from_topk(topk_cfgs):
    # Controlled budget: 2 structure probes per Stage-1 top config.
    probes = [(2, 0.998), (3, 0.999)]
    out = []
    for idx, base in enumerate(topk_cfgs, start=1):
        for pred_depth, momentum in probes:
            cfg = base.copy()
            cfg["stage"] = "s2_structure"
            cfg["anchor"] = f"top{idx}"
            cfg["ijepa_pred_depth"] = int(pred_depth)
            cfg["ijepa_momentum"] = float(momentum)
            out.append(cfg)
    return out


def make_stage3_from_best(best_cfg):
    base = best_cfg.copy()
    best_lr = float(base.get("ijepa_lr", base.get("jepa_lr", 1e-4)))
    best_keep = float(base.get("ijepa_context_keep_ratio", 0.5))
    keep_lo = max(0.1, best_keep - 0.05)
    keep_hi = min(0.9, best_keep + 0.05)
    probes = [
        (0.7 * best_lr, keep_lo),
        (1.0 * best_lr, keep_lo),
        (1.0 * best_lr, keep_hi),
        (1.3 * best_lr, best_keep),
    ]
    out = []
    for lr, keep in probes:
        cfg = base.copy()
        cfg["stage"] = "s3_refine"
        cfg["ijepa_lr"] = float(lr)
        cfg["jepa_lr"] = float(lr)
        cfg["ijepa_context_keep_ratio"] = float(keep)
        out.append(cfg)
    return out


def cfg_key(cfg):
    ijepa_lr = float(cfg.get("ijepa_lr", cfg["jepa_lr"]))
    ijepa_target_scale_min = float(cfg.get("ijepa_target_scale_min", 0.15))
    ijepa_target_scale_max = float(cfg.get("ijepa_target_scale_max", 0.25))
    ijepa_context_keep_ratio = float(cfg.get("ijepa_context_keep_ratio", 0.5))
    ijepa_pred_depth = int(cfg.get("ijepa_pred_depth", 2))
    ijepa_pred_heads = int(cfg.get("ijepa_pred_heads", 8))
    ijepa_pred_mlp_ratio = float(cfg.get("ijepa_pred_mlp_ratio", 4.0))
    ijepa_momentum = float(cfg.get("ijepa_momentum", 0.996))
    ijepa_num_targets = int(cfg.get("ijepa_num_targets", 4))
    ijepa_backbone_arch = str(cfg.get("ijepa_backbone_arch", "vit_base_patch16_224"))
    return (
        int(cfg["jepa_hidden_dim"]),
        round(float(cfg["lambda_jepa_pred"]), 8),
        round(float(cfg["jepa_lr"]), 12),
        round(float(cfg["lambda_cls"]), 8),
        bool(cfg["use_sigreg"]),
        round(float(cfg["lambda_sigreg"]), 8),
        round(float(cfg["clip_LN_lr"]), 12),
        round(float(cfg["prompt_lr"]), 12),
        int(cfg["n_prompts"]),
        round(float(cfg["cls_tau"]), 8),
        bool(cfg["use_clip_cls"]),
        round(ijepa_lr, 12),
        round(ijepa_target_scale_min, 8),
        round(ijepa_target_scale_max, 8),
        round(ijepa_context_keep_ratio, 8),
        ijepa_pred_depth,
        ijepa_pred_heads,
        round(ijepa_pred_mlp_ratio, 8),
        round(ijepa_momentum, 8),
        ijepa_num_targets,
        ijepa_backbone_arch,
    )


def find_best_map_in_dir(exp_dir):
    if not os.path.isdir(exp_dir):
        return None
    best = None
    for name in os.listdir(exp_dir):
        match = BEST_CKPT_RE.search(name)
        if not match:
            continue
        score = float(match.group(1))
        if best is None or score > best:
            best = score
    return best


def find_best_pretrain_loss_in_dir(exp_dir):
    if not os.path.isdir(exp_dir):
        return None
    best = None
    for name in os.listdir(exp_dir):
        match = PRETRAIN_LOSS_RE.search(name)
        if not match:
            continue
        score = float(match.group(1))
        if best is None or score < best:
            best = score
    return best


def compose_runner(args):
    if args.conda_env:
        return ["conda", "run", "-n", args.conda_env, "python"]
    return [args.python_bin]


def run_subprocess(cmd, workspace_root):
    env = os.environ.copy()
    env["PYTHONPATH"] = workspace_root
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return subprocess.run(cmd, cwd=workspace_root, env=env, check=False)


def run_pretrain(args, trial_id, exp_name, cfg, workspace_root):
    runner = compose_runner(args)
    save_dir = pretrain_dir_for(args.save_root, exp_name, args.pretrain_runner)
    last_ckpt = os.path.join(save_dir, "last.ckpt")
    if to_bool(args.reuse_pretrain_if_exists) and os.path.exists(last_ckpt):
        pretrain_loss = find_best_pretrain_loss_in_dir(save_dir)
        print(f"[Trial {trial_id}] reuse existing pretrain ckpt: {last_ckpt}")
        return 0, last_ckpt, True, pretrain_loss

    if args.pretrain_runner == "ijepa":
        ijepa_lr = float(cfg.get("ijepa_lr", cfg["jepa_lr"]))
        ijepa_target_scale_min = float(cfg.get("ijepa_target_scale_min", 0.15))
        ijepa_target_scale_max = float(cfg.get("ijepa_target_scale_max", 0.25))
        ijepa_context_keep_ratio = float(cfg.get("ijepa_context_keep_ratio", 0.5))
        ijepa_pred_depth = int(cfg.get("ijepa_pred_depth", 2))
        ijepa_pred_heads = int(cfg.get("ijepa_pred_heads", 8))
        ijepa_pred_mlp_ratio = float(cfg.get("ijepa_pred_mlp_ratio", 4.0))
        ijepa_momentum = float(cfg.get("ijepa_momentum", 0.996))
        ijepa_num_targets = int(cfg.get("ijepa_num_targets", 4))
        ijepa_backbone_arch = str(cfg.get("ijepa_backbone_arch", "vit_base_patch16_224"))
        cmd = runner + [
            "experiments/train_pretrain_ijepa.py",
            "--exp_name",
            exp_name,
            "--seed",
            str(args.seed + trial_id),
            "--max_epochs",
            str(args.pretrain_max_epochs),
            "--batch_size",
            str(args.pretrain_batch_size),
            "--ijepa_lr",
            str(ijepa_lr),
            "--ijepa_target_scale_min",
            str(ijepa_target_scale_min),
            "--ijepa_target_scale_max",
            str(ijepa_target_scale_max),
            "--ijepa_context_keep_ratio",
            str(ijepa_context_keep_ratio),
            "--ijepa_pred_depth",
            str(ijepa_pred_depth),
            "--ijepa_pred_heads",
            str(ijepa_pred_heads),
            "--ijepa_pred_mlp_ratio",
            str(ijepa_pred_mlp_ratio),
            "--ijepa_momentum",
            str(ijepa_momentum),
            "--ijepa_num_targets",
            str(ijepa_num_targets),
            "--ijepa_backbone_arch",
            ijepa_backbone_arch,
        ]
    else:
        cmd = runner + [
            "experiments/train_pretrain.py",
            "--exp_name",
            exp_name,
            "--seed",
            str(args.seed + trial_id),
            "--max_epochs",
            str(args.pretrain_max_epochs),
            "--batch_size",
            str(args.pretrain_batch_size),
            "--jepa_hidden_dim",
            str(cfg["jepa_hidden_dim"]),
            "--lambda_jepa_pred",
            str(cfg["lambda_jepa_pred"]),
            "--jepa_lr",
            str(cfg["jepa_lr"]),
        ]
    print(f"[Trial {trial_id}] PRETRAIN CMD: {' '.join(cmd)}")
    proc = run_subprocess(cmd, workspace_root)
    pretrain_loss = find_best_pretrain_loss_in_dir(save_dir)
    return proc.returncode, last_ckpt, False, pretrain_loss


def run_finetune(args, trial_id, exp_name, cfg, finetune_init_ckpt, workspace_root):
    runner = compose_runner(args)
    cmd = runner + [
        "experiments/train_finetune.py",
        "--exp_name",
        exp_name,
        "--seed",
        str(args.seed + trial_id),
        "--finetune_init_ckpt",
        finetune_init_ckpt,
        "--max_epochs",
        str(args.finetune_max_epochs),
        "--early_stop_patience",
        str(args.finetune_patience),
        "--batch_size",
        str(args.finetune_batch_size),
        "--use_clip_cls",
        str(cfg["use_clip_cls"]),
        "--lambda_cls",
        str(cfg["lambda_cls"]),
        "--use_sigreg",
        str(cfg["use_sigreg"]),
        "--lambda_sigreg",
        str(cfg["lambda_sigreg"]),
        "--clip_LN_lr",
        str(cfg["clip_LN_lr"]),
        "--prompt_lr",
        str(cfg["prompt_lr"]),
        "--n_prompts",
        str(cfg["n_prompts"]),
        "--cls_tau",
        str(cfg["cls_tau"]),
    ]
    print(f"[Trial {trial_id}] FINETUNE CMD: {' '.join(cmd)}")
    proc = run_subprocess(cmd, workspace_root)
    exp_dir = os.path.join(args.save_root, f"{exp_name}_finetune")
    best_map = find_best_map_in_dir(exp_dir)
    return proc.returncode, best_map


def write_status_snapshot(path, now_ts, trial_id, phase, best, best_pretrain, last_result, grid_profile, total_trials, pretrain_runner):
    lines = [
        "# Auto Two-Stage Pipeline Status",
        "",
        f"- Updated at: `{now_ts}`",
        f"- Last trial id: `{trial_id}`",
        f"- Last phase: `{phase}`",
        f"- Grid profile: `{grid_profile}`",
        f"- Pretrain runner: `{pretrain_runner}`",
        f"- Total candidate configs: `{total_trials}`",
        "",
        "## Last Result",
        f"- exp_name: `{last_result.get('exp_name')}`",
        f"- status: `{last_result.get('status')}`",
        f"- pretrain_rc: `{last_result.get('pretrain_rc')}`",
        f"- pretrain_loss: `{last_result.get('pretrain_loss')}`",
        f"- pretrain_ckpt_dir: `{last_result.get('pretrain_ckpt_dir')}`",
        f"- finetune_rc: `{last_result.get('finetune_rc')}`",
        f"- mAP: `{last_result.get('mAP')}`",
        f"- cfg: `{last_result.get('cfg')}`",
        "",
        "## Best Finetune So Far",
        f"- trial_id: `{best.get('trial_id')}`",
        f"- exp_name: `{best.get('exp_name')}`",
        f"- mAP: `{best.get('mAP')}`",
        f"- cfg: `{best.get('cfg')}`",
        "",
        "## Best Pretrain So Far",
        f"- trial_id: `{best_pretrain.get('trial_id')}`",
        f"- exp_name: `{best_pretrain.get('exp_name')}`",
        f"- train_loss_epoch: `{best_pretrain.get('pretrain_loss')}`",
        f"- cfg: `{best_pretrain.get('cfg')}`",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def cleanup_non_best(args, exp_name):
    pretrain_dir = pretrain_dir_for(args.save_root, exp_name, args.pretrain_runner)
    finetune_dir = os.path.join(args.save_root, f"{exp_name}_finetune")
    shutil.rmtree(pretrain_dir, ignore_errors=True)
    shutil.rmtree(finetune_dir, ignore_errors=True)


def main():
    args = parse_args()
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    status_path = os.path.join(workspace_root, args.status_file)
    os.makedirs(os.path.dirname(status_path), exist_ok=True)

    if args.trial_config_json:
        grid = load_trial_grid_from_json(args.trial_config_json)
        print(f"Grid source: json ({args.trial_config_json})")
    else:
        grid = build_trial_grid(args.grid_profile)
        print(f"Grid profile: {args.grid_profile}")
    print(f"Total candidate configs: {len(grid)}")
    dynamic_stage23 = to_bool(args.dynamic_stage23_from_s1)
    stage1_grid, non_stage1_grid = split_stage1_and_rest(grid)
    use_dynamic_staging = bool(stage1_grid) and dynamic_stage23
    if use_dynamic_staging:
        print(
            f"Dynamic staging enabled: Stage-1={len(stage1_grid)}, "
            f"prelisted non-Stage1={len(non_stage1_grid)} (ignored for auto Stage-2/3)."
        )
    elif dynamic_stage23 and not stage1_grid:
        print("Dynamic staging requested but no Stage-1 labels found; fallback to linear grid.")
    if args.hours <= 0:
        print("Time budget: infinite (hours <= 0)")
        deadline = None
    else:
        print(f"Time budget: {args.hours} hours")
        deadline = time.time() + int(args.hours * 3600)

    best = {"trial_id": None, "exp_name": None, "mAP": -1.0, "cfg": None}
    best_pretrain = {"trial_id": None, "exp_name": None, "pretrain_loss": None, "cfg": None}
    seen_keys = set()
    consecutive_failures = 0
    delete_non_best = to_bool(args.delete_non_best)

    planned_total_trials = len(stage1_grid) if use_dynamic_staging else len(grid)
    write_status_snapshot(
        path=status_path,
        now_ts=datetime.now().isoformat(timespec="seconds"),
        trial_id=0,
        phase="init",
        best=best,
        best_pretrain=best_pretrain,
        last_result={
            "exp_name": None,
            "status": "init",
            "pretrain_rc": None,
            "pretrain_loss": None,
            "pretrain_ckpt_dir": None,
            "finetune_rc": None,
            "mAP": None,
            "cfg": None,
        },
        grid_profile=args.grid_profile,
        total_trials=planned_total_trials,
        pretrain_runner=args.pretrain_runner,
    )

    trial_id = 1
    stage_queue = []
    if use_dynamic_staging:
        stage_queue = [("s1_coarse", stage1_grid)]
    else:
        stage_queue = [("all", grid)]

    queue_idx = 0
    stop_now = False
    while queue_idx < len(stage_queue) and not stop_now:
        stage_name, stage_grid = stage_queue[queue_idx]
        print(f"\n==== RUN STAGE: {stage_name} ({len(stage_grid)} configs) ====")
        stage_records = []
        for cfg in stage_grid:
            if deadline is not None and time.time() >= deadline:
                print("[STOP] Time budget reached.")
                stop_now = True
                break

            key = cfg_key(cfg)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            exp_name = f"{args.base_exp}_t{trial_id:03d}"
            pretrain_ckpt_dir = pretrain_dir_for(args.save_root, exp_name, args.pretrain_runner)
            now_ts = datetime.now().isoformat(timespec="seconds")
            write_status_snapshot(
                path=status_path,
                now_ts=now_ts,
                trial_id=trial_id,
                phase="pretrain",
                best=best,
                best_pretrain=best_pretrain,
                last_result={
                    "exp_name": exp_name,
                    "status": "running_pretrain",
                    "pretrain_rc": None,
                    "pretrain_loss": None,
                    "pretrain_ckpt_dir": pretrain_ckpt_dir,
                    "finetune_rc": None,
                    "mAP": None,
                    "cfg": cfg,
                },
                grid_profile=args.grid_profile,
                total_trials=planned_total_trials,
                pretrain_runner=args.pretrain_runner,
            )

            pretrain_rc, pretrain_ckpt, reused, pretrain_loss = run_pretrain(args, trial_id, exp_name, cfg, workspace_root)
            if pretrain_loss is not None and (
                best_pretrain["pretrain_loss"] is None or pretrain_loss < best_pretrain["pretrain_loss"]
            ):
                best_pretrain = {
                    "trial_id": trial_id,
                    "exp_name": exp_name,
                    "pretrain_loss": pretrain_loss,
                    "cfg": cfg,
                }
                print(f"[NEW BEST PRETRAIN] trial={trial_id} exp={exp_name} loss={pretrain_loss:.4f}")

            if pretrain_rc != 0:
                consecutive_failures += 1
                print(f"[Trial {trial_id}] pretrain failed, rc={pretrain_rc}")
                write_status_snapshot(
                    path=status_path,
                    now_ts=datetime.now().isoformat(timespec="seconds"),
                    trial_id=trial_id,
                    phase="pretrain_failed",
                    best=best,
                    best_pretrain=best_pretrain,
                    last_result={
                        "exp_name": exp_name,
                        "status": "failed_pretrain",
                        "pretrain_rc": pretrain_rc,
                        "pretrain_loss": pretrain_loss,
                        "pretrain_ckpt_dir": pretrain_ckpt_dir,
                        "finetune_rc": None,
                        "mAP": None,
                        "cfg": cfg,
                    },
                    grid_profile=args.grid_profile,
                    total_trials=planned_total_trials,
                    pretrain_runner=args.pretrain_runner,
                )
                if consecutive_failures >= args.stop_after_consecutive_failures:
                    print(f"[STOP] {consecutive_failures} consecutive failures.")
                    stop_now = True
                    break
                trial_id += 1
                continue

            if not os.path.exists(pretrain_ckpt):
                consecutive_failures += 1
                print(f"[Trial {trial_id}] pretrain ckpt missing: {pretrain_ckpt}")
                if consecutive_failures >= args.stop_after_consecutive_failures:
                    print(f"[STOP] {consecutive_failures} consecutive failures.")
                    stop_now = True
                    break
                trial_id += 1
                continue

            now_ts = datetime.now().isoformat(timespec="seconds")
            write_status_snapshot(
                path=status_path,
                now_ts=now_ts,
                trial_id=trial_id,
                phase="finetune",
                best=best,
                best_pretrain=best_pretrain,
                last_result={
                    "exp_name": exp_name,
                    "status": "running_finetune",
                    "pretrain_rc": pretrain_rc,
                    "pretrain_loss": pretrain_loss,
                    "pretrain_ckpt_dir": pretrain_ckpt_dir,
                    "finetune_rc": None,
                    "mAP": None,
                    "cfg": {**cfg, "reused_pretrain": reused},
                },
                grid_profile=args.grid_profile,
                total_trials=planned_total_trials,
                pretrain_runner=args.pretrain_runner,
            )

            finetune_rc, score = run_finetune(args, trial_id, exp_name, cfg, pretrain_ckpt, workspace_root)
            if finetune_rc != 0 and score is None:
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            improved = score is not None and score > best["mAP"]
            if improved:
                best = {"trial_id": trial_id, "exp_name": exp_name, "mAP": score, "cfg": cfg}
                print(f"[NEW BEST] trial={trial_id} exp={exp_name} mAP={score:.4f}")
            elif delete_non_best:
                cleanup_non_best(args, exp_name)
                print(f"[CLEANUP] removed non-best trial dirs for {exp_name}")

            write_status_snapshot(
                path=status_path,
                now_ts=datetime.now().isoformat(timespec="seconds"),
                trial_id=trial_id,
                phase="finished",
                best=best,
                best_pretrain=best_pretrain,
                last_result={
                    "exp_name": exp_name,
                    "status": "finished",
                    "pretrain_rc": pretrain_rc,
                    "pretrain_loss": pretrain_loss,
                    "pretrain_ckpt_dir": pretrain_ckpt_dir,
                    "finetune_rc": finetune_rc,
                    "mAP": score,
                    "cfg": {**cfg, "reused_pretrain": reused},
                },
                grid_profile=args.grid_profile,
                total_trials=planned_total_trials,
                pretrain_runner=args.pretrain_runner,
            )
            stage_records.append(
                {
                    "trial_id": trial_id,
                    "cfg": cfg,
                    "pretrain_loss": pretrain_loss,
                    "mAP": score,
                }
            )

            if consecutive_failures >= args.stop_after_consecutive_failures:
                print(f"[STOP] {consecutive_failures} consecutive failures.")
                stop_now = True
                break
            trial_id += 1

        if stop_now:
            break

        if use_dynamic_staging and stage_name == "s1_coarse":
            ranked_stage1 = rank_pretrain_records(stage_records)
            topk = max(1, int(args.dynamic_stage1_topk))
            top_cfgs = ranked_stage1[:topk]
            if not top_cfgs:
                print("[STOP] Stage-1 produced no valid pretrain scores; cannot build Stage-2/3.")
                break
            stage2_grid = make_stage2_from_topk(top_cfgs)
            planned_total_trials += len(stage2_grid)
            stage_queue.append(("s2_structure", stage2_grid))
            print(f"[Dynamic] Stage-2 generated from Stage-1 top-{len(top_cfgs)}: {len(stage2_grid)} configs")

        elif use_dynamic_staging and stage_name == "s2_structure":
            ranked_stage2 = rank_pretrain_records(stage_records)
            if not ranked_stage2:
                print("[STOP] Stage-2 produced no valid pretrain scores; cannot build Stage-3.")
                break
            best_stage2_cfg = ranked_stage2[0]
            stage3_grid = make_stage3_from_best(best_stage2_cfg)
            planned_total_trials += len(stage3_grid)
            stage_queue.append(("s3_refine", stage3_grid))
            print(f"[Dynamic] Stage-3 generated around Stage-2 best: {len(stage3_grid)} configs")

        queue_idx += 1

    print("\n==== TWO-STAGE SEARCH FINISHED ====")
    if best["trial_id"] is not None:
        print(f"Best trial: {best['trial_id']}")
        print(f"Best exp: {best['exp_name']}")
        print(f"Best mAP: {best['mAP']:.4f}")
        print(f"Best cfg: {best['cfg']}")
    else:
        print("No valid mAP found.")


if __name__ == "__main__":
    main()
