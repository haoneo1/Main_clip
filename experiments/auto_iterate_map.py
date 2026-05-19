import argparse
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from itertools import product


BEST_CKPT_RE = re.compile(r"mAP[=\-](\d+\.\d+)")


def parse_args():
    parser = argparse.ArgumentParser(description="Automated mAP-oriented finetune iteration.")
    parser.add_argument("--hours", type=float, default=10.0, help="Total runtime budget in hours.")
    parser.add_argument(
        "--status_file",
        type=str,
        default="experiments/auto_status_latest.md",
        help="Path (relative to workspace) for live status snapshot.",
    )
    parser.add_argument("--python_bin", type=str, default="python", help="Python executable to use.")
    parser.add_argument("--conda_env", type=str, default="", help="Optional conda env name for training.")
    parser.add_argument(
        "--save_root",
        type=str,
        default="/home/mig/Documents/SBIR_Data/saved_models",
        help="Root dir where train_finetune saves checkpoints.",
    )
    parser.add_argument("--base_exp", type=str, default="auto_jepa", help="Prefix for experiment names.")
    parser.add_argument(
        "--grid_profile",
        type=str,
        default="main",
        choices=["main", "explore"],
        help="Hyperparameter grid profile. Use 'main' for stable baseline sweep, 'explore' for broader search.",
    )
    parser.add_argument("--max_epochs", type=int, default=40, help="Epochs per trial.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience per trial.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for finetune trials.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delete_non_best", type=str, default="true", help="Whether to delete non-best trial ckpts.")
    parser.add_argument(
        "--local_patience",
        type=int,
        default=6,
        help="Max consecutive local-refine trials without improvement before returning to global sweep.",
    )
    parser.add_argument(
        "--local_refine_start_map",
        type=float,
        default=0.73,
        help="Enable local refine only after best mAP reaches this threshold.",
    )
    return parser.parse_args()


def build_trial_grid(profile):
    # Main grid: stable, high-ROI sweep.
    if profile == "main":
        lambda_cls_vals = [0.2, 0.3, 0.5]
        use_sigreg_vals = [True, False]
        lambda_sigreg_vals = [0.05, 0.1, 0.2]
        clip_ln_lrs = [5e-7, 1e-6, 2e-6]
        prompt_lrs = [5e-5, 1e-4]
        n_prompts_vals = [3, 6]
        cls_tau_vals = [0.05, 0.07, 0.1]
    else:
        # Explore grid: broader search for escaping local optima.
        lambda_cls_vals = [0.1, 0.2, 0.4, 0.6]
        use_sigreg_vals = [True, False]
        lambda_sigreg_vals = [0.03, 0.1, 0.2, 0.3]
        clip_ln_lrs = [2e-7, 5e-7, 1e-6, 2e-6]
        prompt_lrs = [2e-5, 5e-5, 1e-4, 2e-4]
        n_prompts_vals = [2, 4, 6, 8]
        cls_tau_vals = [0.03, 0.05, 0.07, 0.1, 0.15]

    trial_grid = []
    for (lambda_cls, use_sigreg, lambda_sigreg, clip_ln_lr, prompt_lr, n_prompts, cls_tau) in product(
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
        trial_grid.append(
            {
                "lambda_cls": lambda_cls,
                "use_sigreg": use_sigreg,
                "lambda_sigreg": lambda_sigreg,
                "clip_LN_lr": clip_ln_lr,
                "prompt_lr": prompt_lr,
                "n_prompts": n_prompts,
                "cls_tau": cls_tau,
            }
        )
    return trial_grid


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


def to_bool(value):
    return str(value).lower() in {"1", "true", "yes", "y", "t"}


def cfg_key(cfg):
    return (
        round(float(cfg["lambda_cls"]), 8),
        bool(cfg["use_sigreg"]),
        round(float(cfg["lambda_sigreg"]), 8),
        round(float(cfg["clip_LN_lr"]), 12),
        round(float(cfg["prompt_lr"]), 12),
        int(cfg["n_prompts"]),
        round(float(cfg["cls_tau"]), 8),
    )


def clip_cfg_bounds(cfg):
    cfg = cfg.copy()
    cfg["lambda_cls"] = float(min(0.8, max(0.0, cfg["lambda_cls"])))
    cfg["lambda_sigreg"] = float(min(0.4, max(0.0, cfg["lambda_sigreg"])))
    cfg["clip_LN_lr"] = float(min(5e-6, max(2e-7, cfg["clip_LN_lr"])))
    cfg["prompt_lr"] = float(min(3e-4, max(2e-5, cfg["prompt_lr"])))
    cfg["n_prompts"] = int(max(1, min(8, cfg["n_prompts"])))
    cfg["cls_tau"] = float(min(0.2, max(0.03, cfg["cls_tau"])))
    return cfg


def build_local_refine_candidates(best_cfg):
    """Generate a small neighborhood around current best config."""
    base = best_cfg.copy()
    candidates = []

    def add(**updates):
        cand = base.copy()
        cand.update(updates)
        candidates.append(clip_cfg_bounds(cand))

    # Loss-weight neighborhood
    add(lambda_cls=base["lambda_cls"] * 0.8)
    add(lambda_cls=base["lambda_cls"] * 1.2)
    if base["use_sigreg"]:
        add(lambda_sigreg=base["lambda_sigreg"] * 0.7)
        add(lambda_sigreg=base["lambda_sigreg"] * 1.3)
    else:
        add(use_sigreg=True, lambda_sigreg=0.05)

    # LR neighborhood
    add(clip_LN_lr=base["clip_LN_lr"] * 0.8)
    add(clip_LN_lr=base["clip_LN_lr"] * 1.2)
    add(prompt_lr=base["prompt_lr"] * 0.8)
    add(prompt_lr=base["prompt_lr"] * 1.2)

    # Structure / temperature neighborhood
    add(n_prompts=base["n_prompts"] + 1)
    add(n_prompts=max(1, base["n_prompts"] - 1))
    add(cls_tau=base["cls_tau"] * 0.85)
    add(cls_tau=base["cls_tau"] * 1.15)

    return candidates


def run_trial(args, trial_id, cfg, workspace_root):
    exp_name = f"{args.base_exp}_t{trial_id:03d}"
    runner = [args.python_bin]
    if args.conda_env:
        runner = ["conda", "run", "-n", args.conda_env, "python"]

    cmd = runner + [
        "experiments/train_finetune.py",
        "--exp_name",
        exp_name,
        "--max_epochs",
        str(args.max_epochs),
        "--early_stop_patience",
        str(args.patience),
        "--seed",
        str(args.seed + trial_id),
        "--batch_size",
        str(args.batch_size),
        "--save_last_ckpt",
        "False",
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
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] Trial {trial_id} start")
    print("CMD:", " ".join(cmd))
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = workspace_root
    run_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.run(cmd, cwd=workspace_root, env=run_env, check=False)

    exp_dir = os.path.join(args.save_root, f"{exp_name}_finetune")
    best_map = find_best_map_in_dir(exp_dir)
    print(f"[Trial {trial_id}] return_code={proc.returncode} best_mAP={best_map} exp_dir={exp_dir}")
    return exp_name, best_map, proc.returncode


def write_status_snapshot(
    path,
    now_ts,
    trial_id,
    phase,
    best,
    last_result,
    queue_len,
    local_refine_start_map,
    grid_profile,
):
    lines = [
        "# Auto Experiment Live Status",
        "",
        f"- Updated at: `{now_ts}`",
        f"- Last trial id: `{trial_id}`",
        f"- Last phase: `{phase}`",
        f"- Grid profile: `{grid_profile}`",
        f"- Local refine queue size: `{queue_len}`",
        f"- Local refine start mAP: `{local_refine_start_map}`",
        "",
        "## Last Result",
        f"- exp_name: `{last_result.get('exp_name')}`",
        f"- status: `{last_result.get('status')}`",
        f"- return_code: `{last_result.get('return_code')}`",
        f"- mAP: `{last_result.get('mAP')}`",
        f"- cfg: `{last_result.get('cfg')}`",
        "",
        "## Best So Far",
        f"- trial_id: `{best.get('trial_id')}`",
        f"- exp_name: `{best.get('exp_name')}`",
        f"- mAP: `{best.get('mAP')}`",
        f"- cfg: `{best.get('cfg')}`",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    infinite_mode = args.hours <= 0
    budget_seconds = int(max(0.0, args.hours) * 3600)
    deadline = time.time() + budget_seconds if not infinite_mode else None

    trial_grid = build_trial_grid(args.grid_profile)
    print(f"Total candidate configs: {len(trial_grid)}")
    print(f"Grid profile: {args.grid_profile}")
    if infinite_mode:
        print("Time budget: infinite (hours<=0)")
    else:
        print(f"Time budget: {args.hours} hours")

    best = {"trial_id": None, "exp_name": None, "mAP": -1.0, "cfg": None}
    results = []
    trial_id = 1
    consecutive_failures = 0
    seen_cfg_keys = set()
    local_refine_queue = []
    local_refine_round = 0
    local_no_improve_streak = 0

    delete_non_best = to_bool(args.delete_non_best)
    write_status_snapshot(
        path=os.path.join(workspace_root, args.status_file),
        now_ts=datetime.now().isoformat(timespec="seconds"),
        trial_id=0,
        phase="init",
        best={**best, "local_refine_start_map": args.local_refine_start_map},
        last_result={"exp_name": None, "return_code": None, "mAP": None, "cfg": None},
        queue_len=0,
        local_refine_start_map=args.local_refine_start_map,
        grid_profile=args.grid_profile,
    )

    while infinite_mode or time.time() < deadline:
        # Priority: local refine queue around new best.
        is_local_phase = bool(local_refine_queue)
        if is_local_phase:
            cfg = local_refine_queue.pop(0)
            phase = "local_refine"
        elif trial_id <= len(trial_grid):
            cfg = trial_grid[trial_id - 1]
            phase = "global_sweep"
        else:
            # Fallback exploitation once global sweep is exhausted.
            if best["cfg"] is None:
                break
            base = best["cfg"].copy()
            # Small perturbation cycle.
            if trial_id % 4 == 0:
                base["lambda_sigreg"] = min(0.3, base["lambda_sigreg"] + 0.05)
            elif trial_id % 4 == 1:
                base["lambda_sigreg"] = max(0.0, base["lambda_sigreg"] - 0.03)
            elif trial_id % 4 == 2:
                base["clip_LN_lr"] = max(3e-7, base["clip_LN_lr"] * 0.8)
            else:
                base["prompt_lr"] = min(2e-4, base["prompt_lr"] * 1.2)
            cfg = base
            phase = "fallback_exploit"

        cfg = clip_cfg_bounds(cfg)
        key = cfg_key(cfg)
        if key in seen_cfg_keys:
            trial_id += 1
            continue
        seen_cfg_keys.add(key)
        print(f"[PHASE] {phase}")
        now_ts = datetime.now().isoformat(timespec="seconds")
        write_status_snapshot(
            path=os.path.join(workspace_root, args.status_file),
            now_ts=now_ts,
            trial_id=trial_id,
            phase=phase,
            best=best,
            last_result={
                "exp_name": f"{args.base_exp}_t{trial_id:03d}",
                "status": "running",
                "return_code": None,
                "mAP": None,
                "cfg": cfg,
            },
            queue_len=len(local_refine_queue),
            local_refine_start_map=args.local_refine_start_map,
            grid_profile=args.grid_profile,
        )

        exp_name, score, return_code = run_trial(args, trial_id, cfg, workspace_root)
        exp_dir = os.path.join(args.save_root, f"{exp_name}_finetune")
        results.append((trial_id, exp_name, score, return_code, cfg))

        if return_code != 0 and score is None:
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        if consecutive_failures >= 3:
            print("[STOP] Encountered 3 consecutive failed trials. Please inspect environment/dependencies.")
            break

        improved = score is not None and score > best["mAP"]
        if improved:
            best = {"trial_id": trial_id, "exp_name": exp_name, "mAP": score, "cfg": cfg}
            print(f"[NEW BEST] trial={trial_id} exp={exp_name} mAP={score:.4f} cfg={cfg}")
            # Stage-2 local refinement trigger only after reaching threshold mAP.
            if score >= args.local_refine_start_map:
                local_refine_round += 1
                local_no_improve_streak = 0
                local_candidates = build_local_refine_candidates(cfg)
                fresh_local = [c for c in local_candidates if cfg_key(c) not in seen_cfg_keys]
                local_refine_queue.extend(fresh_local)
                print(
                    f"[LOCAL REFINE] round={local_refine_round} "
                    f"queued={len(fresh_local)} candidates around trial={trial_id}"
                )
            else:
                print(
                    f"[LOCAL REFINE WAIT] best mAP={score:.4f} < "
                    f"threshold={args.local_refine_start_map:.4f}; keep global sweep."
                )
        elif is_local_phase:
            local_no_improve_streak += 1
            if local_no_improve_streak >= args.local_patience:
                local_refine_queue.clear()
                local_no_improve_streak = 0
                print(
                    f"[LOCAL REFINE STOP] no improvement for {args.local_patience} local trials; "
                    "returning to global sweep."
                )
        else:
            local_no_improve_streak = 0

        if delete_non_best and (not improved) and os.path.isdir(exp_dir):
            shutil.rmtree(exp_dir, ignore_errors=True)
            print(f"[CLEANUP] removed non-best checkpoints: {exp_dir}")

        now_ts = datetime.now().isoformat(timespec="seconds")
        write_status_snapshot(
            path=os.path.join(workspace_root, args.status_file),
            now_ts=now_ts,
            trial_id=trial_id,
            phase=phase,
            best={**best, "local_refine_start_map": args.local_refine_start_map},
            last_result={
                "exp_name": exp_name,
                "status": "finished",
                "return_code": return_code,
                "mAP": score,
                "cfg": cfg,
            },
            queue_len=len(local_refine_queue),
            local_refine_start_map=args.local_refine_start_map,
            grid_profile=args.grid_profile,
        )

        trial_id += 1

    print("\n==== SEARCH FINISHED ====")
    print(f"Total trials: {len(results)}")
    if best["trial_id"] is not None:
        print(f"Best trial: {best['trial_id']}")
        print(f"Best exp: {best['exp_name']}")
        print(f"Best mAP: {best['mAP']:.4f}")
        print(f"Best cfg: {best['cfg']}")
    else:
        print("No valid mAP found from checkpoint names.")


if __name__ == "__main__":
    main()
