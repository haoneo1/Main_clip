import os
import subprocess
import time
from datetime import datetime


WATCH_KEYWORDS = ["auto_live_main", "auto_live_explore"]
POLL_SECONDS = 60


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def has_active_processes():
    out = subprocess.check_output(["ps", "-ef"], text=True)
    for line in out.splitlines():
        if "wait_and_launch_stage2.py" in line:
            continue
        if any(k in line for k in WATCH_KEYWORDS):
            return True
    return False


def run_cmd(command, log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] CMD: {command}\n")
        f.flush()
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            executable="/bin/bash",
        )
    return proc


def main():
    workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(workspace, "experiments", "stage2_logs")
    os.makedirs(logs_dir, exist_ok=True)

    gpu0_log = os.path.join(logs_dir, "gpu0_nocls.log")
    gpu1_log = os.path.join(logs_dir, "gpu1_joint.log")
    controller_log = os.path.join(logs_dir, "controller.log")

    with open(controller_log, "a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] Stage2 queue started. Waiting current jobs to finish.\n")

    while has_active_processes():
        with open(controller_log, "a", encoding="utf-8") as f:
            f.write(f"[{now()}] Current auto jobs still running. Sleep {POLL_SECONDS}s.\n")
        time.sleep(POLL_SECONDS)

    with open(controller_log, "a", encoding="utf-8") as f:
        f.write(f"[{now()}] Current auto jobs finished. Launching stage2 workloads.\n")

    gpu0_cmd = (
        "CUDA_VISIBLE_DEVICES=0 conda run -n pyenv python experiments/train_finetune.py "
        "--exp_name stage2_gpu0_nocls "
        "--max_epochs 60 --early_stop_patience 12 --batch_size 64 "
        "--use_clip_cls False --lambda_cls 0.0 "
        "--use_sigreg True --lambda_sigreg 0.1 "
        "--clip_LN_lr 2e-6 --prompt_lr 5e-5 --n_prompts 3 --cls_tau 0.07"
    )

    gpu1_cmd = (
        "CUDA_VISIBLE_DEVICES=1 conda run -n pyenv python experiments/train_pretrain.py "
        "--exp_name stage2_gpu1_joint "
        "--max_epochs 40 --batch_size 128 "
        "--jepa_hidden_dim 512 --lambda_jepa_pred 1.0 --jepa_lr 1e-4 "
        "&& "
        "CUDA_VISIBLE_DEVICES=1 conda run -n pyenv python experiments/train_finetune.py "
        "--exp_name stage2_gpu1_joint "
        "--finetune_init_ckpt /home/mig/Documents/SBIR_Data/saved_models/stage2_gpu1_joint_pretrain/last.ckpt "
        "--max_epochs 60 --early_stop_patience 12 --batch_size 64 "
        "--use_clip_cls True --lambda_cls 0.2 "
        "--use_sigreg True --lambda_sigreg 0.1 "
        "--clip_LN_lr 1e-6 --prompt_lr 5e-5 --n_prompts 3 --cls_tau 0.07"
    )

    p0 = run_cmd(gpu0_cmd, gpu0_log)
    p1 = run_cmd(gpu1_cmd, gpu1_log)

    with open(controller_log, "a", encoding="utf-8") as f:
        f.write(
            f"[{now()}] Launched: GPU0 PID={p0.pid}, GPU1 pipeline PID={p1.pid}. Waiting completion.\n"
        )

    rc0 = p0.wait()
    rc1 = p1.wait()

    with open(controller_log, "a", encoding="utf-8") as f:
        f.write(f"[{now()}] Stage2 completed. GPU0 rc={rc0}, GPU1 rc={rc1}\n")


if __name__ == "__main__":
    main()
