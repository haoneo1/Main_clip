import glob
import os
import re
import time
from datetime import datetime


SAVE_ROOT = "/home/mig/Documents/SBIR_Data/saved_models"
OUT_FILE = "/home/mig/Documents/SBIR/Main_clip/experiments/stage2_status_live.md"
SLEEP_SECONDS = 20
GPU0_EXP_NAME = "stage2_gpu0_nocls_finetune"
GPU1_EXP_NAME = "stage2_gpu1_joint"


def newest(path_glob: str):
    candidates = glob.glob(path_glob)
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def parse_float_from_name(name: str, pattern: str):
    m = re.search(pattern, name)
    return float(m.group(1)) if m else None


def is_running(keyword: str) -> bool:
    # Lightweight process probe without external dependencies.
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        cmdline_path = os.path.join("/proc", pid, "cmdline")
        try:
            with open(cmdline_path, "rb") as f:
                cmd = f.read().decode("utf-8", errors="ignore").replace("\x00", " ")
            if keyword in cmd:
                return True
        except OSError:
            continue
    return False


def trial_id_from_exp_name(exp_name: str) -> int:
    # Keep a stable trial id in the status markdown.
    # If the exp name uses suffix like *_t076*, this extracts 76.
    m = re.search(r"_t(\d+)", exp_name)
    return int(m.group(1)) if m else 1


def render_status() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gpu0_best = newest(f"{SAVE_ROOT}/stage2_gpu0_nocls_finetune/best-*.ckpt")
    gpu0_last = newest(f"{SAVE_ROOT}/stage2_gpu0_nocls_finetune/last.ckpt")
    gpu0_map = (
        parse_float_from_name(os.path.basename(gpu0_best), r"mAP=([0-9]+\.[0-9]+)")
        if gpu0_best
        else None
    )

    gpu1_pre_best = newest(f"{SAVE_ROOT}/stage2_gpu1_joint_pretrain/pretrain-epoch=*.ckpt")
    gpu1_pre_loss = (
        parse_float_from_name(
            os.path.basename(gpu1_pre_best), r"train_loss_epoch=([0-9]+\.[0-9]+)"
        )
        if gpu1_pre_best
        else None
    )

    gpu1_ft_best = newest(f"{SAVE_ROOT}/stage2_gpu1_joint_finetune/best-*.ckpt")
    gpu1_ft_map = (
        parse_float_from_name(os.path.basename(gpu1_ft_best), r"mAP=([0-9]+\.[0-9]+)")
        if gpu1_ft_best
        else None
    )

    running_gpu0 = is_running("stage2_gpu0_nocls")
    running_gpu1 = is_running("stage2_gpu1_joint")
    gpu0_trial_id = trial_id_from_exp_name(GPU0_EXP_NAME)
    gpu1_trial_id = trial_id_from_exp_name(GPU1_EXP_NAME)

    lines = [
        "# Stage2 Live Status",
        "",
        f"- Updated: `{now}`",
        f"- GPU0 task: `stage2_gpu0_nocls_finetune` ({'running' if running_gpu0 else 'stopped'})",
        f"- GPU1 task: `stage2_gpu1_joint_pretrain -> stage2_gpu1_joint_finetune` ({'running' if running_gpu1 else 'stopped'})",
        "",
        "## GPU0 (Finetune, no class loss)",
        f"- Trial: `{gpu0_trial_id}`",
        f"- Best mAP: `{gpu0_map if gpu0_map is not None else 'N/A'}`",
        f"- Best ckpt: `{os.path.basename(gpu0_best) if gpu0_best else 'N/A'}`",
        f"- Last ckpt mtime: `{datetime.fromtimestamp(os.path.getmtime(gpu0_last)).strftime('%Y-%m-%d %H:%M:%S') if gpu0_last else 'N/A'}`",
        "",
        "## GPU1 (Joint pipeline)",
        f"- Trial: `{gpu1_trial_id}`",
        f"- Pretrain latest loss(epoch): `{gpu1_pre_loss if gpu1_pre_loss is not None else 'N/A'}`",
        f"- Pretrain ckpt: `{os.path.basename(gpu1_pre_best) if gpu1_pre_best else 'N/A'}`",
        f"- Finetune best mAP: `{gpu1_ft_map if gpu1_ft_map is not None else 'N/A (finetune not started)'}`",
        f"- Finetune best ckpt: `{os.path.basename(gpu1_ft_best) if gpu1_ft_best else 'N/A'}`",
    ]
    return "\n".join(lines) + "\n"


def main():
    while True:
        content = render_status()
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
