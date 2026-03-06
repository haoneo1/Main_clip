#!/bin/bash
#SBATCH -J clip_pretrain
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH -o /home/bingxing2/home/scx9951/Main_clip/logs/clip_pretrain_%j.out
#SBATCH -e /home/bingxing2/home/scx9951/Main_clip/logs/clip_pretrain_%j.err

set -e
export PYTHONUNBUFFERED=1

module purge
module load miniforge3/24.1 \
            compilers/cuda/12.1 \
            cudnn/8.9.5.29_cuda12.x \
            compilers/gcc/11.3.0 \
            nccl/2.19.1-1_cuda12.1

source activate py39_env

cd /home/bingxing2/home/scx9951/Main_clip

echo "========== JOB START =========="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Work Dir    : $(pwd)"
echo "Start Time  : $(date)"
echo "Python Path : $(which python)"
echo "Python Ver  : $(python --version)"
echo "Conda Env   : $CONDA_DEFAULT_ENV"
echo "==============================="

python -m experiments.train_pretrain

echo "=========== JOB END ==========="
echo "End Time    : $(date)"
echo "==============================="
