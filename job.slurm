#!/bin/bash
#SBATCH --job-name=openpi_peg_finetune
#SBATCH --partition=gpu_mig40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --account=vkamat0

# 创建日志目录
mkdir -p logs

# 设置环境变量
export OPENPI_DATA_HOME=/nfs/turbo/coe-vkamat/openpi_cache
export HF_HOME=/nfs/turbo/coe-vkamat/huggingface
export HUGGINGFACE_HUB_CACHE=/nfs/turbo/coe-vkamat/huggingface/hub
export HF_DATASETS_CACHE=/nfs/turbo/coe-vkamat/huggingface/datasets
export WANDB_DIR=/nfs/turbo/coe-vkamat/wandb
export WANDB_CACHE_DIR=/nfs/turbo/coe-vkamat/wandb
export PIP_CACHE_DIR=/nfs/turbo/coe-vkamat/pip_cache
export TMPDIR=/nfs/turbo/coe-vkamat/tmp
export XDG_CACHE_HOME=/nfs/turbo/coe-vkamat/openpi_cache
export JAXTYPING_DISABLE=1

# 激活conda环境
source /nfs/turbo/coe-vkamat/miniconda3/etc/profile.d/conda.sh
conda activate openpi

# 切换到工作目录
cd /nfs/turbo/coe-vkamat/openpi

# 打印作业信息
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "======================"

# 先计算norm stats
echo "=== Computing Norm Stats ==="
python scripts/compute_norm_stats.py \
    --config-name pi0_aloha_lora_finetune_peg \
    --max-frames 100

# 检查norm stats是否成功完成
if [ $? -eq 0 ]; then
    echo "=== Norm stats computation completed successfully ==="
else
    echo "=== Norm stats computation failed! Exiting... ==="
    exit 1
fi

# 运行训练脚本
echo "=== Starting Training ==="
python -u scripts/train.py \
    pi0_aloha_lora_finetune_peg \
    --exp-name=peg_finetune_lora_full \
    --overwrite \
    --batch-size 16 \
    --num-workers 2

# 打印结束信息
echo "======================"
echo "End time: $(date)"
echo "Job completed!" 