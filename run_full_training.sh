#!/bin/bash

srun --partition=gpu_mig40 --gres=gpu:1 --cpus-per-task=4 --mem=128G --time=48:00:00 \
     --job-name openpi_full bash -c "
cd /nfs/turbo/coe-vkamat/openpi && \
export OPENPI_DATA_HOME=/nfs/turbo/coe-vkamat/openpi_cache && \
export HF_HOME=/nfs/turbo/coe-vkamat/huggingface && \
export TMPDIR=/nfs/turbo/coe-vkamat/tmp && \
export XDG_CACHE_HOME=/nfs/turbo/coe-vkamat/openpi_cache && \
export JAXTYPING_DISABLE=1 && \
/nfs/turbo/coe-vkamat/miniconda3/envs/openpi/bin/python -u scripts/train.py \
     pi0_aloha_lora_finetune_peg --exp-name=peg_finetune_lora_full --overwrite \
     --batch-size 16 --num-workers 2 \
     2>&1 | tee train_log_full2.txt
" 