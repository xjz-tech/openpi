#!/bin/bash

# 调试Pi0动作处理过程的示例脚本

python scripts/debug_actions.py \
    --repo-id "xiejunz/peg_data" \
    --pi0-checkpoint "checkpoints/pi0_aloha_lora_finetune_peg/peg_finetune_lora_full/19999" \
    --pi0-config "pi0_aloha_lora_finetune_peg" \
    --num-samples 3 \
    --horizon 50
