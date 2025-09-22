#!/usr/bin/env python3
"""
调试脚本：验证训练和验证时的GRU输出维度是否一致
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from joblib import load as joblib_load
import torch
from torch import nn

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi.transforms import flatten_dict as _flatten_dict
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize

from tactile_train_valid import (
    _to_numpy, _marker_to_vector, _prepare_obs_at, 
    LeRobotPi0ActionMarkerDataset, GRURegressor,
    _extract_episode_id, _build_delta_timestamps, _detect_sequence_keys
)


def debug_dimensions():
    """验证训练和验证时的维度一致性"""
    
    # 使用示例参数
    repo_id = "xiejunz/peg_data"
    pca_path = Path("checkpoints/pi0_aloha_lora_finetune_peg/PCA/pca_marker_right.joblib")
    pi0_checkpoint = Path("checkpoints/pi0_aloha_lora_finetune_peg/peg_finetune_lora_full/19999")
    pi0_config_name = "pi0_aloha_lora_finetune_peg"
    horizon = 50
    
    print("=== 调试GRU输出维度 ===")
    
    # 1. 加载Pi0策略
    print("1. 加载Pi0策略...")
    cfg = _config.get_config(pi0_config_name)
    policy = _policy_config.create_trained_policy(cfg, str(pi0_checkpoint))
    
    # 2. 创建训练数据集
    print("2. 创建训练数据集...")
    try:
        # 获取实际的DataConfig对象
        from openpi.models import pi0_config
        model_cfg = pi0_config.Pi0Config()
        data_cfg = cfg.data.create_base_config(cfg.assets_dirs, model_cfg)
        
        cfg_assets = cfg.assets_dirs
        asset_id = data_cfg.asset_id or data_cfg.repo_id
        norm = _normalize.load(cfg_assets / asset_id) if asset_id is not None else None
        actions_mean = None
        actions_std = None
        if norm is not None and "actions" in norm:
            actions_mean = np.asarray(norm["actions"].mean[-7:], dtype=np.float32)
            actions_std = np.asarray(norm["actions"].std[-7:], dtype=np.float32)
            print(f"   已加载归一化统计：mean/std 形状={actions_mean.shape}/{actions_std.shape}")
        else:
            print("   未找到归一化统计")
    except Exception as e:
        print(f"   加载归一化统计失败：{e}")
        actions_mean = None
        actions_std = None
    
    # 创建数据集（跳过预计算，只用于调试维度）
    dataset = LeRobotPi0ActionMarkerDataset(
        repo_id, pca_path, policy, 
        horizon=horizon, 
        precompute_pi0=False,  # 跳过预计算
        actions_norm_mean=actions_mean, 
        actions_norm_std=actions_std,
    )
    
    # 3. 检查训练数据维度
    print("3. 检查训练数据维度...")
    sample_feat, sample_gt, sample_pi0 = dataset[0]
    feature_dim = sample_feat.shape[-1]
    action_dim = sample_gt.shape[-1]
    
    print(f"   训练数据维度:")
    print(f"   - 特征维度 (Pi0后7维+PCA): {feature_dim}")
    print(f"   - 动作维度 (GT): {action_dim}")
    print(f"   - 特征形状: {sample_feat.shape}")
    print(f"   - GT形状: {sample_gt.shape}")
    print(f"   - Pi0形状: {sample_pi0.shape}")
    
    # 4. 创建GRU模型
    print("4. 创建GRU模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRURegressor(input_dim=feature_dim, hidden_dim=32, output_dim=action_dim, num_layers=1).to(device)
    
    # 5. 测试训练时的GRU输出
    print("5. 测试训练时的GRU输出...")
    model.train()
    feat_batch = sample_feat.unsqueeze(0).to(device).float()  # [1, T, F]
    with torch.no_grad():
        pred_train = model(feat_batch)
    
    print(f"   训练时GRU输出:")
    print(f"   - 输入形状: {feat_batch.shape}")
    print(f"   - 输出形状: {pred_train.shape}")
    print(f"   - 输出维度: {pred_train.shape[-1]}")
    
    # 6. 测试验证时的GRU输出
    print("6. 测试验证时的GRU输出...")
    model.eval()
    
    # 模拟验证时的单步推理
    feat_single = feat_batch[0, 0, :].unsqueeze(0).unsqueeze(0)  # [1, 1, F]
    with torch.no_grad():
        pred_valid = model(feat_single)
    
    print(f"   验证时GRU输出:")
    print(f"   - 输入形状: {feat_single.shape}")
    print(f"   - 输出形状: {pred_valid.shape}")
    print(f"   - 输出维度: {pred_valid.shape[-1]}")
    
    # 7. 检查验证时的后处理逻辑
    print("7. 检查验证时的后处理逻辑...")
    gru_output = pred_valid[0, 0].cpu().numpy().astype(np.float32)
    print(f"   - 原始GRU输出形状: {gru_output.shape}")
    
    # 模拟验证脚本中的后处理
    if int(gru_output.shape[0]) == 7:
        gru_right = gru_output
        print(f"   - 情况1 (输出7维): 直接使用 -> {gru_right.shape}")
    elif int(gru_output.shape[0]) == int(action_dim):
        gru_right = gru_output[7:14]
        print(f"   - 情况2 (输出{action_dim}维): 取[7:14] -> {gru_right.shape}")
    else:
        if int(gru_output.shape[0]) > 7:
            gru_right = gru_output[-7:]
            print(f"   - 情况3 (输出{gru_output.shape[0]}维>7): 取后7维 -> {gru_right.shape}")
        else:
            tmp7 = np.zeros((7,), dtype=np.float32)
            tmp7[:int(gru_output.shape[0])] = gru_output
            gru_right = tmp7
            print(f"   - 情况4 (输出{gru_output.shape[0]}维<7): 补零到7维 -> {gru_right.shape}")
    
    # 8. 对比训练loss计算
    print("8. 对比训练loss计算...")
    gt_batch = sample_gt.unsqueeze(0).to(device).float()  # [1, T, A]
    
    # 训练时的loss计算
    loss_train = nn.L1Loss()(pred_train, gt_batch)
    print(f"   训练loss (完整维度): {loss_train.item():.6f}")
    
    # 如果GRU输出是14维，只计算右臂7维的loss
    if pred_train.shape[-1] == 14:
        loss_right_only = nn.L1Loss()(pred_train[..., -7:], gt_batch[..., -7:])
        print(f"   训练loss (仅右臂7维): {loss_right_only.item():.6f}")
    
    # 9. 总结
    print("\n=== 总结 ===")
    print(f"训练时:")
    print(f"  - 输入特征维度: {feature_dim} (Pi0后7维+PCA)")
    print(f"  - GRU输出维度: {pred_train.shape[-1]}")
    print(f"  - GT动作维度: {action_dim}")
    print(f"  - Loss计算: 完整{action_dim}维")
    
    print(f"验证时:")
    print(f"  - 输入特征维度: {feature_dim} (相同)")
    print(f"  - GRU输出维度: {pred_valid.shape[-1]} (相同)")
    print(f"  - 后处理: 提取右臂7维用于绘图")
    print(f"  - 绘图显示: 仅右臂7维")
    
    # 检查一致性
    if pred_train.shape[-1] == pred_valid.shape[-1]:
        print("✓ 训练和验证的GRU输出维度一致")
    else:
        print("✗ 训练和验证的GRU输出维度不一致！")
    
    print("\n=== 建议 ===")
    if action_dim == 14:
        print("- 数据集包含14维动作（左臂7+右臂7）")
        print("- 训练时学习完整14维，验证时只显示右臂7维")
        print("- 这是合理的：训练更全面，验证聚焦关注点")
    else:
        print(f"- 数据集包含{action_dim}维动作")
        print("- 训练和验证都使用相同维度")


if __name__ == "__main__":
    debug_dimensions()
