import argparse
import random
import time
from pathlib import Path
from typing import Any, List
import uuid

import numpy as np
from joblib import dump as joblib_dump, load as joblib_load
from sklearn.decomposition import PCA as SkPCA
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi.transforms import flatten_dict as _flatten_dict
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.models import pi0_config
"""
python scripts/tactile_pca_pi0_gru.py \
    --repo-id "xiejunz/peg_data" \
    --pi0-checkpoint checkpoints/pi0_aloha_lora_finetune_peg/peg_finetune_lora_full/19999 \
    --num-steps 20000 \
    --batch 32 \
    --pi0-precompute-batch 100 \
    --lr 1e-3 \
    --num-layers 1 \
    --hidden 32 \
    --log-interval 100 \
    --save-interval 2000 \
    --wandb-project "marker-pca-pi0-gru" \
    --wandb-enabled \
    --skip-pca \
    --refresh-pi0-cache
"""

def _log(msg: str) -> None:
    print(f"[marker_pca_pi0_gru] {msg}")

# 把在各个device上的数据都变成numpy数组
def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import jax

        if isinstance(x, jax.Array):
            return np.asarray(x)
    except Exception:
        pass
    try:
        import torch as _torch

        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _marker_to_vector(marker: np.ndarray) -> np.ndarray:
    """将marker数据转换为向量形式"""
    x = _to_numpy(marker)
    # marker数据应该是126维向量，直接返回
    if x.ndim == 1:
        return x.astype(np.float32)
    elif x.ndim == 2:
        # 如果是2D，取第一行（假设是单帧数据）
        return x[0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported marker shape: {x.shape}")




def train_pca_on_marker_lerobot(
    repo_id: str,
    out_path: Path,
    k: int = 7,
    max_samples: int = 10_000,
    marker_key_substring: str = "marker_tracking_right_dxdy",
) -> SkPCA:
    marker_key = "observation.marker_tracking_right_dxdy"
    _log(f"使用固定marker键: {marker_key}")

    probe_ds = lerobot_dataset.LeRobotDataset(repo_id)
    X_list: List[np.ndarray] = []
    total = min(len(probe_ds), max_samples)
    _log(f"开始处理 {total} 个样本...")
    for i in range(total):
        s = probe_ds[i]
        flat = _flatten_dict(s)
        marker = _to_numpy(flat[marker_key])
        X_list.append(_marker_to_vector(marker))
    if not X_list:
        raise RuntimeError("未能从 LeRobot 数据集中提取到marker数据")

    X = np.stack(X_list, axis=0).astype(np.float32)
    _log("使用 scikit-learn 进行 PCA 训练（whiten=True）…")
    sk_pca = SkPCA(n_components=k, svd_solver="auto", random_state=0, whiten=True)
    sk_pca.fit(X)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib_dump(sk_pca, out_path)
    _log(f"PCA 训练完成并已保存到: {out_path} (K={k}, D={X.shape[1]}) 来自 repo_id={repo_id}")
    return sk_pca


class LeRobotPi0ActionMarkerDataset(Dataset):  # type: ignore[misc]
    """LeRobot 数据；特征=Pi0 预测动作后7维 + marker PCA；目标=真实动作。"""

    def __init__(
        self,
        repo_id: str,
        pca_path: Path,
        policy: Any,
        *,
        horizon: int = 50,
        marker_key_substring: str = "marker_tracking_right_dxdy",
        precompute_pi0: bool = True,
        precompute_batch_size: int = 8,
        pi0_cache_file: Path | None = None,
        refresh_pi0_cache: bool = False,
        # 可选：对 Pi0 的后7维做 Z-score（来自训练管线 assets 的 actions 统计）
        actions_norm_mean: np.ndarray | None = None,
        actions_norm_std: np.ndarray | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.horizon = horizon
        self.pca: SkPCA = joblib_load(pca_path)
        self.policy = policy
        self._pi0_cache: dict[int, np.ndarray] = {}
        self._precompute_batch_size = precompute_batch_size
        
        # 单文件缓存设置
        self._pi0_cache_file: Path | None = Path(pi0_cache_file) if pi0_cache_file is not None else None
        self._use_disk_cache: bool = pi0_cache_file is not None
        self._refresh_pi0_cache: bool = refresh_pi0_cache
        self._pi0_memmap: np.ndarray | None = None

        # Pi0 归一化统计（仅用于后7维）
        if actions_norm_mean is not None and actions_norm_std is not None:
            self._pi0_mean = actions_norm_mean.astype(np.float32)
            self._pi0_std = actions_norm_std.astype(np.float32)
        else:
            self._pi0_mean = None
            self._pi0_std = None

        # 使用固定的键名
        self.marker_key = "observation.marker_tracking_right_dxdy"
        self.action_key = "action"

        fps = 50.0
        delta_for_seq = [i / fps for i in range(self.horizon)]

        # 探测样本结构确定可用的键
        probe_ds = lerobot_dataset.LeRobotDataset(repo_id)
        probe_sample = probe_ds[0]
        flat_probe = _flatten_dict(probe_sample)

        # 只对需要序列化的键应用 delta_timestamps（参考训练代码的做法）
        # 主要需要序列化的是 action 和 marker_tracking
        sequence_keys = []
        if self.action_key in flat_probe:
            sequence_keys.append(self.action_key)
        if self.marker_key in flat_probe:
            sequence_keys.append(self.marker_key)
        
        delta_timestamps: dict[str, list[float]] = {}
        for k in sequence_keys:
            delta_timestamps[k] = delta_for_seq

        # 使用 delta_timestamps 让数据加载直接返回长度为 horizon 的时间序列窗口
        self.ds = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

        if precompute_pi0:
            self._precompute_all_pi0_actions(batch_size=self._precompute_batch_size)

    def _parse_image(self, image) -> np.ndarray:
        """将输入图像规范为 CHW uint8（Pi0 AlohaInputs 期望 CHW）。"""
        if image is None:
            return np.zeros((3, 224, 224), dtype=np.uint8)

        arr = np.asarray(image)
        if np.issubdtype(arr.dtype, np.floating):
            arr = (255 * arr).astype(np.uint8)

        # 统一到 CHW
        if arr.ndim == 2:  # H, W → 重复到3通道
            arr = np.repeat(arr[None, ...], 3, axis=0)
        elif arr.ndim == 3 and arr.shape[-1] == 3:  # H, W, C → C, H, W
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 3 and arr.shape[0] == 3:  # 已是 C, H, W
            pass
        elif arr.ndim == 3 and arr.shape[0] == 1:  # 1, H, W → 3, H, W
            arr = np.repeat(arr, 3, axis=0)
        elif arr.ndim == 3 and arr.shape[-1] == 1:  # H, W, 1 → 3通道后再转置
            arr = np.repeat(arr, 3, axis=-1)
            arr = np.transpose(arr, (2, 0, 1))
        else:
            # 兜底：尽力转成 CHW
            if arr.ndim >= 3:
                # 取最后三个维度视为 HWC
                hwc = arr.reshape(*arr.shape[-3:])
                if hwc.shape[-1] == 3:
                    arr = np.transpose(hwc, (2, 0, 1))
                else:
                    # 若非3通道，填0到3通道
                    h, w = hwc.shape[0], hwc.shape[1]
                    tmp = np.zeros((3, h, w), dtype=np.uint8)
                    arr = tmp
            else:
                arr = np.zeros((3, 224, 224), dtype=np.uint8)

        return arr.astype(np.uint8)

    def _prepare_obs_at(self, flat: dict, t: int, action_dim: int) -> dict:
        """简化的观察数据准备，参考Pi0训练代码"""
        def _get_time_step(key: str, fallback: Any | None = None) -> Any:
            val = flat.get(key, fallback)
            if val is None:
                return None
            arr = _to_numpy(val)
            return arr if arr.ndim == 1 or arr.ndim == 3 else arr[t]

        # 获取图像数据
        base_img = _get_time_step("observation.images.cam_high")
        if base_img is None:
            base_img = _get_time_step("observation/image")
            
        left_wrist = _get_time_step("observation.images.cam_left_wrist")
        if left_wrist is None:
            left_wrist = _get_time_step("observation/wrist_image")
            
        right_wrist = _get_time_step("observation.images.cam_right_wrist")

        # 解析图像
        base_img = self._parse_image(base_img)
        left_wrist = self._parse_image(left_wrist)
        right_wrist = self._parse_image(right_wrist) if right_wrist is not None else np.zeros_like(base_img)

        # 获取状态数据
        state = _get_time_step("observation.state")
        if state is None:
            state = np.zeros((action_dim,), dtype=np.float32)

        return {
            "images": {
                "cam_high": base_img,
                "cam_left_wrist": left_wrist,
                "cam_right_wrist": right_wrist,
            },
            "state": state.astype(np.float32),
        }


    def _precompute_all_pi0_actions(self, batch_size: int = 8) -> None:
        _log("开始预计算 Pi0 推理结果（全局批量、全时间步缓存）…")
        total = len(self.ds)
        pending_indices: list[int] = []  # episode_idx
        pending_obs: list[dict] = []
        processed_eps = 0

        # 若使用单文件缓存，优先尝试直接加载或创建 memmap
        if self._use_disk_cache:
            assert self._pi0_cache_file is not None
            expected_shape = (total, self.horizon, 7)
            if self._pi0_cache_file.exists() and not self._refresh_pi0_cache:
                try:
                    self._pi0_memmap = np.load(self._pi0_cache_file, mmap_mode="r+")
                    if tuple(self._pi0_memmap.shape) == expected_shape:
                        _log("检测到单文件 Pi0 缓存，跳过预计算")
                        return
                except Exception:
                    pass
            # 创建新的单文件缓存（.npy + memmap）
            self._pi0_memmap = np.lib.format.open_memmap(
                self._pi0_cache_file, mode="w+", dtype=np.float32, shape=expected_shape
            )

        for i in range(total):
            s = self.ds[i]
            flat = _flatten_dict(s)
            gt_full = _to_numpy(flat[self.action_key]).astype(np.float32)
            if gt_full.ndim == 1:
                raise RuntimeError(f"动作数据维度错误：期望2D数组 [T, A]，但得到1D数组 {gt_full.shape}。请检查delta_timestamps配置。")
            T_full, action_dim = gt_full.shape


            # 仅对每个样本做一次推理（以 t=0 的观测为条件）
            obs = self._prepare_obs_at(flat, 0, action_dim)
            pending_indices.append(i)
            pending_obs.append(obs)

            if len(pending_obs) >= batch_size:
                results = (
                    self.policy.batch_infer(pending_obs)
                )
                for ei, out in zip(pending_indices, results):
                    pi0_actions = np.asarray(out["actions"], dtype=np.float32)
                    # 只取后7维（前7维都是0）并保存整段序列
                    pi0_actions = pi0_actions[:, -7:]  # [50, 7]
                    if self._use_disk_cache:
                        assert self._pi0_memmap is not None
                        self._pi0_memmap[ei] = 0.0
                        self._pi0_memmap[ei, : pi0_actions.shape[0], :] = pi0_actions
                    else:
                        self._pi0_cache[ei] = pi0_actions
                pending_indices.clear()
                pending_obs.clear()

            processed_eps += 1
            if (processed_eps % 10) == 0 or processed_eps == total:
                _log(f"预计算进度: {processed_eps}/{total}")

        # flush 剩余未满一批的数据
        if pending_obs:
            results = (
                self.policy.batch_infer(pending_obs)
            )
            for ei, out in zip(pending_indices, results):
                pi0_actions = np.asarray(out["actions"], dtype=np.float32)
                pi0_actions = pi0_actions[:, -7:]  # [50, 7]
                if self._use_disk_cache:
                    assert self._pi0_memmap is not None
                    self._pi0_memmap[ei] = 0.0
                    self._pi0_memmap[ei, : pi0_actions.shape[0], :] = pi0_actions
                else:
                    self._pi0_cache[ei] = pi0_actions
        _log("Pi0 预计算完成")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        s = self.ds[idx]
        flat = _flatten_dict(s)

        # 获取真实动作 [T_full, A]
        gt_actions = _to_numpy(flat[self.action_key]).astype(np.float32) #这个是真值
        T_full, action_dim = gt_actions.shape

        marker_seq = _to_numpy(flat[self.marker_key])   
        if marker_seq.shape[0] != T_full:
            raise RuntimeError(f"marker序列长度 {marker_seq.shape[0]} 与动作长度 {T_full} 不一致")

        # marker -> PCA (用于和Pi0特征拼接)
        xs: List[np.ndarray] = []
        for t in range(T_full):
            xs.append(_marker_to_vector(marker_seq[t]))
        X = np.stack(xs, axis=0)
        Xp = self.pca.transform(X).astype(np.float32)  # [T, K]

        # 这个是从本地磁盘取；
        # 直接从缓存/单文件memmap取整段 [50,7]，再裁剪到 [T_full,7]
        pi0_full = self._pi0_cache.get(idx)
        if pi0_full is None and self._use_disk_cache and self._pi0_memmap is not None:
            pi0_full = np.asarray(self._pi0_memmap[idx])
            self._pi0_cache[idx] = pi0_full
        if pi0_full is None:
            raise RuntimeError(f"Pi0缓存缺失：样本 {idx}。请确保 precompute_pi0=True 或重新运行预计算。")
        pi0_actions = np.asarray(pi0_full[:T_full], dtype=np.float32)

        # 保存原始Pi0动作用于loss计算
        pi0_actions_original = pi0_actions.copy()
        
        # 对 Pi0 的后7维应用训练管线的 Z-score 标准化（仅用于GRU输入特征）
        if self._pi0_mean is not None and self._pi0_std is not None:
            # 形状广播：[T,7] 与 [7]
            pi0_actions_norm = (pi0_actions - self._pi0_mean) / (self._pi0_std + 1e-6)
            print("pi0_actions has been normalized for GRU input")
        else:
            pi0_actions_norm = pi0_actions
            print("pi0_actions has not been normalized for GRU input")

        # 恢复拼接：Pi0 后7维(标准化) + PCA(K)
        feat = np.concatenate([pi0_actions_norm, Xp], axis=-1)  # [T, 7+K]
        return torch.from_numpy(feat), torch.from_numpy(gt_actions), torch.from_numpy(pi0_actions_original)


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        y, _ = self.rnn(x)
        return self.out(y)  # [B, T, A]


def train_gru(
    repo_id: str,
    pca_path: Path,
    out_path: Path,
    *,
    horizon: int = 50,
    hidden_dim: int = 512,
    num_layers: int = 1,
    batch_size: int = 8,
    num_steps: int = 1000,
    lr: float = 1e-3,
    pi0_checkpoint: Path | None = None,
    pi0_config_name: str = "pi0_aloha_lora_finetune_peg",
    default_prompt: str | None = None,
    pi0_precompute_batch: int = 8,
    log_interval: int = 100,
    save_interval: int = 500,
    wandb_enabled: bool = True,
    wandb_project: str = "marker-pca-pi0-gru",
    wandb_name: str | None = None,
    pi0_cache_file: Path | None = None,
    refresh_pi0_cache: bool = False,
    train_steps: int | None = None,
) -> None:
    if pi0_checkpoint is None:
        raise ValueError("必须提供 --pi0-checkpoint，以便使用 Pi0 输出作为 GRU 的输入特征")

    cfg = _config.get_config(pi0_config_name)
    policy = _policy_config.create_trained_policy(cfg, str(pi0_checkpoint), default_prompt=default_prompt)

    # 加载训练管线的 actions 归一化统计（用于对 Pi0 后7维做 Z-score，与 PCA whiten 对齐）
    try:
        # 直接使用绝对路径
        norm_path = Path("/nfs/turbo/coe-vkamat/openpi/assets/pi0_aloha_lora_finetune_peg/xiejunz/peg_data/norm_stats.json")
        _log(f"尝试加载归一化统计: {norm_path}")
        
        if norm_path.exists():
            norm = _normalize.load(norm_path.parent)  # load 需要目录路径
            actions_mean = None
            actions_std = None
            if norm is not None and "actions" in norm:
                actions_mean_full = np.asarray(norm["actions"].mean, dtype=np.float32)
                actions_std_full = np.asarray(norm["actions"].std, dtype=np.float32)
                actions_mean = actions_mean_full[-7:]
                actions_std = actions_std_full[-7:]
                _log(
                    f"已加载 actions 归一化统计：右臂7维 mean/std 形状=({actions_mean.shape[0]},)/({actions_std.shape[0]},) (full=({actions_mean_full.shape[0]},))"
                )
            else:
                _log("归一化文件中未找到 actions 字段，将跳过对 Pi0 的 Z-score")
        else:
            _log(f"归一化统计文件不存在: {norm_path}，将跳过对 Pi0 的 Z-score")
            actions_mean = None
            actions_std = None
    except Exception as e:
        _log(f"加载归一化统计失败：{e}; 将跳过对 Pi0 的 Z-score")
        actions_mean = None
        actions_std = None

    dataset: Dataset = LeRobotPi0ActionMarkerDataset(
        repo_id, pca_path, policy, horizon=horizon, precompute_batch_size=pi0_precompute_batch,
        pi0_cache_file=pi0_cache_file, refresh_pi0_cache=refresh_pi0_cache,
        actions_norm_mean=actions_mean, actions_norm_std=actions_std,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 预取一批确定维度
    sample_feat, sample_gt, sample_pi0 = next(iter(loader))
    feature_dim = sample_feat.shape[-1]
    action_dim = sample_gt.shape[-1]
    effective_train_steps = train_steps if train_steps is not None else sample_feat.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 固定输出维度为 7（仅预测右臂 7 维）
    output_dim = 7
    model = GRURegressor(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
    
    # 显示模型参数信息
    _log("=== GRU模型参数信息 ===")
    _log(f"输入维度 (input_dim): {feature_dim} (Pi0后7维 + PCA{feature_dim-7}维)")
    _log(f"隐藏维度 (hidden_dim): {hidden_dim}")
    _log(f"输出维度 (output_dim): {output_dim}")
    _log(f"层数 (num_layers): {num_layers}")
    _log(f"设备: {device}")
    _log(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    _log(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    # 使用余弦退火学习率调度器（默认启用）
    scheduler = CosineAnnealingLR(optim, T_max=num_steps)

    _log(f"开始 GRU 训练：steps={num_steps}, batch={batch_size}, horizon={horizon}, 有效时间步={effective_train_steps}, feature_dim={feature_dim} (Pi0后7维+PCA), action_dim={action_dim}, num_layers={num_layers}, lr={lr}")
    
    # 初始化wandb
    if wandb_enabled:
        # 如果没有指定wandb_name，自动生成一个随机名称
        if wandb_name is None:
            wandb_name = f"run_{uuid.uuid4().hex[:8]}"
        
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config={
                "repo_id": repo_id,
                "horizon": horizon,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "batch_size": batch_size,
                "num_steps": num_steps,
                "lr": lr,
                "pi0_config": pi0_config_name,
                "feature_dim": feature_dim,
                "action_dim": action_dim,
                "input_mode": "pi0_plus_pca",
                "pi0_precompute_batch": pi0_precompute_batch,
                "log_interval": log_interval,
                "save_interval": save_interval,
                "effective_train_steps": effective_train_steps,
            }
        )
        _log(f"Wandb initialized: project={wandb_project}, name={wandb_name}")
    
    tic = time.time()
    
    # 创建无限循环的数据迭代器
    data_iter = iter(loader)
    model.train()
    
    total_loss = 0.0
    total_pi0_loss = 0.0
    for step in range(1, num_steps + 1):
        try:
            feat, gt, pi0_actions = next(data_iter)
        except StopIteration:
            # 如果数据用完了，重新创建迭代器
            data_iter = iter(loader)
            feat, gt, pi0_actions = next(data_iter)
        
        feat = feat.to(device).float()
        gt = gt.to(device).float()
        pi0_actions = pi0_actions.to(device).float()

        # 仅使用前 effective_train_steps 个时间步
        if effective_train_steps is not None:
            t_use = min(effective_train_steps, feat.shape[1])
            feat = feat[:, :t_use]
            gt = gt[:, :t_use]
            pi0_actions = pi0_actions[:, :t_use]
        
        pred = model(feat)
        # 仅用 GT 的后 7 维计算损失（与固定的 GRU 输出对齐）
        gt_right7 = gt[..., 7:14]
        loss = loss_fn(pred, gt_right7)
        
        # 计算Pi0的loss用于对比（仅与GT的后7维对齐）
        pi0_loss = loss_fn(pi0_actions, gt_right7)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        # 更新学习率
        scheduler.step()
        total_loss += loss.item()
        total_pi0_loss += pi0_loss.item()
        
        # 日志记录
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            avg_pi0_loss = total_pi0_loss / log_interval
            improvement = ((avg_pi0_loss - avg_loss) / avg_pi0_loss) * 100 if avg_pi0_loss > 0 else 0
            
            current_lr = optim.param_groups[0]['lr']
            _log(f"Step {step}: GRU loss={avg_loss:.6f}, Pi0 loss={avg_pi0_loss:.6f}, improvement={improvement:.2f}%, lr={current_lr:.2e}")
            
            # 记录到wandb
            if wandb_enabled:
                wandb.log({
                    "train/gru_loss": avg_loss,
                    "train/pi0_loss": avg_pi0_loss,
                    "train/improvement_percent": improvement,
                    "train/step": step,
                    "train/learning_rate": current_lr,
                }, step=step)
            
            total_loss = 0.0
            total_pi0_loss = 0.0

        # 每 1000 步将一个 batch 的 GRU 输出、Pi0 输出与 GT 写到 outputs/record 目录
        if step % 1000 == 0 and step > 0:
            try:
                from pathlib import Path as _Path
                _out_dir = _Path("outputs/record")
                _out_dir.mkdir(parents=True, exist_ok=True)

                # 取当前张量到 CPU 并转为 numpy
                pred_np = pred.detach().cpu().numpy()
                pi0_np = pi0_actions.detach().cpu().numpy()
                gt_np = gt_right7.detach().cpu().numpy()

                # 记录原始形状并展平为 [B*T, 7]
                def _flatten_bt7(x: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
                    b, t, d = x.shape
                    return x.reshape(b * t, d), (b, t, d)

                pred_flat, pred_shape = _flatten_bt7(pred_np)
                pi0_flat, pi0_shape = _flatten_bt7(pi0_np)
                gt_flat, gt_shape = _flatten_bt7(gt_np)

                dump_path = _out_dir / f"train_batch_step_{step}.txt"
                with open(dump_path, "w", encoding="utf-8") as f:
                    f.write(f"pred_shape={pred_shape}, pi0_shape={pi0_shape}, gt_shape={gt_shape}\n")
                    f.write("# pred (B*T x 7)\n")
                    np.savetxt(f, pred_flat, fmt="%.6f")
                    f.write("\n# pi0 (B*T x 7)\n")
                    np.savetxt(f, pi0_flat, fmt="%.6f")
                    f.write("\n# gt (B*T x 7)\n")
                    np.savetxt(f, gt_flat, fmt="%.6f")
                _log(f"已写入训练批次dump: {dump_path}")
            except Exception as _e:
                _log(f"写入训练批次dump失败(已忽略): {_e}")
        
        # 模型保存
        if step % save_interval == 0:
            # 兼容两种用法：
            # 1) --gru-out 指向目录（推荐）：.../gru/
            # 2) --gru-out 指向文件（不带扩展名也可）：.../gru/iterations
            save_dir = out_path if out_path.suffix == "" else out_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            # 创建带迭代步数的文件名
            iteration_path = save_dir / f"iteration_{step}"
            torch.save({
                "state_dict": model.state_dict(), 
                "input_dim": feature_dim, 
                "hidden_dim": hidden_dim, 
                "num_layers": num_layers,
                "output_dim": output_dim,
                "step": step,
                "optimizer": optim.state_dict(),
            }, iteration_path)
            _log(f"模型已保存到: {iteration_path} (step {step})")
            
            # 记录模型保存到wandb
            if wandb_enabled:
                wandb.log({
                    "checkpoint/save_step": step,
                }, step=step)
    
    # 最终保存（同样兼容目录/文件两种写法）
    save_dir = out_path if out_path.suffix == "" else out_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    # 创建带最终迭代步数的文件名
    final_path = save_dir / f"iteration_{num_steps}"
    torch.save({
        "state_dict": model.state_dict(), 
        "input_dim": feature_dim, 
        "hidden_dim": hidden_dim, 
        "num_layers": num_layers,
        "output_dim": output_dim,
        "step": num_steps,
        "optimizer": optim.state_dict(),
    }, final_path)
    # 记录最终训练时间
    training_time = time.time() - tic
    _log(f"训练完成，用时 {training_time:.1f}s，最终模型已保存到: {final_path}")
    
    # 记录最终结果到wandb
    if wandb_enabled:
        final_gru_loss = total_loss / log_interval if total_loss > 0 else 0
        final_pi0_loss = total_pi0_loss / log_interval if total_pi0_loss > 0 else 0
        final_improvement = ((final_pi0_loss - final_gru_loss) / final_pi0_loss) * 100 if final_pi0_loss > 0 else 0
        
        wandb.log({
            "train/final_gru_loss": final_gru_loss,
            "train/final_pi0_loss": final_pi0_loss,
            "train/final_improvement_percent": final_improvement,
            "train/total_time": training_time,
            "train/total_steps": num_steps,
        }, step=num_steps)
        wandb.finish()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Marker PCA + Pi0 → GRU joint training (inputs = Pi0 actions last 7 dims + marker PCA, target = ground-truth actions)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo_id")
    p.add_argument("--horizon", type=int, default=50, help="Temporal window length")
    p.add_argument("--hidden", type=int, default=256, help="GRU hidden size")
    p.add_argument("--num-layers", type=int, default=1, help="Number of GRU layers")
    p.add_argument("--num-steps", type=int, default=1000, help="Number of training steps")
    p.add_argument("--log-interval", type=int, default=100, help="Log interval in steps")
    p.add_argument("--save-interval", type=int, default=500, help="Save interval in steps")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--pi0-checkpoint", type=Path, required=True, help="Pi0 checkpoint directory (must contain the params subdirectory)")
    p.add_argument("--pi0-config", type=str, default="pi0_aloha_lora_finetune_peg", help="Pi0 config name")
    p.add_argument("--prompt", type=str, default=None, help="Default inference prompt (optional)")
    p.add_argument("--pi0-precompute-batch", type=int, default=8, help="Pi0 预计算时的批大小")
    p.add_argument("--skip-pca", action="store_true", help="Skip PCA training and load from --pca-file")
    p.add_argument("--pca-out", type=Path, default=Path("checkpoints/pi0_aloha_lora_finetune_peg/PCA/pca_marker_right.joblib"), help="Output path to save PCA model (joblib)")
    p.add_argument("--pca-file", type=Path, default=None, help="PCA joblib to load when --skip-pca is set")
    p.add_argument("--gru-out", type=Path, default=Path("checkpoints/pi0_aloha_lora_finetune_peg/gru"), help="Output path to save GRU weights")
    p.add_argument("--wandb-enabled", action="store_true", default=True, help="Enable wandb logging")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--wandb-project", type=str, default="marker-pca-pi0-gru", help="Wandb project name")
    p.add_argument("--wandb-name", type=str, default=None, help="Wandb run name")
    p.add_argument("--pi0-cache-file", type=Path, default=Path("checkpoints/pi0_aloha_lora_finetune_peg/pi0_cache.npy"), help="使用单文件 .npy 缓存（形状为 [N, 50, 7]）")
    p.add_argument("--refresh-pi0-cache", action="store_true", help="强制刷新 Pi0 缓存")
    p.add_argument("--train-steps", type=int, default=25, help="仅在训练与计算loss时使用前N个时间步（<=horizon）")

    args = p.parse_args()

    # 1) 训练或加载 PCA
    if args.skip_pca:
        pca_path = args.pca_file or args.pca_out
        if not Path(pca_path).exists():
            raise FileNotFoundError(f"未找到 PCA 文件: {pca_path}")
        _log(f"跳过 PCA 训练，直接使用: {pca_path}")
    else:
        _log("开始训练 PCA…")
        train_pca_on_marker_lerobot(args.repo_id, args.pca_out)
        pca_path = args.pca_out

    # 2) 训练 GRU（输入=Pi0 动作 + marker PCA，目标=真实动作）
    _log("开始训练 GRU…")
    
    # 处理wandb参数
    wandb_enabled = args.wandb_enabled and not args.no_wandb
    
    train_gru(
        args.repo_id,
        pca_path,
        args.gru_out,
        horizon=args.horizon,
        hidden_dim=args.hidden,
        num_layers=args.num_layers,
        batch_size=args.batch,
        num_steps=args.num_steps,
        lr=args.lr,
        pi0_checkpoint=args.pi0_checkpoint,
        pi0_config_name=args.pi0_config,
        default_prompt=args.prompt,
        pi0_precompute_batch=args.pi0_precompute_batch,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        wandb_enabled=wandb_enabled,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        pi0_cache_file=args.pi0_cache_file,
        refresh_pi0_cache=args.refresh_pi0_cache,
        train_steps=args.train_steps,
    )


if __name__ == "__main__":
    main()