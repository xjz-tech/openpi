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

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi.transforms import flatten_dict as _flatten_dict
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
"""
python scripts/tactile_pca_pi0_gru.py \
    --repo-id "xiejunz/peg_data" \
    --pi0-checkpoint checkpoints/pi0_aloha_lora_finetune_peg/peg_finetune_lora_full/19999 \
    --num-steps 20000 \
    --batch 32 \
    --lr 1e-4 \
    --hidden 32 \
    --log-interval 100 \
    --save-interval 1000 \
    --wandb-project "marker-pca-pi0-gru" \
    --wandb-enabled 
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


def _find_marker_key_from_sample(sample: dict, substring: str = "marker_tracking_right_dxdy") -> str:
    flat = _flatten_dict(sample)
    candidates = [k for k in flat.keys() if substring in k]
    # Prefer observation.* namespace
    candidates.sort(key=lambda k: ("observation" not in k, k))
    if not candidates:
        raise RuntimeError(f"在样本中找不到包含 '{substring}' 的marker键")
    return candidates[0]


def train_pca_on_marker_lerobot(
    repo_id: str,
    out_path: Path,
    k: int = 7,
    max_samples: int = 10_000,
    marker_key_substring: str = "marker_tracking_right_dxdy",
) -> SkPCA:
    probe_ds = lerobot_dataset.LeRobotDataset(repo_id)
    probe_sample = probe_ds[0]
    marker_key = _find_marker_key_from_sample(probe_sample, marker_key_substring)
    _log(f"探测到marker键: {marker_key}")

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
    _log("使用 scikit-learn 进行 PCA 训练…")
    sk_pca = SkPCA(n_components=k, svd_solver="auto", random_state=0)
    sk_pca.fit(X)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib_dump(sk_pca, out_path)
    _log(f"PCA 训练完成并已保存到: {out_path} (K={k}, D={X.shape[1]}) 来自 repo_id={repo_id}")
    return sk_pca


class LeRobotPi0ActionMarkerDataset(Dataset):  # type: ignore[misc]
    """LeRobot 数据；特征=Pi0 预测动作 + marker PCA；目标=真实动作。"""

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
    ) -> None:
        self.repo_id = repo_id
        self.horizon = horizon
        self.pca: SkPCA = joblib_load(pca_path)
        self.policy = policy
        self._pi0_cache: dict[int, dict[int, np.ndarray]] = {}
        self._precompute_batch_size = precompute_batch_size

        probe_ds = lerobot_dataset.LeRobotDataset(repo_id)
        probe_sample = probe_ds[0]
        self.marker_key = _find_marker_key_from_sample(probe_sample, marker_key_substring)
        self.ds = lerobot_dataset.LeRobotDataset(repo_id)

        if precompute_pi0:
            self._precompute_all_pi0_actions(batch_size=self._precompute_batch_size)

    def _as_chw_uint8(self, x: Any, like: np.ndarray | None = None) -> np.ndarray:
        if x is None:
            shape = (3, 224, 224) if like is None else like.shape
            return np.zeros(shape, dtype=np.uint8)
        arr = _to_numpy(x)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            arr = np.transpose(arr, (2, 0, 1))
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.uint8)
        return arr

    def _prepare_obs_at(self, flat: dict, t: int, action_dim: int) -> dict:
        def _get_time_step(key: str, fallback: Any | None = None) -> Any:
            val = flat.get(key, fallback)
            if val is None:
                return None
            arr = _to_numpy(val)
            return arr if arr.ndim == 1 or arr.ndim == 3 else arr[t]

        base_img = _get_time_step("observation.images.cam_high")
        left_wrist = _get_time_step("observation.images.cam_left_wrist")
        right_wrist = _get_time_step("observation.images.cam_right_wrist")
        if base_img is None:
            base_img = _get_time_step("observation/image")
            left_wrist = _get_time_step("observation/wrist_image")

        base_img = self._as_chw_uint8(base_img)
        left_wrist = self._as_chw_uint8(left_wrist, like=base_img)
        right_wrist = self._as_chw_uint8(right_wrist, like=base_img)

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

    def _infer_and_cache(self, idx: int, t: int, obs: dict) -> np.ndarray:
        pi0_out = self.policy.infer(obs)
        pi0_actions = np.asarray(pi0_out["actions"], dtype=np.float32)
        if pi0_actions.ndim == 1:
            pi0_actions = pi0_actions[None, :]
        self._pi0_cache.setdefault(idx, {})[t] = pi0_actions
        return pi0_actions

    def _precompute_all_pi0_actions(self, batch_size: int = 8) -> None:
        _log("开始预计算 Pi0 推理结果（全局批量、全时间步缓存）…")
        total = len(self.ds)
        pending_indices: list[tuple[int, int]] = []  # (episode_idx, t)
        pending_obs: list[dict] = []
        processed_eps = 0

        for i in range(total):
            s = self.ds[i]
            flat = _flatten_dict(s)
            action_key = "action" if "action" in flat else ("actions" if "actions" in flat else None)
            if action_key is None:
                continue
            gt_full = _to_numpy(flat[action_key]).astype(np.float32)
            if gt_full.ndim == 1:
                gt_full = gt_full[None, :]
            T_full, action_dim = gt_full.shape

            for t in range(T_full):
                obs = self._prepare_obs_at(flat, t, action_dim)
                pending_indices.append((i, t))
                pending_obs.append(obs)

                if len(pending_obs) >= batch_size:
                    results = (
                        self.policy.batch_infer(pending_obs)
                        if hasattr(self.policy, "batch_infer")
                        else [self.policy.infer(o) for o in pending_obs]
                    )
                    for (ei, tt), out in zip(pending_indices, results):
                        pi0_actions = np.asarray(out["actions"], dtype=np.float32)
                        if pi0_actions.ndim == 1:
                            pi0_actions = pi0_actions[None, :]
                        self._pi0_cache.setdefault(ei, {})[tt] = pi0_actions
                    pending_indices.clear()
                    pending_obs.clear()

            processed_eps += 1
            if (processed_eps % 10) == 0 or processed_eps == total:
                _log(f"预计算进度: {processed_eps}/{total}")

        # flush 剩余未满一批的数据
        if pending_obs:
            results = (
                self.policy.batch_infer(pending_obs)
                if hasattr(self.policy, "batch_infer")
                else [self.policy.infer(o) for o in pending_obs]
            )
            for (ei, tt), out in zip(pending_indices, results):
                pi0_actions = np.asarray(out["actions"], dtype=np.float32)
                if pi0_actions.ndim == 1:
                    pi0_actions = pi0_actions[None, :]
                self._pi0_cache.setdefault(ei, {})[tt] = pi0_actions
        _log("Pi0 预计算完成")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        s = self.ds[idx]
        flat = _flatten_dict(s)

        # 获取真实动作 [T_full, A]
        action_key = "action" if "action" in flat else ("actions" if "actions" in flat else None)
        if action_key is None:
            raise RuntimeError("样本中未找到 'action' 或 'actions' 键")
        gt_full = _to_numpy(flat[action_key]).astype(np.float32)
        if gt_full.ndim == 1:
            gt_full = gt_full[None, :]
        T_full, action_dim = gt_full.shape

        marker_full = _to_numpy(flat[self.marker_key])
        # 如果是1D (126,)，扩展为 [1, 126]
        if marker_full.ndim == 1:
            marker_full = marker_full[None, :]
        
        if marker_full.shape[0] != T_full:
            raise RuntimeError(f"marker序列长度 {marker_full.shape[0]} 与动作长度 {T_full} 不一致")

        # 随机时间窗（长度=horizon），不足则重复最后一帧
        if T_full > self.horizon:
            start = random.randint(0, T_full - self.horizon)
        else:
            start = 0
        idxs = np.clip(np.arange(start, start + self.horizon), 0, T_full - 1)

        gt_actions = gt_full[idxs]
        marker_seq = marker_full[idxs]

        # marker → PCA
        xs: List[np.ndarray] = []
        for t in range(self.horizon):
            xs.append(_marker_to_vector(marker_seq[t]))
        X = np.stack(xs, axis=0)
        Xp = self.pca.transform(X).astype(np.float32)  # [T, K]

        # 直接使用预计算的Pi0结果
        t0 = int(idxs[0])
        pi0_actions = self._pi0_cache[idx][t0]

        # 以 Pi0 输出长度为准对齐（真实动作/marker特征填充或截断）
        T2 = pi0_actions.shape[0]
        if self.horizon < T2:
            last_gt = gt_actions[-1:]
            gt_actions = np.concatenate([gt_actions, np.tile(last_gt, (T2 - self.horizon, 1))], axis=0)
            last_xp = Xp[-1:]
            Xp = np.concatenate([Xp, np.tile(last_xp, (T2 - self.horizon, 1))], axis=0)
        elif self.horizon > T2:
            gt_actions = gt_actions[:T2]
            Xp = Xp[:T2]

        feat = np.concatenate([pi0_actions, Xp], axis=-1)  # [T2, A+K]
        return torch.from_numpy(feat), torch.from_numpy(gt_actions), torch.from_numpy(pi0_actions)


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
) -> None:
    if pi0_checkpoint is None:
        raise ValueError("必须提供 --pi0-checkpoint，以便使用 Pi0 输出作为 GRU 的输入特征")

    cfg = _config.get_config(pi0_config_name)
    policy = _policy_config.create_trained_policy(cfg, str(pi0_checkpoint), default_prompt=default_prompt)

    dataset: Dataset = LeRobotPi0ActionMarkerDataset(
        repo_id, pca_path, policy, horizon=horizon, precompute_batch_size=pi0_precompute_batch
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 预取一批确定维度
    sample_feat, sample_gt, sample_pi0 = next(iter(loader))
    feature_dim = sample_feat.shape[-1]
    action_dim = sample_gt.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRURegressor(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=action_dim, num_layers=num_layers).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    _log(f"开始 GRU 训练：steps={num_steps}, batch={batch_size}, horizon={horizon}, feature_dim={feature_dim}, action_dim={action_dim}, num_layers={num_layers}, lr={lr}")
    
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
                "pi0_precompute_batch": pi0_precompute_batch,
                "log_interval": log_interval,
                "save_interval": save_interval,
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
        
        pred = model(feat)
        loss = loss_fn(pred, gt)
        
        # 计算Pi0的loss用于对比
        pi0_loss = loss_fn(pi0_actions, gt)
        
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_loss += loss.item()
        total_pi0_loss += pi0_loss.item()
        
        # 日志记录
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            avg_pi0_loss = total_pi0_loss / log_interval
            improvement = ((avg_pi0_loss - avg_loss) / avg_pi0_loss) * 100 if avg_pi0_loss > 0 else 0
            
            _log(f"Step {step}: GRU loss={avg_loss:.6f}, Pi0 loss={avg_pi0_loss:.6f}, improvement={improvement:.2f}%")
            
            # 记录到wandb
            if wandb_enabled:
                wandb.log({
                    "train/gru_loss": avg_loss,
                    "train/pi0_loss": avg_pi0_loss,
                    "train/improvement_percent": improvement,
                    "train/step": step,
                    "train/learning_rate": lr,
                }, step=step)
            
            total_loss = 0.0
            total_pi0_loss = 0.0
        
        # 模型保存
        if step % save_interval == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(), 
                "input_dim": feature_dim, 
                "hidden_dim": hidden_dim, 
                "num_layers": num_layers,
                "output_dim": action_dim,
                "step": step,
                "optimizer": optim.state_dict(),
            }, out_path)
            _log(f"模型已保存到: {out_path} (step {step})")
            
            # 记录模型保存到wandb
            if wandb_enabled:
                wandb.log({
                    "checkpoint/save_step": step,
                }, step=step)
    
    # 最终保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), 
        "input_dim": feature_dim, 
        "hidden_dim": hidden_dim, 
        "num_layers": num_layers,
        "output_dim": action_dim,
        "step": num_steps,
        "optimizer": optim.state_dict(),
    }, out_path)
    # 记录最终训练时间
    training_time = time.time() - tic
    _log(f"训练完成，用时 {training_time:.1f}s，最终模型已保存到: {out_path}")
    
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
        description="Marker PCA + Pi0 → GRU joint training (inputs = Pi0 actions + marker PCA, target = ground-truth actions)",
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
    p.add_argument("--gru-out", type=Path, default=Path("checkpoints/pi0_aloha_lora_finetune_peg/gru/iterations"), help="Output path to save GRU weights")
    p.add_argument("--wandb-enabled", action="store_true", default=True, help="Enable wandb logging")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    p.add_argument("--wandb-project", type=str, default="marker-pca-pi0-gru", help="Wandb project name")
    p.add_argument("--wandb-name", type=str, default=None, help="Wandb run name")

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
    )


if __name__ == "__main__":
    main()

