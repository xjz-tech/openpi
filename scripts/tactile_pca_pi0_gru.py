import argparse
import random
import time
from pathlib import Path
from typing import Any, List

import numpy as np
from joblib import dump as joblib_dump, load as joblib_load
from sklearn.decomposition import PCA as SkPCA

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi.transforms import flatten_dict as _flatten_dict
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config


def _log(msg: str) -> None:
    print(f"[tactile_pca_pi0_gru] {msg}")

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


def _image_to_gray_vector(img: np.ndarray) -> np.ndarray:
    x = _to_numpy(img)
    # Accept HWC or CHW; convert to HWC
    if x.ndim == 3 and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        gray = x.astype(np.float32)
    elif x.ndim == 3 and x.shape[2] == 1:
        gray = x[..., 0].astype(np.float32)
    elif x.ndim == 3 and x.shape[2] == 3:
        x = x.astype(np.float32)
        gray = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    else:
        raise ValueError(f"Unsupported image shape for tactile: {x.shape}")
    return gray.reshape(-1)


def _find_tactile_key_from_sample(sample: dict, substring: str) -> str:
    flat = _flatten_dict(sample)
    candidates = [k for k in flat.keys() if substring in k]
    # Prefer observation.images.* namespace
    candidates.sort(key=lambda k: ("observation.images" not in k, k))
    if not candidates:
        raise RuntimeError(f"在样本中找不到包含 '{substring}' 的触觉键")
    return candidates[0]


def train_pca_on_tactile_lerobot(
    repo_id: str,
    out_path: Path,
    k: int = 64,
    max_images: int = 10_000,
    tactile_key_substring: str = "tactile_right",
) -> SkPCA:
    probe_ds = lerobot_dataset.LeRobotDataset(repo_id)
    probe_sample = probe_ds[0]
    tactile_key = _find_tactile_key_from_sample(probe_sample, tactile_key_substring)
    _log(f"探测到触觉键: {tactile_key}")

    X_list: List[np.ndarray] = []
    total = len(probe_ds)
    _log(f"开始处理全部 {total} 个样本...")
    for i in range(total):
        s = probe_ds[i]
        flat = _flatten_dict(s)
        img = _to_numpy(flat[tactile_key])
        X_list.append(_image_to_gray_vector(img))
    if not X_list:
        raise RuntimeError("未能从 LeRobot 数据集中提取到触觉图像")

    X = np.stack(X_list, axis=0).astype(np.float32)
    _log("使用 scikit-learn 进行 PCA 训练…")
    sk_pca = SkPCA(n_components=k, svd_solver="auto", random_state=0)
    sk_pca.fit(X)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib_dump(sk_pca, out_path)
    _log(f"PCA 训练完成并已保存到: {out_path} (K={k}, D={X.shape[1]}) 来自 repo_id={repo_id}")
    return sk_pca


class LeRobotPi0ActionTactileDataset(Dataset):  # type: ignore[misc]
    """LeRobot 数据；特征=Pi0 预测动作 + 触觉 PCA；目标=真实动作。"""

    def __init__(
        self,
        repo_id: str,
        pca_path: Path,
        policy: Any,
        *,
        horizon: int = 50,
        tactile_key_substring: str = "tactile_right",
    ) -> None:
        self.repo_id = repo_id
        self.horizon = horizon
        self.pca: SkPCA = joblib_load(pca_path)
        self.policy = policy

        probe_ds = lerobot_dataset.LeRobotDataset(repo_id)
        probe_sample = probe_ds[0]
        self.tactile_key = _find_tactile_key_from_sample(probe_sample, tactile_key_substring)
        self.ds = lerobot_dataset.LeRobotDataset(repo_id)

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

        tactile_full = _to_numpy(flat[self.tactile_key])
        if tactile_full.shape[0] != T_full:
            raise RuntimeError(f"触觉序列长度 {tactile_full.shape[0]} 与动作长度 {T_full} 不一致")

        # 随机时间窗（长度=horizon），不足则重复最后一帧
        if T_full > self.horizon:
            start = random.randint(0, T_full - self.horizon)
        else:
            start = 0
        idxs = np.clip(np.arange(start, start + self.horizon), 0, T_full - 1)

        gt_actions = gt_full[idxs]
        tactile_seq = tactile_full[idxs]

        # 触觉 → PCA
        xs: List[np.ndarray] = []
        for t in range(self.horizon):
            xs.append(_image_to_gray_vector(tactile_seq[t]))
        X = np.stack(xs, axis=0)
        Xp = self.pca.transform(X).astype(np.float32)  # [T, K]

        # 构造 Pi0 推理 obs（取该窗口首帧）
        def _get_first(key: str, fallback: Any | None = None) -> Any:
            val = flat.get(key, fallback)
            if val is None:
                return None
            arr = _to_numpy(val)
            return arr if arr.ndim == 1 or arr.ndim == 3 else arr[idxs[0]]

        # Aloha/Libero 两种输入风格都兼容：policy 内部有 repack+normalize
        obs = {}
        # 优先使用 Aloha 风格相机键
        base_img = _get_first("observation.images.cam_high")
        left_wrist = _get_first("observation.images.cam_left_wrist")
        right_wrist = _get_first("observation.images.cam_right_wrist")
        if base_img is None:
            # 退化为 Libero 风格键
            base_img = _get_first("observation/image")
            left_wrist = _get_first("observation/wrist_image")
        # 确保 HWC uint8
        def _as_hwc_uint8(x: Any, like: np.ndarray | None = None) -> np.ndarray:
            if x is None:
                shape = (224, 224, 3) if like is None else like.shape
                return np.zeros(shape, dtype=np.uint8)
            arr = _to_numpy(x)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.uint8)
            return arr

        base_img = _as_hwc_uint8(base_img)
        left_wrist = _as_hwc_uint8(left_wrist, like=base_img)
        right_wrist = _as_hwc_uint8(right_wrist, like=base_img)

        state = _get_first("observation.state")
        if state is None:
            # 若无状态则以动作维度构造零向量
            state = np.zeros((action_dim,), dtype=np.float32)

        obs = {
            # Aloha repack 需要的原始键
            "observation.images.cam_high": base_img,
            "observation.images.cam_left_wrist": left_wrist,
            "observation.images.cam_right_wrist": right_wrist,
            "observation.state": state.astype(np.float32),
        }

        # Pi0 推理，得到预测动作序列 [T2, A]
        pred = self.policy.infer(obs)
        pi0_actions = np.asarray(pred["actions"], dtype=np.float32)
        if pi0_actions.ndim == 1:
            pi0_actions = pi0_actions[None, :]

        # 以 Pi0 输出长度为准对齐（真实动作/触觉特征填充或截断）
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
        return torch.from_numpy(feat), torch.from_numpy(gt_actions)


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
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
    hidden_dim: int = 256,
    batch_size: int = 8,
    epochs: int = 1,
    pi0_checkpoint: Path | None = None,
    pi0_config_name: str = "pi0_aloha_lora_finetune_peg",
    default_prompt: str | None = None,
) -> None:
    if pi0_checkpoint is None:
        raise ValueError("必须提供 --pi0-checkpoint，以便使用 Pi0 输出作为 GRU 的输入特征")

    cfg = _config.get_config(pi0_config_name)
    policy = _policy_config.create_trained_policy(cfg, str(pi0_checkpoint), default_prompt=default_prompt)

    dataset: Dataset = LeRobotPi0ActionTactileDataset(repo_id, pca_path, policy, horizon=horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 预取一批确定维度
    sample_feat, sample_gt = next(iter(loader))
    feature_dim = sample_feat.shape[-1]
    action_dim = sample_gt.shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRURegressor(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=action_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    _log(f"开始 GRU 训练：epochs={epochs}, batch={batch_size}, horizon={horizon}, feature_dim={feature_dim}, action_dim={action_dim}")
    tic = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for feat, gt in loader:
            feat = feat.to(device).float()
            gt = gt.to(device).float()
            pred = model(feat)
            loss = loss_fn(pred, gt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            num_batches += 1
        _log(f"Epoch {epoch}: loss={total_loss / max(1, num_batches):.6f}")
    _log(f"训练完成，用时 {time.time() - tic:.1f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "input_dim": feature_dim, "hidden_dim": hidden_dim, "output_dim": action_dim}, out_path)
    _log(f"GRU 模型已保存到: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tactile PCA + Pi0 → GRU joint training (inputs = Pi0 actions + tactile PCA, target = ground-truth actions)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo_id")
    p.add_argument("--horizon", type=int, default=50, help="Temporal window length")
    p.add_argument("--hidden", type=int, default=256, help="GRU hidden size")
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--pi0-checkpoint", type=Path, required=True, help="Pi0 checkpoint directory (must contain the params subdirectory)")
    p.add_argument("--pi0-config", type=str, default="pi0_aloha_lora_finetune_peg", help="Pi0 config name")
    p.add_argument("--prompt", type=str, default=None, help="Default inference prompt (optional)")
    p.add_argument("--skip-pca", action="store_true", help="Skip PCA training and load from --pca-file")
    p.add_argument("--pca-out", type=Path, default=Path("artifacts/pca_tactile_right.joblib"), help="Output path to save PCA model (joblib)")
    p.add_argument("--pca-file", type=Path, default=None, help="PCA joblib to load when --skip-pca is set")
    p.add_argument("--gru-out", type=Path, default=Path("artifacts/gru.pth"), help="Output path to save GRU weights")

    args = p.parse_args()

    # 1) 训练或加载 PCA
    if args.skip_pca:
        pca_path = args.pca_file or args.pca_out
        if not Path(pca_path).exists():
            raise FileNotFoundError(f"未找到 PCA 文件: {pca_path}")
        _log(f"跳过 PCA 训练，直接使用: {pca_path}")
    else:
        _log("开始训练 PCA…")
        train_pca_on_tactile_lerobot(args.repo_id, args.pca_out)
        pca_path = args.pca_out

    # 2) 训练 GRU（输入=Pi0 动作 + 触觉 PCA，目标=真实动作）
    _log("开始训练 GRU…")
    train_gru(
        args.repo_id,
        pca_path,
        args.gru_out,
        horizon=args.horizon,
        hidden_dim=args.hidden,
        batch_size=args.batch,
        epochs=args.epochs,
        pi0_checkpoint=args.pi0_checkpoint,
        pi0_config_name=args.pi0_config,
        default_prompt=args.prompt,
    )


if __name__ == "__main__":
    main()

