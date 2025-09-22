import argparse
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
import matplotlib.pyplot as plt


"""
python scripts/tactile_train_valid.py \
    --repo-id xiejunz/peg_data \
    --pi0-checkpoint checkpoints/pi0_aloha_lora_finetune_peg/peg_finetune_lora_full/19999 \
    --pi0-config pi0_aloha_lora_finetune_peg \
    --num-steps 20000 \
    --train-steps 25 \
    --batch 32 \
    --hidden 32 \
    --num-layers 1 \
    --lr 1e-3 \
    --pi0-precompute-batch 100 \
    --log-interval 100 \
    --save-interval 2000 \
    --gru-out checkpoints/pi0_aloha_lora_finetune_peg/gru \
    --valid-out outputs/valid \
    --pi0-cache-file checkpoints/pi0_aloha_lora_finetune_peg/pi0_cache.npy \
    --skip-pca \
    --refresh-pi0-cache
"""


def _log(msg: str) -> None:
    print(f"[tactile_train_valid] {msg}")


# ---------- 通用工具 ----------
def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import jax  # type: ignore

        if isinstance(x, jax.Array):
            return np.asarray(x)
    except Exception:
        pass
    try:
        import torch as _torch  # type: ignore

        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _marker_to_vector(marker: np.ndarray) -> np.ndarray:
    x = _to_numpy(marker)
    if x.ndim == 1:
        return x.astype(np.float32)
    elif x.ndim == 2:
        return x[0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported marker shape: {x.shape}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _as_chw_uint8(x: Any, like: Optional[np.ndarray] = None) -> np.ndarray:
    if x is None:
        shape = (3, 224, 224) if like is None else like.shape
        return np.zeros(shape, dtype=np.uint8)
    arr = _to_numpy(x)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (2, 0, 1))
    if arr.dtype != np.uint8:
        arr = (
            (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if np.issubdtype(arr.dtype, np.floating)
            else arr.astype(np.uint8)
        )
    return arr


def _prepare_obs_at(flat: dict, t: int, action_dim_fallback: int) -> dict:
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

    base_img = _as_chw_uint8(base_img)
    left_wrist = _as_chw_uint8(left_wrist, like=base_img)
    right_wrist = _as_chw_uint8(right_wrist, like=base_img)

    state = _get_time_step("observation.state")
    if state is None:
        state = np.zeros((action_dim_fallback,), dtype=np.float32)

    return {
        "images": {
            "cam_high": base_img,
            "cam_left_wrist": left_wrist,
            "cam_right_wrist": right_wrist,
        },
        "state": state.astype(np.float32),
    }


def _find_marker_key_from_sample(sample: dict, substring: str = "marker_tracking_right_dxdy") -> str:
    flat = _flatten_dict(sample)
    candidates = [k for k in flat.keys() if substring in k]
    candidates.sort(key=lambda k: ("observation" not in k, k))
    if not candidates:
        raise RuntimeError(f"在样本中找不到包含 '{substring}' 的marker键")
    return candidates[0]


def _build_right_arm_figure(pi0_right: np.ndarray, gru_right: np.ndarray, gt_right: np.ndarray, title: str):
    T = pi0_right.shape[0]
    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    t = np.arange(T)
    joint_labels = [f"right_joint_{i+1}" for i in range(7)]
    for j in range(7):
        ax = axes[j]
        if T > 0:
            ax.plot(t, pi0_right[:, j], label="Pi0", color="#1f77b4", linewidth=1.5)
            ax.plot(t, gru_right[:, j], label="GRU", color="#d62728", linewidth=1.2)
            ax.plot(t, gt_right[:, j], label="Ground Truth", color="#2ca02c", linewidth=1.0)
        ax.set_ylabel(joint_labels[j])
        ax.grid(True, linestyle=":", alpha=0.4)
        if j == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("timestep")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig


# ---------- 与原脚本相同的 PCA 训练 ----------
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


# ---------- 与原脚本相同的数据集（Pi0后7维 + PCA） ----------
class LeRobotPi0ActionMarkerDataset(Dataset):  # type: ignore[misc]
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
        actions_norm_mean: np.ndarray | None = None,
        actions_norm_std: np.ndarray | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.horizon = horizon
        self.pca: SkPCA = joblib_load(pca_path)
        self.policy = policy
        self._pi0_cache: dict[int, np.ndarray] = {}
        self._precompute_batch_size = precompute_batch_size

        self._pi0_cache_file: Path | None = Path(pi0_cache_file) if pi0_cache_file is not None else None
        self._use_disk_cache: bool = pi0_cache_file is not None
        self._refresh_pi0_cache: bool = refresh_pi0_cache
        self._pi0_memmap: np.ndarray | None = None

        if actions_norm_mean is not None and actions_norm_std is not None:
            self._pi0_mean = actions_norm_mean.astype(np.float32)
            self._pi0_std = actions_norm_std.astype(np.float32)
        else:
            self._pi0_mean = None
            self._pi0_std = None

        self.marker_key = "observation.marker_tracking_right_dxdy"
        self.action_key = "action"

        fps = 50.0
        delta_for_seq = [i / fps for i in range(self.horizon)]

        probe_ds = lerobot_dataset.LeRobotDataset(repo_id)
        probe_sample = probe_ds[0]
        flat_probe = _flatten_dict(probe_sample)

        sequence_keys = []
        if self.action_key in flat_probe:
            sequence_keys.append(self.action_key)
        if self.marker_key in flat_probe:
            sequence_keys.append(self.marker_key)

        delta_timestamps: dict[str, list[float]] = {}
        for k in sequence_keys:
            delta_timestamps[k] = delta_for_seq

        self.ds = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

        if precompute_pi0:
            self._precompute_all_pi0_actions(batch_size=self._precompute_batch_size)

    def _parse_image(self, image) -> np.ndarray:
        if image is None:
            return np.zeros((3, 224, 224), dtype=np.uint8)
        arr = np.asarray(image)
        if np.issubdtype(arr.dtype, np.floating):
            arr = (255 * arr).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.repeat(arr[None, ...], 3, axis=0)
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        elif arr.ndim == 3 and arr.shape[0] == 3:
            pass
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
            arr = np.transpose(arr, (2, 0, 1))
        else:
            if arr.ndim >= 3:
                hwc = arr.reshape(*arr.shape[-3:])
                if hwc.shape[-1] == 3:
                    arr = np.transpose(hwc, (2, 0, 1))
                else:
                    h, w = hwc.shape[0], hwc.shape[1]
                    tmp = np.zeros((3, h, w), dtype=np.uint8)
                    arr = tmp
            else:
                arr = np.zeros((3, 224, 224), dtype=np.uint8)
        return arr.astype(np.uint8)

    def _prepare_obs_at(self, flat: dict, t: int, action_dim: int) -> dict:
        def _get_time_step(key: str, fallback: Any | None = None) -> Any:
            val = flat.get(key, fallback)
            if val is None:
                return None
            arr = _to_numpy(val)
            return arr if arr.ndim == 1 or arr.ndim == 3 else arr[t]

        base_img = _get_time_step("observation.images.cam_high")
        if base_img is None:
            base_img = _get_time_step("observation/image")
        left_wrist = _get_time_step("observation.images.cam_left_wrist")
        if left_wrist is None:
            left_wrist = _get_time_step("observation/wrist_image")
        right_wrist = _get_time_step("observation.images.cam_right_wrist")

        base_img = self._parse_image(base_img)
        left_wrist = self._parse_image(left_wrist)
        right_wrist = self._parse_image(right_wrist) if right_wrist is not None else np.zeros_like(base_img)

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
        pending_indices: list[int] = []
        pending_obs: list[dict] = []
        processed_eps = 0

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
            self._pi0_memmap = np.lib.format.open_memmap(
                self._pi0_cache_file, mode="w+", dtype=np.float32, shape=expected_shape
            )

        for i in range(total):
            s = self.ds[i]
            flat = _flatten_dict(s)
            gt_full = _to_numpy(flat[self.action_key]).astype(np.float32)
            if gt_full.ndim == 1:
                raise RuntimeError(
                    f"动作数据维度错误：期望2D数组 [T, A]，但得到1D数组 {gt_full.shape}。请检查delta_timestamps配置。"
                )
            T_full, action_dim = gt_full.shape

            obs = self._prepare_obs_at(flat, 0, action_dim)
            pending_indices.append(i)
            pending_obs.append(obs)

            if len(pending_obs) >= batch_size:
                results = self.policy.batch_infer(pending_obs)
                for ei, out in zip(pending_indices, results):
                    pi0_actions = np.asarray(out["actions"], dtype=np.float32)
                    pi0_actions = pi0_actions[:, -7:]
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

        if pending_obs:
            results = self.policy.batch_infer(pending_obs)
            for ei, out in zip(pending_indices, results):
                pi0_actions = np.asarray(out["actions"], dtype=np.float32)
                pi0_actions = pi0_actions[:, -7:]
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

        gt_actions = _to_numpy(flat[self.action_key]).astype(np.float32)
        T_full, action_dim = gt_actions.shape

        marker_seq = _to_numpy(flat[self.marker_key])
        if marker_seq.shape[0] != T_full:
            raise RuntimeError(f"marker序列长度 {marker_seq.shape[0]} 与动作长度 {T_full} 不一致")

        xs: List[np.ndarray] = []
        for t in range(T_full):
            xs.append(_marker_to_vector(marker_seq[t]))
        X = np.stack(xs, axis=0)
        Xp = self.pca.transform(X).astype(np.float32)

        pi0_full = self._pi0_cache.get(idx)
        if pi0_full is None and self._use_disk_cache and self._pi0_memmap is not None:
            pi0_full = np.asarray(self._pi0_memmap[idx])
            self._pi0_cache[idx] = pi0_full
        if pi0_full is None:
            raise RuntimeError(
                f"Pi0缓存缺失：样本 {idx}。请确保 precompute_pi0=True 或重新运行预计算。"
            )
        pi0_actions = np.asarray(pi0_full[:T_full], dtype=np.float32)

        pi0_actions_original = pi0_actions.copy()

        if hasattr(self, "_pi0_mean") and hasattr(self, "_pi0_std") and self._pi0_mean is not None and self._pi0_std is not None:
            pi0_actions_norm = (pi0_actions - self._pi0_mean) / (self._pi0_std + 1e-6)
        else:
            pi0_actions_norm = pi0_actions

        feat = np.concatenate([pi0_actions_norm, Xp], axis=-1)
        return torch.from_numpy(feat), torch.from_numpy(gt_actions), torch.from_numpy(pi0_actions_original)


# ---------- 与原脚本相同的 GRU ----------
class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        return self.out(y)


# ---------- 单次验证：随机抽一个 episode，按 eval 流程绘图 ----------
def _extract_episode_id(flat: dict) -> Optional[int]:
    if "episode_index" in flat:
        return int(flat["episode_index"])
    if "episode_idx" in flat:
        return int(flat["episode_idx"])
    if "episode_id" in flat:
        return int(flat["episode_id"])
    return None


def _build_delta_timestamps(fps: float, horizon: int) -> list[float]:
    return [t / float(fps) for t in range(horizon)]


def _detect_sequence_keys(sample: dict, marker_key_substring: str) -> tuple[list[str], str, Optional[str], list[str]]:
    flat = _flatten_dict(sample)
    action_key: Optional[str] = None
    if "action" in flat:
        action_key = "action"
    elif "actions" in flat:
        action_key = "actions"
    marker_key = _find_marker_key_from_sample(sample, marker_key_substring)
    image_keys: list[str] = []
    for k in (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "observation/image",
        "observation/wrist_image",
    ):
        if k in flat:
            image_keys.append(k)
    seq_keys = [k for k in image_keys]
    if "observation.state" in flat:
        seq_keys.append("observation.state")
    if action_key is not None:
        seq_keys.append(action_key)
    if marker_key is not None:
        seq_keys.append(marker_key)
    seen: set[str] = set()
    unique_seq_keys: list[str] = []
    for k in seq_keys:
        if k not in seen:
            seen.add(k)
            unique_seq_keys.append(k)
    return unique_seq_keys, marker_key, action_key, image_keys


@torch.no_grad()
def run_validation_once(
    *,
    repo_id: str,
    out_dir: Path,
    sk_pca: SkPCA,
    gru: nn.Module,
    policy: Any,
    actions_mean_right7: Optional[np.ndarray],
    actions_std_right7: Optional[np.ndarray],
    marker_key_substring: str,
    horizon: int,
    max_start_limit: int = 500,
    iteration: Optional[int] = None,
) -> Optional[Path]:
    _log(f"[valid] 构建验证数据集: {repo_id}")
    probe_ds = lerobot_dataset.LeRobotDataset(repo_id)
    if len(probe_ds) == 0:
        _log("[valid] 验证集为空，跳过")
        return None
    probe_sample = probe_ds[0]
    seq_keys, marker_key_seq, action_key_probe, _ = _detect_sequence_keys(probe_sample, marker_key_substring)
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    fps = float(getattr(meta, "fps", 50.0))
    delta = _build_delta_timestamps(fps, int(horizon))
    delta_timestamps = {k: delta for k in seq_keys}
    ds = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    total_windows = len(ds)
    if total_windows == 0:
        _log("[valid] 无窗口可评估，跳过")
        return None

    # 先扫描一遍，统计每个 episode 在当前约束下的可用窗口数量，然后选择一个可行的 episode
    rnd_idx = 0  # 回退时使用
    rnd_sample = ds[rnd_idx]
    rnd_sample_flat = _flatten_dict(rnd_sample)

    ep_available_counts: dict[int, int] = {}
    for i in range(total_windows):
        sample_scan = ds[i]
        flat_scan = _flatten_dict(sample_scan)
        ep_id_scan_opt = _extract_episode_id(flat_scan)
        ep_id_scan = ep_id_scan_opt if ep_id_scan_opt is not None else -1
        # 与后续一致的起点限制（近似用 i 作为起点估计）
        if i > int(max_start_limit):
            continue
        ep_available_counts[ep_id_scan] = ep_available_counts.get(ep_id_scan, 0) + 1

    # 选择可用窗口最多的 episode；若为空则回退
    if ep_available_counts:
        rnd_ep_id = max(ep_available_counts.items(), key=lambda kv: kv[1])[0]
        _log(f"[valid] 选择可用窗口最多的 episode: {rnd_ep_id} (windows={ep_available_counts[rnd_ep_id]})")
    else:
        rnd_ep_id = -1
        _log("[valid] 未找到任何可用 episode，将回退为单窗口评估")

    # 收集该 episode 的所有窗口（按 stride 相当于连续拼接）
    pi0_chunks: List[np.ndarray] = []
    gru_chunks: List[np.ndarray] = []
    gt_chunks: List[np.ndarray] = []
    dropped_total = 0

    action_key_fallback = action_key_probe
    action_dim_default = 14

    hx: Optional[torch.Tensor] = None
    gru.eval()

    for i in range(total_windows):
        sample = ds[i]
        flat = _flatten_dict(sample)
        ep_id = _extract_episode_id(flat)
        if (ep_id or -1) != rnd_ep_id:
            continue

        # 限制窗口起点
        cnt = i  # 近似作为起点估计
        if cnt > int(max_start_limit):
            continue

        # 取动作序列 (H, A)
        action_key = "action" if "action" in flat else ("actions" if "actions" in flat else action_key_fallback)
        action_dim = action_dim_default
        if action_key is not None and action_key in flat:
            gt_full = _to_numpy(flat[action_key]).astype(np.float32)
            if gt_full.ndim == 1:
                gt_full = gt_full[None, :]
            action_dim = int(gt_full.shape[1])
            gt_actions = gt_full
        else:
            gt_actions = np.zeros((horizon, action_dim), dtype=np.float32)

        # marker 序列 (H, D)
        marker_full = _to_numpy(flat[marker_key_seq])
        if marker_full.ndim == 1:
            marker_full = marker_full[None, :]

        # Pi0：窗口首帧观测做一次 chunk 推理
        obs0 = _prepare_obs_at(flat, 0, action_dim)
        out = policy.infer(obs0)
        pi0_chunk = np.asarray(out["actions"], dtype=np.float32)
        pi0_chunk = pi0_chunk[:, -7:]
        if pi0_chunk.ndim == 1:
            pi0_chunk = pi0_chunk[None, :]
        if pi0_chunk.shape[0] < horizon:
            pad = np.tile(pi0_chunk[-1:], (horizon - pi0_chunk.shape[0], 1))
            pi0_chunk = np.concatenate([pi0_chunk, pad], axis=0)

        T = int(horizon)
        pi0_actions = np.zeros((T, 7), dtype=np.float32)
        gru_actions = np.zeros((T, 7), dtype=np.float32)
        dropped_steps = 0

        for t in range(T):
            pi0_t = pi0_chunk[t]
            pi0_actions[t] = pi0_t.copy()

            if actions_mean_right7 is not None and actions_std_right7 is not None:
                pi0_t_norm = (pi0_t - actions_mean_right7) / (actions_std_right7 + 1e-6)
            else:
                pi0_t_norm = pi0_t

            mv = _to_numpy(marker_full[min(t, marker_full.shape[0] - 1)]).astype(np.float32)
            if mv.ndim != 1:
                mv = mv.reshape(-1)
            if mv.size != 126:
                gru_t = pi0_t_norm
                dropped_steps += 1
            else:
                marker_feat = sk_pca.transform(mv.reshape(1, -1))[0].astype(np.float32)
                feat = np.concatenate([pi0_t_norm, marker_feat], axis=-1)
                feat_t = torch.from_numpy(feat).float().view(1, 1, -1)
                out_any = gru(feat_t)
                out_seq = out_any[0] if isinstance(out_any, tuple) else out_any
                gru_t = out_seq[0, 0].cpu().numpy().astype(np.float32)

            if np.ndim(gru_t) == 0:
                gru_t = np.array([gru_t], dtype=np.float32)
            if int(gru_t.shape[0]) == 7:
                gru_actions[t] = gru_t
            elif int(gru_t.shape[0]) == int(action_dim):
                gru_actions[t] = gru_t[7:14]
            else:
                if int(gru_t.shape[0]) > 7:
                    gru_actions[t] = gru_t[-7:]
                else:
                    tmp7 = np.zeros((7,), dtype=np.float32)
                    tmp7[: int(gru_t.shape[0])] = gru_t
                    gru_actions[t] = tmp7

        pi0_chunks.append(pi0_actions)
        gru_chunks.append(gru_actions)
        gt_chunks.append(gt_actions)
        dropped_total += dropped_steps

    if not pi0_chunks:
        _log("[valid] 未找到该 episode 的窗口，使用随机窗口直接评估（fallback）")
        # 用 rnd_sample 直接构造一次窗口评估
        flat = rnd_sample_flat
        action_key = "action" if "action" in flat else ("actions" if "actions" in flat else None)
        action_dim = 14
        if action_key is not None and action_key in flat:
            gt_full = _to_numpy(flat[action_key]).astype(np.float32)
            if gt_full.ndim == 1:
                gt_full = gt_full[None, :]
            action_dim = int(gt_full.shape[1])
            gt_actions = gt_full
        else:
            gt_actions = np.zeros((horizon, action_dim), dtype=np.float32)

        marker_key = _find_marker_key_from_sample(rnd_sample)
        marker_full = _to_numpy(flat[marker_key])
        if marker_full.ndim == 1:
            marker_full = marker_full[None, :]

        obs0 = _prepare_obs_at(flat, 0, action_dim)
        out = policy.infer(obs0)
        pi0_chunk = np.asarray(out["actions"], dtype=np.float32)
        pi0_chunk = pi0_chunk[:, -7:]
        if pi0_chunk.ndim == 1:
            pi0_chunk = pi0_chunk[None, :]
        if pi0_chunk.shape[0] < horizon:
            pad = np.tile(pi0_chunk[-1:], (horizon - pi0_chunk.shape[0], 1))
            pi0_chunk = np.concatenate([pi0_chunk, pad], axis=0)

        T = int(horizon)
        pi0_actions = np.zeros((T, 7), dtype=np.float32)
        gru_actions = np.zeros((T, 7), dtype=np.float32)
        dropped_steps = 0
        hx = None
        for t in range(T):
            pi0_t = pi0_chunk[t]
            pi0_actions[t] = pi0_t.copy()
            if actions_mean_right7 is not None and actions_std_right7 is not None:
                pi0_t_norm = (pi0_t - actions_mean_right7) / (actions_std_right7 + 1e-6)
            else:
                pi0_t_norm = pi0_t
            mv = _to_numpy(marker_full[min(t, marker_full.shape[0] - 1)]).astype(np.float32)
            if mv.ndim != 1:
                mv = mv.reshape(-1)
            if mv.size != 126:
                gru_t = pi0_t_norm
                dropped_steps += 1
            else:
                marker_feat = sk_pca.transform(mv.reshape(1, -1))[0].astype(np.float32)
                feat = np.concatenate([pi0_t_norm, marker_feat], axis=-1)
                feat_t = torch.from_numpy(feat).float().view(1, 1, -1)
                out_any = gru(feat_t)
                out_seq = out_any[0] if isinstance(out_any, tuple) else out_any
                gru_t = out_seq[0, 0].cpu().numpy().astype(np.float32)
            if np.ndim(gru_t) == 0:
                gru_t = np.array([gru_t], dtype=np.float32)
            if int(gru_t.shape[0]) == 7:
                gru_actions[t] = gru_t
            elif int(gru_t.shape[0]) == int(action_dim):
                gru_actions[t] = gru_t[7:14]
            else:
                if int(gru_t.shape[0]) > 7:
                    gru_actions[t] = gru_t[-7:]
                else:
                    tmp7 = np.zeros((7,), dtype=np.float32)
                    tmp7[: int(gru_t.shape[0])] = gru_t
                    gru_actions[t] = tmp7

        pi0_full = pi0_actions
        gru_right = gru_actions
        gt_full = gt_actions
        right_slice = slice(7, 14)
        mae_pi0 = float(np.nanmean(np.abs(pi0_full - gt_full[:, right_slice])))
        mae_gru = float(np.nanmean(np.abs(gru_right - gt_full[:, right_slice])))
        title = (
            f"episode_{(rnd_ep_id or -1):04d} | Fallback Single Window | steps 0-{len(pi0_full)-1} | "
            f"Pi0-GT MAE={mae_pi0:.4f} | GRU-GT MAE={mae_gru:.4f} | Dropped={dropped_steps}"
        )
        fig = _build_right_arm_figure(pi0_full, gru_right, gt_full[:, right_slice], title)
        _ensure_dir(out_dir)
        iter_tag = f"iteration_{iteration}_" if iteration is not None else ""
        out_file = out_dir / f"{iter_tag}episode_{(rnd_ep_id or -1):04d}_fallback.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        _log(f"[valid] 已保存验证图(回退): {out_file}")
        return out_file

    pi0_full = np.concatenate(pi0_chunks, axis=0)
    gru_right = np.concatenate(gru_chunks, axis=0)
    gt_full = np.concatenate(gt_chunks, axis=0)
    right_slice = slice(7, 14)

    mae_pi0 = float(np.nanmean(np.abs(pi0_full - gt_full[:, right_slice])))
    mae_gru = float(np.nanmean(np.abs(gru_right - gt_full[:, right_slice])))

    title = (
        f"episode_{(rnd_ep_id or -1):04d} | Complete Episode | steps 0-{len(pi0_full)-1} | "
        f"Pi0-GT MAE={mae_pi0:.4f} | GRU-GT MAE={mae_gru:.4f} | Dropped={dropped_total}"
    )
    fig = _build_right_arm_figure(pi0_full, gru_right, gt_full[:, right_slice], title)
    _ensure_dir(out_dir)
    iter_tag = f"iteration_{iteration}_" if iteration is not None else ""
    out_file = out_dir / f"{iter_tag}episode_{(rnd_ep_id or -1):04d}.png"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"[valid] 已保存验证图: {out_file}")
    return out_file


def supports_hx(model: nn.Module) -> bool:
    # 兼容性：如果模型 forward 接受 hx，则返回 True。这里通过简单探测属性名来判断。
    return hasattr(model, "rnn") and isinstance(getattr(model, "rnn"), nn.GRU)


def cast_tuple(x: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(x, tuple) and len(x) == 2:
        return x[0], x[1]
    return x, None


# ---------- 训练（保持与原始脚本一致），并在保存时插入验证 ----------
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
    # 验证相关
    val_repo_id: Optional[str] = None,
    val_out_dir: Path = Path("outputs/valid"),
    val_marker_key_substring: str = "marker_tracking_right_dxdy",
    val_max_start: int = 500,
) -> None:
    if pi0_checkpoint is None:
        raise ValueError("必须提供 --pi0-checkpoint，以便使用 Pi0 输出作为 GRU 的输入特征")

    cfg = _config.get_config(pi0_config_name)
    policy = _policy_config.create_trained_policy(cfg, str(pi0_checkpoint), default_prompt=default_prompt)

    # 与原脚本相同：加载 actions 归一化统计（用于对 Pi0 后7维做 Z-score）
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
        repo_id,
        pca_path,
        policy,
        horizon=horizon,
        precompute_batch_size=pi0_precompute_batch,
        pi0_cache_file=pi0_cache_file,
        refresh_pi0_cache=refresh_pi0_cache,
        actions_norm_mean=actions_mean,
        actions_norm_std=actions_std,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sample_feat, sample_gt, sample_pi0 = next(iter(loader))
    feature_dim = sample_feat.shape[-1]
    action_dim = sample_gt.shape[-1]
    effective_train_steps = train_steps if train_steps is not None else sample_feat.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 固定输出维度为 7（仅预测右臂 7 维）
    output_dim_fixed = 7
    model = GRURegressor(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=output_dim_fixed, num_layers=num_layers).to(device)
    
    # 显示模型参数信息
    _log("=== GRU模型参数信息 ===")
    _log(f"输入维度 (input_dim): {feature_dim} (Pi0后7维 + PCA{feature_dim-7}维)")
    _log(f"隐藏维度 (hidden_dim): {hidden_dim}")
    _log(f"输出维度 (output_dim): {output_dim_fixed}")
    _log(f"层数 (num_layers): {num_layers}")
    _log(f"设备: {device}")
    _log(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    _log(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.L1Loss()
    scheduler = CosineAnnealingLR(optim, T_max=num_steps)

    _log(
        f"开始 GRU 训练：steps={num_steps}, batch={batch_size}, horizon={horizon}, 有效时间步={effective_train_steps}, "
        f"feature_dim={feature_dim} (Pi0后7维+PCA), action_dim={action_dim}, num_layers={num_layers}, lr={lr}"
    )

    if wandb_enabled:
        if wandb_name is None:
            wandb_name = f"run_{int(time.time())}"
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
            },
        )
        _log(f"Wandb initialized: project={wandb_project}, name={wandb_name}")

    tic = time.time()

    data_iter = iter(loader)
    model.train()

    total_loss = 0.0
    total_pi0_loss = 0.0
    for step in range(1, num_steps + 1):
        try:
            feat, gt, pi0_actions = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            feat, gt, pi0_actions = next(data_iter)

        feat = feat.to(device).float()
        gt = gt.to(device).float()
        pi0_actions = pi0_actions.to(device).float()

        if effective_train_steps is not None:
            t_use = min(effective_train_steps, feat.shape[1])
            feat = feat[:, :t_use]
            gt = gt[:, :t_use]
            pi0_actions = pi0_actions[:, :t_use]

        pred = model(feat)
        # 仅用 GT 的后 7 维计算损失（与固定 7 维输出对齐）
        gt_right7 = gt[..., 7:14]
        loss = loss_fn(pred, gt_right7)

        pi0_loss = loss_fn(pi0_actions, gt_right7)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        total_loss += loss.item()
        total_pi0_loss += pi0_loss.item()

        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            avg_pi0_loss = total_pi0_loss / log_interval
            improvement = ((avg_pi0_loss - avg_loss) / avg_pi0_loss) * 100 if avg_pi0_loss > 0 else 0
            current_lr = optim.param_groups[0]["lr"]
            _log(
                f"Step {step}: GRU loss={avg_loss:.6f}, Pi0 loss={avg_pi0_loss:.6f}, improvement={improvement:.2f}%, lr={current_lr:.2e}"
            )
            if wandb_enabled:
                wandb.log(
                    {
                        "train/gru_loss": avg_loss,
                        "train/pi0_loss": avg_pi0_loss,
                        "train/improvement_percent": improvement,
                        "train/step": step,
                        "train/learning_rate": current_lr,
                    },
                    step=step,
                )
            total_loss = 0.0
            total_pi0_loss = 0.0

        # 每 1000 步将一个 batch 的 GRU 输出、Pi0 输出与 GT 写到 outputs/record 目录
        if step % 500 == 0 and step > 0:
            try:
                from pathlib import Path as _Path
                _out_dir = _Path("outputs/record")
                _out_dir.mkdir(parents=True, exist_ok=True)

                pred_np = pred.detach().cpu().numpy()
                pi0_np = pi0_actions.detach().cpu().numpy()
                gt_right7_np = gt_right7.detach().cpu().numpy()

                def _flatten_bt7(x: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
                    b, t, d = x.shape
                    return x.reshape(b * t, d), (b, t, d)

                pred_flat, pred_shape = _flatten_bt7(pred_np)
                pi0_flat, pi0_shape = _flatten_bt7(pi0_np)
                gt_flat, gt_shape = _flatten_bt7(gt_right7_np)

                dump_path = _out_dir / f"train_batch_step_{step}.txt"
                with open(dump_path, "w", encoding="utf-8") as f:
                    f.write(f"pred_shape={pred_shape}, pi0_shape={pi0_shape}, gt_right7_shape={gt_shape}\n")
                    f.write("# pred (B*T x 7)\n")
                    np.savetxt(f, pred_flat, fmt="%.6f")
                    f.write("\n# pi0 (B*T x 7)\n")
                    np.savetxt(f, pi0_flat, fmt="%.6f")
                    f.write("\n# gt_right7 (B*T x 7)\n")
                    np.savetxt(f, gt_flat, fmt="%.6f")
                _log(f"已写入训练批次dump: {dump_path}")
            except Exception as _e:
                _log(f"写入训练批次dump失败(已忽略): {_e}")

        # 保存 + 验证
        if step % save_interval == 0 and step > 0:
            save_dir = out_path if out_path.suffix == "" else out_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            iteration_path = save_dir / f"iteration_{step}"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": feature_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "output_dim": output_dim_fixed,
                    "step": step,
                    "optimizer": optim.state_dict(),
                },
                iteration_path,
            )
            _log(f"模型已保存到: {iteration_path} (step {step})")
            if wandb_enabled:
                wandb.log({"checkpoint/save_step": step}, step=step)

            # 运行一次验证并输出图片：文件名含 iteration
            val_repo = val_repo_id or repo_id
            out_dir_this = val_out_dir / f"iteration_{step}"
            out_dir_this.mkdir(parents=True, exist_ok=True)
            # 使用与训练一致的 Z-score（右臂7维）
            try:
                _ = run_validation_once(
                    repo_id=val_repo,
                    out_dir=out_dir_this,
                    sk_pca=dataset.pca,  # 与训练相同的 PCA
                    gru=model,
                    policy=policy,
                    actions_mean_right7=actions_mean,
                    actions_std_right7=actions_std,
                    marker_key_substring=val_marker_key_substring,
                    horizon=int(effective_train_steps) if effective_train_steps is not None else int(horizon),
                    max_start_limit=int(val_max_start),
                    iteration=step,
                )
                # 验证结束后，恢复训练模式，避免 cudnn RNN backward 报错
                model.train()
            except Exception as e:
                _log(f"[valid] 运行验证失败（已跳过）：{e}")
                # 出错也确保恢复训练模式
                model.train()

    # 训练结束后最终保存
    save_dir = out_path if out_path.suffix == "" else out_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / f"iteration_{num_steps}"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_dim": output_dim_fixed,
            "step": num_steps,
            "optimizer": optim.state_dict(),
        },
        final_path,
    )
    training_time = time.time() - tic
    _log(f"训练完成，用时 {training_time:.1f}s，最终模型已保存到: {final_path}")

    if wandb_enabled:
        final_gru_loss = total_loss / log_interval if total_loss > 0 else 0
        final_pi0_loss = total_pi0_loss / log_interval if total_pi0_loss > 0 else 0
        final_improvement = (
            ((final_pi0_loss - final_gru_loss) / final_pi0_loss) * 100 if final_pi0_loss > 0 else 0
        )
        wandb.log(
            {
                "train/final_gru_loss": final_gru_loss,
                "train/final_pi0_loss": final_pi0_loss,
                "train/final_improvement_percent": final_improvement,
                "train/total_time": training_time,
                "train/total_steps": num_steps,
            },
            step=num_steps,
        )
        wandb.finish()


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Marker PCA + Pi0 → GRU joint training with on-save validation "
            "(inputs = Pi0 actions last 7 dims + marker PCA, target = ground-truth actions)"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo_id (train)")
    p.add_argument("--val-repo-id", type=str, default=None, help="LeRobot dataset repo_id for validation (default=train)")
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
    # 验证参数
    p.add_argument("--valid-out", type=Path, default=Path("outputs/valid"), help="验证图片输出目录（将按 iteration 建子目录）")
    p.add_argument("--valid-marker-key-substring", type=str, default="marker_tracking_right_dxdy", help="验证使用的 marker key 子串")
    p.add_argument("--valid-max-start", type=int, default=500, help="验证时窗口起点最大索引限制")

    args = p.parse_args()

    # 1) 训练或加载 PCA（与原脚本一致）
    if args.skip_pca:
        pca_path = args.pca_file or args.pca_out
        if not Path(pca_path).exists():
            raise FileNotFoundError(f"未找到 PCA 文件: {pca_path}")
        _log(f"跳过 PCA 训练，直接使用: {pca_path}")
    else:
        _log("开始训练 PCA…")
        train_pca_on_marker_lerobot(args.repo_id, args.pca_out)
        pca_path = args.pca_out

    # 2) 训练 GRU，并在保存 checkpoint 时运行验证
    _log("开始训练 GRU…")
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
        val_repo_id=args.val_repo_id,
        val_out_dir=args.valid_out,
        val_marker_key_substring=args.valid_marker_key_substring,
        val_max_start=args.valid_max_start,
    )


if __name__ == "__main__":
    main()


