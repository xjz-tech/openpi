import argparse
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from joblib import load as joblib_load
import matplotlib.pyplot as plt

import torch
from torch import nn

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.transforms import flatten_dict as _flatten_dict
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize

"""
python scripts/eval_pi0_gru_peel.py \
  --repo-id xiejunz/peg_data \
  --out-dir outputs/pi0_gru_eval/peel_gru \
  --pca-path checkpoints/pi0_aloha_lora_finetune_peg/PCA/pca_marker_right.joblib \
  --gru-path checkpoints/pi0_aloha_lora_finetune_peg/gru/iteration_20000 \
  --pi0-checkpoint checkpoints/pi0_aloha_lora_finetune_peg/peg_finetune_lora_full/19999 \
  --pi0-config pi0_aloha_lora_finetune_peg \
  --prompt "put the eraser into the box" \
  --window-size 200 \
  --max-episodes 40
"""
def _log(message: str) -> None:
    print(f"[eval_pi0_gru] {message}")


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


def _find_marker_key_from_sample(sample: dict, substring: str = "marker_tracking_right_dxdy") -> str:
    flat = _flatten_dict(sample)
    candidates = [k for k in flat.keys() if substring in k]
    candidates.sort(key=lambda k: ("observation" not in k, k))
    if not candidates:
        raise RuntimeError(f"在样本中找不到包含 '{substring}' 的marker键")
    return candidates[0]


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


class GRUCorrector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        y, h = self.rnn(x, hx)
        y = self.out(y)
        return y, h


def _infer_pi0_action(policy: Any, obs: dict) -> np.ndarray:
    out = policy.infer(obs)
    actions = np.asarray(out["actions"], dtype=np.float32)
    # 只取后7维（前7维都是0）
    if actions.ndim == 1:
        actions = actions[-7:]
    else:
        actions = actions[:, -7:]
    if actions.ndim == 1:
        return actions
    return actions[0]


def _infer_pi0_chunk(policy: Any, obs: dict) -> np.ndarray:
    """返回 Pi0 的一段动作序列 [H, 7]。若只返回单步，则扩展为 [1, 7]。"""
    out = policy.infer(obs)

    actions = np.asarray(out["actions"], dtype=np.float32)
    actions = actions[:, -7:]
    if actions.ndim == 1:
        actions = actions[None, :]
    return actions


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_right_arm(pi0_actions: np.ndarray, gru_actions: np.ndarray, gt_actions: np.ndarray, out_file: Path, title: str) -> None:
    right_pi0 = pi0_actions[:, 0:7]
    right_gru = gru_actions[:, 7:14]
    right_gt = gt_actions[:, 7:14]
    T = right_pi0.shape[0]

    print(f"绘图调试: T={T}, pi0_shape={right_pi0.shape}, gru_shape={right_gru.shape}, gt_shape={right_gt.shape}")
    print(f"pi0范围: {right_pi0.min():.4f} ~ {right_pi0.max():.4f}")
    print(f"gru范围: {right_gru.min():.4f} ~ {right_gru.max():.4f}")
    print(f"gt范围: {right_gt.min():.4f} ~ {right_gt.max():.4f}")

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    t = np.arange(T)
    joint_labels = [f"right_joint_{i+1}" for i in range(7)]
    
    for j in range(7):
        ax = axes[j]
        if T > 0:
            ax.plot(t, right_pi0[:, j], label="Pi0", color="#1f77b4", linewidth=1.5, marker='o', markersize=2)
            ax.plot(t, right_gru[:, j], label="GRU", color="#d62728", linewidth=1.2, marker='s', markersize=2)
            ax.plot(t, right_gt[:, j], label="Ground Truth", color="#2ca02c", linewidth=1.0, marker='^', markersize=2)
        ax.set_ylabel(joint_labels[j])
        ax.grid(True, linestyle=":", alpha=0.4)
        if j == 0:
            ax.legend(loc="upper right")
    
    axes[-1].set_xlabel("timestep")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _build_right_arm_figure(pi0_actions: np.ndarray, gru_actions: np.ndarray, gt_actions: np.ndarray, title: str):
    right_pi0 = pi0_actions[:, 0:7]
    right_gru = gru_actions[:, 7:14]
    right_gt = gt_actions[:, 7:14]
    T = right_pi0.shape[0]

    fig, axes = plt.subplots(7, 1, figsize=(12, 14), sharex=True)
    t = np.arange(T)
    joint_labels = [f"right_joint_{i+1}" for i in range(7)]
    for j in range(7):
        ax = axes[j]
        if T > 0:
            ax.plot(t, right_pi0[:, j], label="Pi0", color="#1f77b4", linewidth=1.5)
            ax.plot(t, right_gru[:, j], label="GRU", color="#d62728", linewidth=1.2)
            ax.plot(t, right_gt[:, j], label="Ground Truth", color="#2ca02c", linewidth=1.0)
        ax.set_ylabel(joint_labels[j])
        ax.grid(True, linestyle=":", alpha=0.4)
        if j == 0:
            ax.legend(loc="upper right")
    axes[-1].set_xlabel("timestep")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig


def _extract_episode_id(flat: dict) -> Optional[int]:
    """从样本中提取 episode 标识。LeRobot使用episode_data_index管理episode。"""
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
    """从样本中推断需要做序列采样的键集合。

    返回 (sequence_keys, marker_key, action_key, image_keys)
    """
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
    # 去重保持顺序
    seen: set[str] = set()
    unique_seq_keys: list[str] = []
    for k in seq_keys:
        if k not in seen:
            seen.add(k)
            unique_seq_keys.append(k)
    return unique_seq_keys, marker_key, action_key, image_keys


def evaluate(
    repo_id: str,
    out_dir: Path,
    pca_path: Path,
    gru_path: Path,
    pi0_checkpoint: Path,
    pi0_config_name: str,
    prompt: Optional[str],
    marker_key_substring: str = "marker_tracking_right_dxdy",
    window_size: int = 200,
    max_episodes: Optional[int] = None,
    stride: int = 25,
    t_use: int = 25,  # 改为25，与stride一致
    max_start_limit: int = 500,
    gru_num_layers: int = 1,
) -> None:
    _log(f"加载数据集: {repo_id}")
    # 使用序列采样：探测需要的键，基于 fps 构造 delta_timestamps
    probe_ds = lerobot_dataset.LeRobotDataset(repo_id)
    probe_sample = probe_ds[0]
    seq_keys, marker_key_seq, action_key_probe, image_keys = _detect_sequence_keys(probe_sample, marker_key_substring)
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    fps = float(getattr(meta, "fps", 50.0))
    horizon = int(t_use)
    delta = _build_delta_timestamps(fps, horizon)
    delta_timestamps = {k: delta for k in seq_keys}
    ds = lerobot_dataset.LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    total_windows = len(ds)
    _log(f"构建序列数据集完成：窗口长度={horizon}, fps={fps:.1f}, 样本数(窗口)={total_windows}")

    _log("加载 Pi0 策略…")
    cfg = _config.get_config(pi0_config_name)
    policy = _policy_config.create_trained_policy(cfg, str(pi0_checkpoint), default_prompt=prompt)

    _log("加载 PCA 与 GRU…")
    sk_pca = joblib_load(pca_path)
    ckpt = torch.load(gru_path, map_location="cpu")
    input_dim = int(ckpt.get("input_dim"))
    hidden_dim = int(ckpt.get("hidden_dim"))
    output_dim = int(ckpt.get("output_dim"))
    num_layers = int(ckpt.get("num_layers", gru_num_layers))  # 优先使用checkpoint中的层数，否则使用参数
    gru = GRUCorrector(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).eval()
    gru.load_state_dict(ckpt["state_dict"])  # type: ignore[arg-type]

    # 加载训练管线的 actions 归一化统计（用于对 Pi0 后7维做 Z-score，与训练一致）
    actions_mean: Optional[np.ndarray]
    actions_std: Optional[np.ndarray]
    actions_mean = None
    actions_std = None
    try:
        cfg_assets = cfg.assets_dirs
        asset_id = cfg.data.asset_id or cfg.data.repo_id
        norm = _normalize.load(cfg_assets / asset_id) if asset_id is not None else None
        if norm is not None and "actions" in norm:
            actions_mean = np.asarray(norm["actions"].mean[-7:], dtype=np.float32)
            actions_std = np.asarray(norm["actions"].std[-7:], dtype=np.float32)
            _log(f"已加载 actions 归一化统计：mean/std 形状={actions_mean.shape}/{actions_std.shape}")
        else:
            _log("未找到 actions 的归一化统计，评估时将不对 Pi0 做 Z-score")
    except Exception as e:
        _log(f"加载归一化统计失败：{e}; 评估时将不对 Pi0 做 Z-score")

    _ensure_dir(out_dir)

    # 按episode分组处理：先收集一个episode的所有数据，然后生成图片
    ep_window_counters: dict[int, int] = {}
    processed_eps: set[int] = set()
    current_ep_data: dict[int, dict] = {}  # 当前episode的数据
    active_ep_id: Optional[int] = None

    for i in range(total_windows):
        sample = ds[i]
        flat = _flatten_dict(sample)

        ep_id_opt = _extract_episode_id(flat)
        ep_id = ep_id_opt if ep_id_opt is not None else -1

        # episode切换时收尾
        if active_ep_id is None:
            active_ep_id = ep_id
        elif ep_id != active_ep_id:
            prev_id = active_ep_id
            if prev_id in current_ep_data and len(current_ep_data[prev_id]['pi0_actions']) > 0:
                ep_name_prev = f"episode_{prev_id:04d}" if prev_id >= 0 else "episode_unknown"
                pi0_full_prev = np.concatenate(current_ep_data[prev_id]['pi0_actions'], axis=0)
                gru_full_prev = np.concatenate(current_ep_data[prev_id]['gru_actions'], axis=0)
                gt_full_prev = np.concatenate(current_ep_data[prev_id]['gt_actions'], axis=0)
                right_slice_prev = slice(7, 14)
                mae_pi0_gt_prev = float(np.nanmean(np.abs(pi0_full_prev - gt_full_prev[:, right_slice_prev])))
                mae_gru_gt_prev = float(np.nanmean(np.abs((gru_full_prev - gt_full_prev)[:, right_slice_prev])))
                title_prev = (
                    f"{ep_name_prev} | Complete Episode | steps 0-{len(pi0_full_prev)-1} | "
                    f"Pi0-GT MAE={mae_pi0_gt_prev:.4f} | GRU-GT MAE={mae_gru_gt_prev:.4f} | "
                    f"Windows={len(current_ep_data[prev_id]['pi0_actions'])} | Dropped={current_ep_data[prev_id]['dropped_steps']}"
                )
                fig_prev = _build_right_arm_figure(pi0_full_prev, gru_full_prev, gt_full_prev, title_prev)
                png_path_prev = out_dir / f"{ep_name_prev}_complete.png"
                fig_prev.savefig(png_path_prev, dpi=150, bbox_inches='tight')
                plt.close(fig_prev)
                _log(
                    f"{ep_name_prev}: 完整图已保存, 总步数={len(pi0_full_prev)}, "
                    f"Pi0-GT MAE={mae_pi0_gt_prev:.6f}, GRU-GT MAE={mae_gru_gt_prev:.6f}"
                )
                del current_ep_data[prev_id]
            active_ep_id = ep_id

        # 控制评估的 episode 数量
        if max_episodes is not None and ep_id not in processed_eps and len(processed_eps) >= max_episodes:
            continue
        processed_eps.add(ep_id)

        # 窗口筛选与步进
        cnt = ep_window_counters.get(ep_id, 0)
        ep_window_counters[ep_id] = cnt + 1
        start_est = cnt
        if start_est > int(max_start_limit):
            continue
        if cnt % int(stride) != 0:
            continue

        # 动作序列 [H, A]
        action_key = "action" if "action" in flat else ("actions" if "actions" in flat else action_key_probe)
        action_dim = 14
        if action_key is not None and action_key in flat:
            gt_full = _to_numpy(flat[action_key]).astype(np.float32)
            if gt_full.ndim == 1:
                gt_full = gt_full[None, :]
            action_dim = int(gt_full.shape[1])
            gt_actions = gt_full
        else:
            gt_actions = np.zeros((horizon, action_dim), dtype=np.float32)

        # marker 序列 [H, D]
        marker_key = marker_key_seq
        marker_full = _to_numpy(flat[marker_key])
        if marker_full.ndim == 1:
            marker_full = marker_full[None, :]

        # Pi0：窗口首帧观测做一次 chunk 推理
        obs0 = _prepare_obs_at(flat, 0, action_dim)
        pi0_chunk = _infer_pi0_chunk(policy, obs0).astype(np.float32)
        if pi0_chunk.shape[0] < horizon:
            pad = np.tile(pi0_chunk[-1:], (horizon - pi0_chunk.shape[0], 1))
            pi0_chunk = np.concatenate([pi0_chunk, pad], axis=0)

        T_use = horizon
        hx: Optional[torch.Tensor] = None
        pi0_actions = np.zeros((T_use, 7), dtype=np.float32)
        gru_actions = np.zeros((T_use, action_dim), dtype=np.float32)
        dropped_steps = 0

        for t in range(T_use):
            pi0_t = pi0_chunk[t]
            # 对 Pi0 的后7维应用 Z-score（若有统计），与训练一致
            if actions_mean is not None and actions_std is not None:
                pi0_t = (pi0_t - actions_mean) / (actions_std + 1e-6)
            pi0_actions[t] = pi0_t

            # 始终使用：Pi0(7) + PCA(7) 拼接作为 GRU 输入
            mv = _to_numpy(marker_full[min(t, marker_full.shape[0] - 1)]).astype(np.float32)
            if mv.ndim != 1:
                mv = mv.reshape(-1)
            if mv.size != 126:
                gru_t = pi0_t
                dropped_steps += 1
            else:
                marker_feat = sk_pca.transform(mv.reshape(1, -1))[0].astype(np.float32)
                feat = np.concatenate([pi0_t, marker_feat], axis=-1)
                feat_t = torch.from_numpy(feat).float().view(1, 1, -1)
                with torch.no_grad():
                    out_seq, hx = gru(feat_t, hx)
                    gru_t = out_seq[0, 0].cpu().numpy().astype(np.float32)
            gru_actions[t] = gru_t

        # 存储当前窗口数据
        if ep_id not in current_ep_data:
            current_ep_data[ep_id] = {
                'pi0_actions': [],
                'gru_actions': [],
                'gt_actions': [],
                'dropped_steps': 0
            }
        
        current_ep_data[ep_id]['pi0_actions'].append(pi0_actions)
        current_ep_data[ep_id]['gru_actions'].append(gru_actions)
        current_ep_data[ep_id]['gt_actions'].append(gt_actions)
        current_ep_data[ep_id]['dropped_steps'] += dropped_steps

        _log(f"Episode {ep_id}: 收集窗口 {cnt // stride}, dropped_steps={dropped_steps}")

    # 循环结束后，对最后一个episode做收尾
    if active_ep_id is not None and active_ep_id in current_ep_data and len(current_ep_data[active_ep_id]['pi0_actions']) > 0:
        ep_name_last = f"episode_{active_ep_id:04d}" if active_ep_id >= 0 else "episode_unknown"
        pi0_full_last = np.concatenate(current_ep_data[active_ep_id]['pi0_actions'], axis=0)
        gru_full_last = np.concatenate(current_ep_data[active_ep_id]['gru_actions'], axis=0)
        gt_full_last = np.concatenate(current_ep_data[active_ep_id]['gt_actions'], axis=0)
        right_slice_last = slice(7, 14)
        mae_pi0_gt_last = float(np.nanmean(np.abs(pi0_full_last - gt_full_last[:, right_slice_last])))
        mae_gru_gt_last = float(np.nanmean(np.abs((gru_full_last - gt_full_last)[:, right_slice_last])))
        title_last = (
            f"{ep_name_last} | Complete Episode | steps 0-{len(pi0_full_last)-1} | "
            f"Pi0-GT MAE={mae_pi0_gt_last:.4f} | GRU-GT MAE={mae_gru_gt_last:.4f} | "
            f"Windows={len(current_ep_data[active_ep_id]['pi0_actions'])} | Dropped={current_ep_data[active_ep_id]['dropped_steps']}"
        )
        fig_last = _build_right_arm_figure(pi0_full_last, gru_full_last, gt_full_last, title_last)
        png_path_last = out_dir / f"{ep_name_last}_complete.png"
        fig_last.savefig(png_path_last, dpi=150, bbox_inches='tight')
        plt.close(fig_last)
        _log(
            f"{ep_name_last}: 完整图已保存, 总步数={len(pi0_full_last)}, "
            f"Pi0-GT MAE={mae_pi0_gt_last:.6f}, GRU-GT MAE={mae_gru_gt_last:.6f}"
        )

    _log(f"评估完成。图片保存在: {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate Pi0 vs GRU on peel_gru dataset; save per-episode plots for right arm joints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo_id, e.g. xiejunz/peel_gru")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/pi0_gru_eval/peel_gru"), help="Output directory")
    p.add_argument("--pca-path", type=Path, required=True, help="Path to PCA joblib file")
    p.add_argument("--gru-path", type=Path, required=True, help="Path to GRU weights (torch.save file)")
    p.add_argument("--pi0-checkpoint", type=Path, required=True, help="Pi0 checkpoint directory")
    p.add_argument("--pi0-config", type=str, default="pi0_aloha_lora_finetune_peg", help="Pi0 config name")
    p.add_argument("--prompt", type=str, default=None, help="Default inference prompt (optional)")
    p.add_argument("--marker-key-substring", type=str, default="marker_tracking_right_dxdy", help="Marker key substring")
    p.add_argument("--window-size", type=int, default=200, help="Timesteps per plot window")
    p.addendant("--max-episodes", type=int, default=None, help="Limit number of episodes to evaluate")
    p.add_argument("--stride", type=int, default=25, help="Sliding window stride")
    p.add_argument("--t-use", type=int, default=25, help="Fixed window length for each evaluation")
    p.add_argument("--max-start", type=int, default=300, help="Max starting index for windows")
    p.add_argument("--gru-num-layers", type=int, default=1, help="Number of GRU layers (overridden by checkpoint if available)")

    args = p.parse_args()

    evaluate(
        repo_id=args.repo_id,
        out_dir=args.out_dir,
        pca_path=args.pca_path,
        gru_path=args.gru_path,
        pi0_checkpoint=args.pi0_checkpoint,
        pi0_config_name=args.pi0_config,
        prompt=args.prompt,
        marker_key_substring=args.marker_key_substring,
        window_size=args.window_size,
        max_episodes=args.max_episodes,
        stride=args.stride,
        t_use=args.t_use,
        max_start_limit=args.max_start,
        gru_num_layers=args.gru_num_layers,
    )


if __name__ == "__main__":
    main()


