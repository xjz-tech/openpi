import argparse
import threading
from typing import Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

import torch
from torch import nn
from joblib import load as joblib_load

from openpi_client import image_tools
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config


# 仅对 joint5 做限位（弧度制）。
JOINT5_MIN = -1.22
JOINT5_MAX = 1.22


class GRUCorrector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None):
        y, h = self.rnn(x, hx)
        y = self.out(y)
        return y, h


class PiperGRUController:
    """使用 Pi0 + GRU 校正的本地推理控制器。

    流程：
    - 每次从 Pi0 获取 action chunk（默认长度 50）
    - 执行前 25 步（open-loop），每步将当前 marker 向量经 PCA 后与 Pi0 对应步的动作拼接，输入 GRU 输出校正动作
    - 周期性重新进行 Pi0 推理以获得新的 chunk
    """

    def __init__(
        self,
        *,
        pi0_checkpoint_dir: str,
        pi0_config_name: str,
        prompt: str,
        pca_path: str,
        gru_checkpoint_path: str,
        img_front_topic: str,
        img_left_topic: str,
        img_right_topic: str,
        arm_left_state_topic: str,
        arm_right_state_topic: str,
        arm_left_cmd_topic: str,
        arm_right_cmd_topic: str,
        marker_topic: str = "/Marker_Tracking_Right_DXDY",
        marker_dim: int = 126,
        action_horizon: int = 50,
        open_loop_horizon: int = 25,
        rate_hz: int = 15,
    ) -> None:
        # ---------- Pi0 策略加载 ---------- #
        cfg = _config.get_config(pi0_config_name)
        self._policy = _policy_config.create_trained_policy(cfg, pi0_checkpoint_dir, default_prompt=prompt)

        # ---------- PCA 与 GRU 加载 ---------- #
        sk_pca = joblib_load(pca_path)
        ckpt = torch.load(gru_checkpoint_path, map_location="cpu")
        input_dim = int(ckpt.get("input_dim"))
        hidden_dim = int(ckpt.get("hidden_dim"))
        output_dim = int(ckpt.get("output_dim"))
        self._gru = GRUCorrector(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).eval()
        self._gru.load_state_dict(ckpt["state_dict"])
        self._gru_hx: Optional[torch.Tensor] = None
        self._pca = sk_pca

        # ---------- 参数 ---------- #
        self._marker_dim = marker_dim
        self._action_horizon = action_horizon
        self._open_loop_horizon = open_loop_horizon
        self._rate_hz = rate_hz
        self._prompt = prompt

        # ---------- 状态缓存 ---------- #
        self._bridge = CvBridge()
        self._front_image: Optional[np.ndarray] = None
        self._left_image: Optional[np.ndarray] = None
        self._right_image: Optional[np.ndarray] = None
        self._joint_left: Optional[np.ndarray] = None
        self._joint_right: Optional[np.ndarray] = None
        self._marker: Optional[np.ndarray] = None  # shape=(marker_dim,)

        self._pi0_chunk: Optional[np.ndarray] = None  # [H, A]
        self._pi0_step: int = 0

        # ---------- 订阅/发布 ---------- #
        rospy.Subscriber(img_front_topic, Image, self._front_cam_cb, queue_size=1)
        rospy.Subscriber(img_left_topic, Image, self._left_cam_cb, queue_size=1)
        rospy.Subscriber(img_right_topic, Image, self._right_cam_cb, queue_size=1)

        rospy.Subscriber(arm_left_state_topic, JointState, self._left_joint_state_cb, queue_size=1)
        rospy.Subscriber(arm_right_state_topic, JointState, self._right_joint_state_cb, queue_size=1)

        rospy.Subscriber(marker_topic, Float64MultiArray, self._marker_cb, queue_size=1)

        self._left_cmd_pub = rospy.Publisher(arm_left_cmd_topic, JointState, queue_size=1)
        self._right_cmd_pub = rospy.Publisher(arm_right_cmd_topic, JointState, queue_size=1)

    # --------------------- ROS 回调 --------------------- #
    def _front_cam_cb(self, msg: Image) -> None:
        self._front_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")

    def _left_cam_cb(self, msg: Image) -> None:
        self._left_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")

    def _right_cam_cb(self, msg: Image) -> None:
        self._right_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")

    def _left_joint_state_cb(self, msg: JointState) -> None:
        self._joint_left = np.array(msg.position[:7], dtype=np.float32)

    def _right_joint_state_cb(self, msg: JointState) -> None:
        self._joint_right = np.array(msg.position[:7], dtype=np.float32)

    def _marker_cb(self, msg: Float64MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.size >= self._marker_dim:
            self._marker = arr[: self._marker_dim]

    # ----------------------- 主循环 ----------------------- #
    def run(self) -> None:
        rate = rospy.Rate(self._rate_hz)
        while not rospy.is_shutdown():
            if not self._ready():
                rate.sleep()
                continue

            # 需要新的 Pi0 chunk 时重新推理
            if (
                self._pi0_chunk is None
                or self._pi0_step >= self._open_loop_horizon
                or self._pi0_step >= len(self._pi0_chunk)
            ):
                obs = self._build_obs()
                result = self._policy.infer(obs)
                actions = result["actions"]
                if actions.ndim == 1:
                    actions = actions[None, :]
                self._pi0_chunk = np.asarray(actions, dtype=np.float32)
                self._pi0_step = 0
                self._gru_hx = None  # 重置 GRU 隐状态

            # 当前步的 Pi0 动作
            pi0_action = self._pi0_chunk[self._pi0_step]

            # 当前 marker（若缺失则用零向量）
            marker_vec = self._marker if self._marker is not None else np.zeros((self._marker_dim,), dtype=np.float32)
            marker_vec = marker_vec.astype(np.float32, copy=False)
            # PCA -> [K]
            marker_feat = self._pca.transform(marker_vec.reshape(1, -1))[0].astype(np.float32)

            # 组装 GRU 特征：[A+K]
            feat = np.concatenate([pi0_action.astype(np.float32), marker_feat], axis=-1)
            feat_t = torch.from_numpy(feat).float().view(1, 1, -1)  # [B=1, T=1, F]

            with torch.no_grad():
                out_seq, self._gru_hx = self._gru(feat_t, self._gru_hx)
                action = out_seq[0, 0].cpu().numpy()  # [A]

            self._publish_action(action)
            self._pi0_step += 1
            rate.sleep()

    # ----------------------- 工具函数 ----------------------- #
    def _build_obs(self) -> dict:
        def _prep_img(img: Optional[np.ndarray]) -> np.ndarray:
            if img is None:
                return np.zeros((3, 224, 224), dtype=np.uint8)
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
            return np.transpose(img, (2, 0, 1))  # HWC -> CHW

        state = np.concatenate([self._joint_left, self._joint_right]).astype(np.float32)
        return {
            "state": state,
            "images": {
                "cam_high": _prep_img(self._front_image),
                "cam_left_wrist": _prep_img(self._left_image),
                "cam_right_wrist": _prep_img(self._right_image),
            },
        }

    def _publish_action(self, action: np.ndarray) -> None:
        # 允许 (1, 14) 或 (14,)
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]

        if action.shape[0] != 14:
            rospy.logwarn(f"Action 长度应为 14，但得到 {action.shape[0]}，跳过发布")
            return

        # joint5 限位
        action = action.copy()
        action[4] = float(np.clip(action[4], JOINT5_MIN, JOINT5_MAX))
        action[11] = float(np.clip(action[11], JOINT5_MIN, JOINT5_MAX))

        msg_left = JointState()
        msg_left.position = action[:7].tolist()
        self._left_cmd_pub.publish(msg_left)

        msg_right = JointState()
        msg_right.position = action[7:14].tolist()
        self._right_cmd_pub.publish(msg_right)

    def _ready(self) -> bool:
        return (
            self._front_image is not None
            and self._right_image is not None
            and self._joint_left is not None
            and self._joint_right is not None
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi0 + GRU 在线校正推理（Piper 示例）")
    parser.add_argument("--pi0-checkpoint", type=str, default="checkpoints/pi0_aloha_lora_finetune_peg/peg_finetune_lora_full/19999")
    parser.add_argument("--pi0-config", type=str, default="pi0_aloha_lora_finetune_peg")
    parser.add_argument("--prompt", type=str, default="put the eraser into the box")

    parser.add_argument("--pca-path", type=str, default="checkpoints/pi0_aloha_lora_finetune_peg/PCA/pca_marker_right.joblib")
    parser.add_argument("--gru-path", type=str, default="checkpoints/pi0_aloha_lora_finetune_peg/gru/iterations")

    parser.add_argument('--puppet_arm_left_cmd_topic', type=str, default='/left_arm/joint_ctrl_single')
    parser.add_argument('--puppet_arm_right_cmd_topic', type=str, default='/right_arm/joint_ctrl_single')
    parser.add_argument('--puppet_arm_left_topic', type=str, default='/left_arm/joint_states_single')
    parser.add_argument('--puppet_arm_right_topic', type=str, default='/right_arm/joint_states_single')
    parser.add_argument('--img_front_topic', type=str, default='/camera_f/color/image_raw')
    parser.add_argument('--img_left_topic', type=str, default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic', type=str, default='/camera_r/color/image_raw')
    parser.add_argument('--marker_topic', type=str, default='/Marker_Tracking_Right_DXDY')

    parser.add_argument('--action_horizon', type=int, default=50)
    parser.add_argument('--open_loop_horizon', type=int, default=25)
    parser.add_argument('--rate_hz', type=int, default=15)

    args = parser.parse_args()

    rospy.init_node("piper_gru_infer", anonymous=True)

    controller = PiperGRUController(
        pi0_checkpoint_dir=args.pi0_checkpoint,
        pi0_config_name=args.pi0_config,
        prompt=args.prompt,
        pca_path=args.pca_path,
        gru_checkpoint_path=args.gru_path,
        img_front_topic=args.img_front_topic,
        img_left_topic=args.img_left_topic,
        img_right_topic=args.img_right_topic,
        arm_left_state_topic=args.puppet_arm_left_topic,
        arm_right_state_topic=args.puppet_arm_right_topic,
        arm_left_cmd_topic=args.puppet_arm_left_cmd_topic,
        arm_right_cmd_topic=args.puppet_arm_right_cmd_topic,
        marker_topic=args.marker_topic,
        action_horizon=args.action_horizon,
        open_loop_horizon=args.open_loop_horizon,
        rate_hz=args.rate_hz,
    )

    threading.Thread(target=controller.run, daemon=True).start()
    rospy.spin()


if __name__ == "__main__":
    main()


