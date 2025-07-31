import argparse
import threading
from typing import Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray

from openpi_client import image_tools
# 移除 websocket_client_policy 导入，改为本地策略
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
# 不再使用 ActionChunkBroker，而是直接调用模型策略


class PiperController:
    """示例：从 Piper 机器人读取**三个摄像头**与**左右臂关节状态**，1
    使用本地策略进行推理并分别向左右臂发布关节命令。"""

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        prompt: str,
        img_front_topic: str,
        img_left_topic: str,
        img_right_topic: str,
        arm_left_state_topic: str,
        arm_right_state_topic: str,
        arm_left_cmd_topic: str,
        arm_right_cmd_topic: str,
        action_horizon: int = 50,
        open_loop_horizon: int = 25,
    ) -> None:
        # 加载本地策略，直接返回完整动作序列 (action_horizon=50)
        config = _config.get_config("pi0_aloha_lora_finetune_peg")
        self._policy = _policy_config.create_trained_policy(config, checkpoint_dir)
        print("[DEBUG] Policy 加载完成")

        # -------- Open-loop 执行相关 -------- #
        self._open_loop_horizon = open_loop_horizon  # 每多少步重新推理一次
        self._actions_from_chunk_completed: int = 0  # 当前已执行的动作步数
        self._action_chunk: Optional[np.ndarray] = None  # 缓存的动作序列
        self._prompt = prompt
        self._bridge = CvBridge()

        # 最新观测缓存
        self._front_image: Optional[np.ndarray] = None  # 前置摄像头
        self._left_image: Optional[np.ndarray] = None   # 左腕摄像头
        self._right_image: Optional[np.ndarray] = None  # 右腕摄像头
        self._joint_left: Optional[np.ndarray] = None   # 左臂关节
        self._joint_right: Optional[np.ndarray] = None  # 右臂关节

        # ------------------- ROS 订阅 ------------------- #
        rospy.Subscriber(img_front_topic, Image, self._front_cam_cb, queue_size=1)
        rospy.Subscriber(img_left_topic, Image, self._left_cam_cb, queue_size=1)
        rospy.Subscriber(img_right_topic, Image, self._right_cam_cb, queue_size=1)

        rospy.Subscriber(arm_left_state_topic, JointState, self._left_joint_state_cb, queue_size=1)
        rospy.Subscriber(arm_right_state_topic, JointState, self._right_joint_state_cb, queue_size=1)

        # ------------------- ROS 发布 ------------------- #
        self._left_cmd_pub = rospy.Publisher(arm_left_cmd_topic, JointState, queue_size=1)
        self._right_cmd_pub = rospy.Publisher(arm_right_cmd_topic, JointState, queue_size=1)

        # 移动到初始位置
        print("[DEBUG] 开始重置到初始位置...")
        self._reset_to_home_position()
        print("[DEBUG] 初始化完成")

    # --------------------- 回调函数 --------------------- #
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

    # ----------------------- 主循环 ----------------------- #
    def run(self, rate_hz: int = 15) -> None:
        rate = rospy.Rate(rate_hz)
        try:
            while not rospy.is_shutdown():
                if not self._ready():
                    rate.sleep()
                    continue

                # ---------- 构造符合 ALOHA 策略的观测 ---------- #
                def _prep_img(img: Optional[np.ndarray]) -> np.ndarray:
                    if img is None:
                        # 摄像头原始分辨率通常是 640x480，调整到 224x224
                        return np.zeros((3, 224, 224), dtype=np.uint8)
                    # 正常处理图像：从原始分辨率调整到 224x224
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
                    return np.transpose(img, (2, 0, 1))  # HWC -> CHW

                obs = {
                    "state": np.concatenate([self._joint_left, self._joint_right]),
                    "images": {
                        "cam_high": _prep_img(self._front_image),
                        "cam_left_wrist": _prep_img(self._left_image),
                        "cam_right_wrist": _prep_img(self._right_image),
                    },
                    "prompt": self._prompt,
                }

                # ---------------- Open-loop 逻辑 --------------- #
                if (
                    self._action_chunk is None
                    or self._actions_from_chunk_completed >= self._open_loop_horizon
                    or self._actions_from_chunk_completed >= len(self._action_chunk)
                ):

                    print(f"[DEBUG] obs['prompt']: {obs['prompt']}")
                    
                    try:
                        inference_result = self._policy.infer(obs)
                        print(f"[DEBUG] 推理结果类型: {type(inference_result)}")
                        print(f"[DEBUG] 推理结果 keys: {list(inference_result.keys()) if isinstance(inference_result, dict) else 'Not a dict'}")
                        
                        if "actions" not in inference_result:
                            print(f"[ERROR] 推理结果中没有 'actions' 键")
                            print(f"[ERROR] 推理结果: {inference_result}")
                            raise ValueError("推理结果中没有 'actions' 键")
                        
                        self._action_chunk = inference_result["actions"]
                        if self._action_chunk.ndim == 1:
                            # 模型有时只输出单步动作，将其扩展成 (1, action_dim)
                            self._action_chunk = self._action_chunk[None, :]
                        print(f"[DEBUG] action_chunk 类型: {type(self._action_chunk)}")
                        print(f"[DEBUG] action_chunk shape: {self._action_chunk.shape if self._action_chunk is not None else 'None'}")
                        
                        # if self._action_chunk is None:
                        #     print(f"[ERROR] action_chunk 是 None")
                        #     raise ValueError("action_chunk 是 None")
                        
                        self._actions_from_chunk_completed = 0
                        print(f"[DEBUG] 推理完成，获取到 {len(self._action_chunk)} 个动作")
                    except Exception as e:
                        print(f"[ERROR] 推理过程中出现异常: {e}")
                        import traceback
                        traceback.print_exc()
                        raise

                # 取出当前要执行的动作
                action = self._action_chunk[self._actions_from_chunk_completed]
                print(f"[DEBUG] 当前要执行的动作维度: {action.shape}")
                # print(f"[DEBUG] 当前要执行的动作: {action}")
                self._actions_from_chunk_completed += 1

                self._publish_action(action)
                rate.sleep()
        except Exception as e:
            rospy.logerr(f"控制循环异常终止: {e}")
            rospy.signal_shutdown("Controller loop crashed")

    # ----------------------- 辅助函数 ----------------------- #
    def _publish_action(self, action: np.ndarray) -> None:
        """假设 action 长度为 14，其中前 7 维是左臂，后 7 维是右臂。"""
        # 如果是 (1, 14) 或 (14,) 都视为合法
        if action.ndim == 2 and action.shape[0] == 1:
            action = action[0]  # 压平为 (14,)

        if action.shape[0] != 14:
            rospy.logwarn(f"Action 长度应为 14，但得到 {action.shape[0]}，跳过发布")
            return

        msg_left = JointState()
        msg_left.position = action[:7].tolist()
        self._left_cmd_pub.publish(msg_left)

        msg_right = JointState()
        msg_right.position = action[7:14].tolist()
        self._right_cmd_pub.publish(msg_right)

    def _reset_to_home_position(self):
        """将机器人移动到初始位置"""
        left_home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]   # 左臂初始位置
        right_home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]   # 右臂初始位置
        
        # 发布初始位置命令
        msg_left = JointState()
        msg_left.position = left_home
        self._left_cmd_pub.publish(msg_left)
        
        msg_right = JointState()
        msg_right.position = right_home
        self._right_cmd_pub.publish(msg_right)
        
        rospy.sleep(2.0)  # 等待机器人移动到位置

    def _ready(self) -> bool:
        return (
            self._front_image is not None
            and self._right_image is not None
            and self._joint_left is not None
            and self._joint_right is not None
        )


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    # 移除 host 和 port 参数，添加 checkpoint_dir 参数
    parser.add_argument("--checkpoint_dir", type=str, 
                       default="checkpoints/pi0_aloha_lora_finetune_peg/peg_finetune_lora_full/19999",
                       help="模型检查点目录路径")
    parser.add_argument("--prompt", type=str, default="pick up the red block", help="语言指令")

    # ------------ Topic 配置，可根据实际情况调整 ------------- #
    parser.add_argument('--puppet_arm_left_cmd_topic', type=str, default='/left_arm/joint_ctrl_single')
    parser.add_argument('--puppet_arm_right_cmd_topic', type=str, default='/right_arm/joint_ctrl_single')
    parser.add_argument('--puppet_arm_left_topic', type=str, default='/left_arm/joint_states_single')
    parser.add_argument('--puppet_arm_right_topic', type=str, default='/right_arm/joint_states_single')
    parser.add_argument('--img_front_topic', type=str, default='/camera_f/color/image_raw')
    parser.add_argument('--img_left_topic', type=str, default='/camera_l/color/image_raw')
    parser.add_argument('--img_right_topic', type=str, default='/camera_r/color/image_raw')
    parser.add_argument('--action_horizon', type=int, default=50, help='每次推理返回的动作序列长度')
    parser.add_argument('--open_loop_horizon', type=int, default=25, help='在本地执行多少步后重新推理一次 (应 < action_horizon)')

    args = parser.parse_args()

    # 检查模型文件是否存在
    import os
    if not os.path.exists(args.checkpoint_dir):
        print(f"[Piper] 错误：模型检查点目录不存在: {args.checkpoint_dir}")
        print("[Piper] 请确保模型文件已下载到正确位置")
        return

    # print(f"[Piper] 使用本地模型: {args.checkpoint_dir}")
    # print(f"[Piper] 任务指令: {args.prompt}")

    rospy.init_node("piper_openpi_client", anonymous=True)
    print(f"[Piper] 使用本地模型: {args.checkpoint_dir}")
    print(f"[Piper] 任务指令: {args.prompt}")
    print(f"[Piper] 动作序列长度: {args.action_horizon}")
    print(f"[Piper] 重新推理间隔: {args.open_loop_horizon}")

    controller = PiperController(
        checkpoint_dir=args.checkpoint_dir,
        prompt=args.prompt,
        img_front_topic=args.img_front_topic,
        img_left_topic=args.img_left_topic,
        img_right_topic=args.img_right_topic,
        arm_left_state_topic=args.puppet_arm_left_topic,
        arm_right_state_topic=args.puppet_arm_right_topic,
        arm_left_cmd_topic=args.puppet_arm_left_cmd_topic,
        arm_right_cmd_topic=args.puppet_arm_right_cmd_topic,
        action_horizon=args.action_horizon,
        open_loop_horizon=args.open_loop_horizon,
    )

    # 使用线程防止阻塞 Ctrl+C
    threading.Thread(target=controller.run, daemon=True).start()
    rospy.spin()


if __name__ == "__main__":
    main() 