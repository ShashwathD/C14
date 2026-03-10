import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from dreamer.envs.wrappers import (
    FrameSynchronizer,
    RGBDNormalizer,
    preprocess_robot_observation,
)
from dreamer.robot.fallback import DeterministicFallbackPolicy
from dreamer.robot.types import ControlAction, RobotObservation


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(obj, key, default)


@dataclass
class _SimpleSpace:
    shape: Tuple[int, ...]


class MockRGBDCamera:
    def __init__(self, width: int, height: int, fps: int = 15):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.t = 0

    def get_frame(self):
        self.t += 1
        x = np.linspace(0, 1, self.width, dtype=np.float32)
        y = np.linspace(0, 1, self.height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(255 * xv, 0, 255).astype(np.uint8)
        rgb[..., 1] = np.clip(255 * yv, 0, 255).astype(np.uint8)

        # Synthetic depth with a moving obstacle.
        depth = 2.5 * np.ones((self.height, self.width), dtype=np.float32)
        center = int((self.t * 3) % max(self.width, 1))
        left = max(center - self.width // 10, 0)
        right = min(center + self.width // 10, self.width)
        depth[:, left:right] = 0.25

        timestamp = time.time()
        odometry = (0.01 * self.t, 0.0, 0.0)
        done = False
        return rgb, depth, timestamp, odometry, done

    def close(self):
        return None


class OAKDLiteCamera:
    """DepthAI-backed camera adapter."""

    def __init__(self, width: int, height: int, fps: int = 15):
        try:
            import depthai as dai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "depthai is required for camera_source='oakd'"
            ) from exc

        self.dai = dai
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)

        pipeline = dai.Pipeline()

        color = pipeline.createColorCamera()
        color.setPreviewSize(self.width, self.height)
        color.setInterleaved(False)
        color.setFps(self.fps)

        mono_left = pipeline.createMonoCamera()
        mono_right = pipeline.createMonoCamera()
        stereo = pipeline.createStereoDepth()

        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        color.preview.link(xout_rgb.input)

        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        self.device = dai.Device(pipeline)
        self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    def get_frame(self):  # pragma: no cover - hardware path
        rgb_in = self.rgb_queue.get()
        depth_in = self.depth_queue.get()

        rgb = rgb_in.getCvFrame()
        depth = depth_in.getFrame().astype(np.float32) / 1000.0

        timestamp = time.time()
        odometry = (0.0, 0.0, 0.0)
        done = False
        return rgb, depth, timestamp, odometry, done

    def close(self):  # pragma: no cover - hardware path
        self.device.close()


class MockMotorController:
    def __init__(self):
        self.last_action = ControlAction(0.0, 0.0)

    def send(self, action: ControlAction):
        self.last_action = action

    def stop(self):
        self.last_action = ControlAction(0.0, 0.0)

    def close(self):
        return None


class SerialMotorController:
    def __init__(self, port: str, baudrate: int = 115200):
        try:
            import serial  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("pyserial is required for serial motor control") from exc

        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=0.1)

    def send(self, action: ControlAction):  # pragma: no cover - hardware path
        payload = f"{action.linear_vel:.4f},{action.angular_vel:.4f}\n"
        self.serial.write(payload.encode("ascii"))

    def stop(self):  # pragma: no cover - hardware path
        self.send(ControlAction(0.0, 0.0))

    def close(self):  # pragma: no cover - hardware path
        self.serial.close()


class RobotEnv:
    """Runtime env for robot deployment with OAK-D + motor controller abstractions."""

    def __init__(self, config, camera=None, motor_controller=None):
        env_cfg = _cfg_get(config, "environment", None)

        self.width = int(_cfg_get(env_cfg, "width", 64))
        self.height = int(_cfg_get(env_cfg, "height", 64))
        self.max_steps = int(_cfg_get(env_cfg, "max_steps", 300))
        self.camera_source = _cfg_get(env_cfg, "camera_source", "mock")

        self.linear_limit = float(_cfg_get(env_cfg, "action_linear_limit", 0.35))
        self.angular_limit = float(_cfg_get(env_cfg, "action_angular_limit", 1.20))

        depth_min = float(_cfg_get(env_cfg, "depth_min_m", 0.1))
        depth_max = float(_cfg_get(env_cfg, "depth_max_m", 4.0))
        fallback_forward = float(_cfg_get(env_cfg, "fallback_forward_speed", 0.10))
        fallback_turn = float(_cfg_get(env_cfg, "fallback_turn_speed", 0.60))
        fallback_min_depth = float(_cfg_get(env_cfg, "fallback_min_depth_m", 0.35))

        self.normalizer = RGBDNormalizer(depth_min_m=depth_min, depth_max_m=depth_max)
        self.frame_sync = FrameSynchronizer()
        self.fallback_policy = DeterministicFallbackPolicy(
            forward_speed=fallback_forward,
            turn_speed=fallback_turn,
            min_depth_m=fallback_min_depth,
        )

        if camera is not None:
            self.camera = camera
        elif self.camera_source == "oakd":
            self.camera = OAKDLiteCamera(
                width=self.width,
                height=self.height,
                fps=int(_cfg_get(env_cfg, "camera_fps", 15)),
            )
        else:
            self.camera = MockRGBDCamera(
                width=self.width,
                height=self.height,
                fps=int(_cfg_get(env_cfg, "camera_fps", 15)),
            )

        if motor_controller is not None:
            self.motor_controller = motor_controller
        else:
            port = _cfg_get(env_cfg, "serial_port", "")
            self.motor_controller = SerialMotorController(port) if port else MockMotorController()

        self.observation_space = _SimpleSpace((4, self.height, self.width))
        self.action_space = _SimpleSpace((2,))

        self.step_count = 0

    def preprocess_observation(self, observation: RobotObservation) -> np.ndarray:
        return preprocess_robot_observation(observation, normalizer=self.normalizer)

    def observe(self) -> RobotObservation:
        rgb, depth, timestamp, odometry, done = self.camera.get_frame()
        rgb, depth, timestamp = self.frame_sync.sync(rgb, depth, timestamp)
        return RobotObservation(
            rgb=rgb,
            depth=depth,
            timestamp=timestamp,
            odometry=odometry,
            done=done,
        )

    def reset(self) -> RobotObservation:
        self.step_count = 0
        self.motor_controller.stop()
        return self.observe()

    def clamp_action(self, action: ControlAction) -> ControlAction:
        return ControlAction(
            linear_vel=float(np.clip(action.linear_vel, -self.linear_limit, self.linear_limit)),
            angular_vel=float(np.clip(action.angular_vel, -self.angular_limit, self.angular_limit)),
        )

    def tensor_to_action(self, action_tensor) -> ControlAction:
        action = np.asarray(action_tensor.detach().cpu().numpy()).reshape(-1)
        if action.size < 2:
            raise ValueError("Action tensor must contain at least 2 values")
        return self.clamp_action(
            ControlAction(
                linear_vel=float(np.clip(action[0], -1.0, 1.0)) * self.linear_limit,
                angular_vel=float(np.clip(action[1], -1.0, 1.0)) * self.angular_limit,
            )
        )

    def action_to_tensor(self, action: ControlAction, device):
        action = self.clamp_action(action)
        normalized = np.asarray(
            [
                action.linear_vel / max(self.linear_limit, 1e-6),
                action.angular_vel / max(self.angular_limit, 1e-6),
            ],
            dtype=np.float32,
        )
        import torch

        return torch.from_numpy(normalized).to(device).reshape(1, -1)

    def fallback_action(self, observation: RobotObservation) -> ControlAction:
        return self.clamp_action(self.fallback_policy.select_action(observation))

    def step(self, action):
        if isinstance(action, ControlAction):
            control = self.clamp_action(action)
        else:
            array_action = np.asarray(action, dtype=np.float32).reshape(-1)
            control = self.clamp_action(
                ControlAction(
                    linear_vel=float(array_action[0]),
                    angular_vel=float(array_action[1]),
                )
            )

        self.motor_controller.send(control)
        observation = self.observe()
        self.step_count += 1
        done = bool(observation.done or self.step_count >= self.max_steps)
        observation.done = done

        reward = 0.0
        info = {
            "timestamp": observation.timestamp,
            "odometry": observation.odometry,
            "action": control,
        }
        return observation, reward, done, info

    def close(self):
        self.motor_controller.stop()
        self.motor_controller.close()
        self.camera.close()


def make_robot_env(config):
    return RobotEnv(config)


def get_env_infos(env):
    obs_shape = env.observation_space.shape
    discrete_action_bool = False
    action_size = env.action_space.shape[0]
    return obs_shape, discrete_action_bool, action_size
