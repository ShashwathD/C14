from typing import Optional, Tuple

import numpy as np

from dreamer.robot.types import RobotObservation


class FrameSynchronizer:
    """Placeholder synchronizer for RGB and depth frames."""

    def sync(
        self, rgb: np.ndarray, depth: np.ndarray, timestamp: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        return rgb, depth, timestamp


class RGBDNormalizer:
    def __init__(self, depth_min_m: float = 0.1, depth_max_m: float = 4.0):
        self.depth_min_m = float(depth_min_m)
        self.depth_max_m = float(depth_max_m)

    def normalize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        return rgb.astype(np.float32) / 255.0 - 0.5

    def normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        depth = np.nan_to_num(
            depth,
            nan=self.depth_max_m,
            posinf=self.depth_max_m,
            neginf=self.depth_min_m,
        )
        depth = np.clip(depth, self.depth_min_m, self.depth_max_m)
        depth = (depth - self.depth_min_m) / max(self.depth_max_m - self.depth_min_m, 1e-6)
        return depth - 0.5


def stack_rgbd(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    if depth.ndim == 2:
        depth = np.expand_dims(depth, axis=-1)
    rgbd = np.concatenate([rgb, depth], axis=-1)
    return np.transpose(rgbd, (2, 0, 1)).astype(np.float32)


def preprocess_robot_observation(
    observation: RobotObservation,
    normalizer: Optional[RGBDNormalizer] = None,
) -> np.ndarray:
    normalizer = normalizer or RGBDNormalizer()
    rgb = normalizer.normalize_rgb(observation.rgb)
    depth = normalizer.normalize_depth(observation.depth)
    return stack_rgbd(rgb, depth)
