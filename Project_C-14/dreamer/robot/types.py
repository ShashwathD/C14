from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RobotObservation:
    rgb: np.ndarray
    depth: np.ndarray
    timestamp: float
    odometry: Tuple[float, float, float]
    done: bool = False


@dataclass
class ControlAction:
    linear_vel: float
    angular_vel: float

    def to_numpy(self) -> np.ndarray:
        return np.asarray([self.linear_vel, self.angular_vel], dtype=np.float32)


@dataclass
class FallbackDecision:
    use_fallback: bool
    mse: float
    threshold: float
    high_count: int
    low_count: int


@dataclass
class GoalSpec:
    command_id: int
    command_text: str
    goal_vec: np.ndarray
